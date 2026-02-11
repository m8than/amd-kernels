"""
Parity test for HSTU Attention HipKittens kernel.
Tests the HSTU (Hierarchical Sequential Transduction Unit) attention forward pass.

HSTU uses SiLU activation instead of softmax:
  attn_weights = SiLU(Q @ K^T * alpha) / MAX_SEQ_LEN
  O = attn_weights @ V

With masking that supports:
  - Causal masking (future tokens masked)
  - Self-exclusion (diagonal masked out)
  - Multiple targets (last n_targets tokens share position)
  - Contextual sequence length
  - Maximum attention length (sliding window)

Layout: Q,K (total_tokens, H, D_Q), V,O (total_tokens, H, D_V)
Variable-length via seq_offsets: (batch+1,) int32

Usage:
    python test_parity.py                 # Full test (requires HIP GPU)
    python test_parity.py --compile-only  # Just print compile command
    python test_parity.py --shapes-only   # Just print expected shapes
"""

import argparse
import math
import sys


def get_test_config():
    return {
        "num_seqs": 3,
        "seq_lens": [64, 128, 96],
        "H": 16,
        "D_Q": 128,
        "D_V": 128,
        "alpha": 0.125,
        "is_causal": True,
        "has_multiple_targets": False,
        "has_contextual_seq_len": False,
        "has_max_attn_len": False,
        "dtype": "bfloat16",
    }


def print_shapes(cfg):
    total_tokens = sum(cfg["seq_lens"])
    H = cfg["H"]
    D_Q, D_V = cfg["D_Q"], cfg["D_V"]
    print("Expected tensor shapes and dtypes:")
    print(f"  Q:            ({total_tokens}, {H}, {D_Q})   bf16")
    print(f"  K:            ({total_tokens}, {H}, {D_Q})   bf16")
    print(f"  V:            ({total_tokens}, {H}, {D_V})   bf16")
    print(f"  O:            ({total_tokens}, {H}, {D_V})   bf16")
    print(f"  seq_offsets:  ({cfg['num_seqs']+1},)          int32")
    print(f"  Alpha: {cfg['alpha']}")
    print(f"  MAX_SEQ_LEN: {max(cfg['seq_lens'])}")


def print_compile_cmd(cfg):
    H = cfg["H"]
    D_Q, D_V = cfg["D_Q"], cfg["D_V"]
    causal = 1 if cfg["is_causal"] else 0
    mt = 1 if cfg["has_multiple_targets"] else 0
    csl = 1 if cfg["has_contextual_seq_len"] else 0
    mal = 1 if cfg["has_max_attn_len"] else 0
    print("Compile command:")
    print(f"  make ATTN_H={H} BLOCK_D_Q={D_Q} BLOCK_D_V={D_V} "
          f"IS_CAUSAL={causal} HAS_MULTIPLE_TARGETS={mt} "
          f"HAS_CONTEXTUAL_SEQ_LEN={csl} HAS_MAX_ATTN_LEN={mal}")


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    import torch
    return x * torch.sigmoid(x)


def robustness_check(ref, pred):
    import torch
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = diff > (0.001 + 0.05 * denom)
    error_count = mask.sum().item()
    numel = ref.numel()
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), pred.flatten(), dim=0
    ).item()
    return {
        "max_abs": diff.max().item(),
        "rel_error": error_count / numel,
        "l2_error": l2_error,
        "cosine_sim": cos,
        "errors": error_count,
        "total": numel,
    }


def run_parity_test():
    import torch

    cfg = get_test_config()
    H = cfg["H"]
    D_Q, D_V = cfg["D_Q"], cfg["D_V"]
    alpha = cfg["alpha"]
    MAX_SEQ_LEN = max(cfg["seq_lens"])

    torch.manual_seed(42)

    total_tokens = sum(cfg["seq_lens"])

    # Create tensors
    q = torch.randn(total_tokens, H, D_Q, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_tokens, H, D_Q, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total_tokens, H, D_V, dtype=torch.bfloat16, device="cuda")

    # Build seq_offsets
    offsets = [0]
    for sl in cfg["seq_lens"]:
        offsets.append(offsets[-1] + sl)
    seq_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    # num_targets (not used in basic test, but needed for interface)
    num_targets = torch.zeros(cfg["num_seqs"], dtype=torch.int32, device="cuda")

    # PyTorch reference: HSTU attention with SiLU
    ref_out = torch.zeros(total_tokens, H, D_V, dtype=torch.float32, device="cuda")

    for seq in range(cfg["num_seqs"]):
        sl = cfg["seq_lens"][seq]
        s = offsets[seq]
        e = offsets[seq + 1]

        for h in range(H):
            # Q: (sl, D_Q), K: (sl, D_Q), V: (sl, D_V)
            q_h = q[s:e, h, :].float()   # (sl, D_Q)
            k_h = k[s:e, h, :].float()   # (sl, D_Q)
            v_h = v[s:e, h, :].float()   # (sl, D_V)

            # QK^T * alpha
            qk = torch.matmul(q_h, k_h.t()) * alpha  # (sl, sl)

            # Apply SiLU: silu(x) / MAX_SEQ_LEN
            attn = silu(qk) / MAX_SEQ_LEN

            # Apply masking:
            # 1. Self-exclusion: mask out diagonal (m == n)
            diag_mask = torch.eye(sl, dtype=torch.bool, device="cuda")
            # 2. Causal: mask out future (m < n for non-contextual)
            #    In Triton: invalid = (offs_m == offs_n) | (offs_m - offs_n > 0)
            #    Wait - the mask logic is: silu = where(invalid_mask, silu, 0)
            #    So invalid_mask=True means KEEP, invalid_mask=False means ZERO.
            #    Actually re-reading: `invalid_mask = offs_m[:, None] == offs_n[None, :]`
            #    then `invalid_mask = invalid_mask | (offs_m_minus_n > 0)`
            #    then `silu = tl.where(invalid_mask, silu, 0)` — keeps where invalid_mask is True
            #
            #    Wait, that's confusing naming. Let's trace:
            #    - invalid_mask starts as (m == n) — diagonal
            #    - For causal: offs_m_minus_n = offs_m - offs_n, then invalid |= (m-n > 0)
            #      This means: m > n is "invalid". So invalid_mask is True when m >= n (diag + lower tri)
            #    - But then `tl.where(invalid_mask, silu, 0)` KEEPS silu where invalid is True
            #    - So it keeps lower triangle + diagonal, zeros upper triangle
            #
            #    BUT the diagonal self-exclusion was: invalid_mask = (m == n)
            #    Then OR'd with (m - n > 0): that's m > n
            #    Result: invalid_mask = (m == n) | (m > n) = (m >= n) = lower triangle + diag
            #    Then where(invalid_mask, silu, 0) KEEPS lower+diag, zeros upper
            #
            #    Hmm, but that doesn't do self-exclusion at all. Let me re-read...
            #    Actually the variable is named "invalid_mask" but used as the KEEP mask
            #    in tl.where. So it's really a "valid" mask despite the name.
            #    The diagonal IS included in the valid region (m==n is True → kept).
            #
            #    Wait, re-reading more carefully:
            #    `invalid_mask = offs_m[:, None] == offs_n[None, :]`  → True on diagonal
            #    For CAUSAL: `invalid_mask = invalid_mask | (offs_m_minus_n > 0)`
            #      → True on diagonal AND where m > n (strict lower triangle)
            #      → Together: m >= n → lower triangle including diagonal
            #    `silu = tl.where(invalid_mask, silu, 0)` → ZERO where NOT invalid
            #      → This KEEPS the lower triangle+diagonal, ZEROS upper triangle
            #
            #    So despite the name "invalid_mask", this is used as a KEEP/VALID mask.
            #    The self-exclusion diagonal is NOT excluded — it's included!
            #
            #    Actually wait. Let me look again at the non-causal case:
            #    For non-causal: offs_m_minus_n = abs(offs_m - offs_n)
            #    invalid_mask = (m==n) | (abs(m-n) > 0) = (m==n) | (m!=n) = ALL True
            #    So non-causal keeps everything. That makes sense.
            #
            #    For causal: keeps m >= n. The "self-exclusion" comes from HAS_MULTIPLE_TARGETS
            #    and contextual_seq_len adjustments, not from the basic diagonal.
            #
            # Summary for basic causal (no multiple_targets, no contextual_seq_len):
            #   valid_mask = (m >= n) → standard causal (lower triangle including diagonal)

            if cfg["is_causal"]:
                # Causal mask: keep positions where m >= n
                causal_mask = torch.tril(torch.ones(sl, sl, device="cuda", dtype=torch.bool))
                attn = attn * causal_mask.float()
            # Note: in basic mode (no multiple targets), diagonal is NOT excluded

            # O = attn @ V
            ref_out[s:e, h, :] = torch.matmul(attn, v_h)

    ref_out = ref_out.to(torch.bfloat16)

    try:
        import tk_hstu_attn
    except ImportError:
        print("FAIL: tk_hstu_attn not compiled. Run 'make' first.")
        print_compile_cmd(cfg)
        return False

    out = torch.zeros(total_tokens, H, D_V, dtype=torch.bfloat16, device="cuda")
    tk_hstu_attn.dispatch(q, k, v, out, seq_offsets, num_targets)
    torch.cuda.synchronize()

    result = robustness_check(ref_out, out)
    passed = result["cosine_sim"] > 0.999 and result["rel_error"] < 0.01

    print(f"HSTU Attention Parity Test:")
    print(f"  seq_lens={cfg['seq_lens']} H={H} D_Q={D_Q} D_V={D_V}")
    print(f"  alpha={alpha} MAX_SEQ_LEN={MAX_SEQ_LEN} causal={cfg['is_causal']}")
    print(f"  max_abs_diff:  {result['max_abs']:.6f}")
    print(f"  rel_error:     {result['rel_error']:.6f}")
    print(f"  l2_error:      {result['l2_error']:.6f}")
    print(f"  cosine_sim:    {result['cosine_sim']:.6f}")
    print(f"  errors:        {result['errors']}/{result['total']}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="HSTU Attention parity test")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--shapes-only", action="store_true")
    args = parser.parse_args()

    cfg = get_test_config()
    if args.compile_only:
        print_compile_cmd(cfg)
        return
    if args.shapes_only:
        print_shapes(cfg)
        return

    passed = run_parity_test()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
