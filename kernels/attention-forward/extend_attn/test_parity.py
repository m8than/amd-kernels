"""
Parity test for Extend Attention HipKittens kernel.
Tests the two-stage extend attention: prefix (KV buffer) + extend (new tokens).

The Triton kernel handles:
  Stage 1: Attending to prefix KV cache via paged kv_indices
  Stage 2: Attending to newly extended tokens with causal masking

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
        "num_seqs": 2,
        "prefix_lens": [256, 512],    # existing KV cache lengths per sequence
        "extend_lens": [128, 64],     # new tokens being extended per sequence
        "H": 32,
        "H_KV": 8,
        "D": 128,
        "dtype": "bfloat16",
        "causal": True,
    }


def print_shapes(cfg):
    total_extend = sum(cfg["extend_lens"])
    total_kv = sum(cfg["prefix_lens"])
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    print("Expected tensor shapes and dtypes:")
    print(f"  Q_Extend:    ({total_extend}, {H}, {D})    bf16")
    print(f"  K_Extend:    ({total_extend}, {H_KV}, {D}) bf16")
    print(f"  V_Extend:    ({total_extend}, {H_KV}, {D}) bf16")
    print(f"  O_Extend:    ({total_extend}, {H}, {D})    bf16")
    print(f"  K_Buffer:    ({total_kv}, {H_KV}, {D})     bf16")
    print(f"  V_Buffer:    ({total_kv}, {H_KV}, {D})     bf16")
    print(f"  qo_indptr:   ({cfg['num_seqs']+1},)        int32")
    print(f"  kv_indptr:   ({cfg['num_seqs']+1},)        int32")
    print(f"  kv_indices:  ({total_kv},)                  int32")
    print(f"  Scale: {1.0 / math.sqrt(D):.6f}")


def print_compile_cmd(cfg):
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    causal = 1 if cfg["causal"] else 0
    print("Compile command:")
    print(f"  make ATTN_H={H} ATTN_H_KV={H_KV} ATTN_D={D} IS_CAUSAL={causal}")


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
    from torch.nn.functional import scaled_dot_product_attention

    cfg = get_test_config()
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    group_size = H // H_KV

    torch.manual_seed(42)

    # Build the extend test data
    total_extend = sum(cfg["extend_lens"])
    total_prefix = sum(cfg["prefix_lens"])

    q_extend = torch.randn(total_extend, H, D, dtype=torch.bfloat16, device="cuda")
    k_extend = torch.randn(total_extend, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_extend = torch.randn(total_extend, H_KV, D, dtype=torch.bfloat16, device="cuda")
    k_buffer = torch.randn(total_prefix, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_buffer = torch.randn(total_prefix, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # Build indptrs
    qo_starts = [0]
    kv_starts = [0]
    for i, (el, pl) in enumerate(zip(cfg["extend_lens"], cfg["prefix_lens"])):
        qo_starts.append(qo_starts[-1] + el)
        kv_starts.append(kv_starts[-1] + pl)

    qo_indptr = torch.tensor(qo_starts, dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor(kv_starts, dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(total_prefix, dtype=torch.int32, device="cuda")

    # PyTorch reference
    ref_out = torch.zeros_like(q_extend).float()
    for seq in range(cfg["num_seqs"]):
        pl = cfg["prefix_lens"][seq]
        el = cfg["extend_lens"][seq]
        es = qo_starts[seq]
        ps = kv_starts[seq]

        # Concatenate prefix + extend keys/values
        k_all = torch.cat([k_buffer[ps:ps+pl], k_extend[es:es+el]], dim=0)
        v_all = torch.cat([v_buffer[ps:ps+pl], v_extend[es:es+el]], dim=0)

        q_b = q_extend[es:es+el].unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k_all.unsqueeze(0).permute(0, 2, 1, 3)
        v_b = v_all.unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k_b.repeat_interleave(group_size, dim=1)
        v_b = v_b.repeat_interleave(group_size, dim=1)

        # For extend: causal means each new token at position i can attend to
        # all prefix tokens plus extend tokens 0..i
        # Use attn_mask for this
        total_kv = pl + el
        attn_mask = torch.zeros(el, total_kv, dtype=torch.bool, device="cuda")
        for i in range(el):
            attn_mask[i, :pl + i + 1] = True
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(-1, H, -1, -1)

        o_b = scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask)
        ref_out[es:es+el] = o_b.squeeze(0).permute(1, 0, 2).float()

    ref_out = ref_out.to(torch.bfloat16)

    try:
        import tk_extend_attn
    except ImportError:
        print("FAIL: tk_extend_attn not compiled. Run 'make' first.")
        print_compile_cmd(cfg)
        return False

    out = torch.zeros_like(q_extend)
    tk_extend_attn.dispatch(q_extend, k_extend, v_extend, out,
                            k_buffer, v_buffer,
                            qo_indptr, kv_indptr, kv_indices)
    torch.cuda.synchronize()

    result = robustness_check(ref_out, out)
    passed = result["cosine_sim"] > 0.999 and result["rel_error"] < 0.01

    print(f"Extend Attention Parity Test:")
    print(f"  prefix_lens={cfg['prefix_lens']} extend_lens={cfg['extend_lens']}")
    print(f"  max_abs_diff:  {result['max_abs']:.6f}")
    print(f"  rel_error:     {result['rel_error']:.6f}")
    print(f"  l2_error:      {result['l2_error']:.6f}")
    print(f"  cosine_sim:    {result['cosine_sim']:.6f}")
    print(f"  errors:        {result['errors']}/{result['total']}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Extend Attention parity test")
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
