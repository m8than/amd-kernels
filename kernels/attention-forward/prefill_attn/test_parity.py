"""
Parity test for Prefill Attention HipKittens kernel.
Compares against PyTorch SDPA as reference for variable-length prefill.

The Triton kernel uses ragged/packed layout: Q/K/V are (total_tokens, H, D)
with B_Start_Loc and B_Seqlen arrays for batch indexing.

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
        "num_batches": 4,
        "seq_lens": [512, 1024, 768, 256],  # variable lengths
        "H": 64,
        "H_KV": 8,
        "D": 128,
        "dtype": "bfloat16",
        "causal": True,
    }


def print_shapes(cfg):
    total_tokens = sum(cfg["seq_lens"])
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    print("Expected tensor shapes and dtypes:")
    print(f"  Q:           ({total_tokens}, {H}, {D})    bf16")
    print(f"  K:           ({total_tokens}, {H_KV}, {D}) bf16")
    print(f"  V:           ({total_tokens}, {H_KV}, {D}) bf16")
    print(f"  O:           ({total_tokens}, {H}, {D})    bf16")
    print(f"  B_Start_Loc: ({cfg['num_batches']},)       int32")
    print(f"  B_Seqlen:    ({cfg['num_batches']},)       int32")
    print(f"  Seq lengths: {cfg['seq_lens']}")
    print(f"  Scale: {1.0 / math.sqrt(D):.6f}")


def print_compile_cmd(cfg):
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    max_seq = max(cfg["seq_lens"])
    causal = 1 if cfg["causal"] else 0
    print("Compile command:")
    print(f"  make ATTN_H={H} ATTN_H_KV={H_KV} ATTN_D={D} MAX_SEQ_LEN={max_seq} IS_CAUSAL={causal}")


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
    seq_lens = cfg["seq_lens"]
    total_tokens = sum(seq_lens)
    group_size = H // H_KV

    torch.manual_seed(42)

    # Create packed tensors
    q = torch.randn(total_tokens, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_tokens, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total_tokens, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # Compute start locations
    start_locs = [0]
    for s in seq_lens[:-1]:
        start_locs.append(start_locs[-1] + s)

    b_start_loc = torch.tensor(start_locs, dtype=torch.int32, device="cuda")
    b_seqlen = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    # PyTorch reference: process each batch separately
    ref_out = torch.zeros_like(q).float()
    for b in range(len(seq_lens)):
        sl = seq_lens[b]
        start = start_locs[b]
        q_b = q[start:start+sl].unsqueeze(0).permute(0, 2, 1, 3)  # (1, H, sl, D)
        k_b = k[start:start+sl, :H_KV].unsqueeze(0).permute(0, 2, 1, 3)
        v_b = v[start:start+sl, :H_KV].unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k_b.repeat_interleave(group_size, dim=1)
        v_b = v_b.repeat_interleave(group_size, dim=1)
        o_b = scaled_dot_product_attention(q_b, k_b, v_b, is_causal=cfg["causal"])
        ref_out[start:start+sl] = o_b.squeeze(0).permute(1, 0, 2).float()

    ref_out = ref_out.to(torch.bfloat16)

    # HipKittens kernel
    try:
        import tk_prefill_attn
    except ImportError:
        print("FAIL: tk_prefill_attn not compiled. Run 'make' first.")
        print_compile_cmd(cfg)
        return False

    out = torch.zeros_like(q)
    tk_prefill_attn.dispatch(q, k, v, out, b_start_loc, b_seqlen)
    torch.cuda.synchronize()

    result = robustness_check(ref_out, out)
    passed = result["cosine_sim"] > 0.999 and result["rel_error"] < 0.01

    print(f"Prefill Attention Parity Test:")
    print(f"  Config: seq_lens={seq_lens} H={H} H_KV={H_KV} D={D}")
    print(f"  max_abs_diff:  {result['max_abs']:.6f}")
    print(f"  rel_error:     {result['rel_error']:.6f}")
    print(f"  l2_error:      {result['l2_error']:.6f}")
    print(f"  cosine_sim:    {result['cosine_sim']:.6f}")
    print(f"  errors:        {result['errors']}/{result['total']}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Prefill Attention parity test")
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
