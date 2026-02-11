"""
Parity test for MHA Forward Attention HipKittens kernel.
Compares against PyTorch's scaled_dot_product_attention as reference.

Usage:
    python test_parity.py                         # Full test (requires HIP GPU)
    python test_parity.py --compile-only           # Just print compile command
    python test_parity.py --shapes-only            # Just print expected shapes
"""

import argparse
import math
import sys


def get_test_config():
    return {
        "B": 16,       # batch size
        "N": 2048,     # sequence length
        "H": 64,       # query heads
        "H_KV": 8,     # KV heads (GQA)
        "D": 128,      # head dimension
        "dtype": "bfloat16",
        "causal": False,
    }


def print_shapes(cfg):
    B, N, H, H_KV, D = cfg["B"], cfg["N"], cfg["H"], cfg["H_KV"], cfg["D"]
    print("Expected tensor shapes and dtypes:")
    print(f"  Q:   ({B}, {N}, {H}, {D})    bf16")
    print(f"  K:   ({B}, {N}, {H_KV}, {D}) bf16")
    print(f"  V:   ({B}, {N}, {H_KV}, {D}) bf16")
    print(f"  O:   ({B}, {N}, {H}, {D})    bf16")
    print(f"  LSE: ({B}, {H}, 1, {N})      fp32")
    print(f"  Scale: {1.0 / math.sqrt(D):.6f}")


def print_compile_cmd(cfg):
    B, N, H, H_KV, D = cfg["B"], cfg["N"], cfg["H"], cfg["H_KV"], cfg["D"]
    causal = 1 if cfg["causal"] else 0
    print("Compile command:")
    print(f"  make ATTN_B={B} ATTN_H={H} ATTN_H_KV={H_KV} ATTN_N={N} ATTN_D={D} IS_CAUSAL={causal}")


def robustness_check(ref, pred):
    """Compare two tensors with tolerance."""
    import torch
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = diff > (0.001 + 0.05 * denom)
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), pred.flatten(), dim=0
    ).item()
    return {
        "max_abs": diff.max().item(),
        "rel_error": rel_error,
        "l2_error": l2_error,
        "cosine_sim": cos,
        "errors": error_count,
        "total": numel,
    }


def run_parity_test():
    """Run full parity test (requires HIP GPU and compiled kernel)."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    cfg = get_test_config()
    B, N, H, H_KV, D = cfg["B"], cfg["N"], cfg["H"], cfg["H_KV"], cfg["D"]

    torch.manual_seed(42)

    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # PyTorch reference: need to expand KV heads for GQA
    # Reshape to (B, H, N, D) for SDPA
    q_ref = q.permute(0, 2, 1, 3)  # (B, H, N, D)
    k_ref = k.permute(0, 2, 1, 3)  # (B, H_KV, N, D)
    v_ref = v.permute(0, 2, 1, 3)  # (B, H_KV, N, D)

    # Expand KV for GQA
    group_size = H // H_KV
    k_ref = k_ref.repeat_interleave(group_size, dim=1)  # (B, H, N, D)
    v_ref = v_ref.repeat_interleave(group_size, dim=1)  # (B, H, N, D)

    ref_out = scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=cfg["causal"]
    )
    ref_out = ref_out.permute(0, 2, 1, 3)  # (B, N, H, D)

    # HipKittens kernel
    try:
        import tk_mha_fwd
    except ImportError:
        print("FAIL: tk_mha_fwd not compiled. Run 'make' first.")
        print_compile_cmd(cfg)
        return False

    out = torch.zeros(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
    tk_mha_fwd.dispatch(q, k, v, out, lse)
    torch.cuda.synchronize()

    # Compare
    result = robustness_check(ref_out, out)
    passed = result["cosine_sim"] > 0.999 and result["rel_error"] < 0.01

    print(f"MHA Forward Parity Test:")
    print(f"  Config: B={B} N={N} H={H} H_KV={H_KV} D={D}")
    print(f"  max_abs_diff:  {result['max_abs']:.6f}")
    print(f"  rel_error:     {result['rel_error']:.6f}")
    print(f"  l2_error:      {result['l2_error']:.6f}")
    print(f"  cosine_sim:    {result['cosine_sim']:.6f}")
    print(f"  errors:        {result['errors']}/{result['total']}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="MHA Forward parity test")
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
