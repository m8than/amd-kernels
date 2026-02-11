"""
Parity test for gemm_a8wfp4 HipKittens kernel.

Operation: C = dequant(A_int8) @ dequant(B_fp4)^T
  A: (M, K) int8 -> cast to bf16 by host
  B: (N, K) bf16 (pre-dequantized FP4 weights)
  a_scale: (M,) float32 -- per-row scale for A (from e8m0 format)
  C: (M, N) bf16

INT8 activations (e4m3 format) with FP4 weights (e2m1 mxfp4 format).
Both A and B are pre-dequantized to bf16 by the host.
A's per-row scale is applied after full K accumulation.
B's block scales are folded into pre-dequantized values.
"""

import sys
import os


def reference_gemm_a8wfp4(M, N, K, device="cpu"):
    """Compute reference C = (A_int8_as_bf16 @ B_fp4_as_bf16^T) * a_scale."""
    import torch
    torch.manual_seed(42)
    # INT8 activations (simulated as small bf16 values)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    # FP4 weights pre-dequantized (small range typical of FP4)
    B = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5).clamp(-6, 6)
    # Per-row A scale
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.1 + 0.01

    # Reference: matmul then apply A's per-row scale
    accumulator = A.float() @ B.float().T  # (M, N)
    C_ref = (accumulator * a_scale[:, None]).bfloat16()
    return A, B, a_scale, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a8wfp4 as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available. Run 'make' first.")
        return False

    test_sizes = [
        (128, 128, 64),
        (256, 256, 256),
        (1024, 1024, 1024),
    ]

    all_passed = True
    for M, N, K in test_sizes:
        A, B, a_scale, C_ref = reference_gemm_a8wfp4(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        kernel_mod.dispatch(A, B, a_scale, C_out)
        torch.cuda.synchronize()

        max_diff = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
        passed = rel_err < 0.02  # 2% tolerance for bf16

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M}, N={N}, K={K}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        if not passed:
            all_passed = False

    return all_passed


def test_reference_only():
    """Test reference implementation (no hardware needed)."""
    import torch
    print("Reference-only test (no HIP hardware):")

    test_sizes = [
        (128, 128, 64),
        (256, 256, 256),
        (1024, 1024, 1024),
    ]

    for M, N, K in test_sizes:
        A, B, a_scale, C_ref = reference_gemm_a8wfp4(M, N, K)
        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.bfloat16
        assert B.dtype == torch.bfloat16
        assert a_scale.shape == (M,)
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], a_scale: fp32[M], C: bf16[M,N])")
    print("  A is int8 activations pre-cast to bf16")
    print("  B is FP4 weights pre-dequantized to bf16")
    print("  a_scale applied per-row after full K accumulation")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a8wfp4 Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_a8wfp4 (INT8 x FP4 GEMM)")
        print("Operation: C = dequant(A_int8) @ dequant(B_fp4)^T")
        print("Input A: int8 (M, K) -> bf16 via host cast")
        print("Input B: fp4 (N, K) -> bf16 via host dequant")
        print("Scale a_scale: fp32 (M,) -- per-row")
        print("Output C: bf16 (M, N)")
