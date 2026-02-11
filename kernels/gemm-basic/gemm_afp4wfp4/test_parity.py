"""
Parity test for gemm_afp4wfp4 HipKittens kernel.

Operation: C = dequant(A_fp4) @ dequant(B_fp4)^T
  A: (M, K) bf16 (pre-dequantized FP4 activations)
  B: (N, K) bf16 (pre-dequantized FP4 weights)
  C: (M, N) bf16

Both A and B are in FP4 (e2m1 mxfp4) format with e8m0 block scales.
Host pre-dequantizes both to bf16 with scales folded in.
This is the most aggressive quantization variant.

FP4 (e2m1) values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} x {+,-}
Each group of 32 elements shares one e8m0 scale (power-of-2 multiplier).
"""

import sys
import os


def reference_gemm_afp4wfp4(M, N, K, device="cpu"):
    """Compute reference C = A_bf16 @ B_bf16^T (both pre-dequantized)."""
    import torch
    torch.manual_seed(42)
    # FP4 activations pre-dequantized (small range)
    A = (torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5).clamp(-6, 6)
    # FP4 weights pre-dequantized (small range)
    B = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5).clamp(-6, 6)

    # Reference: standard matmul (scales already folded in)
    C_ref = (A.float() @ B.float().T).bfloat16()
    return A, B, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_afp4wfp4 as kernel_mod
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
        A, B, C_ref = reference_gemm_afp4wfp4(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        kernel_mod.dispatch(A, B, C_out)
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
        A, B, C_ref = reference_gemm_afp4wfp4(M, N, K)
        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.bfloat16
        assert B.dtype == torch.bfloat16
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], C: bf16[M,N])")
    print("  A is FP4 activations pre-dequantized to bf16 (scales folded in)")
    print("  B is FP4 weights pre-dequantized to bf16 (scales folded in)")
    print("  Standard bf16 MMA pipeline, no runtime dequantization")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_afp4wfp4 Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_afp4wfp4 (FP4 x FP4 GEMM)")
        print("Operation: C = dequant(A_fp4) @ dequant(B_fp4)^T")
        print("Input A: fp4 (M, K) -> bf16 via host dequant")
        print("Input B: fp4 (N, K) -> bf16 via host dequant")
        print("Output C: bf16 (M, N)")
        print("Both inputs pre-dequantized with scales folded in")
