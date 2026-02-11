"""
Parity test for gemm_a16w16_atomic HipKittens kernel.

Operation: C = A @ B^T (with atomic adds for split-K reduction)
  A: (M, K) bf16
  B: (N, K) bf16
  C: (M, N) fp32 (atomicAdd requires fp32)

The atomic variant partitions the K dimension across multiple thread blocks.
Each block computes a partial sum and atomically adds to the output.
No separate reduce kernel is needed.

Test sizes: 128x128, 512x512, 2048x2048
"""

import subprocess
import sys
import os


def reference_gemm_a16w16_atomic(M, N, K, device="cpu"):
    """Compute reference C = A @ B^T in float32."""
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    C_ref = (A.float() @ B.float().T)  # Keep in fp32 for atomic variant
    return A, B, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a16w16_atomic as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available. Run 'make' first.")
        return False

    test_sizes = [
        (128, 128, 128),
        (512, 512, 512),
        (2048, 2048, 2048),
    ]

    all_passed = True
    for M, N, K in test_sizes:
        A, B, C_ref = reference_gemm_a16w16_atomic(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

        kernel_mod.dispatch(A, B, C_out)
        torch.cuda.synchronize()

        max_diff = (C_out - C_ref).abs().max().item()
        rel_err = max_diff / (C_ref.abs().max().item() + 1e-8)
        passed = rel_err < 0.05  # 5% tolerance for atomic adds (non-deterministic)

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
        (128, 128, 128),
        (512, 512, 512),
        (2048, 2048, 2048),
    ]

    for M, N, K in test_sizes:
        A, B, C_ref = reference_gemm_a16w16_atomic(M, N, K)
        assert C_ref.shape == (M, N)
        assert C_ref.dtype == torch.float32
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}")

    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], C: fp32[M,N])")
    print("  C is zero-initialized, results atomically added")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a16w16_atomic Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_a16w16_atomic (Atomic BF16 GEMM)")
        print("Operation: C = A @ B^T with atomic K-split reduction")
        print("Input A: bf16 (M, K), Input B: bf16 (N, K)")
        print("Output C: fp32 (M, N)")
