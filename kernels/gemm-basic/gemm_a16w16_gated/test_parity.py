"""
Parity test for gemm_a16w16_gated HipKittens kernel.

Operation: C = (A @ B0^T) * silu(A @ B1^T)
  A: (M, K) bf16
  B: (N, K) bf16 where B = [B0; B1], B0 is first N/2 rows, B1 is last N/2 rows
  C: (M, N/2) bf16

silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

This is the gated linear unit pattern used in SwiGLU and similar architectures.
"""

import sys
import os


def reference_gemm_gated(M, N, K, device="cpu"):
    """Compute reference gated GEMM."""
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    half_n = N // 2
    B0 = B[:half_n, :]  # value path
    B1 = B[half_n:, :]  # gate path

    acc0 = A.float() @ B0.float().T  # (M, N/2)
    acc1 = A.float() @ B1.float().T  # (M, N/2)

    # silu(x) = x * sigmoid(x)
    gate = acc1 * torch.sigmoid(acc1)
    C_ref = (acc0 * gate).bfloat16()

    return A, B, C_ref


def test_with_hardware():
    """Full parity test."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a16w16_gated as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available.")
        return False

    test_sizes = [
        (128, 128, 64),   # N=128 -> each half is 64
        (256, 512, 256),  # N=512 -> each half is 256
        (1024, 2048, 1024),
    ]

    all_passed = True
    for M, N, K in test_sizes:
        A, B, C_ref = reference_gemm_gated(M, N, K, device="cuda")
        half_n = N // 2
        C_out = torch.zeros(M, half_n, dtype=torch.bfloat16, device="cuda")

        kernel_mod.dispatch(A, B, C_out)
        torch.cuda.synchronize()

        max_diff = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
        passed = rel_err < 0.05

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M}, N={N}, K={K}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        if not passed:
            all_passed = False

    return all_passed


def test_reference_only():
    """Reference test without hardware."""
    import torch
    print("Reference-only test (no HIP hardware):")

    test_sizes = [
        (128, 128, 64),
        (256, 512, 256),
        (1024, 2048, 1024),
    ]

    for M, N, K in test_sizes:
        A, B, C_ref = reference_gemm_gated(M, N, K)
        half_n = N // 2
        assert C_ref.shape == (M, half_n)
        print(f"  [OK] M={M}, N={N}, K={K}: C shape={C_ref.shape}")

    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], C: bf16[M,N/2])")
    print("  B is split: B0=B[:N/2,:] (value), B1=B[N/2:,:] (gate)")
    print("  C = (A @ B0^T) * silu(A @ B1^T)")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a16w16_gated Parity Test ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_a16w16_gated (Gated BF16 GEMM)")
        print("Operation: C = (A @ B0^T) * silu(A @ B1^T)")
