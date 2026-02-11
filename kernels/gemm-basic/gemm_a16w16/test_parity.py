"""
Parity test for gemm_a16w16 HipKittens kernel.

Operation: C = A @ B^T
  A: (M, K) bf16
  B: (N, K) bf16 (stored row-major, but semantically transposed for mma_ABt)
  C: (M, N) bf16 (accumulated in fp32, cast to bf16 on store)

The HipKittens kernel uses mma_ABt, so B is stored as (N, K) and the
multiply computes A @ B^T = (M,K) x (K,N) = (M,N).

Test sizes: small (256x256), medium (1024x1024), large (4096x4096)
Tolerance: rtol=1e-2, atol=1e-2 for bf16 accumulation
"""

import subprocess
import sys
import os

def reference_gemm_a16w16(M, N, K, device="cpu"):
    """Compute reference C = A @ B^T in float32."""
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    # Reference: cast to float for precision, then compute
    C_ref = (A.float() @ B.float().T).bfloat16()
    return A, B, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        # Try to import the compiled kernel
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a16w16 as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available. Run 'make' first.")
        return False

    test_sizes = [
        (256, 256, 256),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
    ]

    all_passed = True
    for M, N, K in test_sizes:
        # K must be divisible by BLOCK_K=64, M and N by BLOCK_M/N=256
        A, B, C_ref = reference_gemm_a16w16(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        kernel_mod.dispatch(A, B, C_out)
        torch.cuda.synchronize()

        max_diff = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
        passed = rel_err < 0.02  # 2% relative error threshold for bf16

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
        (256, 256, 256),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
    ]

    for M, N, K in test_sizes:
        A, B, C_ref = reference_gemm_a16w16(M, N, K)
        # Verify reference is reasonable
        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], C: bf16[M,N])")
    print("  Constraints: M % 256 == 0, N % 256 == 0, K % 64 == 0")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a16w16 Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("PyTorch not available, showing kernel spec only.")
        print("\nKernel: gemm_a16w16 (BF16 GEMM)")
        print("Operation: C = A @ B^T")
        print("Input A: bf16 (M, K)")
        print("Input B: bf16 (N, K)")
        print("Output C: bf16 (M, N)")
        print("Accumulation: fp32")
        print("Tile: 256x256x64, 8 warps (2x4)")
