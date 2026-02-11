"""
Parity test for gemm_a8w8 HipKittens kernel.

Operation: C = (A_int8 @ B_int8^T) * a_scale * b_scale
  A: (M, K) int8
  B: (N, K) int8 (stored row-major, transposed via mma_ABt)
  a_scale: (M,) float32 -- per-row scale for A
  b_scale: (N,) float32 -- per-row scale for B (applied per-column of output)
  C: (M, N) bf16

The kernel loads int8 A and B, casts to bf16 for MMA, accumulates in fp32,
then applies per-row and per-column dequantization scales before storing
the result as bf16.

Triton reference: B is (K, N) with strides stride_bk, stride_bn.
HipKittens kernel: B is stored as (N, K) and uses mma_ABt, so B^T yields (K, N).

Test sizes: 128x128x128, 512x512x512, 2048x2048x2048
Tolerance: rtol=2% for bf16 output with int8 inputs and fp32 accumulation
"""

import sys
import os


def reference_gemm_a8w8(M, N, K, device="cpu"):
    """Compute reference C = (A_int8 @ B_int8^T) * a_scale[:,None] * b_scale[None,:]."""
    import torch
    torch.manual_seed(42)
    # Random int8 inputs in [-128, 127]
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    # Per-row scales (small positive values typical of quantization)
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.1 + 0.01
    b_scale = torch.rand(N, dtype=torch.float32, device=device) * 0.1 + 0.01
    # Reference: compute in float32, apply scales, cast to bf16
    C_ref = (A.float() @ B.float().T) * a_scale[:, None] * b_scale[None, :]
    C_ref = C_ref.bfloat16()
    return A, B, a_scale, b_scale, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a8w8 as kernel_mod
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
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # The kernel expects A and B cast to bf16 on host side,
        # with separate scale tensors
        A_bf16 = A.to(torch.bfloat16)
        B_bf16 = B.to(torch.bfloat16)
        kernel_mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C_out)
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
        (128, 128, 128),
        (512, 512, 512),
        (2048, 2048, 2048),
    ]

    for M, N, K in test_sizes:
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8(M, N, K)
        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.int8
        assert B.dtype == torch.int8
        assert a_scale.shape == (M,)
        assert b_scale.shape == (N,)
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], a_scale: fp32[M], b_scale: fp32[N], C: bf16[M,N])")
    print("  A and B are int8 inputs cast to bf16 before dispatch")
    print("  Constraints: M % 128 == 0, N % 128 == 0, K % 64 == 0")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a8w8 Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("PyTorch not available, showing kernel spec only.")
        print("\nKernel: gemm_a8w8 (INT8 GEMM with per-tensor dequant)")
        print("Operation: C = (A_int8 @ B_int8^T) * a_scale * b_scale")
        print("Input A: int8 (M, K) -- cast to bf16 for MMA")
        print("Input B: int8 (N, K) -- cast to bf16 for MMA")
        print("Scale a_scale: fp32 (M,) -- per-row scale")
        print("Scale b_scale: fp32 (N,) -- per-column scale")
        print("Output C: bf16 (M, N)")
        print("Accumulation: fp32")
        print("Tile: 128x128x64, 8 warps (2x4)")
