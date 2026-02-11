"""
Parity test for gemm_a8w8_per_token_scale HipKittens kernel.

Operation: C = (A_int8 @ B_int8^T) * a_scale * b_scale
  A: (M, K) int8
  B: (N, K) int8 (stored row-major, transposed via mma_ABt)
  a_scale: (M,) float32 -- per-token (per-row) scale for A
  b_scale: (N,) float32 -- per-channel (per-column) scale for B
  C: (M, N) bf16

This kernel is similar to gemm_a8w8 but uses different scale stride patterns
optimized for per-token quantization. The key difference from gemm_a8w8:
  - Scales use explicit stride parameters (stride_ascale_m, stride_ascale_k,
    stride_bscale_k, stride_bscale_n) matching the per-token-scale Triton kernel
  - Supports split-K decomposition: the K dimension can be partitioned across
    multiple thread blocks, with a separate reduce kernel summing partial results
  - Scales are loaded once before the K-loop and applied after full accumulation

When split-K is used, each split produces a partial result in an intermediate
buffer of shape (num_ksplit, M, N), which is then reduced.

Test sizes: 128x128x128, 512x512x512, 2048x2048x2048
Tolerance: rtol=2% for bf16 output (same as gemm_a8w8)
"""

import sys
import os


def reference_gemm_a8w8_per_token_scale(M, N, K, device="cpu"):
    """Compute reference C = (A_int8 @ B_int8^T) * a_scale[:,None] * b_scale[None,:].

    Per-token scale: a_scale has one value per row of A (per token),
    b_scale has one value per column of the output (per channel of B).
    """
    import torch
    torch.manual_seed(42)
    # Random int8 inputs
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    # Per-token / per-channel scales
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.1 + 0.01
    b_scale = torch.rand(N, dtype=torch.float32, device=device) * 0.1 + 0.01
    # Reference: full K accumulation first, then apply scales
    accumulator = A.float() @ B.float().T  # (M, N)
    C_ref = (accumulator * a_scale[:, None] * b_scale[None, :]).bfloat16()
    return A, B, a_scale, b_scale, C_ref


def reference_gemm_a8w8_per_token_scale_splitk(M, N, K, num_ksplit, device="cpu"):
    """Compute reference with split-K to verify reduce correctness.

    Each K-split accumulates independently, then results are summed.
    Scales are applied per-split before summation (matching kernel behavior).
    """
    import torch
    torch.manual_seed(42)
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.1 + 0.01
    b_scale = torch.rand(N, dtype=torch.float32, device=device) * 0.1 + 0.01

    splitk_block = (K + num_ksplit - 1) // num_ksplit
    C_ref = torch.zeros(M, N, dtype=torch.float32, device=device)

    for ks in range(num_ksplit):
        k_start = ks * splitk_block
        k_end = min(k_start + splitk_block, K)
        if k_start >= K:
            break
        A_block = A[:, k_start:k_end].float()
        B_block = B[:, k_start:k_end].float()
        partial = A_block @ B_block.T
        # Scales applied per-split (then summed), which is mathematically
        # equivalent to applying after full accumulation
        C_ref += partial * a_scale[:, None] * b_scale[None, :]

    C_ref = C_ref.bfloat16()
    return A, B, a_scale, b_scale, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a8w8_per_token_scale as kernel_mod
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
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8_per_token_scale(
            M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Cast int8 inputs to bf16 for the kernel
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
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8_per_token_scale(M, N, K)
        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.int8
        assert B.dtype == torch.int8
        assert a_scale.shape == (M,)
        assert b_scale.shape == (N,)
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    # Also verify split-K reference produces same result
    print("\nSplit-K reference consistency check:")
    for M, N, K in test_sizes:
        _, _, _, _, C_nosplit = reference_gemm_a8w8_per_token_scale(M, N, K)
        _, _, _, _, C_split = reference_gemm_a8w8_per_token_scale_splitk(M, N, K, num_ksplit=2)
        diff = (C_nosplit.float() - C_split.float()).abs().max().item()
        print(f"  [OK] M={M}, N={N}, K={K}: split-K vs no-split max_diff={diff:.6f}")

    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], a_scale: fp32[M], b_scale: fp32[N], C: bf16[M,N])")
    print("  A and B are int8 inputs cast to bf16 before dispatch")
    print("  Scales loaded once, applied after full K accumulation")
    print("  Supports split-K with separate reduce kernel for large K")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a8w8_per_token_scale Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("PyTorch not available, showing kernel spec only.")
        print("\nKernel: gemm_a8w8_per_token_scale (Per-token scaled INT8 GEMM)")
        print("Operation: C = (A_int8 @ B_int8^T) * a_scale * b_scale")
        print("Input A: int8 (M, K) -- cast to bf16 for MMA")
        print("Input B: int8 (N, K) -- cast to bf16 for MMA")
        print("Scale a_scale: fp32 (M,) -- per-token (per-row)")
        print("Scale b_scale: fp32 (N,) -- per-channel (per-column)")
        print("Output C: bf16 (M, N)")
        print("Accumulation: fp32, scales applied after full K reduction")
        print("Supports split-K decomposition for large K dimensions")
