"""
Parity test for gemm_a8w8_blockscale HipKittens kernel.

Operation: Block-scaled INT8 GEMM with per-block dequantization.
  For each K-block: C += (A_block @ B_block^T) * a_block_scale * b_block_scale

  A: (M, K) int8
  B: (N, K) int8 (stored row-major, transposed via mma_ABt)
  a_scale: (M, K // group_k) float32 -- per-row, per-K-group scale for A
  b_scale: (K // group_k, N // group_n) float32 -- per-K-group, per-N-group scale for B
  C: (M, N) bf16

The block-scale approach applies dequantization scales at each K-tile iteration
rather than once at the end. This allows finer-grained quantization where
different blocks of the K dimension can have different scale factors.

Typical parameters: group_k=128, group_n=1 (i.e., b_scale has shape (K//128, N)).

Triton reference supports split-K with a separate reduce kernel.
The HipKittens kernel follows the same block-scale accumulation pattern.

Test sizes: 128x128x128, 512x512x512, 2048x2048x2048
Tolerance: rtol=5% for bf16 output (block-scale accumulation adds rounding)
"""

import sys
import os


# Default block-scale group sizes matching the Triton reference
GROUP_K = 128
GROUP_N = 1


def reference_gemm_a8w8_blockscale(M, N, K, group_k=GROUP_K, group_n=GROUP_N,
                                    device="cpu"):
    """Compute reference block-scaled INT8 GEMM.

    For each K-group, accumulate:
        C += (A_block @ B_block^T) * a_scale_block * b_scale_block
    where scales are indexed per K-group.
    """
    import torch
    torch.manual_seed(42)
    # Random int8 inputs
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

    num_k_groups = (K + group_k - 1) // group_k
    num_n_groups = (N + group_n - 1) // group_n

    # Per-block scales
    a_scale = torch.rand(M, num_k_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    b_scale = torch.rand(num_k_groups, num_n_groups, dtype=torch.float32, device=device) * 0.1 + 0.01

    # Compute reference with per-block scaling
    C_ref = torch.zeros(M, N, dtype=torch.float32, device=device)
    for kg in range(num_k_groups):
        k_start = kg * group_k
        k_end = min(k_start + group_k, K)
        A_block = A[:, k_start:k_end].float()  # (M, block_k)
        B_block = B[:, k_start:k_end].float()  # (N, block_k)

        # Partial matmul for this K-block
        partial = A_block @ B_block.T  # (M, N)

        # Apply block scales: a_scale[:, kg] is (M,), b_scale[kg, :] is (num_n_groups,)
        a_s = a_scale[:, kg]  # (M,)

        # Expand b_scale to full N dimension (group_n elements per group)
        b_s_grouped = b_scale[kg, :]  # (num_n_groups,)
        b_s = b_s_grouped.repeat_interleave(group_n)[:N]  # (N,)

        C_ref += partial * a_s[:, None] * b_s[None, :]

    C_ref = C_ref.bfloat16()
    return A, B, a_scale, b_scale, C_ref


def test_with_hardware():
    """Full parity test when HIP hardware is available."""
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a8w8_blockscale as kernel_mod
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
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8_blockscale(
            M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Cast int8 inputs to bf16 for the kernel
        A_bf16 = A.to(torch.bfloat16)
        B_bf16 = B.to(torch.bfloat16)
        kernel_mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C_out)
        torch.cuda.synchronize()

        max_diff = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
        passed = rel_err < 0.05  # 5% tolerance for block-scale accumulation

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
        A, B, a_scale, b_scale, C_ref = reference_gemm_a8w8_blockscale(M, N, K)
        num_k_groups = (K + GROUP_K - 1) // GROUP_K
        num_n_groups = (N + GROUP_N - 1) // GROUP_N

        assert C_ref.shape == (M, N), f"Shape mismatch: {C_ref.shape} != ({M}, {N})"
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.int8
        assert B.dtype == torch.int8
        assert a_scale.shape == (M, num_k_groups), \
            f"a_scale shape mismatch: {a_scale.shape} != ({M}, {num_k_groups})"
        assert b_scale.shape == (num_k_groups, num_n_groups), \
            f"b_scale shape mismatch: {b_scale.shape} != ({num_k_groups}, {num_n_groups})"
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, "
              f"a_scale={a_scale.shape}, b_scale={b_scale.shape}, "
              f"range=[{C_ref.float().min():.4f}, {C_ref.float().max():.4f}]")

    print(f"\nBlock-scale parameters: group_k={GROUP_K}, group_n={GROUP_N}")
    print("\nExpected compilation command:")
    print("  make GPU_TARGET=CDNA4")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], a_scale: fp32[M,K//group_k],")
    print("           b_scale: fp32[K//group_k,N//group_n], C: bf16[M,N])")
    print("  A and B are int8 inputs cast to bf16 before dispatch")
    print("  Scales applied per K-block inside the accumulation loop")
    print("  Supports split-K with separate reduce kernel")


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a8w8_blockscale Parity Test (with hardware) ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("PyTorch not available, showing kernel spec only.")
        print("\nKernel: gemm_a8w8_blockscale (Block-scaled INT8 GEMM)")
        print("Operation: C += (A_block @ B_block^T) * a_block_scale * b_block_scale")
        print("Input A: int8 (M, K) -- cast to bf16 for MMA")
        print("Input B: int8 (N, K) -- cast to bf16 for MMA")
        print(f"Scale a_scale: fp32 (M, K//{GROUP_K})")
        print(f"Scale b_scale: fp32 (K//{GROUP_K}, N//{GROUP_N})")
        print("Output C: bf16 (M, N)")
        print("Accumulation: fp32, with per-block scale application")
        print(f"Block-scale groups: group_k={GROUP_K}, group_n={GROUP_N}")
