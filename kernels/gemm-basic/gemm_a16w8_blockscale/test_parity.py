"""
Parity test for gemm_a16w8_blockscale HipKittens kernel.

Operation: C = sum_k_blocks[ (A_block @ dequant(B_block)^T) * b_block_scale ]
  A: (M, K) bf16
  B: (N, K) int8 (cast to bf16 for MMA)
  b_scale: (N, K//128) float32
  C: (M, N) bf16

Mixed precision: BF16 activations with block-scaled INT8 weights.
Only B has block scales (A is already in bf16 and needs no dequantization scale).
"""

import sys
import os

GROUP_K = 128

def reference_gemm_a16w8_blockscale(M, N, K, device="cpu"):
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    num_k_groups = (K + GROUP_K - 1) // GROUP_K
    b_scale = torch.rand(N, num_k_groups, dtype=torch.float32, device=device) * 0.1 + 0.01

    C_ref = torch.zeros(M, N, dtype=torch.float32, device=device)
    for kg in range(num_k_groups):
        k_start = kg * GROUP_K
        k_end = min(k_start + GROUP_K, K)
        A_block = A[:, k_start:k_end].float()
        B_block = B[:, k_start:k_end].float()
        partial = A_block @ B_block.T
        b_s = b_scale[:, kg]  # (N,)
        C_ref += partial * b_s[None, :]
    C_ref = C_ref.bfloat16()
    return A, B, b_scale, C_ref

def test_with_hardware():
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a16w8_blockscale as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available.")
        return False
    test_sizes = [(128, 128, 128), (512, 512, 512), (2048, 2048, 2048)]
    all_passed = True
    for M, N, K in test_sizes:
        A, B, b_scale, C_ref = reference_gemm_a16w8_blockscale(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        B_bf16 = B.to(torch.bfloat16)
        kernel_mod.dispatch(A, B_bf16, b_scale, C_out)
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
    import torch
    print("Reference-only test (no HIP hardware):")
    test_sizes = [(128, 128, 128), (512, 512, 512), (2048, 2048, 2048)]
    for M, N, K in test_sizes:
        A, B, b_scale, C_ref = reference_gemm_a16w8_blockscale(M, N, K)
        num_k_groups = (K + GROUP_K - 1) // GROUP_K
        assert C_ref.shape == (M, N)
        assert C_ref.dtype == torch.bfloat16
        assert A.dtype == torch.bfloat16
        assert B.dtype == torch.int8
        assert b_scale.shape == (N, num_k_groups)
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}, b_scale={b_scale.shape}")
    print(f"\nBlock-scale parameter: group_k={GROUP_K}")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], b_scale: fp32[N,K//128], C: bf16[M,N])")

if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a16w8_blockscale Parity Test ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_a16w8_blockscale (Mixed BF16 x INT8 block-scaled GEMM)")
        print("Operation: C = sum_k[ (A_block @ B_block^T) * b_block_scale ]")
