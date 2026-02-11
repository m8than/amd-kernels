"""
Parity test for gemm_a16wfp4 HipKittens kernel.

Operation: C = A_bf16 @ dequant(B_fp4)^T
  A: (M, K) bf16
  B: (N, K) bf16 (pre-dequantized FP4 weights)
  b_scales: (N, K//32) float32 (scales, passed for API compat)
  C: (M, N) bf16

FP4 (e2m1) has values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} x {+,-}
Scale in e8m0 format is a power-of-2 multiplier.

Current implementation expects host-side dequantization of B to bf16.
"""

import sys
import os

SCALE_GROUP = 32

def reference_gemm_a16wfp4(M, N, K, device="cpu"):
    import torch
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # Simulate pre-dequantized FP4 weights (small values typical of FP4 range)
    B = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5).clamp(-6, 6)
    num_scale_groups = (K + SCALE_GROUP - 1) // SCALE_GROUP
    b_scales = torch.ones(N, num_scale_groups, dtype=torch.float32, device=device)
    C_ref = (A.float() @ B.float().T).bfloat16()
    return A, B, b_scales, C_ref

def test_with_hardware():
    import torch
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_gemm_a16wfp4 as kernel_mod
    except ImportError:
        print("SKIP: Compiled kernel not available.")
        return False
    test_sizes = [(128, 128, 64), (256, 256, 256), (1024, 1024, 1024)]
    all_passed = True
    for M, N, K in test_sizes:
        A, B, b_scales, C_ref = reference_gemm_a16wfp4(M, N, K, device="cuda")
        C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        kernel_mod.dispatch(A, B, C_out, b_scales)
        torch.cuda.synchronize()
        max_diff = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
        passed = rel_err < 0.02
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M}, N={N}, K={K}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        if not passed:
            all_passed = False
    return all_passed

def test_reference_only():
    import torch
    print("Reference-only test (no HIP hardware):")
    test_sizes = [(128, 128, 64), (256, 256, 256), (1024, 1024, 1024)]
    for M, N, K in test_sizes:
        A, B, b_scales, C_ref = reference_gemm_a16wfp4(M, N, K)
        assert C_ref.shape == (M, N)
        assert C_ref.dtype == torch.bfloat16
        print(f"  [OK] M={M}, N={N}, K={K}: shape={C_ref.shape}")
    print("\nKernel interface:")
    print("  dispatch(A: bf16[M,K], B: bf16[N,K], C: bf16[M,N], b_scales: fp32[N,K//32])")
    print("  B is FP4 weights pre-dequantized to bf16 by host")

if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            print("=== gemm_a16wfp4 Parity Test ===")
            passed = test_with_hardware()
            sys.exit(0 if passed else 1)
        else:
            test_reference_only()
    except ImportError:
        print("Kernel: gemm_a16wfp4 (BF16 x FP4 GEMM)")
        print("Operation: C = A_bf16 @ dequant(B_fp4)^T")
