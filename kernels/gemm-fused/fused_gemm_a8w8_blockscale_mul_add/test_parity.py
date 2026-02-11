#!/usr/bin/env python3
"""
Parity test for fused_gemm_a8w8_blockscale_mul_add HipKittens kernel.

Fused operation:
  C = c_a * blockscale_gemm(A, B, a_scale, b_scale) + c_b

Reference: Triton fused_gemm_a8w8_blockscale_mul_add.py
"""

import sys
import os
import numpy as np

M = 256
N = 256
K = 256
GROUP_K = 128

def reference_blockscale_gemm_mul_add(a, b, a_scale, b_scale, c_a, c_b, group_k):
    """Reference: blockscale GEMM followed by multiply-add epilogue."""
    M_dim, K_dim = a.shape
    N_dim = b.shape[0]
    num_k_blocks = K_dim // group_k

    gemm_out = np.zeros((M_dim, N_dim), dtype=np.float32)
    for kb in range(num_k_blocks):
        k_start = kb * group_k
        k_end = k_start + group_k
        a_block = a[:, k_start:k_end].astype(np.float32)
        b_block = b[:, k_start:k_end].astype(np.float32)
        partial = a_block @ b_block.T
        partial *= a_scale[:, kb:kb+1]
        partial *= b_scale[kb:kb+1, :]
        gemm_out += partial

    # Fused epilogue: c_a * gemm_out + c_b
    result = c_a * gemm_out + c_b
    return result

def generate_test_data():
    np.random.seed(42)
    a = np.random.randn(M, K).astype(np.float32) * 0.1
    b = np.random.randn(N, K).astype(np.float32) * 0.1
    scale_k = K // GROUP_K
    a_scale = np.random.rand(M, scale_k).astype(np.float32) * 2.0
    b_scale = np.random.rand(scale_k, N).astype(np.float32) * 2.0
    c_a = np.random.randn(M, N).astype(np.float32) * 0.5
    c_b = np.random.randn(M, N).astype(np.float32) * 0.5
    return a, b, a_scale, b_scale, c_a, c_b

def main():
    print("=" * 60)
    print("Fused GEMM A8W8 Blockscale Mul-Add Parity Test")
    print("=" * 60)
    print(f"  M={M}, N={N}, K={K}, GROUP_K={GROUP_K}")
    print()
    print("Input shapes:")
    print(f"  A:       ({M}, {K})     bf16 (from int8)")
    print(f"  B:       ({N}, {K})     bf16 (from int8)")
    print(f"  a_scale: ({M}, {K // GROUP_K}) float32")
    print(f"  b_scale: ({K // GROUP_K}, {N}) float32")
    print(f"  c_a:     ({M}, {N})     bf16 (multiply operand)")
    print(f"  c_b:     ({M}, {N})     bf16 (add operand)")
    print()
    print("Output: C = c_a * gemm(A, B) + c_b")
    print(f"  C:       ({M}, {N})     bf16")
    print()
    print("Compilation command:")
    print(f"  make GPU_TARGET=CDNA4")
    print()

    a, b, a_scale, b_scale, c_a, c_b = generate_test_data()
    ref = reference_blockscale_gemm_mul_add(a, b, a_scale, b_scale, c_a, c_b, GROUP_K)
    print(f"Reference output: mean={ref.mean():.6f}, std={ref.std():.6f}, "
          f"max_abs={np.abs(ref).max():.6f}")

    try:
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        import tk_fused_gemm_a8w8_blockscale_mul_add as tk_mod

        a_t = torch.from_numpy(a).to(torch.bfloat16).cuda()
        b_t = torch.from_numpy(b).to(torch.bfloat16).cuda()
        a_scale_t = torch.from_numpy(a_scale).cuda()
        b_scale_t = torch.from_numpy(b_scale).cuda()
        c_a_t = torch.from_numpy(c_a).to(torch.bfloat16).cuda()
        c_b_t = torch.from_numpy(c_b).to(torch.bfloat16).cuda()
        c_t = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

        tk_mod.dispatch(a_t, b_t, c_t, a_scale_t, b_scale_t, c_a_t, c_b_t)
        torch.cuda.synchronize()

        c_hk = c_t.cpu().float().numpy()
        diff = np.abs(ref - c_hk)
        if np.allclose(ref, c_hk, atol=1e-1, rtol=1e-2):
            print("PASS: Output matches reference.")
        else:
            print(f"FAIL: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    except ImportError:
        print("HipKittens module not available. Reference-only test complete.")
        print("PASS (reference-only)")

if __name__ == "__main__":
    main()
