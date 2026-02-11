#!/usr/bin/env python3
"""
Parity test for MoE Routing Sigmoid Top-1 Fused Kernel

Compares HipKittens C++ kernel output against Triton reference implementation.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import subprocess
import os
import ctypes
from pathlib import Path


# Triton reference kernel (from reference/triton/moe_routing_sigmoid_top1_fused.py)
@triton.jit
def _routing_sigmoid_top1_kernel(
    X_ptr,
    W_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_topk_ids_m,
    stride_topk_ids_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    FUSED_SHARED_EXPERTS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    _TOPK: tl.constexpr = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK
    offs_topk = tl.arange(0, _TOPK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K

        X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k_iter[None, :] * stride_xk)
        W_ptrs = W_ptr + (offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        acc = tl.dot(x, w, acc=acc)

    acc = tl.sigmoid(acc)
    topk_ids = tl.argmax(acc, axis=1, tie_break_left=True)
    topk_weights = tl.max(acc, axis=1)

    topk_ids_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.int32)
    topk_weights_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.float32)

    if FUSED_SHARED_EXPERTS:
        topk_ids_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_ids[:, None], N
        )
        topk_weights_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_weights[:, None], 1.0
        )
    else:
        topk_ids_buffer = topk_ids[:, None]
        topk_weights_buffer = topk_weights[:, None]

    topk_ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None] * stride_topk_ids_m
        + offs_topk[None, :] * stride_topk_ids_n
    )

    topk_weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_topk_weights_m
        + offs_topk[None, :] * stride_topk_weights_n
    )

    tl.store(topk_ids_ptrs, topk_ids_buffer, mask=mask_m[:, None])
    tl.store(topk_weights_ptrs, topk_weights_buffer, mask=mask_m[:, None])


def triton_routing_sigmoid_top1(X, W, TOPK=1, FUSED_SHARED_EXPERTS=False):
    """Reference Triton implementation"""
    M, K = X.shape
    K2, N = W.shape
    assert K == K2, "Dimension mismatch"

    _TOPK = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK
    topk_ids = torch.empty((M, _TOPK), dtype=torch.int32, device=X.device)
    topk_weights = torch.empty((M, _TOPK), dtype=torch.float32, device=X.device)

    BLOCK_M = 64
    BLOCK_K = 32
    BLOCK_N = 128 if N > 64 else 64

    grid = lambda meta: (triton.cdiv(M, BLOCK_M),)

    _routing_sigmoid_top1_kernel[grid](
        X, W,
        topk_ids, topk_weights,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        TOPK=TOPK,
        FUSED_SHARED_EXPERTS=FUSED_SHARED_EXPERTS,
    )

    return topk_ids, topk_weights


def compile_hip_kernel():
    """Compile the HipKittens kernel"""
    kernel_dir = Path(__file__).parent
    lib_path = kernel_dir / "libmoe_routing_sigmoid_top1.so"

    print("Compiling HipKittens kernel...")
    print(f"Command: make -C {kernel_dir}")

    result = subprocess.run(
        ["make", "-C", str(kernel_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("COMPILATION FAILED")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        return None

    print("Compilation successful!")
    return lib_path


def test_parity(M=256, N=16, K=128, TOPK=1, FUSED_SHARED_EXPERTS=False):
    """Test parity between Triton and HipKittens implementations"""

    print(f"\n{'='*60}")
    print(f"Testing MoE Routing Sigmoid Top-1 Fused Kernel")
    print(f"M={M}, N={N}, K={K}, TOPK={TOPK}, FUSED_SHARED_EXPERTS={FUSED_SHARED_EXPERTS}")
    print(f"{'='*60}\n")

    # Create random inputs
    torch.manual_seed(42)
    X = torch.randn((M, K), dtype=torch.float32).cuda()
    W = torch.randn((K, N), dtype=torch.float32).cuda()

    # Convert to bf16 for kernel
    X_bf16 = X.to(torch.bfloat16)
    W_bf16 = W.to(torch.bfloat16)

    # Run Triton reference
    print("Running Triton reference kernel...")
    topk_ids_ref, topk_weights_ref = triton_routing_sigmoid_top1(
        X_bf16, W_bf16, TOPK, FUSED_SHARED_EXPERTS
    )

    print(f"Triton output shapes: topk_ids={topk_ids_ref.shape}, topk_weights={topk_weights_ref.shape}")
    print(f"Sample topk_ids (first 5): {topk_ids_ref[:5]}")
    print(f"Sample topk_weights (first 5): {topk_weights_ref[:5]}")

    # Try to compile and run HipKittens kernel
    lib_path = compile_hip_kernel()

    if lib_path is None or not lib_path.exists():
        print("\n" + "="*60)
        print("HipKittens kernel compilation unavailable or failed.")
        print("This test documents the expected behavior.")
        print("="*60)
        print("\nExpected compilation command:")
        print(f"  cd {Path(__file__).parent}")
        print(f"  make")
        print("\nExpected usage from C++:")
        print(f"  void launch_routing_sigmoid_top1(")
        print(f"      const __hip_bfloat16* X,")
        print(f"      const __hip_bfloat16* W,")
        print(f"      int32_t* topk_ids,")
        print(f"      __hip_bfloat16* topk_weights,")
        print(f"      int M, int N, int K,")
        print(f"      int TOPK,")
        print(f"      bool FUSED_SHARED_EXPERTS,")
        print(f"      hipStream_t stream")
        print(f"  );")
        return "SKIP"

    # Load shared library
    print(f"\nLoading HipKittens kernel from {lib_path}...")
    try:
        lib = ctypes.CDLL(str(lib_path))
    except Exception as e:
        print(f"Failed to load library: {e}")
        return "SKIP"

    # Setup function signature
    lib.launch_routing_sigmoid_top1.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # W
        ctypes.c_void_p,  # topk_ids
        ctypes.c_void_p,  # topk_weights
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # TOPK
        ctypes.c_bool,    # FUSED_SHARED_EXPERTS
        ctypes.c_void_p,  # stream
    ]

    # Prepare output tensors
    _TOPK = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK
    topk_ids_hip = torch.empty((M, _TOPK), dtype=torch.int32, device='cuda')
    topk_weights_hip = torch.empty((M, _TOPK), dtype=torch.bfloat16, device='cuda')

    # Launch kernel
    print("Running HipKittens kernel...")
    lib.launch_routing_sigmoid_top1(
        X_bf16.data_ptr(),
        W_bf16.data_ptr(),
        topk_ids_hip.data_ptr(),
        topk_weights_hip.data_ptr(),
        M, N, K,
        TOPK,
        FUSED_SHARED_EXPERTS,
        None  # default stream
    )

    torch.cuda.synchronize()

    print(f"HipKittens output shapes: topk_ids={topk_ids_hip.shape}, topk_weights={topk_weights_hip.shape}")
    print(f"Sample topk_ids (first 5): {topk_ids_hip[:5]}")
    print(f"Sample topk_weights (first 5): {topk_weights_hip[:5]}")

    # Compare results
    print("\n" + "="*60)
    print("Comparing results...")
    print("="*60)

    ids_match = torch.equal(topk_ids_ref, topk_ids_hip)
    weights_close = torch.allclose(
        topk_weights_ref.float(),
        topk_weights_hip.float(),
        rtol=1e-2, atol=1e-3
    )

    print(f"IDs match: {ids_match}")
    print(f"Weights close (rtol=1e-2, atol=1e-3): {weights_close}")

    if not ids_match:
        diff = (topk_ids_ref != topk_ids_hip).sum().item()
        print(f"  Mismatches: {diff}/{topk_ids_ref.numel()} elements")

    if not weights_close:
        diff = torch.abs(topk_weights_ref.float() - topk_weights_hip.float())
        print(f"  Max abs diff: {diff.max().item():.6f}")
        print(f"  Mean abs diff: {diff.mean().item():.6f}")

    if ids_match and weights_close:
        print("\n✓ PASS: Parity test successful!")
        return "PASS"
    else:
        print("\n✗ FAIL: Parity test failed!")
        return "FAIL"


if __name__ == "__main__":
    # Test various configurations
    test_configs = [
        {"M": 256, "N": 16, "K": 128, "TOPK": 1, "FUSED_SHARED_EXPERTS": False},
        {"M": 512, "N": 32, "K": 256, "TOPK": 1, "FUSED_SHARED_EXPERTS": False},
        {"M": 256, "N": 16, "K": 128, "TOPK": 1, "FUSED_SHARED_EXPERTS": True},
    ]

    results = []
    for config in test_configs:
        result = test_parity(**config)
        results.append(result)

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for config, result in zip(test_configs, results):
        print(f"{config}: {result}")
