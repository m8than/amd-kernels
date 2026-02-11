#!/usr/bin/env python3
"""Test the gemm_hk kernel (128x128 tiles, 64KB LDS, MI325X-optimized)."""

import sys
import importlib.util
import torch

VENV_PYTHON = "/root/aiter-hipkittens/amd-kernels/.venv/bin/python"
SO_PATH = "/root/aiter-hipkittens/amd-kernels/kernels/gemm-basic/gemm_a16w16/gemm_hk_tk.cpython-312-x86_64-linux-gnu.so"

def load_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_gemm_hk():
    print("=" * 60)
    print("Testing gemm_hk_tk (kernel_hk.cpp - 128x128 tiles, 64KB LDS)")
    print("=" * 60)

    try:
        mod = load_module("gemm_hk_tk", SO_PATH)
    except Exception as e:
        print(f"FAIL: Could not load module: {e}")
        return False

    print(f"Module loaded: {mod}")
    print(f"Functions: {dir(mod)}")

    # Test dimensions (must be divisible by 128 for M,N and 64 for K)
    M, N, K = 256, 256, 128

    # kernel_hk uses fp16 (half), not bf16
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")  # fp32 output

    # Reference: C = A @ B^T in fp32
    ref = (A.float() @ B.float().T)

    print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
    print(f"Calling dispatch(A, B, C)...")

    try:
        mod.dispatch(A, B, C)
        torch.cuda.synchronize()
        print("Kernel executed successfully!")
    except Exception as e:
        print(f"FAIL: Kernel execution error: {e}")
        return False

    # Compare
    max_diff = (C - ref).abs().max().item()
    mean_diff = (C - ref).abs().mean().item()
    rel_err = max_diff / ref.abs().max().item() if ref.abs().max().item() > 0 else 0

    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Relative error: {rel_err:.6f}")

    # For fp16 GEMM, tolerance is ~1e-2 to 1e-3
    passed = max_diff < 1.0  # generous threshold first
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"C sample:\n{C[:4, :4]}")
        print(f"Ref sample:\n{ref[:4, :4]}")

    return passed


if __name__ == "__main__":
    result = test_gemm_hk()
    sys.exit(0 if result else 1)
