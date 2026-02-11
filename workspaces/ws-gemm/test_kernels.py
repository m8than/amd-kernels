#!/usr/bin/env python3
"""
Comprehensive parity tests for all GEMM HipKittens kernels.

Tests gemm-basic (10), gemm-batched (4), and gemm-fused (4) kernels.
Each test loads the .so, creates proper inputs on GPU, runs the kernel,
and compares against a PyTorch/numpy reference.
"""

import importlib
import importlib.util
import sys
import os
import traceback
import numpy as np
import torch

# ============================================================================
# Kernel loading utility
# ============================================================================

KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"

def load_kernel(so_path, module_name):
    """Load a compiled .so kernel module."""
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ============================================================================
# Results tracking
# ============================================================================

results = []

def record(name, status, max_diff=None, notes=""):
    results.append({
        "name": name,
        "status": status,
        "max_diff": max_diff,
        "notes": notes,
    })
    symbol = "PASS" if status == "PASS" else ("FAIL" if status == "FAIL" else "SKIP")
    diff_str = f", max_diff={max_diff:.6f}" if max_diff is not None else ""
    note_str = f" ({notes})" if notes else ""
    print(f"  [{symbol}] {name}{diff_str}{note_str}")


# ============================================================================
# gemm-basic kernels
# ============================================================================

def test_gemm_a16w16():
    """C = A @ B^T, all bf16. M,N % 256 == 0, K % 64 == 0."""
    name = "gemm_a16w16"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a16w16/gemm_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a16w16_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 256, 256, 256
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = (A.float() @ B.float().T).bfloat16()

    mod.dispatch(A, B, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a16w16_atomic():
    """C = A @ B^T with atomic fp32 output."""
    name = "gemm_a16w16_atomic"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a16w16_atomic/gemm_a16w16_atomic_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a16w16_atomic_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 128
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    C_ref = A.float() @ B.float().T

    mod.dispatch(A, B, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out - C_ref).abs().max().item()
    rel_err = max_diff / (C_ref.abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a16w16_gated():
    """C = (A @ B0^T) * silu(A @ B1^T), B = [B0; B1]."""
    name = "gemm_a16w16_gated"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a16w16_gated/gemm_a16w16_gated_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a16w16_gated_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 64  # N=128 -> each half is 64
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # N rows, split into halves
    half_n = N // 2
    C_out = torch.zeros(M, half_n, dtype=torch.bfloat16, device="cuda")

    B0 = B[:half_n, :]
    B1 = B[half_n:, :]
    acc0 = A.float() @ B0.float().T
    acc1 = A.float() @ B1.float().T
    gate = acc1 * torch.sigmoid(acc1)
    C_ref = (acc0 * gate).bfloat16()

    mod.dispatch(A, B, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a8w8():
    """C = (A_int8 @ B_int8^T) * a_scale * b_scale."""
    name = "gemm_a8w8"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a8w8/gemm_a8w8_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a8w8_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 128
    torch.manual_seed(42)
    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    b_scale = torch.rand(N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    C_ref = (A_int8.float() @ B_int8.float().T) * a_scale[:, None] * b_scale[None, :]
    C_ref = C_ref.bfloat16()

    A_bf16 = A_int8.to(torch.bfloat16)
    B_bf16 = B_int8.to(torch.bfloat16)
    # pybind order: dispatch(A, B, a_scale, b_scale, C)
    mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a8w8_blockscale():
    """Block-scaled INT8 GEMM. Per-K-block scaling."""
    name = "gemm_a8w8_blockscale"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a8w8_blockscale/gemm_a8w8_blockscale_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a8w8_blockscale_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    GROUP_K = 128
    GROUP_N = 1
    M, N, K = 128, 128, 128
    torch.manual_seed(42)

    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    num_k_groups = K // GROUP_K
    num_n_groups = N // GROUP_N
    a_scale = torch.rand(M, num_k_groups, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    b_scale = torch.rand(num_k_groups, num_n_groups, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Reference
    C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for kg in range(num_k_groups):
        k_s, k_e = kg * GROUP_K, (kg + 1) * GROUP_K
        A_block = A_int8[:, k_s:k_e].float()
        B_block = B_int8[:, k_s:k_e].float()
        partial = A_block @ B_block.T
        a_s = a_scale[:, kg]
        b_s_grouped = b_scale[kg, :]
        b_s = b_s_grouped.repeat_interleave(GROUP_N)[:N]
        C_ref += partial * a_s[:, None] * b_s[None, :]
    C_ref = C_ref.bfloat16()

    A_bf16 = A_int8.to(torch.bfloat16)
    B_bf16 = B_int8.to(torch.bfloat16)

    # pybind order from kernel.cpp: dispatch(A, B, C, a_scale, b_scale)
    # but test_parity.py uses: dispatch(A, B, a_scale, b_scale, C)
    # Try pybind order first
    try:
        mod.dispatch(A_bf16, B_bf16, C_out, a_scale, b_scale)
        torch.cuda.synchronize()
    except Exception:
        C_out.zero_()
        mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C_out)
        torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a8w8_per_token_scale():
    """Per-token-scaled INT8 GEMM."""
    name = "gemm_a8w8_per_token_scale"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a8w8_per_token_scale/gemm_a8w8_per_token_scale_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a8w8_per_token_scale_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 128
    torch.manual_seed(42)
    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    b_scale = torch.rand(N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    C_ref = ((A_int8.float() @ B_int8.float().T) * a_scale[:, None] * b_scale[None, :]).bfloat16()

    A_bf16 = A_int8.to(torch.bfloat16)
    B_bf16 = B_int8.to(torch.bfloat16)

    # pybind order: dispatch(A, B, C, a_scale, b_scale)
    # test_parity.py: dispatch(A, B, a_scale, b_scale, C)
    try:
        mod.dispatch(A_bf16, B_bf16, C_out, a_scale, b_scale)
        torch.cuda.synchronize()
    except Exception:
        C_out.zero_()
        mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C_out)
        torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a16w8_blockscale():
    """Mixed BF16 x INT8 block-scaled GEMM."""
    name = "gemm_a16w8_blockscale"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a16w8_blockscale/gemm_a16w8_blockscale_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a16w8_blockscale_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    GROUP_K = 128
    M, N, K = 128, 128, 128
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    num_k_groups = K // GROUP_K
    b_scale = torch.rand(N, num_k_groups, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Reference
    C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for kg in range(num_k_groups):
        k_s, k_e = kg * GROUP_K, (kg + 1) * GROUP_K
        A_block = A[:, k_s:k_e].float()
        B_block = B_int8[:, k_s:k_e].float()
        partial = A_block @ B_block.T
        b_s = b_scale[:, kg]
        C_ref += partial * b_s[None, :]
    C_ref = C_ref.bfloat16()

    B_bf16 = B_int8.to(torch.bfloat16)

    # pybind order: dispatch(A, B, C, b_scale)
    # test_parity.py: dispatch(A, B_bf16, b_scale, C_out)
    try:
        mod.dispatch(A, B_bf16, C_out, b_scale)
        torch.cuda.synchronize()
    except Exception:
        C_out.zero_()
        mod.dispatch(A, B_bf16, b_scale, C_out)
        torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a16wfp4():
    """BF16 x FP4 GEMM (B pre-dequantized)."""
    name = "gemm_a16wfp4"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a16wfp4/gemm_a16wfp4_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a16wfp4_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    SCALE_GROUP = 32
    M, N, K = 128, 128, 64
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
    num_sg = K // SCALE_GROUP
    b_scales = torch.ones(N, num_sg, dtype=torch.float32, device="cuda")
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    C_ref = (A.float() @ B.float().T).bfloat16()

    # pybind order: dispatch(A, B, C, b_scales)
    # test_parity.py: dispatch(A, B, C_out, b_scales) — same!
    mod.dispatch(A, B, C_out, b_scales)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_a8wfp4():
    """INT8 x FP4 GEMM (both pre-dequantized)."""
    name = "gemm_a8wfp4"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_a8wfp4/gemm_a8wfp4_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_a8wfp4_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 64
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
    a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    C_ref = ((A.float() @ B.float().T) * a_scale[:, None].cuda()).bfloat16()

    # pybind order: dispatch(A, B, a_scale, C)
    mod.dispatch(A, B, a_scale, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_gemm_afp4wfp4():
    """FP4 x FP4 GEMM (both pre-dequantized, simple bf16 GEMM)."""
    name = "gemm_afp4wfp4"
    so = os.path.join(KERNELS_DIR, "gemm-basic/gemm_afp4wfp4/gemm_afp4wfp4_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "gemm_afp4wfp4_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 128, 128, 64
    torch.manual_seed(42)
    A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
    B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    C_ref = (A.float() @ B.float().T).bfloat16()

    mod.dispatch(A, B, C_out)
    torch.cuda.synchronize()

    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.02
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


# ============================================================================
# gemm-batched kernels
# ============================================================================

def test_batched_gemm_bf16():
    """Batched C = A @ B + bias, all bf16. HK layout: (1, B, M/N, K/N)."""
    name = "batched_gemm_bf16"
    so = os.path.join(KERNELS_DIR, "gemm-batched/batched_gemm_bf16/batched_gemm_bf16_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "batched_gemm_bf16_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    BATCH, M, N, K = 4, 256, 256, 128
    torch.manual_seed(42)
    A = torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16, device="cuda")

    C_ref = torch.bmm(A.float(), B.float()).bfloat16()
    C_ref = (C_ref.float() + bias.float()).bfloat16()

    # HK layout: (1, B, M, K), B transposed to (1, B, N, K) for mma_ABt
    A_hk = A.unsqueeze(0).contiguous()
    B_hk = B.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
    bias_hk = bias.unsqueeze(0).contiguous()

    mod.dispatch(A_hk, B_hk, C_hk, bias_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_batched_gemm_a8w8():
    """Batched INT8 GEMM with per-row/col scales + bias."""
    name = "batched_gemm_a8w8"
    so = os.path.join(KERNELS_DIR, "gemm-batched/batched_gemm_a8w8/batched_gemm_a8w8_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "batched_gemm_a8w8_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    BATCH, M, N, K = 4, 256, 256, 128
    torch.manual_seed(42)
    A_int8 = torch.randint(-128, 127, (BATCH, M, K), dtype=torch.int8)
    B_int8 = torch.randint(-128, 127, (BATCH, K, N), dtype=torch.int8)
    a_scale = torch.rand(BATCH, M, 1, dtype=torch.float32) * 0.1
    b_scale = torch.rand(BATCH, 1, N, dtype=torch.float32) * 0.1
    bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16)

    C_ref = torch.bmm(A_int8.float(), B_int8.float()) * a_scale * b_scale
    C_ref = (C_ref + bias.float()).bfloat16()

    # Pre-cast to bf16
    A_bf16 = A_int8.float().bfloat16().cuda()
    B_bf16 = B_int8.float().bfloat16().cuda()
    a_scale_gpu = a_scale.cuda()
    b_scale_gpu = b_scale.cuda()
    bias_gpu = bias.cuda()

    # HK layout
    A_hk = A_bf16.unsqueeze(0).contiguous()
    B_hk = B_bf16.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
    # a_scale: (B, M, 1) -> (1, B, 1, M)
    a_scale_hk = a_scale_gpu.squeeze(-1).unsqueeze(0).unsqueeze(2).contiguous()
    # b_scale: (B, 1, N) -> (1, B, 1, N)
    b_scale_hk = b_scale_gpu.squeeze(-2).unsqueeze(0).unsqueeze(2).contiguous()
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
    bias_hk = bias_gpu.unsqueeze(0).contiguous()

    mod.dispatch(A_hk, B_hk, a_scale_hk, b_scale_hk, C_hk, bias_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    C_ref_gpu = C_ref.cuda()
    max_diff = (C_out.float() - C_ref_gpu.float()).abs().max().item()
    rel_err = max_diff / (C_ref_gpu.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.10  # INT8 batched has larger tolerance
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_batched_gemm_a16wfp4():
    """Batched BF16 x FP4 (B pre-dequantized)."""
    name = "batched_gemm_a16wfp4"
    so = os.path.join(KERNELS_DIR, "gemm-batched/batched_gemm_a16wfp4/batched_gemm_a16wfp4_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "batched_gemm_a16wfp4_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    BATCH, M, N, K = 4, 256, 128, 128
    torch.manual_seed(42)
    A = torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda")
    # Simulate pre-dequantized FP4 B
    B_dequant = (torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)

    C_ref = torch.bmm(A.float(), B_dequant.float()).bfloat16()

    A_hk = A.unsqueeze(0).contiguous()
    B_hk = B_dequant.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")

    mod.dispatch(A_hk, B_hk, C_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


def test_batched_gemm_afp4wfp4():
    """Batched FP4 x FP4 (both pre-dequantized)."""
    name = "batched_gemm_afp4wfp4"
    so = os.path.join(KERNELS_DIR, "gemm-batched/batched_gemm_afp4wfp4/batched_gemm_afp4wfp4_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "batched_gemm_afp4wfp4_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    BATCH, M, N, K = 4, 128, 128, 128
    torch.manual_seed(42)
    A = (torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
    B = (torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)

    C_ref = torch.bmm(A.float(), B.float()).bfloat16()

    A_hk = A.unsqueeze(0).contiguous()
    B_hk = B.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")

    mod.dispatch(A_hk, B_hk, C_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    max_diff = (C_out.float() - C_ref.float()).abs().max().item()
    rel_err = max_diff / (C_ref.float().abs().max().item() + 1e-8)
    passed = rel_err < 0.05
    record(name, "PASS" if passed else "FAIL", max_diff, f"rel_err={rel_err:.6f}")


# ============================================================================
# gemm-fused kernels
# ============================================================================

def test_fused_gemm_a8w8_blockscale_a16w16():
    """Fused: INT8 blockscale GEMM + BF16 GEMM."""
    name = "fused_gemm_a8w8_blockscale_a16w16"
    so = os.path.join(KERNELS_DIR, "gemm-fused/fused_gemm_a8w8_blockscale_a16w16/fused_gemm_a8w8_blockscale_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "fused_gemm_a8w8_blockscale_a16w16_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N_fp8, N_bf16, K = 256, 256, 256, 256
    GROUP_K = 128
    np.random.seed(42)

    a_fp8 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_fp8 = np.random.randn(N_fp8, K).astype(np.float32) * 0.1
    scale_k = K // GROUP_K
    a_scale = np.random.rand(M, scale_k).astype(np.float32) * 2.0
    b_scale = np.random.rand(scale_k, N_fp8).astype(np.float32) * 2.0
    a_bf16 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_bf16 = np.random.randn(N_bf16, K).astype(np.float32) * 0.1

    # Reference: blockscale path
    c_fp8_ref = np.zeros((M, N_fp8), dtype=np.float32)
    for kb in range(scale_k):
        k_s, k_e = kb * GROUP_K, (kb + 1) * GROUP_K
        partial = a_fp8[:, k_s:k_e].astype(np.float32) @ b_fp8[:, k_s:k_e].astype(np.float32).T
        partial *= a_scale[:, kb:kb+1]
        partial *= b_scale[kb:kb+1, :]
        c_fp8_ref += partial

    # Reference: bf16 path
    c_bf16_ref = a_bf16.astype(np.float32) @ b_bf16.astype(np.float32).T

    # Tensors
    a_fp8_t = torch.from_numpy(a_fp8).to(torch.bfloat16).cuda()
    b_fp8_t = torch.from_numpy(b_fp8).to(torch.bfloat16).cuda()
    a_scale_t = torch.from_numpy(a_scale).cuda()
    b_scale_t = torch.from_numpy(b_scale).cuda()
    a_bf16_t = torch.from_numpy(a_bf16).to(torch.bfloat16).cuda()
    b_bf16_t = torch.from_numpy(b_bf16).to(torch.bfloat16).cuda()
    c_fp8_t = torch.zeros(M, N_fp8, dtype=torch.bfloat16, device='cuda')
    c_bf16_t = torch.zeros(M, N_bf16, dtype=torch.bfloat16, device='cuda')

    mod.dispatch(a_fp8_t, b_fp8_t, c_fp8_t, a_scale_t, b_scale_t,
                 a_bf16_t, b_bf16_t, c_bf16_t)
    torch.cuda.synchronize()

    c_fp8_hk = c_fp8_t.cpu().float().numpy()
    c_bf16_hk = c_bf16_t.cpu().float().numpy()

    max_diff_fp8 = np.abs(c_fp8_ref - c_fp8_hk).max()
    max_diff_bf16 = np.abs(c_bf16_ref - c_bf16_hk).max()
    max_diff = max(max_diff_fp8, max_diff_bf16)

    fp8_ok = np.allclose(c_fp8_ref, c_fp8_hk, atol=1e-1, rtol=1e-2)
    bf16_ok = np.allclose(c_bf16_ref, c_bf16_hk, atol=1e-1, rtol=1e-2)

    passed = fp8_ok and bf16_ok
    notes = f"fp8={'ok' if fp8_ok else 'FAIL'}(maxdiff={max_diff_fp8:.4f}), bf16={'ok' if bf16_ok else 'FAIL'}(maxdiff={max_diff_bf16:.4f})"
    record(name, "PASS" if passed else "FAIL", max_diff, notes)


def test_fused_gemm_a8w8_blockscale_mul_add():
    """Fused: C = c_a * blockscale_gemm(A, B) + c_b."""
    name = "fused_gemm_a8w8_blockscale_mul_add"
    so = os.path.join(KERNELS_DIR, "gemm-fused/fused_gemm_a8w8_blockscale_mul_add/fused_gemm_a8w8_blockscale_mul_add_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "fused_gemm_a8w8_blockscale_mul_add_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 256, 256, 256
    GROUP_K = 128
    np.random.seed(42)

    a = np.random.randn(M, K).astype(np.float32) * 0.1
    b = np.random.randn(N, K).astype(np.float32) * 0.1
    scale_k = K // GROUP_K
    a_scale = np.random.rand(M, scale_k).astype(np.float32) * 2.0
    b_scale = np.random.rand(scale_k, N).astype(np.float32) * 2.0
    c_a = np.random.randn(M, N).astype(np.float32) * 0.5
    c_b = np.random.randn(M, N).astype(np.float32) * 0.5

    # Reference
    gemm_out = np.zeros((M, N), dtype=np.float32)
    for kb in range(scale_k):
        k_s, k_e = kb * GROUP_K, (kb + 1) * GROUP_K
        partial = a[:, k_s:k_e].astype(np.float32) @ b[:, k_s:k_e].astype(np.float32).T
        partial *= a_scale[:, kb:kb+1]
        partial *= b_scale[kb:kb+1, :]
        gemm_out += partial
    ref = c_a * gemm_out + c_b

    # Tensors
    a_t = torch.from_numpy(a).to(torch.bfloat16).cuda()
    b_t = torch.from_numpy(b).to(torch.bfloat16).cuda()
    a_scale_t = torch.from_numpy(a_scale).cuda()
    b_scale_t = torch.from_numpy(b_scale).cuda()
    c_a_t = torch.from_numpy(c_a).to(torch.bfloat16).cuda()
    c_b_t = torch.from_numpy(c_b).to(torch.bfloat16).cuda()
    c_t = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    mod.dispatch(a_t, b_t, c_t, a_scale_t, b_scale_t, c_a_t, c_b_t)
    torch.cuda.synchronize()

    c_hk = c_t.cpu().float().numpy()
    max_diff = np.abs(ref - c_hk).max()
    passed = np.allclose(ref, c_hk, atol=1e-1, rtol=1e-2)
    record(name, "PASS" if passed else "FAIL", float(max_diff))


def test_fused_gemm_afp4wfp4_a16w16():
    """Fused: FP4 scaled GEMM + BF16 GEMM."""
    name = "fused_gemm_afp4wfp4_a16w16"
    so = os.path.join(KERNELS_DIR, "gemm-fused/fused_gemm_afp4wfp4_a16w16/fused_gemm_afp4wfp4_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "fused_gemm_afp4wfp4_a16w16_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N_fp4, N_bf16, K = 256, 256, 256, 256
    SCALE_GROUP = 32
    np.random.seed(42)

    a_fp4 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_fp4 = np.random.randn(N_fp4, K).astype(np.float32) * 0.1
    scale_k = K // SCALE_GROUP
    a_fp4_scale = np.random.rand(M, scale_k).astype(np.float32) * 2.0
    b_fp4_scale = np.random.rand(N_fp4, scale_k).astype(np.float32) * 2.0
    a_bf16 = np.random.randn(M, K).astype(np.float32) * 0.1
    b_bf16 = np.random.randn(N_bf16, K).astype(np.float32) * 0.1

    # Reference: FP4 path with per-block scales
    c_fp4_ref = np.zeros((M, N_fp4), dtype=np.float32)
    for kb in range(scale_k):
        k_s, k_e = kb * SCALE_GROUP, (kb + 1) * SCALE_GROUP
        partial = a_fp4[:, k_s:k_e].astype(np.float32) @ b_fp4[:, k_s:k_e].astype(np.float32).T
        partial *= a_fp4_scale[:, kb:kb+1]
        partial *= b_fp4_scale[:, kb:kb+1].T
        c_fp4_ref += partial

    # Reference: BF16 path
    c_bf16_ref = a_bf16.astype(np.float32) @ b_bf16.astype(np.float32).T

    # Tensors
    a_fp4_t = torch.from_numpy(a_fp4).to(torch.bfloat16).cuda()
    b_fp4_t = torch.from_numpy(b_fp4).to(torch.bfloat16).cuda()
    a_fp4_scale_t = torch.from_numpy(a_fp4_scale).cuda()
    b_fp4_scale_t = torch.from_numpy(b_fp4_scale).cuda()
    a_bf16_t = torch.from_numpy(a_bf16).to(torch.bfloat16).cuda()
    b_bf16_t = torch.from_numpy(b_bf16).to(torch.bfloat16).cuda()
    c_fp4_t = torch.zeros(M, N_fp4, dtype=torch.bfloat16, device='cuda')
    c_bf16_t = torch.zeros(M, N_bf16, dtype=torch.bfloat16, device='cuda')

    mod.dispatch(a_fp4_t, b_fp4_t, c_fp4_t, a_fp4_scale_t, b_fp4_scale_t,
                 a_bf16_t, b_bf16_t, c_bf16_t)
    torch.cuda.synchronize()

    c_fp4_hk = c_fp4_t.cpu().float().numpy()
    c_bf16_hk = c_bf16_t.cpu().float().numpy()

    max_diff_fp4 = np.abs(c_fp4_ref - c_fp4_hk).max()
    max_diff_bf16 = np.abs(c_bf16_ref - c_bf16_hk).max()
    max_diff = max(max_diff_fp4, max_diff_bf16)

    fp4_ok = np.allclose(c_fp4_ref, c_fp4_hk, atol=1e-1, rtol=1e-2)
    bf16_ok = np.allclose(c_bf16_ref, c_bf16_hk, atol=1e-1, rtol=1e-2)

    passed = fp4_ok and bf16_ok
    notes = f"fp4={'ok' if fp4_ok else 'FAIL'}(maxdiff={max_diff_fp4:.4f}), bf16={'ok' if bf16_ok else 'FAIL'}(maxdiff={max_diff_bf16:.4f})"
    record(name, "PASS" if passed else "FAIL", max_diff, notes)


def test_fused_gemm_afp4wfp4_mul_add():
    """Fused: C = c_a * fp4_gemm(A, B) + c_b."""
    name = "fused_gemm_afp4wfp4_mul_add"
    so = os.path.join(KERNELS_DIR, "gemm-fused/fused_gemm_afp4wfp4_mul_add/fused_gemm_afp4wfp4_mul_add_tk.cpython-312-x86_64-linux-gnu.so")
    try:
        mod = load_kernel(so, "fused_gemm_afp4wfp4_mul_add_tk")
    except Exception as e:
        record(name, "SKIP", notes=f"load failed: {e}")
        return

    M, N, K = 256, 256, 256
    SCALE_GROUP = 32
    np.random.seed(42)

    a = np.random.randn(M, K).astype(np.float32) * 0.1
    b = np.random.randn(N, K).astype(np.float32) * 0.1
    scale_k = K // SCALE_GROUP
    a_scale = np.random.rand(M, scale_k).astype(np.float32) * 2.0
    b_scale = np.random.rand(N, scale_k).astype(np.float32) * 2.0
    c_a = np.random.randn(M, N).astype(np.float32) * 0.5
    c_b = np.random.randn(M, N).astype(np.float32) * 0.5

    # Reference
    gemm_out = np.zeros((M, N), dtype=np.float32)
    for kb in range(scale_k):
        k_s, k_e = kb * SCALE_GROUP, (kb + 1) * SCALE_GROUP
        partial = a[:, k_s:k_e].astype(np.float32) @ b[:, k_s:k_e].astype(np.float32).T
        partial *= a_scale[:, kb:kb+1]
        partial *= b_scale[:, kb:kb+1].T
        gemm_out += partial
    ref = c_a * gemm_out + c_b

    # Tensors
    a_t = torch.from_numpy(a).to(torch.bfloat16).cuda()
    b_t = torch.from_numpy(b).to(torch.bfloat16).cuda()
    a_scale_t = torch.from_numpy(a_scale).cuda()
    b_scale_t = torch.from_numpy(b_scale).cuda()
    c_a_t = torch.from_numpy(c_a).to(torch.bfloat16).cuda()
    c_b_t = torch.from_numpy(c_b).to(torch.bfloat16).cuda()
    c_t = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    mod.dispatch(a_t, b_t, c_t, a_scale_t, b_scale_t, c_a_t, c_b_t)
    torch.cuda.synchronize()

    c_hk = c_t.cpu().float().numpy()
    max_diff = np.abs(ref - c_hk).max()
    passed = np.allclose(ref, c_hk, atol=1e-1, rtol=1e-2)
    record(name, "PASS" if passed else "FAIL", float(max_diff))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("GEMM Kernel Parity Tests — HipKittens on AMD MI325X (gfx942)")
    print("=" * 70)

    # Basic kernels
    print("\n--- gemm-basic (10 kernels) ---")
    basic_tests = [
        test_gemm_a16w16,
        test_gemm_a16w16_atomic,
        test_gemm_a16w16_gated,
        test_gemm_a8w8,
        test_gemm_a8w8_blockscale,
        test_gemm_a8w8_per_token_scale,
        test_gemm_a16w8_blockscale,
        test_gemm_a16wfp4,
        test_gemm_a8wfp4,
        test_gemm_afp4wfp4,
    ]
    for t in basic_tests:
        try:
            t()
        except Exception as e:
            name = t.__name__.replace("test_", "")
            record(name, "FAIL", notes=f"exception: {e}")
            traceback.print_exc()

    # Batched kernels
    print("\n--- gemm-batched (4 kernels) ---")
    batched_tests = [
        test_batched_gemm_bf16,
        test_batched_gemm_a8w8,
        test_batched_gemm_a16wfp4,
        test_batched_gemm_afp4wfp4,
    ]
    for t in batched_tests:
        try:
            t()
        except Exception as e:
            name = t.__name__.replace("test_", "")
            record(name, "FAIL", notes=f"exception: {e}")
            traceback.print_exc()

    # Fused kernels
    print("\n--- gemm-fused (4 kernels) ---")
    fused_tests = [
        test_fused_gemm_a8w8_blockscale_a16w16,
        test_fused_gemm_a8w8_blockscale_mul_add,
        test_fused_gemm_afp4wfp4_a16w16,
        test_fused_gemm_afp4wfp4_mul_add,
    ]
    for t in fused_tests:
        try:
            t()
        except Exception as e:
            name = t.__name__.replace("test_", "")
            record(name, "FAIL", notes=f"exception: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    print(f"  PASS: {passed}  FAIL: {failed}  SKIP: {skipped}  TOTAL: {len(results)}")

    print(f"\n{'Kernel':<45} {'Status':<6} {'Max Diff':<14} {'Notes'}")
    print("-" * 100)
    for r in results:
        diff_str = f"{r['max_diff']:.6f}" if r['max_diff'] is not None else "N/A"
        print(f"{r['name']:<45} {r['status']:<6} {diff_str:<14} {r['notes']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
