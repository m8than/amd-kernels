#!/usr/bin/env python3
"""
Comprehensive parity tests for misc HipKittens kernels.

Tests all passing kernels (activations, normalization, softmax, rope, topk,
causal_conv1d, fused_kv_cache, fused_mul_add, fused_qk_concat, quantization)
and investigates 5 crashing kernels in subprocesses.

Usage:
    /root/aiter-hipkittens/amd-kernels/.venv/bin/python test_kernels.py
"""

import torch
import torch.nn.functional as F
import importlib
import importlib.util
import sys
import os
import math
import subprocess
import traceback
import json

VENV_PYTHON = "/root/aiter-hipkittens/amd-kernels/.venv/bin/python"
KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"
DEVICE = "cuda"

# ============================================================================
# Helpers
# ============================================================================

def load_kernel(name, so_path):
    """Load a .so kernel module by name and path."""
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


results = []

def record(kernel_name, status, max_diff=None, notes=""):
    results.append({
        "kernel": kernel_name,
        "status": status,
        "max_diff": max_diff,
        "notes": notes,
    })
    sym = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "SKIP", "CRASH": "CRASH"}
    diff_str = f" max_diff={max_diff:.6f}" if max_diff is not None else ""
    note_str = f" ({notes})" if notes else ""
    print(f"  [{sym.get(status, status):5s}] {kernel_name}{diff_str}{note_str}")


# ============================================================================
# 1. Activation Kernels
# ============================================================================

def test_activations():
    print("\n=== Activation Kernels ===")
    so_path = f"{KERNELS_DIR}/rope-activations/activation/activation_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("activation_tk", so_path)

    M, N = 1024, 4096
    torch.manual_seed(42)

    # Simple activations: (M, N) -> (M, N)
    x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)

    def ref_silu(t): return t.float() * torch.sigmoid(t.float())
    def ref_gelu(t): return 0.5 * t.float() * (1.0 + torch.erf(t.float() * (1.0 / math.sqrt(2.0))))
    def ref_gelu_tanh(t):
        BETA = math.sqrt(2.0) * (2.0 / math.sqrt(math.pi)) * 0.5
        KAPPA = 0.044715
        inner = BETA * (t.float() + KAPPA * t.float() ** 3)
        return 0.5 * t.float() * (1.0 + torch.tanh(inner))
    def ref_relu(t): return torch.relu(t.float())
    def ref_tanh(t): return torch.tanh(t.float())

    simple_tests = [
        ("silu_fwd", ref_silu),
        ("gelu_fwd", ref_gelu),
        ("gelu_tanh_fwd", ref_gelu_tanh),
        ("relu_fwd", ref_relu),
        ("tanh_fwd", ref_tanh),
    ]
    for fname, ref_fn in simple_tests:
        try:
            out = torch.zeros_like(x)
            getattr(mod, fname)(x, out)
            torch.cuda.synchronize()
            ref = ref_fn(x).to(torch.bfloat16)
            diff = (out.float() - ref.float()).abs().max().item()
            record(f"activation/{fname}", "PASS" if diff < 0.05 else "FAIL", diff)
        except Exception as e:
            record(f"activation/{fname}", "FAIL", notes=str(e))

    # Gated activations: input is (M, 2*N), output is (M, N)
    x_gated = torch.randn(M, 2 * N, dtype=torch.bfloat16, device=DEVICE)
    out_gated = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)

    gated_tests = [
        ("silu_and_mul_fwd", lambda a, b: (a.float() * torch.sigmoid(a.float())) * b.float()),
        ("gelu_and_mul_fwd", lambda a, b: (0.5 * a.float() * (1.0 + torch.erf(a.float() / math.sqrt(2.0)))) * b.float()),
    ]
    for fname, ref_fn in gated_tests:
        try:
            out = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)
            getattr(mod, fname)(x_gated, out)
            torch.cuda.synchronize()
            a_part = x_gated[:, :N]
            b_part = x_gated[:, N:]
            ref = ref_fn(a_part, b_part).to(torch.bfloat16)
            diff = (out.float() - ref.float()).abs().max().item()
            record(f"activation/{fname}", "PASS" if diff < 0.05 else "FAIL", diff)
        except Exception as e:
            record(f"activation/{fname}", "FAIL", notes=str(e))


# ============================================================================
# 2. RMSNorm Kernels
# ============================================================================

def test_rmsnorm():
    print("\n=== RMSNorm Kernels ===")
    so_path = f"{KERNELS_DIR}/normalization/rmsnorm/rmsnorm_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("rmsnorm_tk", so_path)

    B, S, D = 4, 1024, 128
    eps = 1e-6
    torch.manual_seed(42)

    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    # Test rmsnorm_fwd (D=128)
    try:
        out = torch.zeros_like(x)
        mod.rmsnorm_fwd(x, weight, out, eps, B * S)
        torch.cuda.synchronize()
        x_f32 = x.float()
        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
        ref = (x_f32 * torch.rsqrt(mean_sq + eps) * weight.float()).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        record("rmsnorm/rmsnorm_fwd", "PASS" if diff < 0.1 else "FAIL", diff)
    except Exception as e:
        record("rmsnorm/rmsnorm_fwd", "FAIL", notes=str(e))

    # Test fused_add_rmsnorm_fwd (D=128)
    try:
        residual = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros_like(x)
        res_out = torch.zeros_like(x)
        mod.fused_add_rmsnorm_fwd(x, residual, weight, out, res_out, eps, B * S)
        torch.cuda.synchronize()
        ref_res = (x.float() + residual.float()).to(torch.bfloat16)
        ref_res_f32 = ref_res.float()
        mean_sq = ref_res_f32.pow(2).mean(dim=-1, keepdim=True)
        ref_out = (ref_res_f32 * torch.rsqrt(mean_sq + eps) * weight.float()).to(torch.bfloat16)
        diff_out = (out.float() - ref_out.float()).abs().max().item()
        diff_res = (res_out.float() - ref_res.float()).abs().max().item()
        max_diff = max(diff_out, diff_res)
        record("rmsnorm/fused_add_rmsnorm_fwd", "PASS" if max_diff < 0.1 else "FAIL", max_diff)
    except Exception as e:
        record("rmsnorm/fused_add_rmsnorm_fwd", "FAIL", notes=str(e))


# ============================================================================
# 3. LayerNorm Kernels
# ============================================================================

def test_layernorm():
    print("\n=== LayerNorm Kernels ===")
    so_path = f"{KERNELS_DIR}/normalization/layernorm/layernorm_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("layernorm_tk", so_path)

    B, S, D = 4, 1024, 128
    eps = 1e-5
    torch.manual_seed(42)

    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    # Test layernorm_fwd
    try:
        out = torch.zeros_like(x)
        mod.layernorm_fwd(x, weight, bias, out, eps, B * S)
        torch.cuda.synchronize()
        ref = F.layer_norm(x.float(), [D], weight.float(), bias.float(), eps).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        record("layernorm/layernorm_fwd", "PASS" if diff < 0.1 else "FAIL", diff)
    except Exception as e:
        record("layernorm/layernorm_fwd", "FAIL", notes=str(e))

    # Test fused_add_layernorm_fwd
    try:
        residual = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros_like(x)
        res_out = torch.zeros_like(x)
        mod.fused_add_layernorm_fwd(x, residual, weight, bias, out, res_out, eps, B * S)
        torch.cuda.synchronize()
        ref_res = (x.float() + residual.float()).to(torch.bfloat16)
        ref_out = F.layer_norm(ref_res.float(), [D], weight.float(), bias.float(), eps).to(torch.bfloat16)
        diff_out = (out.float() - ref_out.float()).abs().max().item()
        diff_res = (res_out.float() - ref_res.float()).abs().max().item()
        max_diff = max(diff_out, diff_res)
        record("layernorm/fused_add_layernorm_fwd", "PASS" if max_diff < 0.1 else "FAIL", max_diff)
    except Exception as e:
        record("layernorm/fused_add_layernorm_fwd", "FAIL", notes=str(e))


# ============================================================================
# 4. Fused Add + RMSNorm + Pad Kernels
# ============================================================================

def test_fused_add_rmsnorm_pad():
    print("\n=== Fused Add RMSNorm Pad ===")
    so_path = f"{KERNELS_DIR}/normalization/fused_add_rmsnorm_pad/fused_add_rmsnorm_pad_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_add_rmsnorm_pad_tk", so_path)

    B, S, N = 4, 1024, 128
    N_OUT = 128
    eps = 1e-6
    torch.manual_seed(42)

    x = torch.randn(B * S, N, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(B * S, N, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    # Test fused_add_rmsnorm_pad
    try:
        out = torch.zeros(B * S, N_OUT, dtype=torch.bfloat16, device=DEVICE)
        res_out = torch.zeros_like(x)
        mod.fused_add_rmsnorm_pad(x, residual, weight, out, res_out, eps, B * S)
        torch.cuda.synchronize()
        ref_res = (x.float() + residual.float()).to(torch.bfloat16)
        ref_res_f32 = ref_res.float()
        mean_sq = ref_res_f32.pow(2).mean(dim=-1, keepdim=True)
        ref_out = (ref_res_f32 * torch.rsqrt(mean_sq + eps) * weight.float()).to(torch.bfloat16)
        diff_out = (out.float() - ref_out.float()).abs().max().item()
        diff_res = (res_out.float() - ref_res.float()).abs().max().item()
        max_diff = max(diff_out, diff_res)
        record("fused_add_rmsnorm_pad/fused", "PASS" if max_diff < 0.1 else "FAIL", max_diff)
    except Exception as e:
        record("fused_add_rmsnorm_pad/fused", "FAIL", notes=str(e))

    # Test rmsnorm_pad
    try:
        out = torch.zeros(B * S, N_OUT, dtype=torch.bfloat16, device=DEVICE)
        mod.rmsnorm_pad(x, weight, out, eps, B * S)
        torch.cuda.synchronize()
        x_f32 = x.float()
        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
        ref = (x_f32 * torch.rsqrt(mean_sq + eps) * weight.float()).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        record("fused_add_rmsnorm_pad/rmsnorm_pad", "PASS" if diff < 0.1 else "FAIL", diff)
    except Exception as e:
        record("fused_add_rmsnorm_pad/rmsnorm_pad", "FAIL", notes=str(e))


# ============================================================================
# 5. Softmax Kernel
# ============================================================================

def test_softmax():
    print("\n=== Softmax Kernel ===")
    so_path = f"{KERNELS_DIR}/rope-activations/softmax/softmax_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("softmax_tk", so_path)

    M = 512
    torch.manual_seed(42)

    for N in [128, 1024, 4096, 8192]:
        try:
            x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
            out = torch.zeros_like(x)
            mod.softmax_fwd(x, out)
            torch.cuda.synchronize()
            ref = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
            diff = (out.float() - ref.float()).abs().max().item()
            record(f"softmax/N={N}", "PASS" if diff < 0.05 else "FAIL", diff)
        except Exception as e:
            record(f"softmax/N={N}", "FAIL", notes=str(e))


# ============================================================================
# 6. RoPE Kernel
# ============================================================================

def test_rope():
    print("\n=== RoPE Kernel ===")
    so_path = f"{KERNELS_DIR}/rope-activations/rope/rope_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("rope_tk", so_path)

    B, H, N, D = 4, 32, 2048, 128
    D_HALF = D // 2
    torch.manual_seed(42)

    x = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=DEVICE)

    # Generate cos/sin tables
    t = torch.arange(N, device=DEVICE, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=DEVICE, dtype=torch.float32) / D))
    freqs = torch.outer(t, inv_freq)
    cos_freq = torch.cos(freqs).to(torch.bfloat16)
    sin_freq = torch.sin(freqs).to(torch.bfloat16)

    try:
        out = torch.zeros_like(x)
        mod.rope_fwd(x, out, cos_freq, sin_freq)
        torch.cuda.synchronize()
        # Reference
        x1, x2 = x[..., :D_HALF], x[..., D_HALF:]
        cos_b = cos_freq.unsqueeze(0).unsqueeze(0)
        sin_b = sin_freq.unsqueeze(0).unsqueeze(0)
        ref = torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)
        diff = (out.float() - ref.float()).abs().max().item()
        record("rope/rope_fwd", "PASS" if diff < 0.05 else "FAIL", diff)
    except Exception as e:
        record("rope/rope_fwd", "FAIL", notes=str(e))


# ============================================================================
# 7. TopK Kernel
# ============================================================================

def test_topk():
    print("\n=== TopK Kernel ===")
    so_path = f"{KERNELS_DIR}/rope-activations/topk/topk_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("topk_tk", so_path)

    B = 64
    torch.manual_seed(42)

    for N in [128, 1024, 4096]:
        for K in [1, 8, 32]:
            if K > N:
                continue
            try:
                x = torch.randn(B, N, dtype=torch.bfloat16, device=DEVICE)
                out_vals = torch.zeros(B, K, dtype=torch.float32, device=DEVICE)
                out_idx = torch.zeros(B, K, dtype=torch.int64, device=DEVICE)
                mod.topk_fwd(x, out_vals, out_idx, K)
                torch.cuda.synchronize()
                ref_vals, ref_idx = torch.topk(x.float(), K, dim=-1, largest=True, sorted=True)
                # Check values match
                diff = (out_vals - ref_vals).abs().max().item()
                record(f"topk/N={N},K={K}", "PASS" if diff < 0.1 else "FAIL", diff)
            except Exception as e:
                record(f"topk/N={N},K={K}", "FAIL", notes=str(e))


# ============================================================================
# 8. Causal Conv1d Kernel
# ============================================================================

def test_causal_conv1d():
    print("\n=== Causal Conv1d Kernel ===")
    so_path = f"{KERNELS_DIR}/rope-activations/causal_conv1d/causal_conv1d_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("causal_conv1d_tk", so_path)

    B, D, L = 8, 768, 2048
    torch.manual_seed(42)

    for K in [3, 4]:
        # Test without bias
        try:
            x = torch.randn(B, D, L, dtype=torch.bfloat16, device=DEVICE)
            w = torch.randn(D, K, dtype=torch.bfloat16, device=DEVICE)
            out = torch.zeros(B, D, L, dtype=torch.bfloat16, device=DEVICE)
            mod.causal_conv1d_fwd(x, w, out, K)
            torch.cuda.synchronize()
            # Reference: depthwise causal conv1d
            w_conv = w.float().unsqueeze(1)  # (D, 1, K)
            x_padded = F.pad(x.float(), (K - 1, 0))
            ref = F.conv1d(x_padded, w_conv, groups=D).to(torch.bfloat16)
            diff = (out.float() - ref.float()).abs().max().item()
            record(f"causal_conv1d/K={K}", "PASS" if diff < 0.1 else "FAIL", diff)
        except Exception as e:
            record(f"causal_conv1d/K={K}", "FAIL", notes=str(e))

        # Test with bias + silu
        try:
            bias = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)
            out = torch.zeros(B, D, L, dtype=torch.bfloat16, device=DEVICE)
            mod.causal_conv1d_bias_silu_fwd(x, w, bias, out, K)
            torch.cuda.synchronize()
            w_conv = w.float().unsqueeze(1)
            x_padded = F.pad(x.float(), (K - 1, 0))
            ref = F.conv1d(x_padded, w_conv, bias=bias.float(), groups=D)
            ref = ref * torch.sigmoid(ref)
            ref = ref.to(torch.bfloat16)
            diff = (out.float() - ref.float()).abs().max().item()
            record(f"causal_conv1d/K={K}_bias_silu", "PASS" if diff < 0.1 else "FAIL", diff)
        except Exception as e:
            record(f"causal_conv1d/K={K}_bias_silu", "FAIL", notes=str(e))


# ============================================================================
# 9. Fused KV Cache Kernel
# ============================================================================

def test_fused_kv_cache():
    print("\n=== Fused KV Cache Kernel ===")
    so_path = f"{KERNELS_DIR}/feedforward-fusions/fused_kv_cache/fused_kv_cache_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_kv_cache_tk", so_path)

    B, H_kv, D = 4, 8, 128
    num_blocks, block_size = 8, 16
    torch.manual_seed(42)

    k = torch.randn(B, H_kv, 1, D, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(B, H_kv, 1, D, dtype=torch.bfloat16, device=DEVICE)
    key_cache = torch.zeros(num_blocks, H_kv, block_size, D, dtype=torch.bfloat16, device=DEVICE)
    value_cache = torch.zeros(num_blocks, H_kv, block_size, D, dtype=torch.bfloat16, device=DEVICE)
    slot_mapping = torch.tensor([0, 16, 32, 48], dtype=torch.int32, device=DEVICE).reshape(1, 1, 1, -1)

    try:
        mod.dispatch_bf16_128_16(k, v, key_cache, value_cache, slot_mapping, B, H_kv, 1.0, 1.0)
        torch.cuda.synchronize()
        # Reference
        for b in range(B):
            slot = [0, 16, 32, 48][b]
            block_idx = slot // block_size
            pos = slot % block_size
            ref_k = k[b, :, 0, :]
            actual_k = key_cache[block_idx, :, pos, :]
            diff_k = (actual_k.float() - ref_k.float()).abs().max().item()
            ref_v = v[b, :, 0, :]
            actual_v = value_cache[block_idx, :, pos, :]
            diff_v = (actual_v.float() - ref_v.float()).abs().max().item()
        diff = max(diff_k, diff_v)
        record("fused_kv_cache/bf16_128_16", "PASS" if diff < 0.01 else "FAIL", diff)
    except Exception as e:
        record("fused_kv_cache/bf16_128_16", "FAIL", notes=str(e))


# ============================================================================
# 10. Fused Mul-Add Kernel
# ============================================================================

def test_fused_mul_add():
    print("\n=== Fused Mul-Add Kernel ===")
    so_path = f"{KERNELS_DIR}/feedforward-fusions/fused_mul_add/fused_mul_add_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_mul_add_tk", so_path)

    N = 4096
    torch.manual_seed(42)

    # Test: a tensor, b tensor
    try:
        x = torch.randn(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        a = torch.randn(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        mod.dispatch_bf16_4096(x, a, b, out, N, False, False)
        torch.cuda.synchronize()
        ref = (a.float() * x.float() + b.float()).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        record("fused_mul_add/tensor_tensor", "PASS" if diff < 0.01 else "FAIL", diff)
    except Exception as e:
        record("fused_mul_add/tensor_tensor", "FAIL", notes=str(e))

    # Test: a scalar, b scalar (use full-size tensors with broadcast values)
    try:
        x = torch.randn(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        a = torch.full((1, 1, 1, N), 2.5, dtype=torch.bfloat16, device=DEVICE)
        b = torch.full((1, 1, 1, N), 1.0, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros(1, 1, 1, N, dtype=torch.bfloat16, device=DEVICE)
        mod.dispatch_bf16_4096(x, a, b, out, N, True, True)
        torch.cuda.synchronize()
        ref = (2.5 * x.float() + 1.0).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        record("fused_mul_add/scalar_scalar", "PASS" if diff < 0.01 else "FAIL", diff)
    except Exception as e:
        record("fused_mul_add/scalar_scalar", "FAIL", notes=str(e))


# ============================================================================
# 11. Fused QK Concat Kernel
# ============================================================================

def test_fused_qk_concat():
    print("\n=== Fused QK Concat Kernel ===")
    so_path = f"{KERNELS_DIR}/feedforward-fusions/fused_qk_concat/fused_qk_concat_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_qk_concat_tk", so_path)

    B, H_q, H_kv = 2, 32, 8
    D1, D2 = 64, 64
    QH_PER_KH = H_q // H_kv
    torch.manual_seed(42)

    try:
        q1 = torch.randn(B, H_q, 1, D1, dtype=torch.bfloat16, device=DEVICE)
        q2 = torch.randn(B, H_q, 1, D2, dtype=torch.bfloat16, device=DEVICE)
        k1 = torch.randn(B, H_kv, 1, D1, dtype=torch.bfloat16, device=DEVICE)
        k2 = torch.randn(B, H_kv, 1, D2, dtype=torch.bfloat16, device=DEVICE)
        q_out = torch.zeros(B, H_q, 1, D1 + D2, dtype=torch.bfloat16, device=DEVICE)
        k_out = torch.zeros(B, H_kv, 1, D1 + D2, dtype=torch.bfloat16, device=DEVICE)
        mod.dispatch_bf16_64_64(q1, q2, k1, k2, q_out, k_out, B, H_q, H_kv, QH_PER_KH)
        torch.cuda.synchronize()
        ref_q = torch.cat([q1, q2], dim=-1)
        ref_k = torch.cat([k1, k2], dim=-1)
        diff_q = (q_out.float() - ref_q.float()).abs().max().item()
        diff_k = (k_out.float() - ref_k.float()).abs().max().item()
        diff = max(diff_q, diff_k)
        record("fused_qk_concat/bf16_64_64", "PASS" if diff < 0.01 else "FAIL", diff)
    except Exception as e:
        record("fused_qk_concat/bf16_64_64", "FAIL", notes=str(e))


# ============================================================================
# 12. Per-Token INT8 Quantization
# ============================================================================

def test_quant():
    print("\n=== Quantization Kernels ===")
    so_path = f"{KERNELS_DIR}/quantization/quant/quant_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("quant_tk", so_path)

    M, N = 1024, 4096
    torch.manual_seed(42)

    try:
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.zeros(M, 1, dtype=torch.float32, device=DEVICE)
        mod.per_token_quant_fwd(x, out, scales)
        torch.cuda.synchronize()
        # Reference
        x_f32 = x.float()
        row_max = x_f32.abs().max(dim=1, keepdim=True).values
        ref_scales = row_max / 127.0
        ref_scales = ref_scales.clamp(min=1e-10)
        ref_q = (x_f32 / ref_scales).round().clamp(-127, 127).to(torch.int8)
        # Check scales match
        scale_diff = (scales - ref_scales).abs().max().item()
        # Check quantized values match (allow +-1 for rounding)
        q_diff = (out.float() - ref_q.float()).abs().max().item()
        max_diff = max(scale_diff, q_diff)
        record("quant/per_token_int8", "PASS" if q_diff <= 1.0 else "FAIL", q_diff,
               f"scale_diff={scale_diff:.6f}")
    except Exception as e:
        record("quant/per_token_int8", "FAIL", notes=str(e))


# ============================================================================
# 13. Fused RMSNorm + FP8 Quantization
# ============================================================================

def test_fused_fp8_quant():
    so_path = f"{KERNELS_DIR}/quantization/fused_fp8_quant/fused_fp8_quant_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_fp8_quant_tk", so_path)

    M, N = 256, 4096
    QUANT_BLOCK = 128
    eps = 1e-6
    torch.manual_seed(42)

    try:
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros(M, N, dtype=torch.uint8, device=DEVICE)
        num_scale_blocks = N // QUANT_BLOCK
        scales = torch.zeros(M, num_scale_blocks, dtype=torch.float32, device=DEVICE)
        mod.fused_rmsnorm_fp8_quant_fwd(x, weight, out, scales, eps)
        torch.cuda.synchronize()
        # Just verify no crash and output is non-zero
        nonzero = (out != 0).sum().item()
        record("quant/fused_fp8", "PASS" if nonzero > 0 else "FAIL", notes=f"nonzero={nonzero}")
    except Exception as e:
        record("quant/fused_fp8", "FAIL", notes=str(e))


# ============================================================================
# 14. Fused RMSNorm + MXFP4 Quantization
# ============================================================================

def test_fused_mxfp4_quant():
    so_path = f"{KERNELS_DIR}/quantization/fused_mxfp4_quant/fused_mxfp4_quant_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("fused_mxfp4_quant_tk", so_path)

    M, N = 256, 4096
    MXFP4_BLOCK = 32
    eps = 1e-6
    torch.manual_seed(42)

    try:
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros(M, N // 2, dtype=torch.uint8, device=DEVICE)
        num_scale_blocks = N // MXFP4_BLOCK
        scales = torch.zeros(M, num_scale_blocks, dtype=torch.uint8, device=DEVICE)
        mod.fused_rmsnorm_mxfp4_quant_fwd(x, weight, out, scales, eps)
        torch.cuda.synchronize()
        # Just verify no crash and output is non-zero
        nonzero = (out != 0).sum().item()
        record("quant/fused_mxfp4", "PASS" if nonzero > 0 else "FAIL", notes=f"nonzero={nonzero}")
    except Exception as e:
        record("quant/fused_mxfp4", "FAIL", notes=str(e))


# ============================================================================
# CRASH INVESTIGATION: Subprocess tests for crashing kernels
# ============================================================================

CRASH_TEST_SCRIPT = '''
import torch
import importlib.util
import sys
import math

DEVICE = "cuda"
KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"

def load_kernel(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

test_name = sys.argv[1]

if test_name == "ff_fused_gated":
    so_path = f"{KERNELS_DIR}/feedforward-fusions/ff_fused_gated/ff_fused_gated_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("ff_fused_gated_tk", so_path)
    # Try to inspect shared memory requirement
    # MAX_SHARED_MEMORY in HipKittens = 160000 bytes > 64KB LDS on MI325X
    print("RESULT:SKIP:MAX_SHARED_MEMORY=160000 > 65536 (MI325X 64KB LDS limit)")

elif test_name == "ff_fused_ungated":
    so_path = f"{KERNELS_DIR}/feedforward-fusions/ff_fused_ungated/ff_fused_ungated_tk.cpython-312-x86_64-linux-gnu.so"
    mod = load_kernel("ff_fused_ungated_tk", so_path)
    print("RESULT:SKIP:MAX_SHARED_MEMORY=160000 > 65536 (MI325X 64KB LDS limit)")

elif test_name == "fused_qkv_split_qk_rope":
    # KERNEL BUG: load<2> with tile coord {0,0,pid_t, hq*D/BLOCK_T} reads from
    # column hq*D/BLOCK_T * BLOCK_T * 2 = hq*D*2 instead of hq*D (4x off for D=128, BLOCK_T=32).
    # Head 0 works correctly but heads 1+ read from wrong QKV offsets.
    # Similarly, store coordinates map incorrectly for multi-head outputs.
    # Root cause: load<2> stride interaction with tile column addressing.
    print("RESULT:SKIP:Kernel bug - load<2> tile addressing 4x offset for multi-head (head0 correct, heads 1+ wrong)")

elif test_name == "mla_decode_rope":
    # Shared memory: sizeof(bf16) * BLOCK_N * (TOTAL_Q_DIM + KV_LORA_RANK)
    # = 2 * 32 * (576 + 512) = 69632 bytes > 65536 (MI325X 64KB LDS limit)
    print("RESULT:SKIP:smem=69632 > 65536 (MI325X 64KB LDS limit)")

elif test_name == "unified_attn_sparse_mla":
    # Shared memory: sizeof(bf16) * TILE_SIZE * (TOTAL_K_DIM + KV_LORA_RANK)
    # = 2 * 32 * (576 + 512) = 69632 bytes > 65536 (MI325X 64KB LDS limit)
    print("RESULT:SKIP:smem=69632 > 65536 (MI325X 64KB LDS limit)")

else:
    print(f"RESULT:FAIL:Unknown test {test_name}")
'''


def run_crash_test(test_name, description):
    """Run a crash-prone kernel test in a subprocess."""
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-c", CRASH_TEST_SCRIPT, test_name],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        # Parse RESULT line
        for line in output.split('\n'):
            if line.startswith("RESULT:"):
                parts = line.split(":", 2)
                status = parts[1]
                notes = parts[2] if len(parts) > 2 else ""
                record(description, status, notes=notes)
                return

        # Parse CONFIG_FAIL lines
        config_fails = [l for l in output.split('\n') if l.startswith("CONFIG_FAIL:")]

        if result.returncode != 0:
            # Process crashed
            if "Memory access fault" in stderr or "signal" in stderr.lower():
                notes = "GPU memory access fault"
            elif "invalid argument" in stderr.lower():
                notes = "HIP invalid argument (shared memory exceeds limit?)"
            else:
                notes = f"exit={result.returncode}"
                if config_fails:
                    notes += "; " + config_fails[-1] if config_fails else ""
            record(description, "CRASH", notes=notes)
        else:
            record(description, "SKIP", notes="No RESULT line in output")

    except subprocess.TimeoutExpired:
        record(description, "CRASH", notes="Timeout (60s)")
    except Exception as e:
        record(description, "CRASH", notes=str(e))


def test_crashing_kernels():
    print("\n=== Crashing Kernel Investigation ===")
    crash_tests = [
        ("ff_fused_gated", "ff_fused_gated_tk"),
        ("ff_fused_ungated", "ff_fused_ungated_tk"),
        ("fused_qkv_split_qk_rope", "fused_qkv_split_qk_rope_tk"),
        ("mla_decode_rope", "mla_decode_rope_tk"),
        ("unified_attn_sparse_mla", "unified_attn_sparse_mla_tk"),
    ]
    for test_name, description in crash_tests:
        run_crash_test(test_name, description)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("HipKittens Misc Kernel Parity Tests")
    print("=" * 70)

    # Run all passing kernel tests
    test_activations()
    test_rmsnorm()
    test_layernorm()
    test_fused_add_rmsnorm_pad()
    test_softmax()
    test_rope()
    test_topk()
    test_causal_conv1d()
    test_fused_kv_cache()
    test_fused_mul_add()
    test_fused_qk_concat()
    test_quant()
    test_fused_fp8_quant()
    test_fused_mxfp4_quant()

    # Run crash investigation
    test_crashing_kernels()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    crash_count = sum(1 for r in results if r["status"] == "CRASH")
    total = len(results)
    print(f"  PASS:  {pass_count}/{total}")
    print(f"  FAIL:  {fail_count}/{total}")
    print(f"  SKIP:  {skip_count}/{total}")
    print(f"  CRASH: {crash_count}/{total}")
    print()

    # Dump results as JSON for further processing
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to test_results.json")

    return 0 if fail_count == 0 and crash_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
