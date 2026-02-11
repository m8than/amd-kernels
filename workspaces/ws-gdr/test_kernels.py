#!/usr/bin/env python3
"""
GPU parity tests for all GDR (Gated Delta Rule) kernels.
Tests both decode and prefill variants.
"""

import importlib.util
import sys
import os
import traceback
import torch
import torch.nn.functional as F
import math

KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"
DEVICE = "cuda"

results = []


def load_module(name, so_path):
    """Load a compiled .so pybind11 module."""
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(kernel_name, status, max_diff=None, notes=""):
    """Record test result."""
    results.append({
        "kernel": kernel_name,
        "status": status,
        "max_diff": max_diff,
        "notes": notes,
    })
    diff_str = f"{max_diff:.6e}" if max_diff is not None else "N/A"
    print(f"  [{status}] {kernel_name}: max_diff={diff_str} {notes}")


# =============================================================================
# GDR-DECODE Kernels
# =============================================================================

# --- 1. gdr_utils_tk ---
def test_gdr_utils():
    print("\n=== gdr_utils_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-decode/gdr_utils/gdr_utils_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("gdr_utils_tk", so)

    N = 1024
    torch.manual_seed(42)

    # abs_diff
    try:
        x = torch.randn(N, dtype=torch.float32, device=DEVICE)
        y = torch.randn(N, dtype=torch.float32, device=DEVICE)
        out = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        lib.abs_diff(x, y, out, N)
        ref = torch.abs(x - y)
        diff = (out - ref).abs().max().item()
        report("gdr_utils/abs_diff", "PASS" if diff < 1e-5 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/abs_diff", "ERROR", notes=str(e))

    # squared_diff
    try:
        out = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        lib.squared_diff(x, y, out, N)
        ref = (x - y) ** 2
        diff = (out - ref).abs().max().item()
        report("gdr_utils/squared_diff", "PASS" if diff < 1e-5 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/squared_diff", "ERROR", notes=str(e))

    # max_reduce
    try:
        inp = torch.randn(N, dtype=torch.float32, device=DEVICE)
        out = torch.zeros(1, dtype=torch.float32, device=DEVICE)
        lib.max_reduce(inp, out, N)
        ref = inp.max()
        diff = (out.item() - ref.item())
        diff = abs(diff)
        report("gdr_utils/max_reduce", "PASS" if diff < 1e-5 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/max_reduce", "ERROR", notes=str(e))

    # sum_reduce
    try:
        out = torch.zeros(1, dtype=torch.float32, device=DEVICE)
        lib.sum_reduce(inp, out, N)
        ref = inp.sum()
        diff = abs(out.item() - ref.item())
        report("gdr_utils/sum_reduce", "PASS" if diff < 1e-3 else "FAIL", diff,
               notes="float32 accumulation tolerance")
    except Exception as e:
        report("gdr_utils/sum_reduce", "ERROR", notes=str(e))

    # bf16_to_float
    try:
        inp_bf16 = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
        out_f32 = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        lib.bf16_to_float(inp_bf16, out_f32, N)
        ref = inp_bf16.float()
        diff = (out_f32 - ref).abs().max().item()
        report("gdr_utils/bf16_to_float", "PASS" if diff < 1e-7 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/bf16_to_float", "ERROR", notes=str(e))

    # float_to_bf16
    try:
        inp_f32 = torch.randn(N, dtype=torch.float32, device=DEVICE)
        out_bf16 = torch.zeros(N, dtype=torch.bfloat16, device=DEVICE)
        lib.float_to_bf16(inp_f32, out_bf16, N)
        ref = inp_f32.bfloat16()
        diff = (out_bf16.float() - ref.float()).abs().max().item()
        report("gdr_utils/float_to_bf16", "PASS" if diff < 1e-7 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/float_to_bf16", "ERROR", notes=str(e))

    # l2_normalize
    try:
        num_vecs = 16
        vec_len = 64
        inp = torch.randn(num_vecs * vec_len, dtype=torch.float32, device=DEVICE)
        out = torch.zeros_like(inp)
        lib.l2_normalize(inp, out, num_vecs, vec_len)
        # Reference: normalize each vector
        inp_2d = inp.view(num_vecs, vec_len)
        ref_2d = F.normalize(inp_2d, p=2, dim=1)
        ref = ref_2d.view(-1)
        diff = (out - ref).abs().max().item()
        report("gdr_utils/l2_normalize", "PASS" if diff < 1e-5 else "FAIL", diff)
    except Exception as e:
        report("gdr_utils/l2_normalize", "ERROR", notes=str(e))


# --- 2. causal_conv1d_split_qkv_tk (decode) ---
def test_causal_conv1d_split_qkv_decode():
    print("\n=== causal_conv1d_split_qkv_tk (decode) ===")
    so = os.path.join(KERNELS_DIR, "gdr-decode/causal_conv1d_split_qkv/causal_conv1d_split_qkv_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("causal_conv1d_split_qkv_tk", so)

    try:
        batch, key_dim, value_dim = 4, 64, 128
        dim = 2 * key_dim + value_dim  # 256
        seqlen = 1
        width = 4
        state_len = width - 1

        torch.manual_seed(42)
        x = torch.randn(batch, dim, seqlen, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.randn(dim, width, dtype=torch.bfloat16, device=DEVICE) * 0.1
        bias = torch.randn(dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
        conv_state = torch.randn(batch, dim, state_len, dtype=torch.bfloat16, device=DEVICE) * 0.1

        q = torch.zeros(batch, key_dim, seqlen, dtype=torch.bfloat16, device=DEVICE)
        k_out = torch.zeros(batch, key_dim, seqlen, dtype=torch.bfloat16, device=DEVICE)
        v_out = torch.zeros(batch, value_dim, seqlen, dtype=torch.bfloat16, device=DEVICE)

        # Reference: causal conv1d + silu + split
        # conv_state holds the last (width-1) inputs per feature
        # new_state = [state[:, :, 1:], x]
        # conv_output = sum(new_state * weight) + bias, then silu
        conv_state_ref = conv_state.clone()
        new_state = torch.cat([conv_state_ref[:, :, 1:], x], dim=2)  # (batch, dim, width)
        # conv: dot product along width dimension
        conv_out = (new_state * weight.unsqueeze(0)).sum(dim=2) + bias.unsqueeze(0)  # (batch, dim)
        conv_out_silu = conv_out * torch.sigmoid(conv_out)  # silu
        ref_q = conv_out_silu[:, :key_dim].unsqueeze(2)
        ref_k = conv_out_silu[:, key_dim:2*key_dim].unsqueeze(2)
        ref_v = conv_out_silu[:, 2*key_dim:].unsqueeze(2)

        lib.causal_conv1d_update_split_qkv(
            x, weight, bias, conv_state, conv_state,  # conv_state_indices=conv_state (unused)
            q, k_out, v_out,
            key_dim, value_dim,
            batch, dim, seqlen, width,
            batch,  # num_cache_lines
            True, True, False)

        diff_q = (q.float() - ref_q.float()).abs().max().item()
        diff_k = (k_out.float() - ref_k.float()).abs().max().item()
        diff_v = (v_out.float() - ref_v.float()).abs().max().item()
        max_diff = max(diff_q, diff_k, diff_v)
        report("causal_conv1d_split_qkv (decode)", "PASS" if max_diff < 0.05 else "FAIL",
               max_diff, f"q={diff_q:.4e} k={diff_k:.4e} v={diff_v:.4e}")
    except Exception as e:
        report("causal_conv1d_split_qkv (decode)", "ERROR", notes=str(e))


# --- 3. fused_qkvzba_split_tk ---
def test_fused_qkvzba_split():
    print("\n=== fused_qkvzba_split_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-decode/fused_qkvzba_split/fused_qkvzba_split_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("fused_qkvzba_split_tk", so)

    try:
        batch = 4
        num_heads_qk = 4
        num_heads_v = 16
        head_qk = 128
        head_v = 128
        v_per_qk = num_heads_v // num_heads_qk  # 4

        qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2  # 1280
        ba_dim_t = v_per_qk * 2  # 8
        qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v  # 3072

        torch.manual_seed(42)
        mixed_qkvz = torch.randn(batch, num_heads_qk * qkvz_dim_t, dtype=torch.bfloat16, device=DEVICE)
        mixed_ba = torch.randn(batch, num_heads_qk * ba_dim_t, dtype=torch.bfloat16, device=DEVICE)

        mixed_qkv = torch.zeros(batch, qkv_dim_t, dtype=torch.bfloat16, device=DEVICE)
        z = torch.zeros(batch, num_heads_v * head_v, dtype=torch.bfloat16, device=DEVICE)
        b_out = torch.zeros(batch, num_heads_v, dtype=torch.bfloat16, device=DEVICE)
        a_out = torch.zeros(batch, num_heads_v, dtype=torch.bfloat16, device=DEVICE)

        # Reference: manually reshuffle
        # mixed_qkvz layout per QK-head: [q(head_qk), k(head_qk), v(v_per_qk*head_v), z(v_per_qk*head_v)]
        ref_all_q = []
        ref_all_k = []
        ref_all_v = []
        ref_all_z = []
        ref_all_b = []
        ref_all_a = []

        for h in range(num_heads_qk):
            offset = h * qkvz_dim_t
            q_h = mixed_qkvz[:, offset:offset+head_qk]
            offset += head_qk
            k_h = mixed_qkvz[:, offset:offset+head_qk]
            offset += head_qk
            v_h = mixed_qkvz[:, offset:offset+v_per_qk*head_v]
            offset += v_per_qk * head_v
            z_h = mixed_qkvz[:, offset:offset+v_per_qk*head_v]

            ref_all_q.append(q_h)
            ref_all_k.append(k_h)
            # v and z are split into v_per_qk sub-heads
            for vi in range(v_per_qk):
                ref_all_v.append(v_h[:, vi*head_v:(vi+1)*head_v])
                ref_all_z.append(z_h[:, vi*head_v:(vi+1)*head_v])

            ba_offset = h * ba_dim_t
            b_h = mixed_ba[:, ba_offset:ba_offset+v_per_qk]
            a_h = mixed_ba[:, ba_offset+v_per_qk:ba_offset+2*v_per_qk]
            for vi in range(v_per_qk):
                ref_all_b.append(b_h[:, vi:vi+1])
                ref_all_a.append(a_h[:, vi:vi+1])

        ref_qkv = torch.cat(ref_all_q + ref_all_k + ref_all_v, dim=1)
        ref_z = torch.cat(ref_all_z, dim=1)
        ref_b = torch.cat(ref_all_b, dim=1)
        ref_a = torch.cat(ref_all_a, dim=1)

        lib.fused_qkvzba_split(
            mixed_qkv, z, b_out, a_out,
            mixed_qkvz, mixed_ba,
            batch, num_heads_qk, num_heads_v, head_qk, head_v)

        diff_qkv = (mixed_qkv.float() - ref_qkv.float()).abs().max().item()
        diff_z = (z.float() - ref_z.float()).abs().max().item()
        diff_b = (b_out.float() - ref_b.float()).abs().max().item()
        diff_a = (a_out.float() - ref_a.float()).abs().max().item()
        max_diff = max(diff_qkv, diff_z, diff_b, diff_a)
        report("fused_qkvzba_split", "PASS" if max_diff < 1e-5 else "FAIL",
               max_diff, f"qkv={diff_qkv:.4e} z={diff_z:.4e} b={diff_b:.4e} a={diff_a:.4e}")
    except Exception as e:
        report("fused_qkvzba_split", "ERROR", notes=str(e))


# --- 4. fused_recurrent_tk ---
def test_fused_recurrent():
    print("\n=== fused_recurrent_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-decode/fused_recurrent/fused_recurrent_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("fused_recurrent_tk", so)

    try:
        B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
        scale = K ** -0.5

        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=DEVICE)
        beta = (torch.rand(B, T, HV, dtype=torch.bfloat16, device=DEVICE) + 0.5)
        g = torch.randn(B, T, HV, dtype=torch.float32, device=DEVICE) * 0.1
        h0 = torch.randn(B * HV, K, V, dtype=torch.float32, device=DEVICE) * 0.01

        o = torch.zeros(B, T, HV, V, dtype=torch.bfloat16, device=DEVICE)
        ht = torch.zeros(B * HV, K, V, dtype=torch.float32, device=DEVICE)

        # cu_seqlens for non-packed mode (not used, but kernel reads it)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)

        # Reference: recurrent gated delta rule
        # For each batch, time step:
        #   For each HV head:
        #     gate = exp(g[b,t,hv])
        #     h = gate * h + beta[b,t,hv] * outer(k[b,t,h_mapped,K], v[b,t,hv,:])
        #     o[b,t,hv,:] = scale * q[b,t,h_mapped,:] @ h
        # Note: H may differ from HV; q/k use H heads mapped to HV
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        beta_f = beta.float()

        ref_o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=DEVICE)
        ref_ht = torch.zeros(B * HV, K, V, dtype=torch.float32, device=DEVICE)

        for b in range(B):
            for hv in range(HV):
                h_state = h0[b * HV + hv].clone()
                h_idx = hv % H  # map HV head to H head for q/k
                for t in range(T):
                    gate = torch.exp(g[b, t, hv])
                    h_state = gate * h_state + beta_f[b, t, hv] * torch.outer(k_f[b, t, h_idx], v_f[b, t, hv])
                    ref_o[b, t, hv] = scale * q_f[b, t, h_idx] @ h_state
                ref_ht[b * HV + hv] = h_state

        lib.fused_recurrent_fwd(
            q, k, v,
            g, g, g,  # gk and gv passed but not used
            beta, o,
            h0, ht,
            cu_seqlens,
            scale,
            T, B, H, HV, K, V,
            True, False, False,  # use_g, use_gk, use_gv
            True,   # is_beta_headwise
            True,   # use_initial_state
            True)   # store_final_state

        diff_o = (o.float() - ref_o).abs().max().item()
        diff_ht = (ht - ref_ht).abs().max().item()
        max_diff = max(diff_o, diff_ht)
        report("fused_recurrent", "PASS" if max_diff < 0.1 else "FAIL",
               max_diff, f"o={diff_o:.4e} ht={diff_ht:.4e}")
    except Exception as e:
        report("fused_recurrent", "ERROR", notes=str(e))


# --- 5. fused_sigmoid_gating_recurrent_tk ---
def test_fused_sigmoid_gating_recurrent():
    print("\n=== fused_sigmoid_gating_recurrent_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-decode/fused_sigmoid_gating_recurrent/fused_sigmoid_gating_recurrent_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("fused_sigmoid_gating_recurrent_tk", so)

    try:
        B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
        scale = K ** -0.5
        sp_beta_val = 1.0
        sp_thresh = 20.0

        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=DEVICE)
        k_t = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=DEVICE)
        v_t = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=DEVICE)
        b_t = torch.randn(B, T, HV, dtype=torch.bfloat16, device=DEVICE)
        a_t = torch.randn(B, T, HV, dtype=torch.bfloat16, device=DEVICE) * 0.1
        A_log = torch.randn(HV, dtype=torch.bfloat16, device=DEVICE) * 0.5
        dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=DEVICE) * 0.1

        num_states = B
        h0_source = torch.randn(num_states, HV, K, V, dtype=torch.float32, device=DEVICE) * 0.01
        h0_indices = torch.arange(B, dtype=torch.int32, device=DEVICE)

        o = torch.zeros(1, B * T, HV, V, dtype=torch.bfloat16, device=DEVICE)
        cu_seqlens = torch.tensor([0, T] * B, dtype=torch.int32, device=DEVICE)[:B+1]
        # Fix cu_seqlens for packed sequences
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
        for i in range(B):
            cu_seqlens[i + 1] = cu_seqlens[i] + T

        # Reference implementation
        q_f = q.float()
        k_f = k_t.float()
        v_f = v_t.float()
        b_f = b_t.float()
        a_f = a_t.float()
        A_log_f = A_log.float()
        dt_bias_f = dt_bias.float()

        ref_o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=DEVICE)
        h0_ref = h0_source.clone()

        for b_idx in range(B):
            for hv in range(HV):
                h_state = h0_ref[h0_indices[b_idx], hv].clone()
                h_idx = hv % H
                for t in range(T):
                    # Gating: g = -exp(A_log) * softplus(a + dt_bias)
                    x_val = a_f[b_idx, t, hv] + dt_bias_f[hv]
                    bx = sp_beta_val * x_val
                    if bx <= sp_thresh:
                        sp = torch.log(1.0 + torch.exp(bx)) / sp_beta_val
                    else:
                        sp = x_val
                    gate = torch.exp(-torch.exp(A_log_f[hv]) * sp)

                    # Beta from sigmoid
                    beta_val = torch.sigmoid(b_f[b_idx, t, hv])

                    h_state = gate * h_state + beta_val * torch.outer(k_f[b_idx, t, h_idx], v_f[b_idx, t, hv])
                    ref_o[b_idx, t, hv] = scale * q_f[b_idx, t, h_idx] @ h_state

        lib.fused_sigmoid_gating_recurrent(
            A_log, a_t, dt_bias,
            sp_beta_val, sp_thresh,
            q, k_t, v_t,
            b_t, o,
            h0_source, h0_indices,
            cu_seqlens,
            scale,
            T, B, H, HV, K, V,
            True, False)  # use_initial_state, use_qk_l2norm

        o_reshaped = o.view(B, T, HV, V)
        diff_o = (o_reshaped.float() - ref_o).abs().max().item()
        report("fused_sigmoid_gating_recurrent", "PASS" if diff_o < 0.1 else "FAIL",
               diff_o)
    except Exception as e:
        report("fused_sigmoid_gating_recurrent", "ERROR", notes=str(e))


# =============================================================================
# GDR-PREFILL Kernels
# =============================================================================

# --- 1. l2norm_tk ---
def test_l2norm():
    print("\n=== l2norm_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/l2norm/l2norm_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("l2norm_tk", so)

    T, D = 512, 128
    eps = 1e-6
    torch.manual_seed(42)

    # Forward
    try:
        x = torch.randn(T, D, dtype=torch.float32, device=DEVICE)
        y = torch.zeros_like(x)
        rstd = torch.zeros(T, dtype=torch.float32, device=DEVICE)

        lib.l2norm_fwd(x, y, rstd, eps)

        norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)
        ref_y = x / norm
        ref_rstd = (1.0 / norm).squeeze(-1)

        diff_y = (y - ref_y).abs().max().item()
        diff_rstd = (rstd - ref_rstd).abs().max().item()
        max_diff = max(diff_y, diff_rstd)
        report("l2norm_fwd", "PASS" if max_diff < 1e-5 else "FAIL",
               max_diff, f"y={diff_y:.4e} rstd={diff_rstd:.4e}")
    except Exception as e:
        report("l2norm_fwd", "ERROR", notes=str(e))

    # Backward
    try:
        dy = torch.randn(T, D, dtype=torch.float32, device=DEVICE)
        dx = torch.zeros_like(dy)

        lib.l2norm_bwd(y, rstd, dy, dx, eps)

        # Reference: dx = dy * rstd - sum(dy * y, dim=-1, keepdim=True) * y * rstd
        dot = torch.sum(dy * y, dim=-1, keepdim=True)
        ref_dx = dy * rstd.unsqueeze(-1) - dot * y * rstd.unsqueeze(-1)

        diff_dx = (dx - ref_dx).abs().max().item()
        report("l2norm_bwd", "PASS" if diff_dx < 1e-4 else "FAIL", diff_dx)
    except Exception as e:
        report("l2norm_bwd", "ERROR", notes=str(e))


# --- 2. cumsum_tk ---
def test_cumsum():
    print("\n=== cumsum_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/cumsum/cumsum_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("cumsum_tk", so)

    # Scalar cumsum
    try:
        B, T, H = 2, 256, 4
        BT = 64
        torch.manual_seed(42)

        s = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE)
        o = torch.zeros_like(s)

        lib.cumsum_scalar(s, o, B, T, H, BT)

        # Reference: chunk-local cumsum
        NT = T // BT
        ref = torch.zeros_like(s)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    start = nt * BT
                    end = start + BT
                    ref[b, start:end, h] = torch.cumsum(s[b, start:end, h], dim=0)

        diff = (o - ref).abs().max().item()
        report("cumsum_scalar", "PASS" if diff < 1e-4 else "FAIL", diff)
    except Exception as e:
        report("cumsum_scalar", "ERROR", notes=str(e))

    # Scalar cumsum reverse
    try:
        o_rev = torch.zeros_like(s)
        lib.cumsum_scalar(s, o_rev, B, T, H, BT, 1.0, False, True)

        ref_rev = torch.zeros_like(s)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    start = nt * BT
                    end = start + BT
                    ref_rev[b, start:end, h] = torch.cumsum(s[b, start:end, h].flip(0), dim=0).flip(0)

        diff = (o_rev - ref_rev).abs().max().item()
        report("cumsum_scalar_reverse", "PASS" if diff < 1e-4 else "FAIL", diff)
    except Exception as e:
        report("cumsum_scalar_reverse", "ERROR", notes=str(e))

    # Scalar cumsum with scale
    try:
        o_scaled = torch.zeros_like(s)
        sc = 2.0
        lib.cumsum_scalar(s, o_scaled, B, T, H, BT, sc, True, False)

        ref_sc = torch.zeros_like(s)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    start = nt * BT
                    end = start + BT
                    ref_sc[b, start:end, h] = torch.cumsum(s[b, start:end, h] * sc, dim=0)

        diff = (o_scaled - ref_sc).abs().max().item()
        report("cumsum_scalar_scaled", "PASS" if diff < 1e-3 else "FAIL", diff)
    except Exception as e:
        report("cumsum_scalar_scaled", "ERROR", notes=str(e))

    # Vector cumsum
    try:
        S = 32  # vector dimension
        sv = torch.randn(B, T, H, S, dtype=torch.float32, device=DEVICE)
        ov = torch.zeros_like(sv)

        lib.cumsum_vector(sv, ov, B, T, H, S, BT)

        ref_v = torch.zeros_like(sv)
        for b in range(B):
            for h in range(H):
                for si in range(S):
                    for nt in range(NT):
                        start = nt * BT
                        end = start + BT
                        ref_v[b, start:end, h, si] = torch.cumsum(sv[b, start:end, h, si], dim=0)

        diff = (ov - ref_v).abs().max().item()
        report("cumsum_vector", "PASS" if diff < 1e-4 else "FAIL", diff)
    except Exception as e:
        report("cumsum_vector", "ERROR", notes=str(e))


# --- 3. index_tk ---
def test_index():
    print("\n=== index_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/index/index_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("index_tk", so)

    # prepare_lens
    try:
        cu_seqlens = [0, 128, 320, 448]
        lens = lib.prepare_lens(cu_seqlens)
        ref_lens = [128, 192, 128]
        assert list(lens) == ref_lens, f"Expected {ref_lens}, got {list(lens)}"
        report("index/prepare_lens", "PASS", 0.0)
    except Exception as e:
        report("index/prepare_lens", "ERROR", notes=str(e))

    # prepare_cu_seqlens_from_lens
    try:
        result = lib.prepare_cu_seqlens_from_lens([128, 192, 128])
        ref = [0, 128, 320, 448]
        assert list(result) == ref, f"Expected {ref}, got {list(result)}"
        report("index/prepare_cu_seqlens_from_lens", "PASS", 0.0)
    except Exception as e:
        report("index/prepare_cu_seqlens_from_lens", "ERROR", notes=str(e))

    # prepare_position_ids
    try:
        cu_seqlens_short = [0, 3, 5]
        pos_ids = lib.prepare_position_ids(cu_seqlens_short)
        ref = [0, 1, 2, 0, 1]
        assert list(pos_ids) == ref, f"Expected {ref}, got {list(pos_ids)}"
        report("index/prepare_position_ids", "PASS", 0.0)
    except Exception as e:
        report("index/prepare_position_ids", "ERROR", notes=str(e))

    # prepare_sequence_ids
    try:
        seq_ids = lib.prepare_sequence_ids(cu_seqlens_short)
        ref = [0, 0, 0, 1, 1]
        assert list(seq_ids) == ref, f"Expected {ref}, got {list(seq_ids)}"
        report("index/prepare_sequence_ids", "PASS", 0.0)
    except Exception as e:
        report("index/prepare_sequence_ids", "ERROR", notes=str(e))

    # prepare_chunk_offsets
    try:
        chunk_offsets = lib.prepare_chunk_offsets(cu_seqlens, 64)
        ref = [0, 2, 5, 7]
        assert list(chunk_offsets) == ref, f"Expected {ref}, got {list(chunk_offsets)}"
        report("index/prepare_chunk_offsets", "PASS", 0.0)
    except Exception as e:
        report("index/prepare_chunk_offsets", "ERROR", notes=str(e))

    # get_max_num_splits
    try:
        max_splits = lib.get_max_num_splits(cu_seqlens, 64)
        ref = 3  # max(ceil(128/64), ceil(192/64), ceil(128/64)) = max(2, 3, 2) = 3
        assert max_splits == ref, f"Expected {ref}, got {max_splits}"
        report("index/get_max_num_splits", "PASS", 0.0)
    except Exception as e:
        report("index/get_max_num_splits", "ERROR", notes=str(e))

    # gpu_prepare_chunk_indices
    try:
        cu_seqlens_t = torch.tensor([0, 128, 320, 448], dtype=torch.int32, device=DEVICE)
        chunk_size = 64
        N = 3  # num sequences

        chunk_offsets_ref = [0, 2, 5, 7]
        total_chunks = chunk_offsets_ref[-1]
        chunk_indices = torch.zeros(total_chunks, 2, dtype=torch.int32, device=DEVICE)
        chunk_offsets_t = torch.tensor(chunk_offsets_ref, dtype=torch.int32, device=DEVICE)

        lib.gpu_prepare_chunk_indices(cu_seqlens_t, chunk_indices, chunk_offsets_t, N, chunk_size)

        # Reference: chunk_indices[i] = (seq_id, chunk_within_seq)
        ref_indices = [(0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,1)]
        indices_cpu = chunk_indices.cpu().tolist()
        match = all(indices_cpu[i] == list(ref_indices[i]) for i in range(total_chunks))
        report("index/gpu_prepare_chunk_indices", "PASS" if match else "FAIL", 0.0,
               f"indices={indices_cpu[:4]}...")
    except Exception as e:
        report("index/gpu_prepare_chunk_indices", "ERROR", notes=str(e))


# --- 4. op_tk ---
def test_ops():
    print("\n=== op_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/op/op_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("op_tk", so)

    try:
        N = 1024
        torch.manual_seed(42)
        inp = torch.randn(N, dtype=torch.float32, device=DEVICE)

        out_exp = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_exp2 = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_log = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_log2 = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_safe_exp = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_softplus = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_sigmoid = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        out_silu = torch.zeros(N, dtype=torch.float32, device=DEVICE)

        lib.test_ops(inp, out_exp, out_exp2, out_log, out_log2,
                     out_safe_exp, out_softplus, out_sigmoid, out_silu, N)

        # Reference values
        ref_exp = torch.exp(inp)
        ref_exp2 = torch.exp2(inp)
        ref_log = torch.log(inp)  # will have NaN for negative values
        ref_log2 = torch.log2(inp)
        ref_safe_exp = torch.where(inp <= 0, torch.exp(inp), torch.zeros_like(inp))
        ref_softplus = F.softplus(inp)
        ref_sigmoid = torch.sigmoid(inp)
        ref_silu = F.silu(inp)

        # For log/log2 only compare valid (positive) values
        pos_mask = inp > 0

        results_ops = {
            "exp": (out_exp, ref_exp, None),
            "exp2": (out_exp2, ref_exp2, None),
            "log": (out_log, ref_log, pos_mask),
            "log2": (out_log2, ref_log2, pos_mask),
            "safe_exp": (out_safe_exp, ref_safe_exp, None),
            "softplus": (out_softplus, ref_softplus, None),
            "sigmoid": (out_sigmoid, ref_sigmoid, None),
            "silu": (out_silu, ref_silu, None),
        }

        for name, (out, ref, mask) in results_ops.items():
            if mask is not None:
                d = (out[mask] - ref[mask]).abs().max().item()
            else:
                d = (out - ref).abs().max().item()
            report(f"op/{name}", "PASS" if d < 1e-4 else "FAIL", d)
    except Exception as e:
        report("op/test_ops", "ERROR", notes=str(e))


# --- 5. solve_tril_tk ---
def test_solve_tril():
    print("\n=== solve_tril_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/solve_tril/solve_tril_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("solve_tril_tk", so)

    for BT in [16, 32, 64]:
        try:
            B_dim, T_dim, H_dim = 1, BT, 2  # T must be multiple of BT
            torch.manual_seed(42)

            # A: strictly lower triangular, stored as (B, T, H, BT)
            A = torch.randn(B_dim, T_dim, H_dim, BT, dtype=torch.float32, device=DEVICE) * 0.1
            # Zero out upper triangle + diagonal within each BT chunk
            for t_start in range(0, T_dim, BT):
                for i in range(BT):
                    A[:, t_start + i, :, i:] = 0.0  # zero diagonal and above

            Ai = torch.zeros_like(A)
            lib.solve_tril(A, Ai, B_dim, T_dim, H_dim, BT)

            # Reference: (I + A)^{-1}
            # Reconstruct full matrix and invert
            max_err = 0.0
            for b in range(B_dim):
                for h in range(H_dim):
                    for nt in range(T_dim // BT):
                        cs = nt * BT
                        # Extract BT x BT lower triangular from A
                        A_block = torch.zeros(BT, BT, dtype=torch.float32, device=DEVICE)
                        for i in range(BT):
                            A_block[i, :i] = A[b, cs + i, h, :i]
                        M = torch.eye(BT, device=DEVICE) + A_block
                        M_inv = torch.linalg.inv(M)
                        # Extract kernel result
                        Ai_block = torch.zeros(BT, BT, dtype=torch.float32, device=DEVICE)
                        for i in range(BT):
                            Ai_block[i, :BT] = Ai[b, cs + i, h, :BT]
                        err = (Ai_block - M_inv).abs().max().item()
                        max_err = max(max_err, err)

            report(f"solve_tril (BT={BT})", "PASS" if max_err < 1e-3 else "FAIL", max_err)
        except Exception as e:
            report(f"solve_tril (BT={BT})", "ERROR", notes=str(e))


# --- 6. wy_representation_tk ---
def test_wy_representation():
    print("\n=== wy_representation_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/wy_representation/wy_representation_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("wy_representation_tk", so)

    # chunk_scaled_dot_kkt
    try:
        B, T, H, K, BT = 1, 128, 2, 64, 64
        torch.manual_seed(42)

        k = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5
        g = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5
        beta = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5

        A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEVICE)
        lib.chunk_scaled_dot_kkt(k, g, beta, A, B, T, H, K, BT, True)

        # Reference
        NT = T // BT
        ref_A = torch.zeros_like(A)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    cs = nt * BT
                    for i in range(BT):
                        for j in range(i):
                            dot = (k[b, cs+i, h] * k[b, cs+j, h]).sum()
                            gd = torch.exp(g[b, cs+i, h] - g[b, cs+j, h])
                            ref_A[b, cs+i, h, j] = beta[b, cs+i, h] * dot * gd

        diff = (A - ref_A).abs().max().item()
        report("wy_representation/kkt", "PASS" if diff < 1e-3 else "FAIL", diff)
    except Exception as e:
        report("wy_representation/kkt", "ERROR", notes=str(e))

    # recompute_w_u
    try:
        V_dim = 64
        v = torch.randn(B, T, H, V_dim, dtype=torch.float32, device=DEVICE) * 0.5
        # Need solved A (lower tri) for recompute_w_u
        # Use identity-like A for simplicity
        A_solved = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEVICE)
        for t_idx in range(T):
            local_idx = t_idx % BT
            A_solved[:, t_idx, :, local_idx] = 1.0  # diagonal = 1 (I)

        w = torch.zeros(B, T, H, K, dtype=torch.float32, device=DEVICE)
        u = torch.zeros(B, T, H, V_dim, dtype=torch.float32, device=DEVICE)

        lib.recompute_w_u(k, v, beta, A_solved, g, w, u, B, T, H, K, V_dim, BT, True)

        # With A = I (identity), w_i = beta_i * k_i, u_i = beta_i * v_i (approximately)
        # This is a smoke test - the exact reference depends on the full formula
        report("wy_representation/recompute_w_u", "PASS" if not torch.isnan(w).any() else "FAIL",
               notes="smoke test, no NaN in output")
    except Exception as e:
        report("wy_representation/recompute_w_u", "ERROR", notes=str(e))


# --- 7. fused_gdn_gating_prefill_tk ---
def test_fused_gdn_gating_prefill():
    print("\n=== fused_gdn_gating_prefill_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/fused_gdn_gating_prefill/fused_gdn_gating_prefill_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("fused_gdn_gating_prefill_tk", so)

    try:
        S, H = 256, 16
        sp_beta_val, sp_thresh = 1.0, 20.0

        torch.manual_seed(42)
        A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5
        a = torch.randn(S, H, dtype=torch.float32, device=DEVICE)
        b = torch.randn(S, H, dtype=torch.float32, device=DEVICE)
        dt_bias = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.1

        g = torch.zeros(S, H, dtype=torch.float32, device=DEVICE)
        beta_out = torch.zeros(S, H, dtype=torch.float32, device=DEVICE)

        lib.fused_gdn_gating(A_log, a, b, dt_bias, g, beta_out, S, H, sp_beta_val, sp_thresh)

        # Reference
        x = a + dt_bias.unsqueeze(0)
        bx = sp_beta_val * x
        sp = torch.where(bx <= sp_thresh, torch.log(1.0 + torch.exp(bx)) / sp_beta_val, x)
        ref_g = -torch.exp(A_log.unsqueeze(0)) * sp
        ref_beta = torch.sigmoid(b)

        diff_g = (g - ref_g).abs().max().item()
        diff_beta = (beta_out - ref_beta).abs().max().item()
        max_diff = max(diff_g, diff_beta)
        report("fused_gdn_gating_prefill", "PASS" if max_diff < 1e-4 else "FAIL",
               max_diff, f"g={diff_g:.4e} beta={diff_beta:.4e}")
    except Exception as e:
        report("fused_gdn_gating_prefill", "ERROR", notes=str(e))


# --- 8. fused_cumsum_kkt_tk ---
def test_fused_cumsum_kkt():
    print("\n=== fused_cumsum_kkt_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/fused_cumsum_kkt/fused_cumsum_kkt_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("fused_cumsum_kkt_tk", so)

    try:
        B, T, H, Hg, K, BT = 1, 128, 4, 4, 64, 64
        torch.manual_seed(42)

        g = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.1
        k = torch.randn(B, T, Hg, K, dtype=torch.float32, device=DEVICE) * 0.5
        beta = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5

        g_cumsum = torch.zeros(B, T, H, dtype=torch.float32, device=DEVICE)
        A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEVICE)

        lib.fused_cumsum_kkt(g, k, beta, g_cumsum, A, B, T, H, Hg, K, BT)

        # Reference
        NT = T // BT
        ref_g_cumsum = torch.zeros_like(g_cumsum)
        ref_A = torch.zeros_like(A)

        for b in range(B):
            for h in range(H):
                hg = h // (H // Hg)
                for nt in range(NT):
                    cs = nt * BT
                    running = 0.0
                    for i in range(BT):
                        running += g[b, cs + i, h].item()
                        ref_g_cumsum[b, cs + i, h] = running

                    for i in range(BT):
                        for j in range(i):
                            dot = (k[b, cs+i, hg] * k[b, cs+j, hg]).sum().item()
                            gd = ref_g_cumsum[b, cs+i, h].item() - ref_g_cumsum[b, cs+j, h].item()
                            se = math.exp(gd) if gd <= 0 else 0.0
                            ref_A[b, cs+i, h, j] = beta[b, cs+i, h].item() * dot * se

        diff_g = (g_cumsum - ref_g_cumsum).abs().max().item()
        diff_A = (A - ref_A).abs().max().item()
        max_diff = max(diff_g, diff_A)
        report("fused_cumsum_kkt", "PASS" if max_diff < 1e-3 else "FAIL",
               max_diff, f"g_cumsum={diff_g:.4e} A={diff_A:.4e}")
    except Exception as e:
        report("fused_cumsum_kkt", "ERROR", notes=str(e))


# --- 9. chunk_tk ---
def test_chunk():
    print("\n=== chunk_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/chunk/chunk_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("chunk_tk", so)

    # Test chunk_cumsum
    try:
        B, T, H, K, V, BT = 1, 128, 2, 64, 64, 64
        torch.manual_seed(42)

        g = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.1
        g_cumsum = torch.zeros_like(g)

        lib.chunk_cumsum(g, g_cumsum, B, T, H, BT)

        NT = T // BT
        ref = torch.zeros_like(g)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    cs = nt * BT
                    ref[b, cs:cs+BT, h] = torch.cumsum(g[b, cs:cs+BT, h], dim=0)

        diff = (g_cumsum - ref).abs().max().item()
        report("chunk/cumsum", "PASS" if diff < 1e-4 else "FAIL", diff)
    except Exception as e:
        report("chunk/cumsum", "ERROR", notes=str(e))

    # Test chunk_compute_A
    try:
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5
        beta = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5
        A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEVICE)

        # Use the cumsum we just computed
        lib.chunk_compute_A(k, g_cumsum, beta, A, B, T, H, K, BT)

        # Reference
        ref_A = torch.zeros_like(A)
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    cs = nt * BT
                    for i in range(BT):
                        for j in range(i):
                            dot = (k[b, cs+i, h] * k[b, cs+j, h]).sum()
                            gd = torch.exp(g_cumsum[b, cs+i, h] - g_cumsum[b, cs+j, h])
                            ref_A[b, cs+i, h, j] = beta[b, cs+i, h] * dot * gd

        diff = (A - ref_A).abs().max().item()
        report("chunk/compute_A", "PASS" if diff < 1e-3 else "FAIL", diff)
    except Exception as e:
        report("chunk/compute_A", "ERROR", notes=str(e))

    # Test chunk_solve_tril (in-place)
    try:
        A_for_solve = A.clone()
        lib.chunk_solve_tril(A_for_solve, B, T, H, BT)

        # Verify: (I+A) @ A_solved should be I
        max_err = 0.0
        for b in range(B):
            for h in range(H):
                for nt in range(NT):
                    cs = nt * BT
                    A_block = torch.zeros(BT, BT, dtype=torch.float32, device=DEVICE)
                    Ai_block = torch.zeros(BT, BT, dtype=torch.float32, device=DEVICE)
                    for i in range(BT):
                        A_block[i, :i] = A[b, cs+i, h, :i]
                        Ai_block[i, :BT] = A_for_solve[b, cs+i, h, :BT]
                    M = torch.eye(BT, device=DEVICE) + A_block
                    check = M @ Ai_block
                    err = (check - torch.eye(BT, device=DEVICE)).abs().max().item()
                    max_err = max(max_err, err)

        report("chunk/solve_tril", "PASS" if max_err < 1e-3 else "FAIL", max_err)
    except Exception as e:
        report("chunk/solve_tril", "ERROR", notes=str(e))

    # Test full pipeline
    try:
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE) * 0.5
        g2 = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.1
        g_cumsum2 = torch.zeros_like(g2)
        beta2 = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5
        A2 = torch.zeros(B, T, H, BT, dtype=torch.float32, device=DEVICE)
        k2 = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5
        w_out = torch.zeros(B, T, H, K, dtype=torch.float32, device=DEVICE)
        u_out = torch.zeros(B, T, H, V, dtype=torch.float32, device=DEVICE)

        lib.chunk_pipeline(k2, v, g2, g_cumsum2, beta2, A2, w_out, u_out, B, T, H, K, V, BT)

        # Smoke test: outputs should not contain NaN
        has_nan = torch.isnan(w_out).any() or torch.isnan(u_out).any()
        report("chunk/pipeline", "PASS" if not has_nan else "FAIL",
               notes="smoke test, no NaN" if not has_nan else "NaN in output")
    except Exception as e:
        report("chunk/pipeline", "ERROR", notes=str(e))


# --- 10. chunk_delta_h_tk ---
def test_chunk_delta_h():
    print("\n=== chunk_delta_h_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/chunk_delta_h/chunk_delta_h_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("chunk_delta_h_tk", so)

    try:
        B, T, H, K, V, BT = 1, 128, 2, 64, 64, 64
        NT = T // BT
        torch.manual_seed(42)

        k = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
        w = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
        u = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE) * 0.1
        g = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.1

        h0 = torch.zeros(B, H, K, V, dtype=torch.float32, device=DEVICE)
        h = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=DEVICE)
        v_new = torch.zeros(B, T, H, V, dtype=torch.float32, device=DEVICE)
        ht = torch.zeros(B, H, K, V, dtype=torch.float32, device=DEVICE)

        lib.chunk_delta_h_fwd(k, w, u, g,
                              h0, h, v_new, ht,
                              B, T, H, K, V, BT,
                              True, False, True, True)

        # Smoke test: check no NaN
        has_nan = (torch.isnan(h).any() or torch.isnan(v_new).any() or
                   torch.isnan(ht).any())
        report("chunk_delta_h", "PASS" if not has_nan else "FAIL",
               notes="smoke test, no NaN" if not has_nan else "NaN in output")
    except Exception as e:
        report("chunk_delta_h", "ERROR", notes=str(e))


# --- 11. chunk_o_tk ---
def test_chunk_o():
    print("\n=== chunk_o_tk ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/chunk_o/chunk_o_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("chunk_o_tk", so)

    try:
        B, T, H, K, V, BT = 1, 128, 2, 64, 64, 64
        NT = T // BT
        scale = 1.0 / math.sqrt(K)
        torch.manual_seed(42)

        q = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.5
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE) * 0.5
        g = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE) * 0.1
        h = torch.randn(B, NT, H, K, V, dtype=torch.float32, device=DEVICE) * 0.1

        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=DEVICE)

        lib.chunk_fwd_o(q, k, v, h, g, o, scale, B, T, H, K, V, BT, True)

        # Reference
        ref_o = torch.zeros_like(o)
        for b in range(B):
            for hh in range(H):
                for nt in range(NT):
                    cs = nt * BT
                    for i in range(BT):
                        ti = cs + i
                        gi = g[b, ti, hh]
                        # Inter-chunk
                        inter = q[b, ti, hh] @ h[b, nt, hh] * torch.exp(gi)
                        # Intra-chunk
                        intra = torch.zeros(V, dtype=torch.float32, device=DEVICE)
                        for j in range(i + 1):
                            tj = cs + j
                            qk_val = (q[b, ti, hh] * k[b, tj, hh]).sum()
                            gj = g[b, tj, hh]
                            qk_val = qk_val * torch.exp(gi - gj)
                            intra = intra + qk_val * v[b, tj, hh]
                        ref_o[b, ti, hh] = (inter + intra) * scale

        diff = (o - ref_o).abs().max().item()
        report("chunk_fwd_o", "PASS" if diff < 0.05 else "FAIL", diff)
    except Exception as e:
        report("chunk_fwd_o", "ERROR", notes=str(e))


# --- 12. causal_conv1d_fwd_split_qkv_tk (prefill) ---
def test_causal_conv1d_fwd_split_qkv_prefill():
    print("\n=== causal_conv1d_fwd_split_qkv_tk (prefill) ===")
    so = os.path.join(KERNELS_DIR, "gdr-prefill/causal_conv1d_fwd_split_qkv/causal_conv1d_fwd_split_qkv_tk.cpython-312-x86_64-linux-gnu.so")
    lib = load_module("causal_conv1d_fwd_split_qkv_tk", so)

    try:
        num_seqs = 2
        seq_lens = [32, 48]
        total_tokens = sum(seq_lens)
        k_dim, v_dim = 64, 64
        dim = 2 * k_dim + v_dim  # 192
        kernel_width = 4
        max_seqlen = max(seq_lens)

        torch.manual_seed(42)
        # x is (dim, total_tokens) - dim-major
        x = torch.randn(dim, total_tokens, dtype=torch.float32, device=DEVICE)
        w = torch.randn(dim, kernel_width, dtype=torch.float32, device=DEVICE) * 0.5
        bias = torch.randn(dim, dtype=torch.float32, device=DEVICE) * 0.1
        query_start_loc = torch.tensor([0, 32, 80], dtype=torch.int32, device=DEVICE)

        q_out = torch.zeros(total_tokens, k_dim, dtype=torch.float32, device=DEVICE)
        k_out = torch.zeros(total_tokens, k_dim, dtype=torch.float32, device=DEVICE)
        v_out = torch.zeros(total_tokens, v_dim, dtype=torch.float32, device=DEVICE)

        stride_x_dim = total_tokens  # stride along dim axis
        stride_x_token = 1  # stride along token axis

        lib.causal_conv1d_fwd_split(
            x, w, bias, query_start_loc,
            q_out, k_out, v_out,
            dim, k_dim, v_dim, kernel_width,
            total_tokens, num_seqs, max_seqlen,
            stride_x_dim, stride_x_token,
            True, True)

        # Reference: per-sequence causal conv1d + silu + split
        ref_q = torch.zeros(total_tokens, k_dim, dtype=torch.float32, device=DEVICE)
        ref_k = torch.zeros(total_tokens, k_dim, dtype=torch.float32, device=DEVICE)
        ref_v = torch.zeros(total_tokens, v_dim, dtype=torch.float32, device=DEVICE)

        for s in range(num_seqs):
            seq_start = query_start_loc[s].item()
            seqlen = query_start_loc[s + 1].item() - seq_start
            for feat in range(dim):
                state = [0.0] * (kernel_width - 1)
                for t in range(seqlen):
                    gt = seq_start + t
                    x_curr = x[feat, gt].item()
                    acc = bias[feat].item()
                    for wi in range(kernel_width - 1):
                        acc += state[wi] * w[feat, wi].item()
                    acc += x_curr * w[feat, kernel_width - 1].item()
                    # Shift state
                    state = state[1:] + [x_curr]
                    # SiLU
                    acc = acc / (1.0 + math.exp(-acc)) if abs(acc) < 50 else (acc if acc > 0 else 0.0)
                    if feat < k_dim:
                        ref_q[gt, feat] = acc
                    elif feat < 2 * k_dim:
                        ref_k[gt, feat - k_dim] = acc
                    else:
                        ref_v[gt, feat - 2 * k_dim] = acc

        diff_q = (q_out - ref_q).abs().max().item()
        diff_k = (k_out - ref_k).abs().max().item()
        diff_v = (v_out - ref_v).abs().max().item()
        max_diff = max(diff_q, diff_k, diff_v)
        report("causal_conv1d_fwd_split_qkv (prefill)", "PASS" if max_diff < 1e-3 else "FAIL",
               max_diff, f"q={diff_q:.4e} k={diff_k:.4e} v={diff_v:.4e}")
    except Exception as e:
        report("causal_conv1d_fwd_split_qkv (prefill)", "ERROR", notes=str(e))


# =============================================================================
# Main
# =============================================================================

def print_summary():
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Kernel':<50} {'Status':<8} {'Max Diff':<15} {'Notes'}")
    print("-" * 80)
    pass_count = 0
    fail_count = 0
    error_count = 0
    for r in results:
        diff_str = f"{r['max_diff']:.6e}" if r['max_diff'] is not None else "N/A"
        print(f"{r['kernel']:<50} {r['status']:<8} {diff_str:<15} {r['notes']}")
        if r['status'] == 'PASS':
            pass_count += 1
        elif r['status'] == 'FAIL':
            fail_count += 1
        else:
            error_count += 1
    print("-" * 80)
    print(f"Total: {len(results)} | PASS: {pass_count} | FAIL: {fail_count} | ERROR: {error_count}")


if __name__ == "__main__":
    torch.manual_seed(42)

    print("GDR Kernel Parity Tests")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Simple kernels
    test_gdr_utils()
    test_ops()
    test_l2norm()
    test_cumsum()
    test_index()

    # Medium kernels
    test_solve_tril()
    test_wy_representation()
    test_fused_gdn_gating_prefill()
    test_fused_cumsum_kkt()

    # Complex decode kernels
    test_causal_conv1d_split_qkv_decode()
    test_fused_qkvzba_split()
    test_fused_recurrent()
    test_fused_sigmoid_gating_recurrent()

    # Complex prefill kernels
    test_chunk()
    test_chunk_delta_h()
    test_chunk_o()
    test_causal_conv1d_fwd_split_qkv_prefill()

    print_summary()
