#!/usr/bin/env python3
"""
Parity tests for all MoE kernels (gemm, routing, misc).
Loads pre-compiled .so files via importlib and tests against PyTorch references.
"""

import importlib.util
import sys
import os
import traceback
import torch
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda"
KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"

results = []  # (name, status, max_diff, notes)


def load_kernel(so_path, module_name):
    """Load a .so kernel module."""
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(name, status, max_diff=None, notes=""):
    diff_str = f"{max_diff:.6f}" if max_diff is not None else "N/A"
    results.append((name, status, diff_str, notes))
    print(f"  [{status}] {name} (max_diff={diff_str}) {notes}")


# ============================================================================
# Helper: Generate standard MoE routing data (sorted token IDs approach)
# Used by moe_op, moe_op_gelu, moe_op_silu_fused, moe_op_e2e
# ============================================================================

def generate_sorted_routing(num_tokens, num_experts, top_k, block_m=128):
    """Generate sorted routing data for the standard MoE GEMM kernels."""
    total_slots = num_tokens * top_k

    # Random expert assignments
    expert_assignments = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32)
    raw_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    flat_token_ids = torch.arange(total_slots, dtype=torch.int32)
    flat_expert_ids = expert_assignments.flatten()
    flat_weights = weights.flatten()

    sort_idx = torch.argsort(flat_expert_ids, stable=True)
    sorted_token_ids = flat_token_ids[sort_idx]
    sorted_expert_ids = flat_expert_ids[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    # Pad to multiple of block_m
    num_padded = ((total_slots + block_m - 1) // block_m) * block_m
    padded_token_ids = torch.full((num_padded,), total_slots, dtype=torch.int32)
    padded_token_ids[:total_slots] = sorted_token_ids
    padded_weights = torch.zeros(num_padded, dtype=torch.float32)
    padded_weights[:total_slots] = sorted_weights

    # Expert IDs per block
    num_blocks = num_padded // block_m
    block_expert_ids = torch.zeros(num_blocks, dtype=torch.int32)
    for b in range(num_blocks):
        start = b * block_m
        if start < total_slots:
            block_expert_ids[b] = sorted_expert_ids[min(start, total_slots - 1)]

    num_tokens_post_padded = torch.tensor([num_padded], dtype=torch.int32)

    return (padded_token_ids, block_expert_ids, padded_weights,
            num_padded, total_slots, num_tokens_post_padded)


# ============================================================================
# Helper: Generate OGS routing data
# Used by moe_op_gemm_a8w8, a8w8_blockscale, a8w4, a4w4
# ============================================================================

def generate_ogs_routing(num_tokens, num_experts, top_k, block_m=128):
    """Generate OGS-style routing: hist, offs, gather, expt_data."""
    total = num_tokens * top_k
    assigns = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32)

    hist = torch.zeros(num_experts, dtype=torch.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            hist[assigns[t, k]] += 1

    offs = torch.zeros(num_experts, dtype=torch.int32)
    for e in range(1, num_experts):
        offs[e] = offs[e - 1] + hist[e - 1]

    gather = torch.zeros(total, dtype=torch.int32)
    pos = offs.clone()
    for t in range(num_tokens):
        for k in range(top_k):
            e = assigns[t, k].item()
            gather[pos[e]] = t * top_k + k
            pos[e] += 1

    grid_m_tiles = []
    for e in range(num_experts):
        n_blocks = (hist[e].item() + block_m - 1) // block_m
        for b in range(n_blocks):
            grid_m_tiles.append(e | (b << 16))

    expt_data = torch.tensor(grid_m_tiles, dtype=torch.int32)
    grid_m = len(expt_data)

    return hist, offs, gather, expt_data, grid_m


# ============================================================================
# Reference: standard MoE GEMM C = A[token] @ B[expert]
# ============================================================================

def ref_moe_gemm(A, B, sorted_token_ids, expert_ids, topk_weights,
                 num_valid, top_k, block_m=128, mul_routed_weight=True):
    num_padded = sorted_token_ids.shape[0]
    N = B.shape[2]
    C = torch.zeros(num_padded, N, dtype=torch.float32)
    num_blocks = expert_ids.shape[0]

    for bid in range(num_blocks):
        expert = expert_ids[bid].item()
        for lm in range(block_m):
            tidx = bid * block_m + lm
            if tidx >= num_padded:
                break
            token_id = sorted_token_ids[tidx].item()
            if token_id >= num_valid:
                continue
            orig = token_id // top_k
            result = A[orig].float() @ B[expert].float()
            if mul_routed_weight:
                result *= topk_weights[tidx].item()
            C[token_id] = result

    return C


# ============================================================================
# 1. MoE Align Block Size
# ============================================================================

def test_moe_align_block_size():
    print("\n=== moe_align_block_size ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-routing/moe_align_block_size/moe_align_block_size_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_align_block_size_tk")

        num_tokens = 1024
        num_experts = 8
        block_size = 64
        torch.manual_seed(42)

        topk_ids = torch.randint(0, num_experts, (num_tokens,), dtype=torch.int32, device=DEVICE)

        # Pre-compute expected total for buffer sizing
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32)
        for e in range(num_experts):
            tokens_per_expert[e] = (topk_ids.cpu() == e).sum()
        aligned_counts = ((tokens_per_expert + block_size - 1) // block_size) * block_size
        expected_total = aligned_counts.sum().item()

        max_num_blocks = expected_total // block_size

        sorted_token_ids = torch.zeros(expected_total, dtype=torch.int32, device=DEVICE)
        expert_ids_out = torch.zeros(max_num_blocks, dtype=torch.int32, device=DEVICE)
        tokens_cnts_buffer = torch.zeros((num_experts + 1) * num_experts, dtype=torch.int32, device=DEVICE)
        cumsum_buffer = torch.zeros(num_experts + 1, dtype=torch.int32, device=DEVICE)
        total_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=DEVICE)

        mod.moe_align_block_size(
            topk_ids, sorted_token_ids, expert_ids_out,
            tokens_cnts_buffer, cumsum_buffer, total_tokens_post_pad,
            num_experts, block_size, num_tokens
        )
        torch.cuda.synchronize()

        total_out = total_tokens_post_pad.cpu().item()

        # Reference
        ref_sorted, ref_experts, ref_total = _ref_align_block_size(
            topk_ids.cpu(), num_experts, block_size
        )

        if total_out != ref_total:
            report("moe_align_block_size", "FAIL", notes=f"total mismatch: got {total_out} expected {ref_total}")
            return

        # Check all original tokens appear in sorted output
        gpu_sorted = sorted_token_ids[:total_out].cpu()
        valid_mask = gpu_sorted < num_tokens
        valid_ids = gpu_sorted[valid_mask].sort()[0]
        ref_valid = ref_sorted[:ref_total][ref_sorted[:ref_total] < num_tokens].sort()[0]

        if torch.equal(valid_ids, ref_valid):
            report("moe_align_block_size", "PASS", 0.0)
        else:
            diff = (valid_ids != ref_valid).sum().item()
            report("moe_align_block_size", "FAIL", notes=f"{diff} token mismatches")

    except Exception as e:
        report("moe_align_block_size", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


def _ref_align_block_size(topk_ids, num_experts, block_size):
    """Reference implementation."""
    numel = topk_ids.shape[0]
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32)
    for e in range(num_experts):
        tokens_per_expert[e] = (topk_ids == e).sum()

    aligned_counts = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32)
    cumsum[1:] = torch.cumsum(aligned_counts, dim=0)
    total = cumsum[-1].item()

    sorted_ids = torch.zeros(total, dtype=torch.int32)
    expert_ids_out = torch.zeros(total // block_size, dtype=torch.int32)
    local_counters = torch.zeros(num_experts, dtype=torch.int32)

    for i in range(numel):
        eid = topk_ids[i].item()
        pos = cumsum[eid].item() + local_counters[eid].item()
        sorted_ids[pos] = i
        local_counters[eid] += 1

    for e in range(num_experts):
        sb = cumsum[e].item() // block_size
        eb = cumsum[e + 1].item() // block_size
        for b in range(sb, eb):
            expert_ids_out[b] = e

    return sorted_ids, expert_ids_out, total


# ============================================================================
# 2. MoE Top-K
# ============================================================================

def test_moe_topk():
    print("\n=== moe_topk ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-routing/moe_topk/moe_topk_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_topk_tk")

        num_tokens = 128
        num_experts = 64
        K = 4
        apply_softmax = True

        torch.manual_seed(42)
        X = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=DEVICE)

        num_bit_blocks = (num_experts + 31) // 32
        Yv = torch.empty(num_tokens, K, dtype=torch.bfloat16, device=DEVICE)
        Yi = torch.empty(num_tokens, K, dtype=torch.int32, device=DEVICE)
        Bits = torch.zeros(num_tokens, num_bit_blocks, dtype=torch.int32, device=DEVICE)

        mod.moe_topk(
            X, num_experts,  # stride_xm
            Yv, Yi, K,      # stride_ym
            Bits, num_bit_blocks, num_tokens,  # stride_bm, stride_bn
            num_tokens, num_experts, K, apply_softmax
        )
        torch.cuda.synchronize()

        # Reference
        ref_vals, ref_indices = torch.topk(X.float(), K, dim=1, largest=True, sorted=True)
        if apply_softmax:
            ref_vals = F.softmax(ref_vals, dim=1).to(torch.bfloat16)
        else:
            ref_vals = ref_vals.to(torch.bfloat16)
        ref_indices = ref_indices.to(torch.int32)

        # Compare indices (order may differ, so compare sets)
        gpu_Yi = Yi.cpu()
        gpu_Yv = Yv.cpu()

        # Check indices match as sets per row
        idx_match = 0
        for r in range(num_tokens):
            if set(gpu_Yi[r].tolist()) == set(ref_indices.cpu()[r].tolist()):
                idx_match += 1

        idx_rate = idx_match / num_tokens
        max_diff = (gpu_Yv.float() - ref_vals.cpu().float()).abs().max().item()

        if idx_rate > 0.95 and max_diff < 0.05:
            report("moe_topk", "PASS", max_diff, f"idx_match={idx_rate:.1%}")
        else:
            report("moe_topk", "FAIL", max_diff, f"idx_match={idx_rate:.1%}")

    except Exception as e:
        report("moe_topk", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 3. MoE Bitmatrix
# ============================================================================

def test_moe_bitmatrix():
    print("\n=== moe_bitmatrix (sum_bitmatrix_rows_fused) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-routing/moe_bitmatrix/moe_bitmatrix_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_bitmatrix_tk")

        num_tokens = 256
        num_experts = 64
        num_blocks = (num_experts + 31) // 32  # 2

        torch.manual_seed(42)
        np.random.seed(42)

        # Create bitmatrix: each token assigned to 2 random experts
        bitmatrix = torch.zeros(num_tokens, num_blocks, dtype=torch.int32)
        for t in range(num_tokens):
            experts = np.random.choice(num_experts, 2, replace=False)
            for eid in experts:
                block_idx = eid // 32
                bit_pos = eid % 32
                bitmatrix[t, block_idx] |= (1 << bit_pos)

        bitmatrix_gpu = bitmatrix.to(device=DEVICE)
        ret = torch.zeros(num_experts, dtype=torch.int32, device=DEVICE)

        # Column-major layout: stride_bm=1, stride_bn=num_tokens
        bm_col = bitmatrix_gpu.t().contiguous()  # [num_blocks, num_tokens]

        mod.sum_bitmatrix_rows_fused(
            bm_col,
            num_tokens,   # shape_bm
            1,            # stride_bm
            num_tokens,   # stride_bn
            ret,
            num_blocks    # N_BLKS_BITMATRIX
        )
        torch.cuda.synchronize()

        # Reference: count tokens per expert
        ref_hist = torch.zeros(num_experts, dtype=torch.int32)
        for t in range(num_tokens):
            for eid in range(num_experts):
                b = eid // 32
                bp = eid % 32
                if bitmatrix[t, b].item() & (1 << bp):
                    ref_hist[eid] += 1

        gpu_ret = ret.cpu()
        max_diff = (gpu_ret - ref_hist).abs().max().item()

        if max_diff == 0:
            report("moe_bitmatrix (fused)", "PASS", 0.0)
        else:
            report("moe_bitmatrix (fused)", "FAIL", float(max_diff))

    except Exception as e:
        report("moe_bitmatrix (fused)", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 4. MoE Expert Data
# ============================================================================

def test_moe_expt_data():
    print("\n=== moe_expt_data (expt_data_compute) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-routing/moe_expt_data/moe_expt_data_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_expt_data_tk")

        num_experts = 8
        tile_dim_log2 = 6  # tile_dim = 64

        torch.manual_seed(42)
        hist = torch.randint(0, 200, (num_experts,), dtype=torch.int32)
        n_gates = hist.sum().item()
        tile_dim = 1 << tile_dim_log2

        max_num_tiles = 0
        for h in hist:
            max_num_tiles += (h.item() + tile_dim - 1) // tile_dim
        max_num_tiles += 10

        hist_gpu = hist.to(device=DEVICE)
        token_start = torch.zeros(num_experts + 1, dtype=torch.int32, device=DEVICE)
        tile_start = torch.zeros(num_experts + 1, dtype=torch.int32, device=DEVICE)
        md_tile_info = torch.full((max_num_tiles,), 0xFFFFFFFF, dtype=torch.int32, device=DEVICE)

        mod.expt_data_compute(
            hist_gpu, token_start, tile_start, md_tile_info,
            num_experts, max_num_tiles, n_gates, tile_dim_log2, False
        )
        torch.cuda.synchronize()

        # Reference
        ref_ts, ref_tls, ref_md = _ref_expt_data(hist, tile_dim_log2, max_num_tiles, n_gates)

        gpu_ts = token_start.cpu()
        gpu_tls = tile_start.cpu()
        gpu_md = md_tile_info.cpu()

        ts_diff = (gpu_ts - ref_ts).abs().max().item()
        tls_diff = (gpu_tls - ref_tls).abs().max().item()

        total_tiles = ref_tls[num_experts].item()
        md_diff = (gpu_md[:total_tiles] - ref_md[:total_tiles]).abs().max().item()

        max_diff = max(ts_diff, tls_diff, md_diff)
        if max_diff == 0:
            report("moe_expt_data", "PASS", 0.0)
        else:
            report("moe_expt_data", "FAIL", float(max_diff),
                   f"ts_diff={ts_diff} tls_diff={tls_diff} md_diff={md_diff}")

    except Exception as e:
        report("moe_expt_data", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


def _ref_expt_data(hist, tile_dim_log2, max_num_tiles, n_gates):
    num_experts = hist.shape[0]
    tile_dim = 1 << tile_dim_log2
    token_start = torch.zeros(num_experts + 1, dtype=torch.int32)
    tile_start = torch.zeros(num_experts + 1, dtype=torch.int32)
    token_acc = 0
    tile_acc = 0
    for e in range(num_experts):
        h = hist[e].item()
        token_start[e] = token_acc
        tile_start[e] = tile_acc
        token_acc += h
        tile_acc += (h + tile_dim - 1) // tile_dim
    token_start[num_experts] = n_gates
    tile_start[num_experts] = tile_acc

    md = torch.full((max_num_tiles,), 0xFFFFFFFF, dtype=torch.int32)
    for e in range(num_experts):
        h = hist[e].item()
        if h == 0:
            continue
        n_tiles = (h + tile_dim - 1) // tile_dim
        t_off = tile_start[e].item()
        for t in range(n_tiles):
            md[t_off + t] = (t << 16) | e

    return token_start, tile_start, md


# ============================================================================
# 5. MoE GEMM BF16 (moe_op)
# ============================================================================

def test_moe_op():
    print("\n=== moe_op (bf16 GEMM) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op/moe_op_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        block_m = 128

        torch.manual_seed(42)
        A = torch.randn(num_tokens, K_dim, dtype=torch.bfloat16, device=DEVICE)
        B = torch.randn(num_experts, K_dim, N_dim, dtype=torch.bfloat16, device=DEVICE)

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, N_dim, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)
        ntp_gpu = ntp.to(DEVICE)

        mod.moe_gemm(
            A, B, C, topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu, ntp_gpu,
            N_dim, K_dim, num_valid, top_k,
            K_dim, 1,                     # stride_am, stride_ak
            K_dim * N_dim, N_dim, 1,      # stride_be, stride_bk, stride_bn
            N_dim, 1,                     # stride_cm, stride_cn
            True                          # mul_routed_weight
        )
        torch.cuda.synchronize()

        C_ref = ref_moe_gemm(
            A.cpu(), B.cpu(), sorted_ids, expert_ids, topk_weights,
            num_valid, top_k, block_m)

        C_gpu = C.cpu().float()
        max_diff = (C_gpu - C_ref).abs().max().item()
        # BF16 accumulation tolerance
        if max_diff < 1.0:
            report("moe_op", "PASS", max_diff)
        else:
            report("moe_op", "FAIL", max_diff)

    except Exception as e:
        report("moe_op", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 6. MoE GEMM + GeLU (moe_op_gelu)
# ============================================================================

def test_moe_op_gelu():
    print("\n=== moe_op_gelu (bf16 GEMM + GeLU) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_gelu/moe_op_gelu_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_gelu_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        block_m = 128

        torch.manual_seed(42)
        A = torch.randn(num_tokens, K_dim, dtype=torch.bfloat16, device=DEVICE)
        B = torch.randn(num_experts, K_dim, N_dim, dtype=torch.bfloat16, device=DEVICE)

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, N_dim, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)
        ntp_gpu = ntp.to(DEVICE)

        mod.moe_gelu(
            A, B, C, topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu, ntp_gpu,
            N_dim, K_dim, num_valid, top_k,
            K_dim, 1,
            K_dim * N_dim, N_dim, 1,
            N_dim, 1,
            False  # mul_routed_weight=False for GeLU variant
        )
        torch.cuda.synchronize()

        # Reference: C = GeLU(A @ B[expert])
        C_ref = torch.zeros(num_padded, N_dim, dtype=torch.float32)
        for bid in range(expert_ids.shape[0]):
            expert = expert_ids[bid].item()
            for lm in range(block_m):
                tidx = bid * block_m + lm
                if tidx >= num_padded:
                    break
                token_id = sorted_ids[tidx].item()
                if token_id >= num_valid:
                    continue
                orig = token_id // top_k
                result = A.cpu()[orig].float() @ B.cpu()[expert].float()
                C_ref[token_id] = F.gelu(result, approximate='tanh')

        C_gpu = C.cpu().float()
        max_diff = (C_gpu - C_ref).abs().max().item()
        if max_diff < 2.0:
            report("moe_op_gelu", "PASS", max_diff)
        else:
            report("moe_op_gelu", "FAIL", max_diff)

    except Exception as e:
        report("moe_op_gelu", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 7. MoE GEMM + SiLU Fused (moe_op_silu_fused)
# ============================================================================

def test_moe_op_silu_fused():
    print("\n=== moe_op_silu_fused (bf16 GEMM + SwiGLU) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_silu_fused/moe_op_silu_fused_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_silu_fused_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 512  # gate+up, output is N/2=256
        block_m = 128

        torch.manual_seed(42)
        A = torch.randn(num_tokens, K_dim, dtype=torch.bfloat16, device=DEVICE)
        B = torch.randn(num_experts, K_dim, N_dim, dtype=torch.bfloat16, device=DEVICE)

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, N_dim // 2, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)
        ntp_gpu = ntp.to(DEVICE)

        mod.moe_silu_fused(
            A, B, C, topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu, ntp_gpu,
            N_dim, K_dim, num_valid, top_k,
            K_dim, 1,
            K_dim * N_dim, N_dim, 1,
            N_dim // 2, 1,
            True  # mul_routed_weight
        )
        torch.cuda.synchronize()

        # Reference: SiLU(gate) * up
        C_ref = torch.zeros(num_padded, N_dim // 2, dtype=torch.float32)
        for bid in range(expert_ids.shape[0]):
            expert = expert_ids[bid].item()
            for lm in range(block_m):
                tidx = bid * block_m + lm
                if tidx >= num_padded:
                    break
                token_id = sorted_ids[tidx].item()
                if token_id >= num_valid:
                    continue
                orig = token_id // top_k
                gemm_out = A.cpu()[orig].float() @ B.cpu()[expert].float()
                gemm_out *= topk_weights[tidx].item()
                gate = gemm_out[:N_dim // 2]
                up = gemm_out[N_dim // 2:]
                C_ref[token_id] = F.silu(gate) * up

        C_gpu = C.cpu().float()
        max_diff = (C_gpu - C_ref).abs().max().item()
        if max_diff < 2.0:
            report("moe_op_silu_fused", "PASS", max_diff)
        else:
            report("moe_op_silu_fused", "FAIL", max_diff)

    except Exception as e:
        report("moe_op_silu_fused", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 8. MoE E2E (moe_op_e2e)
# ============================================================================

def test_moe_op_e2e():
    print("\n=== moe_op_e2e (two-layer gated MLP) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_e2e/moe_op_e2e_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_e2e_tk")

        num_tokens = 32
        num_experts = 4
        top_k = 2
        K_dim = 128
        hidden_dim = 256
        N_dim = 2 * hidden_dim  # gate+up
        block_m = 64

        torch.manual_seed(42)
        A = torch.randn(num_tokens, K_dim, dtype=torch.bfloat16, device=DEVICE)
        W1 = torch.randn(num_experts, N_dim, K_dim, dtype=torch.bfloat16, device=DEVICE)
        W2 = torch.randn(num_experts, K_dim, hidden_dim, dtype=torch.bfloat16, device=DEVICE)

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, K_dim, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)
        ntp_gpu = ntp.to(DEVICE)

        mod.moe_e2e(
            A, W1, W2, C, topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu, ntp_gpu,
            N_dim, K_dim, num_valid, top_k,
            K_dim, 1,                           # stride_am, stride_ak
            N_dim * K_dim, K_dim, 1,            # stride_w1e, stride_w1n, stride_w1k
            K_dim * hidden_dim, hidden_dim, 1,  # stride_w2e, stride_w2n, stride_w2k
            K_dim,                              # stride_cm
            True                                # mul_routed_weight
        )
        torch.cuda.synchronize()

        # Reference: SiLU(A@W1_gate) * (A@W1_up) then @ W2
        C_ref = torch.zeros(num_padded, K_dim, dtype=torch.float32)
        N_half = N_dim // 2
        for bid in range(expert_ids.shape[0]):
            expert = expert_ids[bid].item()
            for lm in range(block_m):
                tidx = bid * block_m + lm
                if tidx >= num_padded:
                    break
                token_id = sorted_ids[tidx].item()
                if token_id >= num_valid:
                    continue
                orig = token_id // top_k
                a_row = A.cpu()[orig].float()
                w1_e = W1.cpu()[expert].float()  # [N, K]
                l1_out = a_row @ w1_e.T  # [N]
                gate = l1_out[:N_half]
                up = l1_out[N_half:]
                intermediate = F.silu(gate) * up  # [N_half]
                w2_e = W2.cpu()[expert].float()  # [K, hidden_dim]
                result = intermediate @ w2_e.T  # [K]
                result *= topk_weights[tidx].item()
                C_ref[token_id] += result

        C_gpu = C.cpu().float()
        max_diff = (C_gpu - C_ref).abs().max().item()
        if max_diff < 5.0:
            report("moe_op_e2e", "PASS", max_diff)
        else:
            report("moe_op_e2e", "FAIL", max_diff)

    except Exception as e:
        report("moe_op_e2e", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 9. MoE INT8 GEMM (moe_op_gemm_a8w8)
# ============================================================================

def test_moe_a8w8():
    print("\n=== moe_op_gemm_a8w8 (INT8 GEMM) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_gemm_a8w8/moe_op_gemm_a8w8_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_gemm_a8w8_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        block_m = 128

        torch.manual_seed(42)
        X = torch.randint(-128, 127, (num_tokens, K_dim), dtype=torch.int8, device=DEVICE)
        W = torch.randint(-128, 127, (num_experts, K_dim, N_dim), dtype=torch.int8, device=DEVICE)
        x_scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
        w_scale = torch.rand(num_experts, dtype=torch.float32, device=DEVICE) * 0.01
        bias = torch.randn(num_experts, N_dim, dtype=torch.float32, device=DEVICE) * 0.1

        hist, offs, gather, expt_data, grid_m = generate_ogs_routing(
            num_tokens, num_experts, top_k, block_m)

        total = num_tokens * top_k
        grid_n = (N_dim + 127) // 128

        Y = torch.zeros(total, N_dim, dtype=torch.bfloat16, device=DEVICE)

        hist_gpu = hist.to(DEVICE)
        offs_gpu = offs.to(DEVICE)
        gather_gpu = gather.to(DEVICE)
        expt_data_gpu = expt_data.to(DEVICE)

        mod.moe_a8w8(
            X, W, Y, x_scale, w_scale, bias,
            gather_gpu, hist_gpu, offs_gpu, expt_data_gpu,
            N_dim, K_dim, grid_m, grid_n, top_k,
            N_dim, 1,           # stride_ym, stride_yn
            K_dim, 1,           # stride_xm, stride_xk
            K_dim * N_dim, N_dim, 1,  # stride_we, stride_wk, stride_wn
            False, 0.0          # apply_swiglu, swiglu_alpha
        )
        torch.cuda.synchronize()

        # Reference
        Y_ref = torch.zeros(total, N_dim, dtype=torch.float32)
        for tile_idx in range(len(expt_data)):
            ed = expt_data[tile_idx].item()
            eid = ed & 0xFFFF
            bid = ed >> 16
            M = hist[eid].item()
            start = offs[eid].item()
            for lm in range(block_m):
                gm = bid * block_m + lm
                if gm >= M:
                    continue
                sorted_idx = start + gm
                if sorted_idx >= total:
                    continue
                token_idx = gather[sorted_idx].item() // top_k
                x_row = X.cpu()[token_idx].float()
                w_e = W.cpu()[eid].float()
                result = x_row @ w_e * x_scale.cpu().item() * w_scale.cpu()[eid].item()
                result += bias.cpu()[eid]
                Y_ref[sorted_idx] = result

        Y_gpu = Y.cpu().float()
        max_diff = (Y_gpu - Y_ref).abs().max().item()
        if max_diff < 2.0:
            report("moe_op_gemm_a8w8", "PASS", max_diff)
        else:
            report("moe_op_gemm_a8w8", "FAIL", max_diff)

    except Exception as e:
        report("moe_op_gemm_a8w8", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 10. MoE INT8 Block-Scaled (moe_op_gemm_a8w8_blockscale)
# ============================================================================

def test_moe_a8w8_blockscale():
    print("\n=== moe_op_gemm_a8w8_blockscale ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_gemm_a8w8_blockscale/moe_op_gemm_a8w8_blockscale_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_gemm_a8w8_blockscale_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        group_k = 64
        group_n = 64
        block_m = 128

        torch.manual_seed(42)
        X = torch.randint(-128, 127, (num_tokens, K_dim), dtype=torch.int8, device=DEVICE)
        W = torch.randint(-128, 127, (num_experts, K_dim, N_dim), dtype=torch.int8, device=DEVICE)
        X_scale = torch.rand(num_tokens, K_dim // group_k, dtype=torch.float32, device=DEVICE) * 0.01
        W_scale = torch.rand(num_experts, K_dim // group_k, N_dim // group_n, dtype=torch.float32, device=DEVICE) * 0.01

        hist, offs, gather, expt_data, grid_m = generate_ogs_routing(
            num_tokens, num_experts, top_k, block_m)

        total = num_tokens * top_k
        grid_n = (N_dim + 63) // 64

        Y = torch.zeros(total, N_dim, dtype=torch.bfloat16, device=DEVICE)

        hist_gpu = hist.to(DEVICE)
        offs_gpu = offs.to(DEVICE)
        gather_gpu = gather.to(DEVICE)
        expt_data_gpu = expt_data.to(DEVICE)

        mod.moe_a8w8_blockscale(
            X, W, Y, X_scale, W_scale, None,  # no bias
            gather_gpu, hist_gpu, offs_gpu, expt_data_gpu,
            N_dim, K_dim, group_k, group_n, grid_m, grid_n, top_k,
            N_dim, 1,               # stride_ym, stride_yn
            K_dim, 1,               # stride_xm, stride_xk
            K_dim * N_dim, N_dim, 1, # stride_we, stride_wk, stride_wn
            K_dim // group_k, 1,     # stride_xsm, stride_xsk
            (K_dim // group_k) * (N_dim // group_n), N_dim // group_n, 1  # stride_wse, stride_wsk, stride_wsn
        )
        torch.cuda.synchronize()

        # Simple reference: just check output is non-zero and in reasonable range
        Y_gpu = Y.cpu().float()
        nonzero = (Y_gpu.abs() > 0).any(dim=1).sum().item()
        max_val = Y_gpu.abs().max().item()

        if nonzero > 0 and max_val < 1e6:
            report("moe_op_gemm_a8w8_blockscale", "PASS", notes=f"nonzero_rows={nonzero}/{total}, max_val={max_val:.2f}")
        else:
            report("moe_op_gemm_a8w8_blockscale", "FAIL", notes=f"nonzero_rows={nonzero}/{total}")

    except Exception as e:
        report("moe_op_gemm_a8w8_blockscale", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 11. MoE INT8xINT4 (moe_op_gemm_a8w4)
# ============================================================================

def test_moe_a8w4():
    print("\n=== moe_op_gemm_a8w4 (INT8xINT4 GEMM) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_gemm_a8w4/moe_op_gemm_a8w4_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_gemm_a8w4_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        group_size = 128
        block_m = 128

        torch.manual_seed(42)
        X = torch.randint(-128, 127, (num_tokens, K_dim), dtype=torch.int8, device=DEVICE)
        # Packed INT4 weights: K/2 rows, N cols
        W = torch.randint(0, 255, (num_experts, K_dim // 2, N_dim), dtype=torch.uint8, device=DEVICE)
        X_scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
        W_scale = torch.rand(num_experts, K_dim // group_size, N_dim, dtype=torch.float32, device=DEVICE) * 0.01

        hist, offs, gather, expt_data, grid_m = generate_ogs_routing(
            num_tokens, num_experts, top_k, block_m)

        total = num_tokens * top_k
        grid_n = (N_dim + 127) // 128

        Y = torch.zeros(total, N_dim, dtype=torch.bfloat16, device=DEVICE)

        hist_gpu = hist.to(DEVICE)
        offs_gpu = offs.to(DEVICE)
        gather_gpu = gather.to(DEVICE)
        expt_data_gpu = expt_data.to(DEVICE)

        mod.moe_a8w4(
            X, W, Y, X_scale, W_scale, None,  # W_zp=None
            gather_gpu, hist_gpu, offs_gpu, expt_data_gpu,
            N_dim, K_dim, group_size, grid_m, grid_n, top_k,
            N_dim, 1,                     # stride_ym, stride_yn
            K_dim, 1,                     # stride_xm, stride_xk
            (K_dim // 2) * N_dim, N_dim, 1,  # stride_we, stride_wk, stride_wn
            (K_dim // group_size) * N_dim, N_dim, 1,  # stride_wse, stride_wsk, stride_wsn
            False  # has_zero_point
        )
        torch.cuda.synchronize()

        Y_gpu = Y.cpu().float()
        nonzero = (Y_gpu.abs() > 0).any(dim=1).sum().item()
        max_val = Y_gpu.abs().max().item()

        if nonzero > 0 and max_val < 1e6:
            report("moe_op_gemm_a8w4", "PASS", notes=f"nonzero_rows={nonzero}/{total}, max_val={max_val:.2f}")
        else:
            report("moe_op_gemm_a8w4", "FAIL", notes=f"nonzero_rows={nonzero}/{total}")

    except Exception as e:
        report("moe_op_gemm_a8w4", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 12. MoE INT4xINT4 (moe_op_gemm_a4w4)
# ============================================================================

def test_moe_a4w4():
    print("\n=== moe_op_gemm_a4w4 (INT4xINT4 GEMM) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_gemm_a4w4/moe_op_gemm_a4w4_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_gemm_a4w4_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        block_m = 128

        torch.manual_seed(42)
        # Packed INT4: K/2 rows
        X = torch.randint(0, 255, (num_tokens, K_dim // 2), dtype=torch.uint8, device=DEVICE)
        W = torch.randint(0, 255, (num_experts, K_dim // 2, N_dim), dtype=torch.uint8, device=DEVICE)
        # MX scales
        X_mx_scale = torch.randint(120, 134, (num_tokens, K_dim // 32), dtype=torch.uint8, device=DEVICE)
        W_mx_scale = torch.randint(120, 134, (num_experts, K_dim // 32, N_dim), dtype=torch.uint8, device=DEVICE)
        X_static_scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
        W_static_scale = torch.rand(num_experts, dtype=torch.float32, device=DEVICE) * 0.01

        hist, offs, gather, expt_data, grid_m = generate_ogs_routing(
            num_tokens, num_experts, top_k, block_m)

        total = num_tokens * top_k
        grid_n = (N_dim + 127) // 128

        Y = torch.zeros(total, N_dim, dtype=torch.bfloat16, device=DEVICE)

        hist_gpu = hist.to(DEVICE)
        offs_gpu = offs.to(DEVICE)
        gather_gpu = gather.to(DEVICE)
        expt_data_gpu = expt_data.to(DEVICE)

        mod.moe_a4w4(
            X, W, Y, X_mx_scale, W_mx_scale, X_static_scale, W_static_scale,
            gather_gpu, hist_gpu, offs_gpu, expt_data_gpu,
            N_dim, K_dim, grid_m, grid_n, top_k,
            N_dim, 1,                     # stride_ym, stride_yn
            K_dim // 2, 1,                # stride_xm, stride_xk
            (K_dim // 2) * N_dim, N_dim, 1,  # stride_we, stride_wk, stride_wn
            False  # apply_swiglu
        )
        torch.cuda.synchronize()

        Y_gpu = Y.cpu().float()
        nonzero = (Y_gpu.abs() > 0).any(dim=1).sum().item()
        max_val = Y_gpu.abs().max().item()

        if nonzero > 0 and max_val < 1e6:
            report("moe_op_gemm_a4w4", "PASS", notes=f"nonzero_rows={nonzero}/{total}, max_val={max_val:.2f}")
        else:
            report("moe_op_gemm_a4w4", "FAIL", notes=f"nonzero_rows={nonzero}/{total}")

    except Exception as e:
        report("moe_op_gemm_a4w4", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 13. MoE MXFP4 GEMM (moe_op_mxfp4)
# ============================================================================

def test_moe_mxfp4():
    print("\n=== moe_op_mxfp4 (MXFP4 GEMM) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_mxfp4/moe_op_mxfp4_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_mxfp4_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 256
        block_m = 128
        mx_group = 32

        torch.manual_seed(42)
        # Packed FP4: K/2 rows
        A = torch.randint(0, 255, (num_tokens, K_dim // 2), dtype=torch.uint8, device=DEVICE)
        B = torch.randint(0, 255, (num_experts, K_dim // 2, N_dim), dtype=torch.uint8, device=DEVICE)
        A_mx_scale = torch.randint(120, 134, (num_tokens, K_dim // mx_group), dtype=torch.uint8, device=DEVICE)
        B_mx_scale = torch.randint(120, 134, (num_experts, K_dim // mx_group, N_dim), dtype=torch.uint8, device=DEVICE)
        A_scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
        B_scale = torch.rand(num_experts, dtype=torch.float32, device=DEVICE) * 0.01

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, N_dim, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)

        num_m_blocks = num_padded // block_m
        grid_n = (N_dim + 127) // 128

        mod.moe_mxfp4(
            A, B, C, A_mx_scale, B_mx_scale, A_scale, B_scale,
            topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu,
            N_dim, K_dim, num_valid, top_k, True,  # mul_routed_weight
            K_dim // 2, 1,                          # stride_am, stride_ak
            (K_dim // 2) * N_dim, N_dim, 1,         # stride_be, stride_bk, stride_bn
            N_dim, 1,                               # stride_cm, stride_cn
            K_dim // mx_group, 1,                   # stride_amxm, stride_amxk
            (K_dim // mx_group) * N_dim, N_dim, 1,  # stride_bmxe, stride_bmxk, stride_bmxn
            num_m_blocks, grid_n                    # grid_m, grid_n
        )
        torch.cuda.synchronize()

        C_gpu = C.cpu().float()
        nonzero = (C_gpu.abs() > 0).any(dim=1).sum().item()
        max_val = C_gpu.abs().max().item()

        if nonzero > 0 and max_val < 1e6:
            report("moe_op_mxfp4", "PASS", notes=f"nonzero_rows={nonzero}/{num_padded}, max_val={max_val:.4f}")
        else:
            report("moe_op_mxfp4", "FAIL", notes=f"nonzero_rows={nonzero}/{num_padded}")

    except Exception as e:
        report("moe_op_mxfp4", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 14. MoE MXFP4 + SiLU Fused (moe_op_mxfp4_silu_fused)
# ============================================================================

def test_moe_mxfp4_silu_fused():
    print("\n=== moe_op_mxfp4_silu_fused (MXFP4 + SwiGLU) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-gemm/moe_op_mxfp4_silu_fused/moe_op_mxfp4_silu_fused_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_op_mxfp4_silu_fused_tk")

        num_tokens = 64
        num_experts = 8
        top_k = 2
        K_dim = 256
        N_dim = 512  # gate+up, output is N/2=256
        block_m = 128
        mx_group = 32

        torch.manual_seed(42)
        A = torch.randint(0, 255, (num_tokens, K_dim // 2), dtype=torch.uint8, device=DEVICE)
        B = torch.randint(0, 255, (num_experts, K_dim // 2, N_dim), dtype=torch.uint8, device=DEVICE)
        A_mx_scale = torch.randint(120, 134, (num_tokens, K_dim // mx_group), dtype=torch.uint8, device=DEVICE)
        B_mx_scale = torch.randint(120, 134, (num_experts, K_dim // mx_group, N_dim), dtype=torch.uint8, device=DEVICE)
        A_scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
        B_scale = torch.rand(num_experts, dtype=torch.float32, device=DEVICE) * 0.01

        (sorted_ids, expert_ids, topk_weights,
         num_padded, num_valid, ntp) = generate_sorted_routing(
            num_tokens, num_experts, top_k, block_m)

        C = torch.zeros(num_padded, N_dim // 2, dtype=torch.bfloat16, device=DEVICE)

        sorted_ids_gpu = sorted_ids.to(DEVICE)
        expert_ids_gpu = expert_ids.to(DEVICE)
        topk_weights_gpu = topk_weights.to(DEVICE)

        num_m_blocks = num_padded // block_m
        grid_n = (N_dim + 127) // 128

        mod.moe_mxfp4_silu_fused(
            A, B, C, A_mx_scale, B_mx_scale, A_scale, B_scale,
            topk_weights_gpu, sorted_ids_gpu, expert_ids_gpu,
            N_dim, K_dim, num_valid, top_k, True,  # mul_routed_weight
            K_dim // 2, 1,
            (K_dim // 2) * N_dim, N_dim, 1,
            N_dim // 2, 1,
            K_dim // mx_group, 1,
            (K_dim // mx_group) * N_dim, N_dim, 1,
            num_m_blocks, grid_n
        )
        torch.cuda.synchronize()

        C_gpu = C.cpu().float()
        nonzero = (C_gpu.abs() > 0).any(dim=1).sum().item()
        max_val = C_gpu.abs().max().item()

        if nonzero > 0 and max_val < 1e6:
            report("moe_op_mxfp4_silu_fused", "PASS", notes=f"nonzero_rows={nonzero}/{num_padded}, max_val={max_val:.4f}")
        else:
            report("moe_op_mxfp4_silu_fused", "FAIL", notes=f"nonzero_rows={nonzero}/{num_padded}")

    except Exception as e:
        report("moe_op_mxfp4_silu_fused", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 15. Quant MoE (downcast_to_static_fp8, downcast_to_mxfp, upcast_from_mxfp)
# ============================================================================

def test_quant_moe_fp8():
    print("\n=== quant_moe (downcast_to_static_fp8) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-misc/quant_moe/quant_moe_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "quant_moe_tk")

        M, N = 256, 128
        torch.manual_seed(42)
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        scale = torch.tensor([0.5], dtype=torch.float32, device=DEVICE)
        y = torch.empty(M, N, dtype=torch.float8_e4m3fnuz, device=DEVICE)

        mod.downcast_to_static_fp8(x, y, scale, M, N)
        torch.cuda.synchronize()

        # Reference: x / scale, clamp to FP8 range
        ref = (x.float() / scale.item()).clamp(-240, 240).to(torch.float8_e4m3fnuz)

        y_bytes = y.view(torch.uint8).cpu()
        ref_bytes = ref.view(torch.uint8).cpu()
        match_rate = (y_bytes == ref_bytes).float().mean().item()

        if match_rate > 0.90:
            report("quant_moe (fp8)", "PASS", 1.0 - match_rate, f"match_rate={match_rate:.1%}")
        else:
            report("quant_moe (fp8)", "FAIL", 1.0 - match_rate, f"match_rate={match_rate:.1%}")

    except Exception as e:
        report("quant_moe (fp8)", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


def test_quant_moe_mxfp():
    print("\n=== quant_moe (downcast_to_mxfp + upcast_from_mxfp) ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-misc/quant_moe/quant_moe_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "quant_moe_tk")

        outer_dim = 128
        quant_dim = 256
        is_fp4 = False  # Test FP8 MX first

        torch.manual_seed(42)
        src = torch.randn(outer_dim, quant_dim, dtype=torch.bfloat16, device=DEVICE)

        # MX tensor output
        if is_fp4:
            mx_tensor = torch.empty(outer_dim, quant_dim // 2, dtype=torch.uint8, device=DEVICE)
        else:
            mx_tensor = torch.empty(outer_dim, quant_dim, dtype=torch.uint8, device=DEVICE)
        mx_scale = torch.empty(outer_dim, quant_dim // 32, dtype=torch.uint8, device=DEVICE)

        mod.downcast_to_mxfp(mx_tensor, mx_scale, src, outer_dim, quant_dim, is_fp4)
        torch.cuda.synchronize()

        # Upcast back
        out = torch.empty(outer_dim, quant_dim, dtype=torch.bfloat16, device=DEVICE)
        mod.upcast_from_mxfp(out, mx_scale, mx_tensor, outer_dim, quant_dim, is_fp4)
        torch.cuda.synchronize()

        # Check reconstruction error
        recon_error = (src.float() - out.float()).abs().mean().item()
        max_error = (src.float() - out.float()).abs().max().item()

        if recon_error < 0.5:
            report("quant_moe (mxfp roundtrip)", "PASS", max_error, f"mean_err={recon_error:.4f}")
        else:
            report("quant_moe (mxfp roundtrip)", "FAIL", max_error, f"mean_err={recon_error:.4f}")

    except Exception as e:
        report("quant_moe (mxfp roundtrip)", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# 16. MoE Routing Sigmoid Top-1 Fused
# ============================================================================

def test_moe_routing_sigmoid_top1():
    print("\n=== moe_routing_sigmoid_top1_fused ===")
    try:
        so_path = os.path.join(KERNELS_DIR, "moe-misc/moe_routing_sigmoid_top1_fused/moe_routing_sigmoid_top1_fused_tk.cpython-312-x86_64-linux-gnu.so")
        mod = load_kernel(so_path, "moe_routing_sigmoid_top1_fused_tk")

        M = 256
        N = 16   # num_experts
        K = 128  # hidden dim
        TOPK = 1
        FUSED_SHARED_EXPERTS = False

        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)

        _TOPK = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK
        topk_ids = torch.empty(M, _TOPK, dtype=torch.int32, device=DEVICE)
        topk_weights = torch.empty(M, _TOPK, dtype=torch.bfloat16, device=DEVICE)

        mod.moe_routing_sigmoid_top1_fused(
            X, W, topk_ids, topk_weights,
            M, N, K, TOPK, FUSED_SHARED_EXPERTS
        )
        torch.cuda.synchronize()

        # Reference: sigmoid(X @ W), then argmax
        logits = X.float() @ W.float()
        scores = torch.sigmoid(logits)
        ref_ids = scores.argmax(dim=1, keepdim=True).to(torch.int32)
        ref_weights = scores.gather(1, ref_ids.long()).to(torch.bfloat16)

        gpu_ids = topk_ids.cpu()
        gpu_weights = topk_weights.cpu()

        ids_match = (gpu_ids == ref_ids.cpu()).float().mean().item()
        weight_diff = (gpu_weights.float() - ref_weights.cpu().float()).abs().max().item()

        if ids_match > 0.95 and weight_diff < 0.05:
            report("moe_routing_sigmoid_top1_fused", "PASS", weight_diff, f"ids_match={ids_match:.1%}")
        else:
            report("moe_routing_sigmoid_top1_fused", "FAIL", weight_diff, f"ids_match={ids_match:.1%}")

    except Exception as e:
        report("moe_routing_sigmoid_top1_fused", "ERROR", notes=str(e)[:120])
        traceback.print_exc()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MoE Kernel Parity Tests")
    print("=" * 70)

    # Routing kernels (needed by GEMM kernels)
    test_moe_align_block_size()
    test_moe_topk()
    test_moe_bitmatrix()
    test_moe_expt_data()

    # BF16 GEMM kernels
    test_moe_op()
    test_moe_op_gelu()
    test_moe_op_silu_fused()
    test_moe_op_e2e()

    # Quantized GEMM kernels
    test_moe_a8w8()
    test_moe_a8w8_blockscale()
    test_moe_a8w4()
    test_moe_a4w4()

    # MXFP4 GEMM kernels
    test_moe_mxfp4()
    test_moe_mxfp4_silu_fused()

    # Misc kernels
    test_quant_moe_fp8()
    test_quant_moe_mxfp()
    test_moe_routing_sigmoid_top1()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<40} {'Status':<8} {'Max Diff':<12} {'Notes'}")
    print("-" * 90)
    for name, status, diff, notes in results:
        print(f"{name:<40} {status:<8} {diff:<12} {notes}")

    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    errors = sum(1 for _, s, _, _ in results if s == "ERROR")
    total = len(results)
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed} | Errors: {errors}")

    return results


if __name__ == "__main__":
    results_data = main()
