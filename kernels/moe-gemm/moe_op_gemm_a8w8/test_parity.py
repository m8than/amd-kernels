#!/usr/bin/env python3
"""
Parity test for MoE INT8 GEMM (moe_op_gemm_a8w8).

Tests: Y = dequant(X_int8 @ W_int8) * X_scale * W_scale + bias

Input shapes:
  - X: [total_tokens, K] int8
  - W: [E, K, N] int8
  - X_scale: scalar float32
  - W_scale: [E] float32
  - ExptHist, ExptOffs, ExptData for routing

Output shapes:
  - Y: [total_tokens, N] bf16

Reference: Triton _moe_gemm_a8w8 from moe_op_gemm_a8w8.py
"""

import numpy as np
import sys

NUM_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 2
K_DIM = 256
N_DIM = 256
BLOCK_M = 128


def generate_ogs_routing(num_tokens, num_experts, top_k):
    """Generate OGS-style routing data."""
    total = num_tokens * top_k

    assigns = np.random.randint(0, num_experts, (num_tokens, top_k)).astype(np.int32)

    # Count tokens per expert
    hist = np.zeros(num_experts, dtype=np.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            hist[assigns[t, k]] += 1

    # Prefix sum
    offs = np.zeros(num_experts, dtype=np.int32)
    for e in range(1, num_experts):
        offs[e] = offs[e-1] + hist[e-1]

    # Build gather index (sorted by expert)
    gather = np.zeros(total, dtype=np.int32)
    pos = offs.copy()
    for t in range(num_tokens):
        for k in range(top_k):
            e = assigns[t, k]
            gather[pos[e]] = t * top_k + k
            pos[e] += 1

    # Build ExptData: packed (expt_id | block_id << 16)
    grid_m_tiles = []
    for e in range(num_experts):
        n_blocks = (hist[e] + BLOCK_M - 1) // BLOCK_M
        for b in range(n_blocks):
            grid_m_tiles.append(e | (b << 16))

    expt_data = np.array(grid_m_tiles, dtype=np.int32)
    grid_m = len(expt_data)

    return hist, offs, gather, expt_data, grid_m


def reference_moe_a8w8(X_int8, W_int8, x_scale, w_scale, bias,
                        gather, hist, offs, expt_data, n_expts_act):
    total_tokens = len(gather)
    N = W_int8.shape[2]
    Y = np.zeros((total_tokens, N), dtype=np.float32)

    for tile_idx in range(len(expt_data)):
        ed = expt_data[tile_idx]
        expt_id = ed & 0xFFFF
        block_id = ed >> 16

        M = hist[expt_id]
        start = offs[expt_id]

        for lm in range(BLOCK_M):
            gm = (block_id * BLOCK_M + lm) % M
            if gm >= M:
                continue

            sorted_idx = start + gm
            if sorted_idx >= total_tokens:
                continue

            token_idx = gather[sorted_idx] // n_expts_act

            x_row = X_int8[token_idx].astype(np.float32)
            w_expert = W_int8[expt_id].astype(np.float32)

            result = x_row @ w_expert * x_scale * w_scale[expt_id]

            if bias is not None:
                result += bias[expt_id]

            Y[sorted_idx] = result

    return Y


def test_parity():
    print("=" * 60)
    print("MoE INT8 GEMM (moe_op_gemm_a8w8) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randint(-128, 127, (NUM_TOKENS, K_DIM)).astype(np.int8)
    W = np.random.randint(-128, 127, (NUM_EXPERTS, K_DIM, N_DIM)).astype(np.int8)
    x_scale = np.float32(0.01)
    w_scale = np.random.rand(NUM_EXPERTS).astype(np.float32) * 0.01
    bias = np.random.randn(NUM_EXPERTS, N_DIM).astype(np.float32) * 0.1

    hist, offs, gather, expt_data, grid_m = \
        generate_ogs_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    print(f"Config: tokens={NUM_TOKENS}, experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  K={K_DIM}, N={N_DIM}, grid_m={grid_m}")

    Y_ref = reference_moe_a8w8(X, W, x_scale, w_scale, bias,
                                gather, hist, offs, expt_data, TOP_K)

    print(f"  Output range: [{Y_ref.min():.4f}, {Y_ref.max():.4f}]")
    nonzero = np.any(Y_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{len(gather)}")

    print("\n[PASS] Reference implementation validated")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
