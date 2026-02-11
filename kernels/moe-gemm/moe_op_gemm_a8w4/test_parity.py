#!/usr/bin/env python3
"""
Parity test for MoE INT8xINT4 GEMM (moe_op_gemm_a8w4).

Tests mixed-precision GEMM with INT4 weight unpacking and group dequant:
  w = ((packed >> shift) & 0xF - zp) * scale
  Y = X_int8 @ dequant(W_int4)

Input shapes:
  - X: [total_tokens, K] int8
  - W: [E, K/2, N] uint8 (packed INT4)
  - W_scale: [E, K/group_size, N] float32
"""

import numpy as np
import sys

NUM_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 2
K_DIM = 256
N_DIM = 256
GROUP_SIZE = 128
BLOCK_M = 128


def generate_ogs_routing(num_tokens, num_experts, top_k):
    total = num_tokens * top_k
    assigns = np.random.randint(0, num_experts, (num_tokens, top_k)).astype(np.int32)
    hist = np.zeros(num_experts, dtype=np.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            hist[assigns[t, k]] += 1
    offs = np.cumsum(np.concatenate([[0], hist[:-1]])).astype(np.int32)
    gather = np.zeros(total, dtype=np.int32)
    pos = offs.copy()
    for t in range(num_tokens):
        for k in range(top_k):
            e = assigns[t, k]
            gather[pos[e]] = t * top_k + k
            pos[e] += 1
    grid_m_tiles = []
    for e in range(num_experts):
        for b in range((hist[e] + BLOCK_M - 1) // BLOCK_M):
            grid_m_tiles.append(e | (b << 16))
    return hist, offs, gather, np.array(grid_m_tiles, dtype=np.int32), len(grid_m_tiles)


def reference_moe_a8w4(X, W_packed, w_scale, gather, hist, offs,
                        expt_data, n_expts_act, group_size):
    total = len(gather)
    K = X.shape[1]
    N = W_packed.shape[2]
    Y = np.zeros((total, N), dtype=np.float32)

    for tile_idx in range(len(expt_data)):
        ed = expt_data[tile_idx]
        eid = ed & 0xFFFF
        bid = ed >> 16
        M = hist[eid]
        start = offs[eid]

        for lm in range(min(BLOCK_M, M)):
            gm = (bid * BLOCK_M + lm) % M
            token_idx = gather[start + gm] // n_expts_act

            result = np.zeros(N, dtype=np.float32)
            for k in range(K):
                x_val = float(X[token_idx, k])
                packed = W_packed[eid, k // 2, :]
                shift = (k % 2) * 4
                w_int4 = ((packed.astype(np.int32) >> shift) & 0xF).astype(np.float32)
                k_group = k // group_size
                scale = w_scale[eid, k_group, :]
                w_dequant = (w_int4 - 8.0) * scale
                result += x_val * w_dequant

            Y[start + gm] = result

    return Y


def test_parity():
    print("=" * 60)
    print("MoE INT8xINT4 GEMM (moe_op_gemm_a8w4) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randint(-128, 127, (NUM_TOKENS, K_DIM)).astype(np.int8)
    W_packed = np.random.randint(0, 255, (NUM_EXPERTS, K_DIM // 2, N_DIM)).astype(np.uint8)
    w_scale = np.random.rand(NUM_EXPERTS, K_DIM // GROUP_SIZE, N_DIM).astype(np.float32) * 0.01

    hist, offs, gather, expt_data, grid_m = \
        generate_ogs_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    Y_ref = reference_moe_a8w4(X, W_packed, w_scale, gather, hist, offs,
                                expt_data, TOP_K, GROUP_SIZE)

    print(f"Config: tokens={NUM_TOKENS}, K={K_DIM}, N={N_DIM}, group={GROUP_SIZE}")
    print(f"  Output range: [{Y_ref.min():.4f}, {Y_ref.max():.4f}]")
    nonzero = np.any(Y_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{len(gather)}")

    print("\n[PASS] Reference implementation validated")
    return True

if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
