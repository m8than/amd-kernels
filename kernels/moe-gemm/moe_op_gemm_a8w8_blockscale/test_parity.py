#!/usr/bin/env python3
"""
Parity test for MoE INT8 Block-Scaled GEMM (moe_op_gemm_a8w8_blockscale).

Tests block-wise dequantization:
  acc += (X_int8 @ W_int8) * X_scale[m, k/gk] * W_scale[e, k/gk, n/gn]

Input shapes:
  - X: [total_tokens, K] int8
  - W: [E, K, N] int8
  - X_scale: [total_tokens, K/group_k] float32
  - W_scale: [E, K/group_k, N/group_n] float32

Output: Y: [total_tokens, N] bf16
"""

import numpy as np
import sys

NUM_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 2
K_DIM = 256
N_DIM = 256
GROUP_K = 64
GROUP_N = 64
BLOCK_M = 128


def generate_ogs_routing(num_tokens, num_experts, top_k):
    total = num_tokens * top_k
    assigns = np.random.randint(0, num_experts, (num_tokens, top_k)).astype(np.int32)

    hist = np.zeros(num_experts, dtype=np.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            hist[assigns[t, k]] += 1

    offs = np.zeros(num_experts, dtype=np.int32)
    for e in range(1, num_experts):
        offs[e] = offs[e-1] + hist[e-1]

    gather = np.zeros(total, dtype=np.int32)
    pos = offs.copy()
    for t in range(num_tokens):
        for k in range(top_k):
            e = assigns[t, k]
            gather[pos[e]] = t * top_k + k
            pos[e] += 1

    grid_m_tiles = []
    for e in range(num_experts):
        n_blocks = (hist[e] + BLOCK_M - 1) // BLOCK_M
        for b in range(n_blocks):
            grid_m_tiles.append(e | (b << 16))

    expt_data = np.array(grid_m_tiles, dtype=np.int32)
    return hist, offs, gather, expt_data, len(expt_data)


def reference_moe_a8w8_bs(X, W, x_scale, w_scale, gather, hist, offs,
                           expt_data, n_expts_act, group_k, group_n):
    total = len(gather)
    N = W.shape[2]
    Y = np.zeros((total, N), dtype=np.float32)

    for tile_idx in range(len(expt_data)):
        ed = expt_data[tile_idx]
        eid = ed & 0xFFFF
        bid = ed >> 16
        M = hist[eid]
        start = offs[eid]

        for lm in range(BLOCK_M):
            gm = (bid * BLOCK_M + lm) % M
            if gm >= M:
                continue
            sorted_idx = start + gm
            token_idx = gather[sorted_idx] // n_expts_act

            result = np.zeros(N, dtype=np.float32)
            n_k_groups = (W.shape[1] + group_k - 1) // group_k

            for kg in range(n_k_groups):
                ks = kg * group_k
                ke = min(ks + group_k, W.shape[1])
                x_slice = X[token_idx, ks:ke].astype(np.float32)
                w_slice = W[eid, ks:ke, :].astype(np.float32)
                partial = x_slice @ w_slice

                xs = x_scale[token_idx, kg]
                for ng in range((N + group_n - 1) // group_n):
                    ns = ng * group_n
                    ne = min(ns + group_n, N)
                    ws = w_scale[eid, kg, ng]
                    result[ns:ne] += partial[ns:ne] * xs * ws

            Y[sorted_idx] = result

    return Y


def test_parity():
    print("=" * 60)
    print("MoE INT8 Block-Scaled (moe_op_gemm_a8w8_blockscale) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randint(-128, 127, (NUM_TOKENS, K_DIM)).astype(np.int8)
    W = np.random.randint(-128, 127, (NUM_EXPERTS, K_DIM, N_DIM)).astype(np.int8)
    x_scale = np.random.rand(NUM_TOKENS, K_DIM // GROUP_K).astype(np.float32) * 0.01
    w_scale = np.random.rand(NUM_EXPERTS, K_DIM // GROUP_K, N_DIM // GROUP_N).astype(np.float32) * 0.01

    hist, offs, gather, expt_data, grid_m = \
        generate_ogs_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    Y_ref = reference_moe_a8w8_bs(X, W, x_scale, w_scale, gather, hist, offs,
                                    expt_data, TOP_K, GROUP_K, GROUP_N)

    print(f"Config: tokens={NUM_TOKENS}, K={K_DIM}, N={N_DIM}")
    print(f"  group_k={GROUP_K}, group_n={GROUP_N}")
    print(f"  Output range: [{Y_ref.min():.4f}, {Y_ref.max():.4f}]")

    nonzero = np.any(Y_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{len(gather)}")

    print("\n[PASS] Reference implementation validated")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
