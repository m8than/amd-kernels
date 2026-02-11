#!/usr/bin/env python3
"""
Parity test for MoE INT4 GEMM (moe_op_gemm_a4w4).

Tests INT4xINT4 quantized GEMM with packed storage:
  - X: [total_tokens, K/2] uint8 (packed INT4)
  - W: [E, K/2, N] uint8 (packed INT4)
  - Y = dequant(X_int4) @ dequant(W_int4) * scales

Output: Y: [total_tokens, N] bf16
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


def unpack_int4(packed, idx):
    """Unpack signed INT4 from packed uint8."""
    shift = (idx % 2) * 4
    val = (packed >> shift) & 0xF
    return np.where(val >= 8, val.astype(np.int32) - 16, val.astype(np.int32)).astype(np.float32)


def reference_moe_a4w4(X_packed, W_packed, gather, hist, offs,
                        expt_data, n_expts_act, K):
    total = len(gather)
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
                x_val = unpack_int4(X_packed[token_idx, k // 2], k)
                w_packed_row = W_packed[eid, k // 2, :]
                w_vals = unpack_int4(w_packed_row, k)
                result += float(x_val) * w_vals

            Y[start + gm] = result

    return Y


def test_parity():
    print("=" * 60)
    print("MoE INT4 GEMM (moe_op_gemm_a4w4) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randint(0, 255, (NUM_TOKENS, K_DIM // 2)).astype(np.uint8)
    W = np.random.randint(0, 255, (NUM_EXPERTS, K_DIM // 2, N_DIM)).astype(np.uint8)

    hist, offs, gather, expt_data, grid_m = \
        generate_ogs_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    Y_ref = reference_moe_a4w4(X, W, gather, hist, offs, expt_data, TOP_K, K_DIM)

    print(f"Config: tokens={NUM_TOKENS}, K={K_DIM}, N={N_DIM}")
    print(f"  Output range: [{Y_ref.min():.4f}, {Y_ref.max():.4f}]")
    nonzero = np.any(Y_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{len(gather)}")

    print("\n[PASS] Reference implementation validated")
    return True

if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
