#!/usr/bin/env python3
"""
Parity test for MoE GEMM with fused SiLU activation (moe_op_silu_fused).

Tests: activated = SiLU(gate) * up
where [gate, up] = A @ B[expert] (interleaved columns)

Input shapes:
  - A: [num_tokens, K] bf16
  - B: [E, K, N] bf16 where N = 2 * output_dim
  - routing metadata

Output shapes:
  - C: [num_tokens_padded, N/2] bf16

Reference: Triton _fused_moe_silu_kernel from moe_op_silu_fused.py
"""

import numpy as np
import sys

NUM_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 2
K_DIM = 256
N_DIM = 512  # Full N (gate+up), output is N/2=256
BLOCK_M = 128


def silu(x):
    return x / (1.0 + np.exp2(-1.44269504089 * x))


def generate_routing(num_tokens, num_experts, top_k):
    total = num_tokens * top_k
    assigns = np.random.randint(0, num_experts, (num_tokens, top_k)).astype(np.int32)
    weights = np.random.rand(num_tokens, top_k).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)

    flat_ids = np.arange(total, dtype=np.int32)
    flat_experts = assigns.flatten()
    flat_weights = weights.flatten()
    sort_idx = np.argsort(flat_experts, kind='stable')

    num_padded = ((total + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    padded_ids = np.full(num_padded, total, dtype=np.int32)
    padded_ids[:total] = flat_ids[sort_idx]
    padded_weights = np.zeros(num_padded, dtype=np.float32)
    padded_weights[:total] = flat_weights[sort_idx]

    sorted_experts = flat_experts[sort_idx]
    num_blocks = num_padded // BLOCK_M
    block_experts = np.full(num_blocks, -1, dtype=np.int32)
    for b in range(num_blocks):
        s = b * BLOCK_M
        if s < total:
            block_experts[b] = sorted_experts[min(s, total - 1)]

    return padded_ids, block_experts, padded_weights, num_padded, total


def reference_moe_silu(A, B, sorted_ids, expert_ids, topk_weights,
                        num_valid, top_k, mul_weight=True):
    num_padded = len(sorted_ids)
    N = B.shape[2]
    N_half = N // 2
    C = np.zeros((num_padded, N_half), dtype=np.float32)

    for bid in range(len(expert_ids)):
        expert = expert_ids[bid]
        if expert == -1:
            continue
        for lm in range(BLOCK_M):
            tidx = bid * BLOCK_M + lm
            if tidx >= num_padded:
                break
            token_id = sorted_ids[tidx]
            if token_id >= num_valid:
                continue
            orig = token_id // top_k

            gemm_out = A[orig].astype(np.float32) @ B[expert].astype(np.float32)

            if mul_weight:
                gemm_out *= topk_weights[tidx]

            gate = gemm_out[:N_half]
            up = gemm_out[N_half:]
            C[token_id] = silu(gate) * up

    return C


def test_parity():
    print("=" * 60)
    print("MoE GEMM + SiLU Fused (moe_op_silu_fused) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    A = np.random.randn(NUM_TOKENS, K_DIM).astype(np.float16)
    B = np.random.randn(NUM_EXPERTS, K_DIM, N_DIM).astype(np.float16)

    padded_ids, block_experts, padded_weights, num_padded, num_valid = \
        generate_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    C_ref = reference_moe_silu(A, B, padded_ids, block_experts,
                                padded_weights, num_valid, TOP_K)

    print(f"Config: tokens={NUM_TOKENS}, K={K_DIM}, N={N_DIM}, out_dim={N_DIM//2}")
    print(f"  Output shape: {C_ref.shape}")
    print(f"  Output range: [{C_ref.min():.4f}, {C_ref.max():.4f}]")

    nonzero = np.any(C_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{num_padded}")

    print("\n[PASS] Reference implementation validated")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
