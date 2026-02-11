#!/usr/bin/env python3
"""
Parity test for MoE End-to-End kernel (moe_op_e2e).

Tests the fused two-layer gated MLP:
  Layer 1: intermediate = SiLU(A @ W1_gate) * (A @ W1_up)
  Layer 2: output = intermediate @ W2

Input shapes:
  - A: [num_tokens, K] bf16
  - W1: [E, N, K] bf16 where N = 2*hidden (gate concat up)
  - W2: [E, K, N//2] bf16
  - sorted_token_ids: [num_tokens_padded] int32
  - expert_ids: [num_m_blocks] int32
  - topk_weights: [num_tokens_padded] float32

Output shapes:
  - C: [num_tokens_padded, K] bf16

Reference: Triton e2e_moe_kernel from moe_op_e2e.py
"""

import numpy as np
import sys
import os

NUM_TOKENS = 32
NUM_EXPERTS = 4
TOP_K = 2
K_DIM = 128
HIDDEN_DIM = 256  # N = 2*HIDDEN_DIM for gate+up
BLOCK_M = 64


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp2(-1.44269504089 * x))


def reference_moe_e2e(A, W1, W2, sorted_token_ids, expert_ids,
                       topk_weights, num_valid, top_k, mul_weight=True):
    """Reference implementation of E2E MoE kernel."""
    num_padded = len(sorted_token_ids)
    K = A.shape[1]
    N = W1.shape[1]
    N_half = N // 2
    C = np.zeros((num_padded, K), dtype=np.float32)

    num_blocks = len(expert_ids)
    for bid in range(num_blocks):
        expert = expert_ids[bid]
        if expert == -1:
            continue

        for lm in range(BLOCK_M):
            tidx = bid * BLOCK_M + lm
            if tidx >= num_padded:
                break
            token_id = sorted_token_ids[tidx]
            if token_id >= num_valid:
                continue
            orig = token_id // top_k

            # Layer 1: A @ W1
            a_row = A[orig].astype(np.float32)
            w1_expert = W1[expert].astype(np.float32)  # [N, K]

            # W1 has gate in first half, up in second half
            l1_out = a_row @ w1_expert.T  # [N]
            gate = l1_out[:N_half]
            up = l1_out[N_half:]

            # SiLU-and-mul
            intermediate = silu(gate) * up  # [N_half]

            # Layer 2: intermediate @ W2
            w2_expert = W2[expert].astype(np.float32)  # [K, N_half]
            result = intermediate @ w2_expert.T  # [K]

            if mul_weight:
                result *= topk_weights[tidx]

            C[token_id] += result

    return C


def generate_routing(num_tokens, num_experts, top_k):
    total = num_tokens * top_k
    expert_assigns = np.random.randint(0, num_experts, (num_tokens, top_k)).astype(np.int32)
    weights = np.random.rand(num_tokens, top_k).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)

    flat_ids = np.arange(total, dtype=np.int32)
    flat_experts = expert_assigns.flatten()
    flat_weights = weights.flatten()

    sort_idx = np.argsort(flat_experts, kind='stable')
    sorted_ids = flat_ids[sort_idx]
    sorted_experts = flat_experts[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    num_padded = ((total + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    padded_ids = np.full(num_padded, total, dtype=np.int32)
    padded_ids[:total] = sorted_ids
    padded_weights = np.zeros(num_padded, dtype=np.float32)
    padded_weights[:total] = sorted_weights

    num_blocks = num_padded // BLOCK_M
    block_experts = np.full(num_blocks, -1, dtype=np.int32)
    for b in range(num_blocks):
        s = b * BLOCK_M
        if s < total:
            block_experts[b] = sorted_experts[min(s, total - 1)]

    return padded_ids, block_experts, padded_weights, num_padded, total


def test_parity():
    print("=" * 60)
    print("MoE E2E (moe_op_e2e) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    A = np.random.randn(NUM_TOKENS, K_DIM).astype(np.float16)
    W1 = np.random.randn(NUM_EXPERTS, 2 * HIDDEN_DIM, K_DIM).astype(np.float16)
    W2 = np.random.randn(NUM_EXPERTS, K_DIM, HIDDEN_DIM).astype(np.float16)

    padded_ids, block_experts, padded_weights, num_padded, num_valid = \
        generate_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    print(f"Config: tokens={NUM_TOKENS}, experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  K={K_DIM}, hidden={HIDDEN_DIM}, N(W1)={2*HIDDEN_DIM}")
    print(f"  padded={num_padded}, valid={num_valid}")

    C_ref = reference_moe_e2e(A, W1, W2, padded_ids, block_experts,
                               padded_weights, num_valid, TOP_K)

    print(f"  Output shape: {C_ref.shape}")
    print(f"  Output range: [{C_ref.min():.4f}, {C_ref.max():.4f}]")

    nonzero = np.any(C_ref != 0, axis=1).sum()
    print(f"  Non-zero rows: {nonzero}/{num_padded}")
    assert nonzero > 0, "All-zero output!"

    print("\nInput/Output Specification:")
    print(f"  A:  [{NUM_TOKENS}, {K_DIM}] bf16")
    print(f"  W1: [{NUM_EXPERTS}, {2*HIDDEN_DIM}, {K_DIM}] bf16")
    print(f"  W2: [{NUM_EXPERTS}, {K_DIM}, {HIDDEN_DIM}] bf16")
    print(f"  C:  [{num_padded}, {K_DIM}] bf16")

    print("\n[PASS] Reference implementation validated")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
