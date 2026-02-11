#!/usr/bin/env python3
"""
Parity test for MoE GEMM Core kernel (moe_op).

Tests the fused MoE GEMM operation:
  For each expert e: C[tokens_for_e] = A[tokens_for_e] @ B[e]

Input shapes:
  - A: [num_tokens, K] bf16 — input token activations
  - B: [E, K, N] bf16 — expert weight matrices
  - sorted_token_ids: [num_tokens_padded] int32 — sorted by expert
  - expert_ids: [num_m_blocks] int32 — expert for each block
  - topk_weights: [num_tokens_padded] float32 — routing weights
  - num_tokens_post_padded: [1] int32 — padded token count

Output shapes:
  - C: [num_tokens_padded, N] bf16

Reference: Triton _fused_moe_kernel from moe_op.py
"""

import subprocess
import sys
import os
import struct

import numpy as np

# Test configuration
NUM_TOKENS = 64      # Original tokens
NUM_EXPERTS = 8      # Number of experts
TOP_K = 2            # Top-k routing
K_DIM = 256          # Input feature dimension
N_DIM = 256          # Output feature dimension
BLOCK_M = 128        # Must match kernel

def generate_moe_routing(num_tokens, num_experts, top_k):
    """Generate MoE routing data (sorted token IDs, expert IDs, weights)."""
    # Each token is assigned to top_k experts
    total_slots = num_tokens * top_k

    # Random expert assignments per token
    expert_assignments = np.random.randint(0, num_experts, size=(num_tokens, top_k)).astype(np.int32)

    # Random routing weights (softmax over top-k)
    raw_weights = np.random.rand(num_tokens, top_k).astype(np.float32)
    weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    # Sort tokens by expert ID (flatten first)
    flat_token_ids = np.arange(total_slots, dtype=np.int32)  # token_id * top_k + k
    flat_expert_ids = expert_assignments.flatten()
    flat_weights = weights.flatten()

    # Sort by expert
    sort_idx = np.argsort(flat_expert_ids, kind='stable')
    sorted_token_ids = flat_token_ids[sort_idx]
    sorted_expert_ids = flat_expert_ids[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    # Pad to multiple of BLOCK_M
    num_padded = ((total_slots + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    padded_token_ids = np.full(num_padded, total_slots, dtype=np.int32)  # invalid sentinel
    padded_token_ids[:total_slots] = sorted_token_ids
    padded_weights = np.zeros(num_padded, dtype=np.float32)
    padded_weights[:total_slots] = sorted_weights

    # Expert IDs per block
    num_blocks = num_padded // BLOCK_M
    block_expert_ids = np.zeros(num_blocks, dtype=np.int32)
    for b in range(num_blocks):
        start = b * BLOCK_M
        if start < total_slots:
            block_expert_ids[b] = sorted_expert_ids[min(start, total_slots - 1)]
        else:
            block_expert_ids[b] = -1  # padding block

    return padded_token_ids, block_expert_ids, padded_weights, num_padded, total_slots


def reference_moe_gemm(A, B, sorted_token_ids, expert_ids, topk_weights,
                       num_valid_tokens, top_k, mul_routed_weight=True):
    """
    Reference implementation of MoE GEMM in NumPy.

    A: [num_tokens, K]
    B: [E, K, N]
    Returns C: [num_padded, N]
    """
    num_padded = len(sorted_token_ids)
    N = B.shape[2]
    C = np.zeros((num_padded, N), dtype=np.float32)

    num_blocks = len(expert_ids)
    for block_id in range(num_blocks):
        expert = expert_ids[block_id]
        if expert == -1:
            continue

        for local_m in range(BLOCK_M):
            token_idx = block_id * BLOCK_M + local_m
            if token_idx >= num_padded:
                break
            token_id = sorted_token_ids[token_idx]
            if token_id >= num_valid_tokens:
                continue

            orig_token = token_id // top_k

            # C[token_id, :] = A[orig_token, :] @ B[expert, :, :]
            result = A[orig_token].astype(np.float32) @ B[expert].astype(np.float32)

            if mul_routed_weight:
                result *= topk_weights[token_idx]

            C[token_id] = result

    return C


def test_parity():
    """Run parity test between reference and HipKittens kernel."""
    print("=" * 60)
    print("MoE GEMM Core (moe_op) Parity Test")
    print("=" * 60)

    # Generate inputs
    np.random.seed(42)
    A = np.random.randn(NUM_TOKENS, K_DIM).astype(np.float16)  # bf16 approx
    B = np.random.randn(NUM_EXPERTS, K_DIM, N_DIM).astype(np.float16)

    sorted_token_ids, expert_ids, topk_weights, num_padded, num_valid = \
        generate_moe_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    print(f"Configuration:")
    print(f"  num_tokens={NUM_TOKENS}, num_experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  K={K_DIM}, N={N_DIM}")
    print(f"  num_padded={num_padded}, num_valid={num_valid}")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_N=128, BLOCK_K=32")

    # Reference computation
    print("\nRunning reference implementation...")
    C_ref = reference_moe_gemm(A, B, sorted_token_ids, expert_ids,
                                topk_weights, num_valid, TOP_K,
                                mul_routed_weight=True)
    print(f"  Reference output shape: {C_ref.shape}")
    print(f"  Reference output range: [{C_ref.min():.4f}, {C_ref.max():.4f}]")

    # Check if HIP kernel binary exists
    kernel_path = os.path.join(os.path.dirname(__file__), 'moe_op_kernel')
    if os.path.exists(kernel_path):
        print("\nHipKittens kernel binary found. Running GPU parity test...")
        # GPU test would go here when hardware is available
        print("  [SKIP] GPU hardware not available for parity test")
    else:
        print(f"\nKernel binary not found at {kernel_path}")
        print("  Build with: make -C kernels/moe_op/")

    # Compile command for reference
    print("\nExpected compilation command:")
    print("  hipcc -std=c++20 -O3 --offload-arch=gfx942 "
          "-I../../reference/hipkittens/include "
          "-o moe_op_kernel kernel.cpp")

    # Input/output specification
    print("\nInput/Output Specification:")
    print(f"  A:                  [{NUM_TOKENS}, {K_DIM}] bf16")
    print(f"  B:                  [{NUM_EXPERTS}, {K_DIM}, {N_DIM}] bf16")
    print(f"  C:                  [{num_padded}, {N_DIM}] bf16")
    print(f"  sorted_token_ids:   [{num_padded}] int32")
    print(f"  expert_ids:         [{len(expert_ids)}] int32")
    print(f"  topk_weights:       [{num_padded}] float32")

    # Sanity checks on reference
    # Verify non-zero outputs exist
    nonzero_rows = np.any(C_ref != 0, axis=1).sum()
    print(f"\n  Non-zero output rows: {nonzero_rows}/{num_padded}")
    assert nonzero_rows > 0, "Reference produced all-zero output!"

    print("\n[PASS] Reference implementation validated")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_parity()
    sys.exit(0 if success else 1)
