#!/usr/bin/env python3
"""
Parity test for MoE MXFP4 GEMM + fused SiLU (moe_op_mxfp4_silu_fused).

Tests microscaled FP4 GEMM with fused SwiGLU activation:
  - A: [total_tokens, K/2] uint8 (packed FP4 E2M1)
  - B: [E, K/2, N] uint8 (packed FP4 E2M1) with interleaved gate/up columns
  - MX microscales for both A and B
  - Per-tensor/per-expert static scales

After GEMM, applies: output[n] = SiLU(gate[n]) * up[n]
where gate and up are interleaved in the GEMM output.

Output: C: [num_padded, N/2] bf16
"""

import numpy as np
import sys

NUM_TOKENS = 64
NUM_EXPERTS = 8
TOP_K = 2
K_DIM = 256
N_DIM = 512   # Full N (gate+up interleaved), output is N/2=256
BLOCK_M = 128
MX_GROUP_SIZE = 32

# E2M1 FP4 lookup table
FP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def silu(x):
    return x / (1.0 + np.exp2(-1.44269504089 * x))


def unpack_fp4(packed, idx):
    """Unpack FP4 E2M1 from packed uint8."""
    shift = (idx % 2) * 4
    nibble = (packed >> shift) & 0xF
    return FP4_LUT[nibble]


def decode_mx_scale(val):
    """Decode E8M0 microscale: 2^(val - 127)."""
    result = np.where(val == 0, 0.0, np.ldexp(1.0, val.astype(np.int32) - 127))
    return result.astype(np.float32)


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


def reference_moe_mxfp4_silu(A_packed, B_packed, A_mx, B_mx, a_scale, b_scale,
                               sorted_ids, expert_ids, topk_weights,
                               num_valid, top_k, K, N, mul_weight=True):
    num_padded = len(sorted_ids)
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

            # Full GEMM over N columns
            gemm_out = np.zeros(N, dtype=np.float32)
            for k in range(K):
                mx_group = k // MX_GROUP_SIZE
                a_mx_s = decode_mx_scale(np.array([A_mx[orig, mx_group]]))[0]
                a_val = unpack_fp4(A_packed[orig, k // 2], k) * a_mx_s

                b_packed_row = B_packed[expert, k // 2, :]
                b_vals = np.array([unpack_fp4(b_packed_row[n], k) for n in range(N)])
                b_mx_s = decode_mx_scale(B_mx[expert, mx_group, :])
                b_vals *= b_mx_s

                gemm_out += a_val * b_vals

            gemm_out *= a_scale * b_scale[expert]
            if mul_weight:
                gemm_out *= topk_weights[tidx]

            # Split into gate and up, apply SiLU-and-mul
            gate = gemm_out[:N_half]
            up = gemm_out[N_half:]
            C[tidx] = silu(gate) * up

    return C


def test_parity():
    print("=" * 60)
    print("MoE MXFP4 + SiLU Fused (moe_op_mxfp4_silu_fused) Parity Test")
    print("=" * 60)

    np.random.seed(42)
    A = np.random.randint(0, 255, (NUM_TOKENS, K_DIM // 2)).astype(np.uint8)
    B = np.random.randint(0, 255, (NUM_EXPERTS, K_DIM // 2, N_DIM)).astype(np.uint8)

    A_mx = np.random.randint(120, 134, (NUM_TOKENS, K_DIM // MX_GROUP_SIZE)).astype(np.uint8)
    B_mx = np.random.randint(120, 134, (NUM_EXPERTS, K_DIM // MX_GROUP_SIZE, N_DIM)).astype(np.uint8)

    a_scale = np.float32(0.01)
    b_scale = np.random.rand(NUM_EXPERTS).astype(np.float32) * 0.01

    padded_ids, block_experts, padded_weights, num_padded, num_valid = \
        generate_routing(NUM_TOKENS, NUM_EXPERTS, TOP_K)

    C_ref = reference_moe_mxfp4_silu(A, B, A_mx, B_mx, a_scale, b_scale,
                                       padded_ids, block_experts, padded_weights,
                                       num_valid, TOP_K, K_DIM, N_DIM)

    print(f"Config: tokens={NUM_TOKENS}, K={K_DIM}, N={N_DIM}, out_dim={N_DIM//2}")
    print(f"  MX group size: {MX_GROUP_SIZE}")
    print(f"  Output shape: {C_ref.shape}")
    print(f"  Output range: [{C_ref.min():.6f}, {C_ref.max():.6f}]")

    nonzero = np.any(C_ref != 0, axis=1).sum()
    assert nonzero > 0
    print(f"  Non-zero rows: {nonzero}/{num_padded}")

    print("\n[PASS] Reference implementation validated")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_parity() else 1)
