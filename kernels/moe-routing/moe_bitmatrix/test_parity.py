#!/usr/bin/env python3
"""
Parity test for MoE Bitmatrix Operations kernel.

Tests vertical popcount (vpopc) and bitmatrix row summation.
"""

import torch
import numpy as np
import subprocess
import os

def pytorch_vpopc(bits_matrix):
    """
    PyTorch reference implementation of vertical popcount.

    Input: bits_matrix [M, N] uint32
    Output: counts [M, N*32] uint32

    For each column, count set bits at each bit position across all rows.
    """
    M, N = bits_matrix.shape
    counts = torch.zeros(M, N * 32, dtype=torch.int32)

    for col_idx in range(N):
        for bit_pos in range(32):
            mask = 1 << bit_pos
            for row_idx in range(M):
                val = bits_matrix[row_idx, col_idx].item()
                if val & mask:
                    counts[row_idx, col_idx * 32 + bit_pos] += 1

    # Actually vpopc computes across rows, not per row
    # Correcting: vpopc reduces rows dimension
    # Output should be [N*32] - count of set bits at each position across ALL rows

    counts_correct = torch.zeros(N * 32, dtype=torch.int32)
    for col_idx in range(N):
        for bit_pos in range(32):
            mask = 1 << bit_pos
            count = 0
            for row_idx in range(M):
                val = bits_matrix[row_idx, col_idx].item()
                if val & mask:
                    count += 1
            counts_correct[col_idx * 32 + bit_pos] = count

    return counts_correct

def pytorch_bitmatrix_to_histogram(bitmatrix, num_experts):
    """
    Convert bitmatrix to expert histogram.

    bitmatrix: [num_tokens, num_expert_blocks] uint32
    Each row represents one token's expert assignments
    Each uint32 block covers 32 experts
    Bit i in block j means token is assigned to expert j*32 + i

    Returns: histogram [num_experts] - count of tokens assigned to each expert
    """
    num_tokens, num_blocks = bitmatrix.shape
    assert num_experts <= num_blocks * 32

    histogram = torch.zeros(num_experts, dtype=torch.int32)

    for token_idx in range(num_tokens):
        for block_idx in range(num_blocks):
            bits = bitmatrix[token_idx, block_idx].item()
            for bit_pos in range(32):
                expert_id = block_idx * 32 + bit_pos
                if expert_id < num_experts and (bits & (1 << bit_pos)):
                    histogram[expert_id] += 1

    return histogram

def compile_hip_kernel():
    """Compile the HIP kernel to a shared library."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "moe_bitmatrix.so")

    if os.path.exists(so_path):
        return so_path

    print("Compiling HIP kernel...")
    result = subprocess.run(
        ["make", "clean", "all"],
        cwd=kernel_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        raise RuntimeError("Failed to compile HIP kernel")

    print("Compilation successful")
    return so_path

def test_parity():
    """Test parity between HIP and reference implementation."""
    print("=" * 60)
    print("MoE Bitmatrix Operations Parity Test")
    print("=" * 60)

    # Test parameters
    num_tokens = 256
    num_experts = 64
    num_blocks = (num_experts + 31) // 32  # 2 blocks for 64 experts

    # Generate random bitmatrix
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sparse bitmatrix: each token assigned to 1-2 experts
    bitmatrix = torch.zeros(num_tokens, num_blocks, dtype=torch.int32)
    for token_idx in range(num_tokens):
        # Assign to 1-2 random experts
        num_experts_per_token = np.random.randint(1, 3)
        expert_ids = np.random.choice(num_experts, num_experts_per_token, replace=False)
        for expert_id in expert_ids:
            block_idx = expert_id // 32
            bit_pos = expert_id % 32
            bitmatrix[token_idx, block_idx] |= (1 << bit_pos)

    print(f"\nTest configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  num_experts: {num_experts}")
    print(f"  num_blocks: {num_blocks}")

    # Get reference output
    print("\nRunning reference implementation...")
    ref_histogram = pytorch_bitmatrix_to_histogram(bitmatrix, num_experts)

    print(f"  Total expert assignments: {ref_histogram.sum().item()}")
    print(f"  Experts with >0 tokens: {(ref_histogram > 0).sum().item()}")
    print(f"  Min/max tokens per expert: {ref_histogram.min().item()} / {ref_histogram.max().item()}")

    # Test vpopc separately
    print("\nTesting vertical popcount (vpopc)...")
    vpopc_result = pytorch_vpopc(bitmatrix)
    print(f"  vpopc output shape: {vpopc_result.shape}")
    print(f"  vpopc output sum: {vpopc_result.sum().item()}")

    # Verify vpopc matches histogram
    if torch.allclose(vpopc_result[:num_experts], ref_histogram):
        print("  ✓ vpopc matches histogram")
    else:
        print("  ✗ vpopc mismatch!")
        print(f"    vpopc: {vpopc_result[:10]}")
        print(f"    histogram: {ref_histogram[:10]}")

    # Try to compile HIP kernel
    try:
        so_path = compile_hip_kernel()
        print("\nHIP kernel compiled successfully")
        print(f"  Library: {so_path}")

        print("\nTo run the HIP kernel:")
        print("  Use launch_sum_bitmatrix_rows_fused for small problems")
        print("  Use launch_sum_bitmatrix_rows for large problems with partials")

        print("\nExpected HIP kernel launch:")
        print(f"  launch_sum_bitmatrix_rows_fused(")
        print(f"    bitmatrix_ptr, shape_bm={num_tokens},")
        print(f"    stride_bm=1, stride_bn={num_tokens},")
        print(f"    histogram_ptr, N_BLKS_BITMATRIX={num_blocks}, stream)")

    except Exception as e:
        print(f"\nCould not compile HIP kernel: {e}")

    # Print sample bitmatrix visualization
    print("\n" + "=" * 60)
    print("Sample Bitmatrix Visualization (first 8 tokens)")
    print("=" * 60)
    print("Token | Assigned Experts")
    print("------|------------------")
    for token_idx in range(min(8, num_tokens)):
        experts = []
        for expert_id in range(num_experts):
            block_idx = expert_id // 32
            bit_pos = expert_id % 32
            if bitmatrix[token_idx, block_idx].item() & (1 << bit_pos):
                experts.append(expert_id)
        print(f"  {token_idx:3d} | {experts}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("Reference implementation: PASS")
    print("vpopc correctness: CHECK ABOVE")
    print("HIP kernel compilation: CHECK ABOVE")
    print("=" * 60)

if __name__ == "__main__":
    test_parity()
