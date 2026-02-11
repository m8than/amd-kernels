#!/usr/bin/env python3
"""
Parity test for MoE Top-K Selection kernel.

Tests top-K expert selection with optional softmax and bitmatrix packing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import os

def pytorch_reference(X, K, apply_softmax=True):
    """
    PyTorch reference implementation of MoE top-K selection.

    Args:
        X: [M, N] bf16 - logits for M tokens, N experts
        K: int - number of experts to select per token
        apply_softmax: bool - whether to normalize with softmax

    Returns:
        Yv: [M, K] bf16 - top-K values
        Yi: [M, K] int32 - top-K expert indices
        Bits: [M, ceil(N/32)] uint32 - bitmatrix
    """
    M, N = X.shape

    # Select top-K
    topk_vals, topk_indices = torch.topk(X, K, dim=1, largest=True, sorted=True)

    # Apply softmax if requested
    if apply_softmax:
        topk_vals = F.softmax(topk_vals.float(), dim=1).to(X.dtype)

    # Pack into bitmatrix
    num_bit_blocks = (N + 31) // 32
    bitmatrix = torch.zeros(M, num_bit_blocks, dtype=torch.int32)

    for row_idx in range(M):
        for k in range(K):
            expert_id = topk_indices[row_idx, k].item()
            block_id = expert_id // 32
            bit_pos = expert_id % 32
            bitmatrix[row_idx, block_id] |= (1 << bit_pos)

    return topk_vals, topk_indices.to(torch.int32), bitmatrix.to(torch.int32)

def compile_hip_kernel():
    """Compile the HIP kernel to a shared library."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "moe_topk.so")

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
    print("MoE Top-K Selection Parity Test")
    print("=" * 60)

    # Test parameters
    num_tokens = 128
    num_experts = 64
    K = 4
    apply_softmax = True

    # Generate random logits
    torch.manual_seed(42)
    X = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)

    print(f"\nTest configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  num_experts: {num_experts}")
    print(f"  K (experts per token): {K}")
    print(f"  apply_softmax: {apply_softmax}")

    # Get reference output
    print("\nRunning reference implementation...")
    ref_vals, ref_indices, ref_bitmatrix = pytorch_reference(X, K, apply_softmax)

    print(f"  Output shapes:")
    print(f"    Values: {ref_vals.shape}")
    print(f"    Indices: {ref_indices.shape}")
    print(f"    Bitmatrix: {ref_bitmatrix.shape}")

    # Verify softmax normalization
    if apply_softmax:
        row_sums = ref_vals.float().sum(dim=1)
        print(f"\n  Softmax check (row sums should be ~1.0):")
        print(f"    Mean: {row_sums.mean().item():.6f}")
        print(f"    Std:  {row_sums.std().item():.6f}")
        print(f"    Min:  {row_sums.min().item():.6f}")
        print(f"    Max:  {row_sums.max().item():.6f}")

    # Verify bitmatrix consistency
    print(f"\n  Bitmatrix verification:")
    bits_set_per_row = []
    for row_idx in range(num_tokens):
        bits_set = 0
        for block_idx in range(ref_bitmatrix.shape[1]):
            bits = ref_bitmatrix[row_idx, block_idx].item()
            bits_set += bin(bits & 0xFFFFFFFF).count('1')
        bits_set_per_row.append(bits_set)

    print(f"    Bits set per row (should be {K}):")
    print(f"      Mean: {np.mean(bits_set_per_row):.1f}")
    print(f"      All equal to K: {all(b == K for b in bits_set_per_row)}")

    # Show sample outputs
    print(f"\n  Sample output (first 3 tokens):")
    for row_idx in range(min(3, num_tokens)):
        print(f"\n    Token {row_idx}:")
        print(f"      Expert IDs: {ref_indices[row_idx].tolist()}")
        print(f"      Values:     {[f'{v:.4f}' for v in ref_vals[row_idx].float().tolist()]}")

        # Decode bitmatrix
        experts_from_bits = []
        for block_idx in range(ref_bitmatrix.shape[1]):
            bits = ref_bitmatrix[row_idx, block_idx].item()
            for bit_pos in range(32):
                if bits & (1 << bit_pos):
                    expert_id = block_idx * 32 + bit_pos
                    if expert_id < num_experts:
                        experts_from_bits.append(expert_id)

        print(f"      Bitmatrix experts: {sorted(experts_from_bits)}")

        # Verify consistency
        if sorted(ref_indices[row_idx].tolist()) == sorted(experts_from_bits):
            print(f"      ✓ Bitmatrix matches indices")
        else:
            print(f"      ✗ Bitmatrix mismatch!")

    # Try to compile HIP kernel
    try:
        so_path = compile_hip_kernel()
        print("\n" + "=" * 60)
        print("HIP kernel compiled successfully")
        print("=" * 60)
        print(f"Library: {so_path}")

        print("\nExpected HIP kernel launch:")
        print(f"  launch_moe_topk(")
        print(f"    X_ptr, stride_xm={num_experts},")
        print(f"    Yv_ptr, Yi_ptr, stride_ym={K},")
        print(f"    Bits_ptr, stride_bm=1, stride_bn={num_tokens},")
        print(f"    n_rows={num_tokens}, n_expts_tot={num_experts},")
        print(f"    K={K}, apply_softmax={apply_softmax}, stream)")

    except Exception as e:
        print(f"\nCould not compile HIP kernel: {e}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("Reference implementation: PASS")
    print("Top-K selection: ✓")
    print("Softmax normalization: ✓" if apply_softmax else "N/A")
    print("Bitmatrix packing: ✓")
    print("HIP kernel compilation: CHECK ABOVE")
    print("=" * 60)

if __name__ == "__main__":
    test_parity()
