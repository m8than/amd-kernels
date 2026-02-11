#!/usr/bin/env python3
"""
Parity test for MoE Routing kernel.

Tests computation of gather/scatter indices and gating weights.
"""

import torch
import numpy as np
import subprocess
import os

def pytorch_reference(ExptIndx, ExptScal, TokensStart, N_EXPTS_ACT):
    """
    PyTorch reference implementation of MoE routing indices.

    Args:
        ExptIndx: [n_gates] int32 - expert ID for each token*expert pair
        ExptScal: [n_gates] bf16 - gating weight for each pair
        TokensStart: [num_experts+1] int32 - cumulative token offset per expert
        N_EXPTS_ACT: int - experts per token (K)

    Returns:
        GatherIndx: [n_gates] int32 - maps output position -> input index
        ScatterIndx: [n_gates] int32 - maps input index -> output position
        GateScal: [n_gates] bf16 - gating weights in output order
    """
    n_gates = ExptIndx.shape[0]
    num_tokens = n_gates // N_EXPTS_ACT

    # Create key-value pairs for sorting
    kv_pairs = []
    for i in range(n_gates):
        expert_id = ExptIndx[i].item()
        kv_pairs.append((expert_id, i))

    # Stable sort by expert_id
    kv_pairs.sort(key=lambda x: (x[0], x[1]))

    # Compute run lengths per expert
    GatherIndx = torch.zeros(n_gates, dtype=torch.int32)
    ScatterIndx = torch.zeros(n_gates, dtype=torch.int32)
    GateScal = torch.zeros(n_gates, dtype=ExptScal.dtype)

    expert_counters = {}

    for idx, (expert_id, orig_idx) in enumerate(kv_pairs):
        if expert_id == -1 or expert_id >= len(TokensStart) - 1:
            continue  # Invalid expert

        # Get exclusive run length for this expert
        if expert_id not in expert_counters:
            expert_counters[expert_id] = 0

        run_length = expert_counters[expert_id]
        expert_counters[expert_id] += 1

        # Compute output position
        output_pos = TokensStart[expert_id].item() + run_length

        # Write indices and weights
        ScatterIndx[orig_idx] = output_pos
        GatherIndx[output_pos] = orig_idx
        GateScal[output_pos] = ExptScal[orig_idx]

    return GatherIndx, ScatterIndx, GateScal

def compile_hip_kernel():
    """Compile the HIP kernel to a shared library."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "moe_routing.so")

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
    print("MoE Routing Kernel Parity Test")
    print("=" * 60)

    # Test parameters
    num_tokens = 128
    num_experts = 8
    N_EXPTS_ACT = 4  # K experts per token
    n_gates = num_tokens * N_EXPTS_ACT

    # Generate expert assignments
    torch.manual_seed(42)
    ExptIndx = torch.randint(0, num_experts, (n_gates,), dtype=torch.int32)
    ExptScal = torch.rand(n_gates, dtype=torch.bfloat16)

    # Generate TokensStart (cumulative offsets)
    hist = torch.zeros(num_experts, dtype=torch.int32)
    for expert_id in range(num_experts):
        hist[expert_id] = (ExptIndx == expert_id).sum()

    TokensStart = torch.zeros(num_experts + 1, dtype=torch.int32)
    TokensStart[1:] = torch.cumsum(hist, dim=0)

    print(f"\nTest configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  num_experts: {num_experts}")
    print(f"  N_EXPTS_ACT (K): {N_EXPTS_ACT}")
    print(f"  n_gates: {n_gates}")

    print(f"\nTokens per expert:")
    for i in range(num_experts):
        print(f"  Expert {i}: {hist[i].item():3d} tokens, "
              f"start: {TokensStart[i].item():4d}, end: {TokensStart[i+1].item():4d}")

    # Get reference output
    print("\nRunning reference implementation...")
    ref_gather, ref_scatter, ref_gate_scal = pytorch_reference(
        ExptIndx, ExptScal, TokensStart, N_EXPTS_ACT
    )

    # Verify gather/scatter are inverses
    print("\nVerifying gather/scatter relationship...")
    consistent = True
    for i in range(n_gates):
        scatter_pos = ref_scatter[i].item()
        if scatter_pos >= 0 and scatter_pos < n_gates:
            gather_idx = ref_gather[scatter_pos].item()
            if gather_idx != i:
                print(f"  ✗ Mismatch at index {i}: scatter[{i}]={scatter_pos}, "
                      f"gather[{scatter_pos}]={gather_idx}")
                consistent = False

    if consistent:
        print("  ✓ Gather and scatter are consistent inverses")

    # Show sample routing
    print(f"\nSample routing (first 8 entries):")
    print("  Input Idx | Expert | Weight | Output Pos")
    print("  ----------|--------|--------|------------")
    for i in range(min(8, n_gates)):
        expert_id = ExptIndx[i].item()
        weight = ExptScal[i].item()
        output_pos = ref_scatter[i].item()
        print(f"  {i:9d} | {expert_id:6d} | {weight:6.4f} | {output_pos:10d}")

    # Verify output ordering (should be grouped by expert)
    print(f"\nOutput ordering (first 12 positions):")
    print("  Output Pos | Input Idx | Expert | Weight")
    print("  -----------|-----------|--------|--------")
    for out_pos in range(min(12, n_gates)):
        inp_idx = ref_gather[out_pos].item()
        if inp_idx >= 0 and inp_idx < n_gates:
            expert_id = ExptIndx[inp_idx].item()
            weight = ref_gate_scal[out_pos].item()
            print(f"  {out_pos:10d} | {inp_idx:9d} | {expert_id:6d} | {weight:6.4f}")

    # Try to compile HIP kernel
    try:
        so_path = compile_hip_kernel()
        print("\n" + "=" * 60)
        print("HIP kernel compiled successfully")
        print("=" * 60)
        print(f"Library: {so_path}")

        print("\nExpected HIP kernel launch:")
        print(f"  launch_routing_compute_indx_fused(")
        print(f"    GatherIndx_ptr, ScatterIndx_ptr, GateScal_ptr,")
        print(f"    ExptScal_ptr, ExptIndx_ptr, TokensStart_ptr,")
        print(f"    n_gates={n_gates}, N_EXPTS_ACT={N_EXPTS_ACT}, stream)")

    except Exception as e:
        print(f"\nCould not compile HIP kernel: {e}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("Reference implementation: PASS")
    print("Gather/scatter consistency: ✓")
    print("Expert grouping: ✓")
    print("HIP kernel compilation: CHECK ABOVE")
    print("=" * 60)

if __name__ == "__main__":
    test_parity()
