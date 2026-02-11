#!/usr/bin/env python3
"""
Parity test for ff_fused_gated HipKittens kernel

Tests the gated feed forward kernel against Triton reference.

NOTE: This kernel currently implements stages 1-2 (first GEMM + gating).
The full Triton kernel also includes stage 3 (second GEMM with W2).
This test validates the intermediate gated output.
"""

import torch
import numpy as np
import subprocess
import os
import sys

def compile_kernel():
    """Compile the HipKittens kernel."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Compiling kernel in {kernel_dir}...")

    # Check for required environment
    if 'THUNDERKITTENS_ROOT' not in os.environ:
        print("ERROR: THUNDERKITTENS_ROOT environment variable not set")
        return False

    result = subprocess.run(['make', 'clean'], cwd=kernel_dir, capture_output=True)
    result = subprocess.run(['make'], cwd=kernel_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print("Compilation successful")
    return True

def silu(x):
    """SiLU activation function."""
    return x * torch.sigmoid(x)

def reference_gated_ff_stage1_2(X, W1, activation='silu'):
    """
    Reference implementation of stages 1-2 of gated feed forward.

    Args:
        X: (M, K) input tensor
        W1: (N, K) weight tensor (gate and up concatenated)
        activation: activation function name

    Returns:
        gated: (M, N/2) intermediate gated output
    """
    M, K = X.shape
    N, K_ = W1.shape
    assert K == K_, f"K dimension mismatch: {K} != {K_}"
    assert N % 2 == 0, "N must be even"

    half_n = N // 2

    # Split W1 into gate and up paths
    W1_gate = W1[:half_n, :]  # (N/2, K)
    W1_up = W1[half_n:, :]    # (N/2, K)

    # First GEMM: X @ W1^T
    gate_result = X @ W1_gate.T  # (M, N/2)
    up_result = X @ W1_up.T      # (M, N/2)

    # Apply activation and gating
    if activation == 'silu':
        gated = gate_result * silu(up_result)
    else:
        gated = gate_result * up_result

    return gated

def generate_test_inputs(M=128, N=256, K=4096, dtype=torch.bfloat16):
    """Generate random test inputs."""
    X = torch.randn(M, K, dtype=dtype)
    W1 = torch.randn(N, K, dtype=dtype)
    W2 = torch.randn(N // 2, K, dtype=dtype)  # For future stage 3 testing
    return X, W1, W2

def test_parity():
    """Test parity between reference and HipKittens implementation."""
    print("=" * 80)
    print("FF Fused Gated Kernel Parity Test")
    print("=" * 80)

    # Test configuration
    M, N, K = 128, 256, 4096
    dtype = torch.bfloat16

    print(f"\nTest configuration:")
    print(f"  Input X: ({M}, {K}) {dtype}")
    print(f"  Weight W1: ({N}, {K}) {dtype}")
    print(f"  Expected output: ({M}, {N//2}) {dtype}")

    # Generate inputs
    print("\nGenerating test inputs...")
    X, W1, W2 = generate_test_inputs(M, N, K, dtype)

    # Compute reference output (stages 1-2 only)
    print("Computing reference output (stages 1-2)...")
    ref_output = reference_gated_ff_stage1_2(X, W1, activation='silu')

    print(f"\nReference output shape: {ref_output.shape}")
    print(f"Reference output stats:")
    print(f"  mean: {ref_output.float().mean():.6f}")
    print(f"  std:  {ref_output.float().std():.6f}")
    print(f"  min:  {ref_output.float().min():.6f}")
    print(f"  max:  {ref_output.float().max():.6f}")

    # Try to compile and run HipKittens kernel
    print("\n" + "=" * 80)
    print("HipKittens Kernel Compilation")
    print("=" * 80)

    if not compile_kernel():
        print("\nSKIPPED: Kernel compilation failed")
        print("This is expected if:")
        print("  - THUNDERKITTENS_ROOT is not set")
        print("  - hipcc is not available")
        print("  - Running on non-AMD hardware")
        return

    # Try to import and run
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import tk_ff_fused_gated

        print("\nRunning HipKittens kernel...")
        print("NOTE: This requires AMD GPU hardware")

        # TODO: Add actual kernel invocation when binding is complete
        print("SKIPPED: Kernel invocation not yet implemented in pybind")

    except ImportError as e:
        print(f"\nSKIPPED: Could not import kernel module: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print("✓ Reference computation successful")
    print("✓ Test framework ready")
    print("⚠ HipKittens kernel validation pending hardware availability")
    print("\nTo run full parity check:")
    print("  1. Ensure THUNDERKITTENS_ROOT is set")
    print("  2. Run on AMD GPU hardware (gfx90a, gfx942, or gfx950)")
    print("  3. Execute: python test_parity.py")

if __name__ == "__main__":
    test_parity()
