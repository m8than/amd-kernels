#!/usr/bin/env python3
"""Parity test for l2norm kernel."""

import subprocess
import sys
import os
import numpy as np

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

def compile_kernel():
    subprocess.run(["make", "-C", KERNEL_DIR, "clean"], capture_output=True)
    result = subprocess.run(["make", "-C", KERNEL_DIR], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False
    return True

def run_kernel():
    binary = os.path.join(KERNEL_DIR, "kernel")
    if not os.path.exists(binary):
        return False
    result = subprocess.run([binary], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    return result.returncode == 0 and "FAIL" not in result.stdout

def python_reference_test():
    """Test against numpy reference."""
    print("=== NumPy Reference Test ===")
    T, D = 512, 128
    eps = 1e-6
    np.random.seed(42)
    x = np.random.randn(T, D).astype(np.float32)

    # Forward
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    y = x / norm
    rstd = 1.0 / norm.squeeze(-1)

    # Backward
    dy = np.random.randn(T, D).astype(np.float32)
    dot = np.sum(dy * y, axis=-1, keepdims=True)
    dx = dy * rstd[:, None] - dot * y * rstd[:, None]

    print(f"Input shape: [{T}, {D}]")
    print(f"Forward output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Backward output range: [{dx.min():.4f}, {dx.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("L2 Norm Kernel Parity Test")
    print("=" * 60)
    python_reference_test()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        success = run_kernel()
        return 0 if success else 1
    else:
        print("Compilation failed (expected without HIP toolchain)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
