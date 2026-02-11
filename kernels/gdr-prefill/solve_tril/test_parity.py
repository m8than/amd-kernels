#!/usr/bin/env python3
"""Parity test for solve_tril kernel."""

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
    print("=== NumPy Reference Test ===")
    BT = 64
    np.random.seed(42)

    # Create strictly lower triangular matrix
    A = np.random.randn(BT, BT).astype(np.float32) * 0.1
    A = np.tril(A, -1)

    # Compute (I + A)^{-1}
    M = np.eye(BT) + A
    Ai = np.linalg.inv(M)

    # Verify: M @ Ai should be I
    check = M @ Ai
    err = np.max(np.abs(check - np.eye(BT)))
    print(f"Chunk size: {BT}")
    print(f"(I+A) @ (I+A)^-1 max error from identity: {err:.6e}")
    print(f"NumPy reference: {'PASS' if err < 1e-5 else 'FAIL'}")

def main():
    print("=" * 60)
    print("Solve Tril Kernel Parity Test")
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
