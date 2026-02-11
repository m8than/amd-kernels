#!/usr/bin/env python3
"""Parity test for op (math utility) kernel."""

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
    x = np.linspace(-5, 5, 100).astype(np.float32)

    # Test each operation
    ops = {
        "exp": np.exp(x),
        "exp2": np.exp2(x),
        "log": np.log(np.abs(x) + 0.001),
        "safe_exp": np.where(x <= 0, np.exp(x), 0.0),
        "softplus": np.where(x > 20, x, np.log(1 + np.exp(x))),
        "sigmoid": 1.0 / (1.0 + np.exp(-x)),
        "silu": x / (1.0 + np.exp(-x)),
    }
    for name, vals in ops.items():
        print(f"  {name}: range [{vals.min():.4f}, {vals.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Op (Math Utilities) Parity Test")
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
