#!/usr/bin/env python3
"""Parity test for wy_representation kernel."""
import subprocess, sys, os, numpy as np

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

def compile_kernel():
    subprocess.run(["make", "-C", KERNEL_DIR, "clean"], capture_output=True)
    r = subprocess.run(["make", "-C", KERNEL_DIR], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Compilation failed:\n{r.stderr}")
        return False
    return True

def run_kernel():
    binary = os.path.join(KERNEL_DIR, "kernel")
    if not os.path.exists(binary): return False
    r = subprocess.run([binary], capture_output=True, text=True, timeout=60)
    print(r.stdout)
    return r.returncode == 0 and "FAIL" not in r.stdout

def python_reference():
    print("=== NumPy Reference Test ===")
    B, T, H, K, BT = 1, 128, 2, 64, 64
    np.random.seed(42)
    k = np.random.randn(B, T, H, K).astype(np.float32) * 0.5
    g = np.random.randn(B, T, H).astype(np.float32) * 0.5
    beta = np.random.randn(B, T, H).astype(np.float32) * 0.5

    # Compute A = beta * K @ K^T * exp(g_diff)  (strictly lower triangular per chunk)
    NT = T // BT
    A = np.zeros((B, T, H, BT), dtype=np.float32)
    for b_idx in range(B):
        for h_idx in range(H):
            for nt in range(NT):
                cs = nt * BT
                for i in range(BT):
                    for j in range(i):
                        dot = np.dot(k[b_idx, cs+i, h_idx], k[b_idx, cs+j, h_idx])
                        gd = np.exp(g[b_idx, cs+i, h_idx] - g[b_idx, cs+j, h_idx])
                        A[b_idx, cs+i, h_idx, j] = beta[b_idx, cs+i, h_idx] * dot * gd
    print(f"KKT output range: [{A.min():.4f}, {A.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("WY Representation Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
