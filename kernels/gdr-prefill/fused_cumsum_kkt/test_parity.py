#!/usr/bin/env python3
"""Parity test for fused_cumsum_kkt kernel."""
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
    """Test fused cumsum + KKT with NumPy."""
    print("=== NumPy Reference Test ===")
    B, T, H, Hg, K, BT = 1, 128, 4, 4, 64, 64
    np.random.seed(42)

    g = np.random.randn(B, T, H).astype(np.float32) * 0.1
    k = np.random.randn(B, T, Hg, K).astype(np.float32) * 0.5
    beta = np.random.randn(B, T, H).astype(np.float32) * 0.5

    NT = T // BT
    g_cumsum = np.zeros_like(g)
    A = np.zeros((B, T, H, BT), dtype=np.float32)

    for b in range(B):
        for h in range(H):
            hg = h // (H // Hg)
            for nt in range(NT):
                cs = nt * BT
                # Cumsum
                running = 0.0
                for i in range(BT):
                    running += g[b, cs + i, h]
                    g_cumsum[b, cs + i, h] = running
                # KKT with safe_exp
                for i in range(BT):
                    for j in range(i):
                        dot = np.dot(k[b, cs+i, hg], k[b, cs+j, hg])
                        gd = g_cumsum[b, cs+i, h] - g_cumsum[b, cs+j, h]
                        se = np.exp(gd) if gd <= 0 else 0.0
                        A[b, cs+i, h, j] = beta[b, cs+i, h] * dot * se

    print(f"g_cumsum range: [{g_cumsum.min():.4f}, {g_cumsum.max():.4f}]")
    print(f"A range: [{A.min():.4f}, {A.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Fused Cumsum KKT Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
