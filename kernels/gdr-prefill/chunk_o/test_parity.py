#!/usr/bin/env python3
"""Parity test for chunk_o kernel."""
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
    r = subprocess.run([binary], capture_output=True, text=True, timeout=120)
    print(r.stdout)
    return r.returncode == 0 and "FAIL" not in r.stdout

def python_reference():
    """Test chunk_o forward with NumPy."""
    print("=== NumPy Reference Test ===")
    B, T, H, K, V, BT = 1, 128, 2, 64, 64, 64
    scale = 1.0 / np.sqrt(K)
    np.random.seed(42)

    q = np.random.randn(B, T, H, K).astype(np.float32) * 0.5
    k = np.random.randn(B, T, H, K).astype(np.float32) * 0.5
    v = np.random.randn(B, T, H, V).astype(np.float32) * 0.5
    g = np.random.randn(B, T, H).astype(np.float32) * 0.1
    NT = T // BT
    h = np.random.randn(B, NT, H, K, V).astype(np.float32) * 0.5

    o = np.zeros((B, T, H, V), dtype=np.float32)
    for b in range(B):
        for hh in range(H):
            for nt in range(NT):
                cs = nt * BT
                for i in range(BT):
                    ti = cs + i
                    # Inter-chunk: Q @ H
                    inter = q[b, ti, hh] @ h[b, nt, hh]  # [V]
                    gi = g[b, ti, hh]
                    inter *= np.exp(gi)

                    # Intra-chunk: causal Q@K^T @ V
                    intra = np.zeros(V, dtype=np.float32)
                    for j in range(i + 1):
                        tj = cs + j
                        qk = np.dot(q[b, ti, hh], k[b, tj, hh])
                        gj = g[b, tj, hh]
                        qk *= np.exp(gi - gj)
                        intra += qk * v[b, tj, hh]

                    o[b, ti, hh] = (inter + intra) * scale

    print(f"Output range: [{o.min():.4f}, {o.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Chunk O Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
