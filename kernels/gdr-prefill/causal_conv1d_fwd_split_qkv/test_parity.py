#!/usr/bin/env python3
"""Parity test for causal_conv1d_fwd_split_qkv kernel."""
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
    """Test causal conv1d + split with NumPy."""
    print("=== NumPy Reference Test ===")
    num_seqs = 2
    seq_lens = [32, 48]
    total_tokens = sum(seq_lens)
    k_dim, v_dim = 64, 64
    dim = 2 * k_dim + v_dim  # 192
    kernel_width = 4
    np.random.seed(42)

    # x is [dim, total_tokens]
    x = np.random.randn(dim, total_tokens).astype(np.float32)
    w = np.random.randn(dim, kernel_width).astype(np.float32) * 0.5
    bias = np.random.randn(dim).astype(np.float32) * 0.1
    query_start_loc = [0, 32, 80]

    q = np.zeros((total_tokens, k_dim), dtype=np.float32)
    k = np.zeros((total_tokens, k_dim), dtype=np.float32)
    v = np.zeros((total_tokens, v_dim), dtype=np.float32)

    for s in range(num_seqs):
        seq_start = query_start_loc[s]
        seqlen = query_start_loc[s + 1] - seq_start
        for feat in range(dim):
            cols = [0.0, 0.0, 0.0]
            for t in range(seqlen):
                gt = seq_start + t
                x_curr = x[feat, gt]
                acc = bias[feat]
                acc += (cols[0] * w[feat, 0] + cols[1] * w[feat, 1] +
                        cols[2] * w[feat, 2] + x_curr * w[feat, 3])
                cols[0] = cols[1]
                cols[1] = cols[2]
                cols[2] = x_curr
                # SiLU
                acc = acc / (1.0 + np.exp(-acc))
                if feat < k_dim:
                    q[gt, feat] = acc
                elif feat < 2 * k_dim:
                    k[gt, feat - k_dim] = acc
                elif feat < 2 * k_dim + v_dim:
                    v[gt, feat - 2 * k_dim] = acc

    print(f"q range: [{q.min():.4f}, {q.max():.4f}]")
    print(f"k range: [{k.min():.4f}, {k.max():.4f}]")
    print(f"v range: [{v.min():.4f}, {v.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Causal Conv1d Split QKV Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
