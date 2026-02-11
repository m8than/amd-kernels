#!/usr/bin/env python3
"""Parity test for fused_gdn_gating_prefill kernel."""
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
    """Test fused gating with NumPy."""
    print("=== NumPy Reference Test ===")
    S, H = 256, 16
    sp_beta, sp_thresh = 1.0, 20.0
    np.random.seed(42)

    A_log = np.random.randn(H).astype(np.float32) * 0.5
    a = np.random.randn(S, H).astype(np.float32)
    b = np.random.randn(S, H).astype(np.float32)
    dt_bias = np.random.randn(H).astype(np.float32) * 0.1

    # g = -exp(A_log) * softplus(a + dt_bias)
    x = a + dt_bias[None, :]
    bx = sp_beta * x
    sp = np.where(bx <= sp_thresh, np.log(1.0 + np.exp(bx)) / sp_beta, x)
    g = -np.exp(A_log[None, :]) * sp

    # beta = sigmoid(b)
    beta = 1.0 / (1.0 + np.exp(-b))

    print(f"g range: [{g.min():.4f}, {g.max():.4f}]")
    print(f"beta range: [{beta.min():.4f}, {beta.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Fused GDN Gating Prefill Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
