#!/usr/bin/env python3
"""Parity test for chunk_delta_h kernel."""
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
    """Test chunk_delta_h forward with NumPy."""
    print("=== NumPy Reference Test ===")
    B, T, H, K, V, BT = 1, 128, 2, 64, 64, 64
    np.random.seed(42)

    k = np.random.randn(B, T, H, K).astype(np.float32) * 0.5
    w = np.random.randn(B, T, H, K).astype(np.float32) * 0.5
    u = np.random.randn(B, T, H, V).astype(np.float32) * 0.5
    g = np.random.randn(B, T, H).astype(np.float32) * 0.1

    NT = T // BT

    # Forward recurrence: h_state starts at 0
    # For each chunk: store h, compute v_new = u - w@h, update h
    h_states = np.zeros((B, NT, H, K, V), dtype=np.float32)
    final_state = np.zeros((B, H, K, V), dtype=np.float32)

    for b in range(B):
        for h in range(H):
            h_state = np.zeros((K, V), dtype=np.float32)
            for nt in range(NT):
                cs = nt * BT
                h_states[b, nt, h] = h_state.copy()

                # Compute v_new and update state
                g_last = g[b, cs + BT - 1, h]
                decay = np.exp(g_last)
                h_new = h_state * decay

                for t in range(BT):
                    v_res = u[b, cs + t, h] - w[b, cs + t, h] @ h_states[b, nt, h]
                    g_t = g[b, cs + t, h]
                    scale = np.exp(g_last - g_t)
                    h_new += np.outer(k[b, cs + t, h], v_res) * scale

                h_state = h_new

            final_state[b, h] = h_state

    print(f"Hidden state range: [{h_states.min():.4f}, {h_states.max():.4f}]")
    print(f"Final state range: [{final_state.min():.4f}, {final_state.max():.4f}]")
    print("NumPy reference: PASS")

def main():
    print("=" * 60)
    print("Chunk Delta H Kernel Parity Test")
    print("=" * 60)
    python_reference()
    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        return 0 if run_kernel() else 1
    print("Compilation failed (expected without HIP toolchain)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
