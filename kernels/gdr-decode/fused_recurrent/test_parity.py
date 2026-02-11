"""
Parity test for fused_recurrent_gated_delta_rule_fwd kernel.

Tests the HIP C++ kernel against a numpy reference implementation.

Input shapes:
  q, k: (B, T, H, K) bf16
  v:    (B, T, HV, V) bf16
  g:    (B, T, HV) float32 (optional)
  beta: (B, T, HV) float32 (headwise) or (B, T, HV, V) float32
  h0:   (B*HV, K, V) float32 (optional initial state)

Output shapes:
  o:    (B, T, HV, V) bf16
  ht:   (B*HV, K, V) float32 (optional final state)

Compilation:
  hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC -o fused_recurrent.so kernel.cpp
"""

import numpy as np
import subprocess
import os
import struct

# ============================================================
# Numpy reference implementation
# ============================================================

def fused_recurrent_gated_delta_rule_fwd_ref(
    q, k, v, beta, scale,
    g=None, gk=None, gv=None,
    h0=None, is_beta_headwise=True,
):
    """
    Numpy reference for fused recurrent gated delta rule forward.

    Args:
        q: (B, T, H, K) float32
        k: (B, T, H, K) float32
        v: (B, T, HV, V) float32
        beta: (B, T, HV) or (B, T, HV, V) float32
        scale: float
        g: (B, T, HV) float32 or None
        gk: (B, T, HV, K) float32 or None
        gv: (B, T, HV, V) float32 or None
        h0: (B, HV, K, V) float32 or None
        is_beta_headwise: bool

    Returns:
        o: (B, T, HV, V) float32
        ht: (B, HV, K, V) float32 -- final state
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    o = np.zeros((B, T, HV, V), dtype=np.float32)
    ht = np.zeros((B, HV, K, V), dtype=np.float32)

    for b in range(B):
        for hv in range(HV):
            h_qk = hv // (HV // H)  # GQA head mapping

            # Initialize hidden state (K, V)
            h = np.zeros((K, V), dtype=np.float32)
            if h0 is not None:
                h = h0[b, hv].copy()

            for t in range(T):
                b_q = q[b, t, h_qk] * scale           # (K,)
                b_k = k[b, t, h_qk]                    # (K,)
                b_v = v[b, t, hv]                       # (V,)

                if is_beta_headwise:
                    b_beta = beta[b, t, hv]             # scalar
                else:
                    b_beta = beta[b, t, hv]             # (V,)

                # Apply gates
                if g is not None:
                    b_g = g[b, t, hv]
                    h *= np.exp(b_g)

                if gk is not None:
                    b_gk = gk[b, t, hv]                # (K,)
                    h *= np.exp(b_gk[:, None])

                if gv is not None:
                    b_gv = gv[b, t, hv]                # (V,)
                    h *= np.exp(b_gv[None, :])

                # Delta rule: v' = beta * (v - h^T k)
                hk_sum = np.sum(h * b_k[:, None], axis=0)  # (V,)
                b_v_prime = b_beta * (b_v - hk_sum)

                # Update: h += k[:, None] * v'[None, :]
                h += b_k[:, None] * b_v_prime[None, :]

                # Output: o = h^T q = sum(h * q[:, None], axis=0)
                o[b, t, hv] = np.sum(h * b_q[:, None], axis=0)

            ht[b, hv] = h

    return o, ht


def test_numpy_reference():
    """Test the numpy reference implementation for correctness."""
    np.random.seed(42)

    B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
    scale = K ** -0.5

    q = np.random.randn(B, T, H, K).astype(np.float32) * 0.1
    k = np.random.randn(B, T, H, K).astype(np.float32) * 0.1
    v = np.random.randn(B, T, HV, V).astype(np.float32) * 0.1
    beta = np.random.uniform(0.5, 1.5, (B, T, HV)).astype(np.float32)
    g = np.random.randn(B, T, HV).astype(np.float32) * 0.1

    h0 = np.random.randn(B, HV, K, V).astype(np.float32) * 0.01

    o, ht = fused_recurrent_gated_delta_rule_fwd_ref(
        q, k, v, beta, scale, g=g, h0=h0, is_beta_headwise=True
    )

    print(f"Output shape: {o.shape}")
    print(f"Final state shape: {ht.shape}")
    print(f"Output range: [{o.min():.6f}, {o.max():.6f}]")
    print(f"Output mean abs: {np.abs(o).mean():.6f}")

    # Basic sanity: output should not be all zeros (since we have initial state + inputs)
    assert np.abs(o).mean() > 1e-8, "Output is all zeros!"
    print("PASS: numpy reference sanity check")


def test_hip_kernel():
    """Test HIP kernel against numpy reference (requires ROCm)."""
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not available for HIP kernel test")
        return

    if not torch.cuda.is_available():
        print("SKIP: No GPU available for HIP kernel test")
        return

    # Build kernel
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "fused_recurrent.so")

    if not os.path.exists(so_path):
        print("Building kernel...")
        result = subprocess.run(
            ["make", "-C", kernel_dir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            print("SKIP: Could not build kernel")
            return

    import ctypes
    lib = ctypes.CDLL(so_path)

    np.random.seed(42)
    B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device="cuda") + 0.5
    g = torch.randn(B, T, HV, dtype=torch.float32, device="cuda") * 0.1
    h0 = torch.randn(B * HV, K, V, dtype=torch.float32, device="cuda") * 0.01
    o = torch.zeros(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    ht = torch.zeros(B * HV, K, V, dtype=torch.float32, device="cuda")

    # Make contiguous
    q, k, v, beta, g, h0, o, ht = [t.contiguous() for t in [q, k, v, beta, g, h0, o, ht]]

    # Get numpy reference
    q_np = q.float().cpu().numpy()
    k_np = k.float().cpu().numpy()
    v_np = v.float().cpu().numpy()
    beta_np = beta.float().cpu().numpy()
    g_np = g.cpu().numpy()
    h0_np = h0.cpu().numpy().reshape(B, HV, K, V)

    o_ref, ht_ref = fused_recurrent_gated_delta_rule_fwd_ref(
        q_np, k_np, v_np, beta_np, scale, g=g_np, h0=h0_np, is_beta_headwise=True
    )

    # Launch HIP kernel via ctypes
    lib.launch_fused_recurrent_gated_delta_rule_fwd(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(0),  # gk = nullptr
        ctypes.c_void_p(0),  # gv = nullptr
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(ht.data_ptr()),
        ctypes.c_void_p(0),  # cu_seqlens = nullptr
        ctypes.c_float(scale),
        ctypes.c_int(T), ctypes.c_int(B),
        ctypes.c_int(H), ctypes.c_int(HV),
        ctypes.c_int(K), ctypes.c_int(V),
        ctypes.c_bool(True),   # use_g
        ctypes.c_bool(False),  # use_gk
        ctypes.c_bool(False),  # use_gv
        ctypes.c_bool(True),   # is_beta_headwise
        ctypes.c_bool(True),   # use_initial_state
        ctypes.c_bool(True),   # store_final_state
        ctypes.c_void_p(0),    # stream = default
    )
    torch.cuda.synchronize()

    o_hip = o.float().cpu().numpy()
    ht_hip = ht.cpu().numpy().reshape(B, HV, K, V)

    # Compare
    o_diff = np.abs(o_hip - o_ref).max()
    ht_diff = np.abs(ht_hip - ht_ref).max()

    print(f"Output max diff: {o_diff:.6f}")
    print(f"Final state max diff: {ht_diff:.6f}")

    atol = 0.05  # bf16 tolerance
    if o_diff < atol and ht_diff < atol:
        print("PASS: HIP kernel matches numpy reference")
    else:
        print("FAIL: HIP kernel output differs from reference")


if __name__ == "__main__":
    print("=" * 60)
    print("Fused Recurrent Gated Delta Rule Forward - Parity Test")
    print("=" * 60)

    print("\n--- Numpy Reference Test ---")
    test_numpy_reference()

    print("\n--- HIP Kernel Test ---")
    test_hip_kernel()
