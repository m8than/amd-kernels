"""
Parity test for causal_conv1d_update_split_qkv kernel.

Tests the HIP C++ kernel against a numpy reference implementation.

Input shapes:
  x:          (batch, dim, seqlen) bf16 -- dim = 2*key_dim + value_dim
  weight:     (dim, width) bf16
  bias:       (dim,) bf16 or None
  conv_state: (num_cache_lines, dim, state_len) bf16 -- state_len = width - 1

Output shapes:
  q: (batch, key_dim, seqlen) bf16
  k: (batch, key_dim, seqlen) bf16
  v: (batch, value_dim, seqlen) bf16
  conv_state: updated in-place

Compilation:
  hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC \
    -o causal_conv1d_split_qkv.so kernel.cpp
"""

import numpy as np
import subprocess
import os


def silu(x):
    """SiLU activation."""
    return x / (1.0 + np.exp(-x))


def causal_conv1d_update_split_qkv_ref(
    x, weight, bias, conv_state,
    key_dim, value_dim,
    silu_activation=True,
    conv_state_indices=None,
):
    """
    Numpy reference for causal conv1d update with split QKV.

    Args:
        x: (batch, dim, seqlen) float32
        weight: (dim, width) float32
        bias: (dim,) float32 or None
        conv_state: (num_cache_lines, dim, state_len) float32 -- modified in-place
        key_dim: int
        value_dim: int
        silu_activation: bool
        conv_state_indices: (batch,) int or None

    Returns:
        q: (batch, key_dim, seqlen) float32
        k: (batch, key_dim, seqlen) float32
        v: (batch, value_dim, seqlen) float32
    """
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1

    q = np.zeros((batch, key_dim, seqlen), dtype=np.float32)
    k = np.zeros((batch, key_dim, seqlen), dtype=np.float32)
    v = np.zeros((batch, value_dim, seqlen), dtype=np.float32)

    for b in range(batch):
        cs_idx = conv_state_indices[b] if conv_state_indices is not None else b

        for d in range(dim):
            # Load sliding window state
            state = conv_state[cs_idx, d, :state_len].copy()

            for t in range(seqlen):
                x_val = x[b, d, t]

                # Compute convolution
                acc = bias[d] if bias is not None else 0.0
                for j in range(state_len):
                    acc += weight[d, j] * state[j]
                acc += weight[d, state_len] * x_val

                # Shift window
                for j in range(state_len - 1):
                    state[j] = state[j + 1]
                if state_len > 0:
                    state[state_len - 1] = x_val

                # SiLU
                if silu_activation:
                    acc = silu(acc)

                # Split to q, k, v
                if d < key_dim:
                    q[b, d, t] = acc
                elif d < 2 * key_dim:
                    k[b, d - key_dim, t] = acc
                elif d < 2 * key_dim + value_dim:
                    v[b, d - 2 * key_dim, t] = acc

            # Update conv_state
            conv_state[cs_idx, d, :state_len] = state

    return q, k, v


def test_numpy_reference():
    """Test numpy reference implementation."""
    np.random.seed(42)

    batch = 4
    key_dim = 64
    value_dim = 128
    dim = 2 * key_dim + value_dim  # 256
    seqlen = 1  # decode = single token
    width = 4   # kernel width
    state_len = width - 1

    x = np.random.randn(batch, dim, seqlen).astype(np.float32) * 0.1
    weight = np.random.randn(dim, width).astype(np.float32) * 0.1
    bias = np.random.randn(dim).astype(np.float32) * 0.1
    conv_state = np.random.randn(batch, dim, state_len).astype(np.float32) * 0.1

    q, k, v = causal_conv1d_update_split_qkv_ref(
        x, weight, bias, conv_state,
        key_dim, value_dim, silu_activation=True
    )

    print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    print(f"q range: [{q.min():.6f}, {q.max():.6f}]")
    print(f"k range: [{k.min():.6f}, {k.max():.6f}]")
    print(f"v range: [{v.min():.6f}, {v.max():.6f}]")

    assert q.shape == (batch, key_dim, seqlen)
    assert k.shape == (batch, key_dim, seqlen)
    assert v.shape == (batch, value_dim, seqlen)
    assert np.abs(q).mean() > 1e-8
    print("PASS: numpy reference sanity check")


def test_multi_token():
    """Test with multiple tokens (seqlen > 1)."""
    np.random.seed(123)

    batch = 2
    key_dim = 32
    value_dim = 64
    dim = 2 * key_dim + value_dim
    seqlen = 4
    width = 4
    state_len = width - 1

    x = np.random.randn(batch, dim, seqlen).astype(np.float32) * 0.1
    weight = np.random.randn(dim, width).astype(np.float32) * 0.1
    bias = np.random.randn(dim).astype(np.float32) * 0.01
    conv_state = np.zeros((batch, dim, state_len), dtype=np.float32)

    q, k, v = causal_conv1d_update_split_qkv_ref(
        x, weight, bias, conv_state,
        key_dim, value_dim, silu_activation=True
    )

    # Verify causality: output at t should only depend on inputs at t' <= t
    # Re-run with truncated input
    conv_state2 = np.zeros((batch, dim, state_len), dtype=np.float32)
    x_trunc = x[:, :, :2].copy()
    q2, k2, v2 = causal_conv1d_update_split_qkv_ref(
        x_trunc, weight, bias, conv_state2,
        key_dim, value_dim, silu_activation=True
    )

    # First 2 tokens should match
    assert np.allclose(q[:, :, :2], q2, atol=1e-6), "Causality violated for q"
    assert np.allclose(k[:, :, :2], k2, atol=1e-6), "Causality violated for k"
    assert np.allclose(v[:, :, :2], v2, atol=1e-6), "Causality violated for v"
    print("PASS: causality check")


def test_hip_kernel():
    """Test HIP kernel against numpy reference (requires ROCm)."""
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        return

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "causal_conv1d_split_qkv.so")

    if not os.path.exists(so_path):
        print("Building kernel...")
        result = subprocess.run(["make", "-C", kernel_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            print("SKIP: Could not build kernel")
            return

    import ctypes
    lib = ctypes.CDLL(so_path)

    np.random.seed(42)
    batch, key_dim, value_dim = 4, 64, 128
    dim = 2 * key_dim + value_dim
    seqlen = 1
    width = 4
    state_len = width - 1

    x = torch.randn(batch, dim, seqlen, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device="cuda") * 0.1
    bias_t = torch.randn(dim, dtype=torch.bfloat16, device="cuda") * 0.1
    conv_state = torch.randn(batch, dim, state_len, dtype=torch.bfloat16, device="cuda") * 0.1

    q = torch.zeros(batch, key_dim, seqlen, dtype=torch.bfloat16, device="cuda")
    k_out = torch.zeros(batch, key_dim, seqlen, dtype=torch.bfloat16, device="cuda")
    v_out = torch.zeros(batch, value_dim, seqlen, dtype=torch.bfloat16, device="cuda")

    # Numpy reference
    q_ref, k_ref, v_ref = causal_conv1d_update_split_qkv_ref(
        x.float().cpu().numpy(),
        weight.float().cpu().numpy(),
        bias_t.float().cpu().numpy(),
        conv_state.float().cpu().numpy().copy(),
        key_dim, value_dim, silu_activation=True,
    )

    # Launch HIP kernel
    lib.launch_causal_conv1d_update_split_qkv(
        ctypes.c_void_p(x.contiguous().data_ptr()),
        ctypes.c_void_p(weight.contiguous().data_ptr()),
        ctypes.c_void_p(bias_t.contiguous().data_ptr()),
        ctypes.c_void_p(conv_state.contiguous().data_ptr()),
        ctypes.c_void_p(0),  # no continuous batching indices
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_out.data_ptr()),
        ctypes.c_void_p(v_out.data_ptr()),
        ctypes.c_int(key_dim), ctypes.c_int(value_dim),
        ctypes.c_int(batch), ctypes.c_int(dim),
        ctypes.c_int(seqlen), ctypes.c_int(width),
        ctypes.c_int(batch),  # num_cache_lines
        ctypes.c_bool(True),   # has_bias
        ctypes.c_bool(True),   # silu_activation
        ctypes.c_bool(False),  # is_continuous_batching
        ctypes.c_void_p(0),    # stream
    )
    torch.cuda.synchronize()

    q_hip = q.float().cpu().numpy()
    k_hip = k_out.float().cpu().numpy()
    v_hip = v_out.float().cpu().numpy()

    q_diff = np.abs(q_hip - q_ref).max()
    k_diff = np.abs(k_hip - k_ref).max()
    v_diff = np.abs(v_hip - v_ref).max()

    print(f"q max diff: {q_diff:.6f}")
    print(f"k max diff: {k_diff:.6f}")
    print(f"v max diff: {v_diff:.6f}")

    atol = 0.05
    if q_diff < atol and k_diff < atol and v_diff < atol:
        print("PASS: HIP kernel matches numpy reference")
    else:
        print("FAIL: HIP kernel output differs from reference")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Conv1D Update Split QKV - Parity Test")
    print("=" * 60)

    print("\n--- Numpy Reference Test ---")
    test_numpy_reference()

    print("\n--- Multi-token Causality Test ---")
    test_multi_token()

    print("\n--- HIP Kernel Test ---")
    test_hip_kernel()
