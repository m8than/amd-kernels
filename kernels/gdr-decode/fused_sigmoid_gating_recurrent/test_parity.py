"""
Parity test for fused_sigmoid_gating_delta_rule_update kernel.

Tests the HIP C++ kernel against a numpy reference implementation.

Input shapes:
  A_log:   (HV,) bf16 -- log-scale gating parameter
  a:       (B, T, HV) bf16 -- gating input
  dt_bias: (HV,) bf16 -- gating bias
  q, k:    (B, T, H, K) bf16
  v:       (B, T, HV, V) bf16
  b:       (B, T, HV) bf16 -- pre-sigmoid beta input
  h0_source: (num_states, HV, K, V) float32 -- state bank
  h0_indices: (B,) int32 -- batch-to-state mapping

Output shapes:
  o:       (B, T, HV, V) bf16
  h0_source updated in-place with final states

Compilation:
  hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC \
    -o fused_sigmoid_gating_recurrent.so kernel.cpp
"""

import numpy as np
import subprocess
import os


def softplus(x, beta=1.0, threshold=20.0):
    """Numpy softplus with numerical stability."""
    bx = beta * x
    return np.where(bx <= threshold, (1.0 / beta) * np.log(1.0 + np.exp(bx)), x)


def sigmoid(x):
    """Numpy sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def fused_sigmoid_gating_delta_rule_update_ref(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b_input, scale,
    h0_source=None, h0_indices=None,
    use_qk_l2norm=False,
):
    """
    Numpy reference for fused sigmoid gating delta rule update.

    Args:
        A_log: (HV,) float32
        a: (B, T, HV) float32
        dt_bias: (HV,) float32
        softplus_beta: float
        softplus_threshold: float
        q: (B, T, H, K) float32
        k: (B, T, H, K) float32
        v: (B, T, HV, V) float32
        b_input: (B, T, HV) float32  -- pre-sigmoid
        scale: float
        h0_source: (num_states, HV, K, V) float32 or None
        h0_indices: (B,) int32 or None
        use_qk_l2norm: bool

    Returns:
        o: (B, T, HV, V) float32
        h0_source: updated in-place
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    o = np.zeros((B, T, HV, V), dtype=np.float32)

    for bn in range(B):
        for hv in range(HV):
            h_qk = hv // (HV // H)

            # Initialize hidden state
            h = np.zeros((K, V), dtype=np.float32)
            state_idx = -1
            if h0_source is not None and h0_indices is not None:
                state_idx = h0_indices[bn]
                if state_idx >= 0:
                    h = h0_source[state_idx, hv].copy()

            b_A_log = A_log[hv]
            b_dt_bias = dt_bias[hv]

            for t in range(T):
                b_q = q[bn, t, h_qk].copy()  # (K,)
                b_k = k[bn, t, h_qk].copy()  # (K,)
                b_v = v[bn, t, hv].copy()     # (V,)
                b_b = b_input[bn, t, hv]      # scalar
                b_a = a[bn, t, hv]            # scalar

                # L2 norm (optional)
                if use_qk_l2norm:
                    b_q = b_q / (np.sqrt(np.sum(b_q * b_q) + 1e-6))
                    b_k = b_k / (np.sqrt(np.sum(b_k * b_k) + 1e-6))

                b_q *= scale

                # Compute sigmoid gating
                sp = softplus(b_a + b_dt_bias, softplus_beta, softplus_threshold)
                b_g = -np.exp(b_A_log) * sp
                b_beta = sigmoid(b_b)

                # Apply gate
                h *= np.exp(b_g)

                # Delta rule
                hk_sum = np.sum(h * b_k[:, None], axis=0)  # (V,)
                b_v_prime = b_beta * (b_v - hk_sum)
                h += b_k[:, None] * b_v_prime[None, :]

                # Output
                o[bn, t, hv] = np.sum(h * b_q[:, None], axis=0)

            # Store final state
            if h0_source is not None and state_idx >= 0:
                h0_source[state_idx, hv] = h

    return o


def test_numpy_reference():
    """Test the numpy reference implementation for correctness."""
    np.random.seed(42)

    B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
    scale = K ** -0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    q = np.random.randn(B, T, H, K).astype(np.float32) * 0.1
    k = np.random.randn(B, T, H, K).astype(np.float32) * 0.1
    v = np.random.randn(B, T, HV, V).astype(np.float32) * 0.1
    b_input = np.random.randn(B, T, HV).astype(np.float32)
    a = np.random.randn(B, T, HV).astype(np.float32) * 0.1
    A_log = np.random.randn(HV).astype(np.float32) * 0.5
    dt_bias = np.random.randn(HV).astype(np.float32) * 0.1

    num_states = B
    h0_source = np.random.randn(num_states, HV, K, V).astype(np.float32) * 0.01
    h0_indices = np.arange(B, dtype=np.int32)

    o = fused_sigmoid_gating_delta_rule_update_ref(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b_input, scale,
        h0_source=h0_source, h0_indices=h0_indices,
    )

    print(f"Output shape: {o.shape}")
    print(f"Output range: [{o.min():.6f}, {o.max():.6f}]")
    print(f"Output mean abs: {np.abs(o).mean():.6f}")
    assert np.abs(o).mean() > 1e-8, "Output is all zeros!"
    print("PASS: numpy reference sanity check")


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
    so_path = os.path.join(kernel_dir, "fused_sigmoid_gating_recurrent.so")

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
    B, T, H, HV, K, V = 2, 4, 2, 4, 64, 32
    scale = K ** -0.5

    # Create tensors
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k_t = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v_t = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    b_t = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    a_t = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda") * 0.1
    A_log = torch.randn(HV, dtype=torch.bfloat16, device="cuda") * 0.5
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda") * 0.1

    num_states = B
    h0_source = torch.randn(num_states, HV, K, V, dtype=torch.float32, device="cuda") * 0.01
    h0_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    o = torch.zeros(1, B * T, HV, V, dtype=torch.bfloat16, device="cuda")

    # Get reference
    o_ref = fused_sigmoid_gating_delta_rule_update_ref(
        A_log.float().cpu().numpy(), a_t.float().cpu().numpy(),
        dt_bias.float().cpu().numpy(), 1.0, 20.0,
        q.float().cpu().numpy(), k_t.float().cpu().numpy(),
        v_t.float().cpu().numpy(), b_t.float().cpu().numpy(), scale,
        h0_source=h0_source.cpu().numpy().copy(),
        h0_indices=h0_indices.cpu().numpy(),
    )

    # Launch kernel
    lib.launch_fused_sigmoid_gating_delta_rule_update(
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(a_t.contiguous().data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_float(1.0), ctypes.c_float(20.0),
        ctypes.c_void_p(q.contiguous().data_ptr()),
        ctypes.c_void_p(k_t.contiguous().data_ptr()),
        ctypes.c_void_p(v_t.contiguous().data_ptr()),
        ctypes.c_void_p(b_t.contiguous().data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(h0_source.data_ptr()),
        ctypes.c_void_p(h0_indices.data_ptr()),
        ctypes.c_void_p(0),  # cu_seqlens = nullptr
        ctypes.c_float(scale),
        ctypes.c_int(T), ctypes.c_int(B),
        ctypes.c_int(H), ctypes.c_int(HV),
        ctypes.c_int(K), ctypes.c_int(V),
        ctypes.c_bool(True),   # use_initial_state
        ctypes.c_bool(False),  # use_qk_l2norm
        ctypes.c_void_p(0),    # stream
    )
    torch.cuda.synchronize()

    o_hip = o.squeeze(0).reshape(B, T, HV, V).float().cpu().numpy()
    o_diff = np.abs(o_hip - o_ref).max()
    print(f"Output max diff: {o_diff:.6f}")

    atol = 0.05
    if o_diff < atol:
        print("PASS: HIP kernel matches numpy reference")
    else:
        print("FAIL: HIP kernel output differs from reference")


if __name__ == "__main__":
    print("=" * 60)
    print("Fused Sigmoid Gating Delta Rule Update - Parity Test")
    print("=" * 60)

    print("\n--- Numpy Reference Test ---")
    test_numpy_reference()

    print("\n--- HIP Kernel Test ---")
    test_hip_kernel()
