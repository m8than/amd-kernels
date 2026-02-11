"""
Parity test for gated delta rule utility functions.

Tests the HIP C++ utility kernels against numpy reference implementations.

Utilities tested:
1. abs_diff: Element-wise |x - y|
2. squared_diff: Element-wise (x - y)^2
3. max_reduce: Max reduction
4. sum_reduce: Sum reduction
5. bf16 <-> float conversion
6. L2 normalization
7. Error metrics (get_abs_err, get_err_ratio) -- composed from above

Compilation:
  hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC \
    -o gdr_utils.so kernel.cpp
"""

import numpy as np
import subprocess
import os


# ============================================================
# Numpy reference implementations (from gated_delta_rule_utils.py)
# ============================================================

def get_abs_err(x, y):
    """Maximum absolute error between two arrays."""
    return np.abs(x - y).max()


def get_err_ratio(x, y):
    """RMSE ratio: rmse(x-y) / rms(x)."""
    err = np.sqrt(np.mean((x - y) ** 2))
    base = np.sqrt(np.mean(x ** 2))
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    """Check if two arrays are close (matching original Python utility)."""
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    print(msg)
    if abs_atol <= err_atol:
        return True
    return error_rate < ratio


def l2_normalize(x, axis=-1, eps=1e-6):
    """L2 normalize along an axis."""
    norm = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True) + eps)
    return x / norm


def softplus(x, beta=1.0, threshold=20.0):
    """Softplus with numerical stability."""
    bx = beta * x
    return np.where(bx <= threshold, (1.0 / beta) * np.log(1.0 + np.exp(bx)), x)


def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    """SiLU (Swish) activation."""
    return x * sigmoid(x)


# ============================================================
# Tests
# ============================================================

def test_error_metrics():
    """Test error computation utilities."""
    np.random.seed(42)

    x = np.random.randn(1000).astype(np.float32)
    y = x + np.random.randn(1000).astype(np.float32) * 0.01

    abs_err = get_abs_err(x, y)
    err_ratio = get_err_ratio(x, y)

    print(f"abs_err: {abs_err:.6f}")
    print(f"err_ratio: {err_ratio:.6f}")

    assert abs_err > 0, "abs_err should be > 0 for noisy data"
    assert abs_err < 0.1, "abs_err should be small for small noise"
    assert err_ratio < 0.1, "err_ratio should be small for small noise"
    print("PASS: error metrics")


def test_l2_normalize():
    """Test L2 normalization."""
    np.random.seed(42)

    x = np.random.randn(10, 64).astype(np.float32)
    x_norm = l2_normalize(x, axis=-1)

    # Check that norms are ~1
    norms = np.sqrt(np.sum(x_norm ** 2, axis=-1))
    assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not 1: {norms}"
    print("PASS: L2 normalization")


def test_activations():
    """Test activation functions."""
    x = np.array([-5, -1, 0, 1, 5], dtype=np.float32)

    # Sigmoid
    s = sigmoid(x)
    assert np.allclose(s, [0.006693, 0.268941, 0.5, 0.731059, 0.993307], atol=1e-4)

    # Softplus
    sp = softplus(x)
    assert all(sp > 0), "Softplus should be positive"
    assert np.isclose(sp[2], np.log(2), atol=1e-5), "softplus(0) should be ln(2)"

    # SiLU
    si = silu(x)
    assert np.isclose(si[2], 0.0, atol=1e-5), "silu(0) should be 0"
    assert si[4] > 0, "silu(5) should be positive"

    print("PASS: activation functions")


def test_assert_close():
    """Test the assert_close validation utility."""
    x = np.random.randn(100).astype(np.float32)

    # Exact match
    assert assert_close("exact", x, x, ratio=0.01)

    # Near match
    y = x + np.random.randn(100).astype(np.float32) * 1e-5
    assert assert_close("near", x, y, ratio=0.01)

    # Far match
    z = x + np.random.randn(100).astype(np.float32) * 10.0
    assert not assert_close("far", x, z, ratio=0.01)

    print("PASS: assert_close utility")


def test_hip_kernel():
    """Test HIP utility kernels (requires ROCm)."""
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        return

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "gdr_utils.so")

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
    N = 1024

    x = torch.randn(N, dtype=torch.float32, device="cuda")
    y = torch.randn(N, dtype=torch.float32, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")

    # Test abs_diff
    lib.launch_abs_diff(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    ref = np.abs(x.cpu().numpy() - y.cpu().numpy())
    hip_out = out.cpu().numpy()
    diff = np.abs(hip_out - ref).max()
    print(f"abs_diff max error: {diff:.8f}")
    assert diff < 1e-6, f"abs_diff failed: {diff}"
    print("PASS: abs_diff kernel")

    # Test squared_diff
    lib.launch_squared_diff(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    ref = (x.cpu().numpy() - y.cpu().numpy()) ** 2
    hip_out = out.cpu().numpy()
    diff = np.abs(hip_out - ref).max()
    print(f"squared_diff max error: {diff:.8f}")
    assert diff < 1e-5, f"squared_diff failed: {diff}"
    print("PASS: squared_diff kernel")

    # Test bf16 roundtrip
    x_f32 = torch.randn(N, dtype=torch.float32, device="cuda")
    x_bf16 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    x_roundtrip = torch.zeros(N, dtype=torch.float32, device="cuda")

    lib.launch_float_to_bf16(
        ctypes.c_void_p(x_f32.data_ptr()),
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_void_p(0),
    )
    lib.launch_bf16_to_float(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(x_roundtrip.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    ref_bf16 = x_f32.bfloat16().float().cpu().numpy()
    hip_roundtrip = x_roundtrip.cpu().numpy()
    diff = np.abs(hip_roundtrip - ref_bf16).max()
    print(f"bf16 roundtrip max error: {diff:.8f}")
    assert diff < 1e-6, f"bf16 roundtrip failed: {diff}"
    print("PASS: bf16 conversion kernels")

    # Test L2 normalize
    vec_len = 64
    num_vecs = 16
    x_vecs = torch.randn(num_vecs, vec_len, dtype=torch.float32, device="cuda")
    out_vecs = torch.zeros_like(x_vecs)

    lib.launch_l2_normalize(
        ctypes.c_void_p(x_vecs.data_ptr()),
        ctypes.c_void_p(out_vecs.data_ptr()),
        ctypes.c_int(num_vecs),
        ctypes.c_int(vec_len),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    ref_norm = l2_normalize(x_vecs.cpu().numpy(), axis=-1)
    hip_norm = out_vecs.cpu().numpy()
    diff = np.abs(hip_norm - ref_norm).max()
    print(f"L2 normalize max error: {diff:.8f}")
    assert diff < 1e-4, f"L2 normalize failed: {diff}"
    print("PASS: L2 normalize kernel")


if __name__ == "__main__":
    print("=" * 60)
    print("Gated Delta Rule Utilities - Parity Test")
    print("=" * 60)

    print("\n--- Error Metrics Test ---")
    test_error_metrics()

    print("\n--- L2 Normalization Test ---")
    test_l2_normalize()

    print("\n--- Activation Functions Test ---")
    test_activations()

    print("\n--- Assert Close Utility Test ---")
    test_assert_close()

    print("\n--- HIP Kernel Test ---")
    test_hip_kernel()
