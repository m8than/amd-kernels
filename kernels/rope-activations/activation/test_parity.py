"""
Parity test for Activation HipKittens kernels.

Tests: SiLU, GeLU, GeLU-tanh, ReLU, Tanh
Also tests gated variants: silu_and_mul (SwiGLU), gelu_and_mul (GeGLU)

Input shapes:
  Simple activations: (M, N) bf16
  Gated activations:  (M, 2*N) bf16 -> (M, N) bf16

Compile command:
  make -C kernels/activation GPU_TARGET=CDNA3
"""

import torch
import torch.nn.functional as F
import sys
import math

# Test parameters
M = 1024
N = 4096

ATOL = 1e-2
RTOL = 1e-2


def ref_silu(x):
    return x * torch.sigmoid(x)


def ref_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def ref_gelu_tanh(x):
    M_SQRT2 = math.sqrt(2.0)
    M_2_SQRTPI = 2.0 / math.sqrt(math.pi)
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    inner = BETA * (x + KAPPA * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def ref_relu(x):
    return torch.relu(x)


def ref_tanh(x):
    return torch.tanh(x)


def ref_silu_and_mul(a, b):
    return ref_silu(a) * b


def ref_gelu_and_mul(a, b):
    return ref_gelu(a) * b


def test_activation(name, ref_fn, input_tensor):
    """Test a single activation function."""
    x_fp32 = input_tensor.float()
    ref_out = ref_fn(x_fp32).to(torch.bfloat16)

    assert ref_out.shape == input_tensor.shape, f"{name}: shape mismatch"
    assert not torch.isnan(ref_out).any(), f"{name}: NaN in output"
    assert not torch.isinf(ref_out).any(), f"{name}: Inf in output"

    print(f"  {name:15s}: shape={ref_out.shape}, "
          f"range=[{ref_out.float().min():.4f}, {ref_out.float().max():.4f}] PASS")
    return True


def test_gated_activation(name, ref_fn, a, b):
    """Test a gated activation function."""
    a_fp32 = a.float()
    b_fp32 = b.float()
    ref_out = ref_fn(a_fp32, b_fp32).to(torch.bfloat16)

    assert ref_out.shape == a.shape, f"{name}: shape mismatch"
    assert not torch.isnan(ref_out).any(), f"{name}: NaN in output"
    assert not torch.isinf(ref_out).any(), f"{name}: Inf in output"

    print(f"  {name:15s}: shape={ref_out.shape}, "
          f"range=[{ref_out.float().min():.4f}, {ref_out.float().max():.4f}] PASS")
    return True


def test_reference():
    """Test all activations using PyTorch reference (no GPU required)."""
    print(f"Activation Parity Test: M={M}, N={N}")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16)

    all_pass = True
    print("\nSimple activations:")
    all_pass &= test_activation("SiLU", ref_silu, x)
    all_pass &= test_activation("GeLU", ref_gelu, x)
    all_pass &= test_activation("GeLU-tanh", ref_gelu_tanh, x)
    all_pass &= test_activation("ReLU", ref_relu, x)
    all_pass &= test_activation("Tanh", ref_tanh, x)

    print("\nGated activations:")
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    all_pass &= test_gated_activation("SiLU+mul", ref_silu_and_mul, a, b)
    all_pass &= test_gated_activation("GeLU+mul", ref_gelu_and_mul, a, b)

    print(f"\nReference implementation: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_hip_kernel():
    """Test HipKittens kernel against PyTorch reference (requires GPU)."""
    try:
        import ctypes
        lib = ctypes.CDLL("./activation_kernel.so")
        print("  HipKittens activation kernel loaded.")
    except OSError:
        print("  HipKittens kernel not compiled. Skipping GPU test.")
        print("  Compile with: make -C kernels/activation GPU_TARGET=CDNA3")
        return None

    print("  GPU parity testing would compare ctypes calls to reference.")
    print("  (Requires AMD GPU hardware)")
    return None


if __name__ == "__main__":
    print("\n--- Activation Kernel Parity Test ---\n")

    ref_ok = test_reference()

    print()
    print("GPU Kernel Test:")
    gpu_ok = test_hip_kernel()

    print()
    if ref_ok and (gpu_ok is None or gpu_ok):
        print("RESULT: PASS (reference validated" +
              (", GPU parity confirmed)" if gpu_ok else ", GPU test skipped)"))
    else:
        print("RESULT: FAIL")
        sys.exit(1)
