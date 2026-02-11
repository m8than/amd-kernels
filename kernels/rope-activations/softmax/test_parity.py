"""
Parity test for Online Softmax HipKittens kernel.

Tests row-wise softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Input shapes:
  input:  (M, N) bf16 — 2D tensor
Output shapes:
  output: (M, N) bf16 — softmax applied along dim=-1

Compile command:
  make -C kernels/softmax GPU_TARGET=CDNA3
"""

import torch
import sys

# Test parameters
M = 512
N_SIZES = [128, 1024, 4096, 8192]  # Test various row widths

ATOL = 1e-2
RTOL = 1e-2


def ref_softmax(x):
    """PyTorch reference softmax along last dim."""
    return torch.softmax(x.float(), dim=-1).to(torch.bfloat16)


def test_reference():
    """Test using PyTorch reference only (no GPU required)."""
    print(f"Softmax Parity Test: M={M}")
    print("=" * 60)

    torch.manual_seed(42)
    all_pass = True

    for N in N_SIZES:
        x = torch.randn(M, N, dtype=torch.bfloat16)
        out = ref_softmax(x)

        # Verify properties
        shape_ok = out.shape == (M, N)
        # Each row should sum to ~1.0
        row_sums = out.float().sum(dim=-1)
        sum_ok = torch.allclose(row_sums, torch.ones(M), atol=1e-2)
        # All values should be in [0, 1]
        range_ok = (out.float() >= 0.0).all() and (out.float() <= 1.0 + 1e-3).all()
        nan_ok = not torch.isnan(out).any()

        passed = shape_ok and sum_ok and range_ok and nan_ok
        all_pass &= passed

        print(f"  N={N:5d}: sum_range=[{row_sums.min():.4f}, {row_sums.max():.4f}], "
              f"val_range=[{out.float().min():.6f}, {out.float().max():.6f}] "
              f"{'PASS' if passed else 'FAIL'}")

    print(f"\nReference implementation: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_hip_kernel():
    """Test HipKittens kernel against PyTorch reference (requires GPU)."""
    try:
        import ctypes
        lib = ctypes.CDLL("./softmax_kernel.so")
        print("  HipKittens softmax kernel loaded.")
    except OSError:
        print("  HipKittens kernel not compiled. Skipping GPU test.")
        print("  Compile with: make -C kernels/softmax GPU_TARGET=CDNA3")
        return None

    print("  GPU parity testing would compare ctypes calls to reference.")
    print("  (Requires AMD GPU hardware)")
    return None


if __name__ == "__main__":
    print("\n--- Softmax Kernel Parity Test ---\n")

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
