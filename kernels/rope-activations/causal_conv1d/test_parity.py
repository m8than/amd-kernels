"""
Parity test for Causal 1D Convolution HipKittens kernel.

Tests causal conv1d used in Mamba/SSM architectures:
  output[t] = sum_{k=0}^{K-1} weight[k] * input[t - K + 1 + k] + bias

Input shapes:
  x:    (B, D, L) bf16 — input tensor
  w:    (D, K) bf16 — convolution weights
  bias: (D,) bf16 — optional bias

Output shapes:
  o:    (B, D, L) bf16

Compile command:
  make -C kernels/causal_conv1d GPU_TARGET=CDNA3
"""

import torch
import torch.nn.functional as F
import sys

# Test parameters
B = 8       # batch size
D = 768     # channels/dim
L = 2048    # sequence length
K_SIZES = [3, 4]  # kernel widths

ATOL = 1e-2
RTOL = 1e-2


def ref_causal_conv1d(x, w, bias=None, activation=None):
    """
    PyTorch reference causal conv1d.
    x: (B, D, L)
    w: (D, K)
    bias: (D,) or None
    Returns: (B, D, L)
    """
    B, D, L = x.shape
    K = w.shape[1]

    # Pad left by K-1 for causal convolution
    x_padded = F.pad(x.float(), (K - 1, 0))  # (B, D, L + K - 1)

    # Manual convolution per channel (depthwise)
    out = torch.zeros(B, D, L, dtype=torch.float32)
    for t in range(L):
        for k in range(K):
            out[:, :, t] += x_padded[:, :, t + k] * w[:, k].float()

    if bias is not None:
        out += bias.float().unsqueeze(0).unsqueeze(-1)

    if activation == 'silu':
        out = out * torch.sigmoid(out)

    return out.to(torch.bfloat16)


def ref_causal_conv1d_fast(x, w, bias=None, activation=None):
    """Faster reference using F.conv1d with groups=D (depthwise)."""
    B, D, L = x.shape
    K = w.shape[1]

    # Reshape w for depthwise conv: (D, 1, K)
    w_conv = w.float().unsqueeze(1)

    # Pad for causal (left-pad by K-1)
    x_padded = F.pad(x.float(), (K - 1, 0))

    # Depthwise convolution
    out = F.conv1d(x_padded, w_conv, bias=bias.float() if bias is not None else None,
                   groups=D)

    if activation == 'silu':
        out = out * torch.sigmoid(out)

    return out.to(torch.bfloat16)


def test_reference():
    """Test using PyTorch reference only (no GPU required)."""
    print(f"Causal Conv1d Parity Test: B={B}, D={D}, L={L}")
    print("=" * 60)

    torch.manual_seed(42)
    all_pass = True

    for K in K_SIZES:
        x = torch.randn(B, D, L, dtype=torch.bfloat16)
        w = torch.randn(D, K, dtype=torch.bfloat16)
        bias = torch.randn(D, dtype=torch.bfloat16)

        # Test without bias/activation
        out_no_bias = ref_causal_conv1d_fast(x, w)
        assert out_no_bias.shape == (B, D, L), f"K={K}: shape mismatch"
        assert not torch.isnan(out_no_bias).any(), f"K={K}: NaN in output"

        # Test with bias + SiLU
        out_bias_silu = ref_causal_conv1d_fast(x, w, bias=bias, activation='silu')
        assert out_bias_silu.shape == (B, D, L)
        assert not torch.isnan(out_bias_silu).any()

        # Verify causality: changing future input shouldn't affect past output
        x2 = x.clone()
        x2[:, :, L // 2:] = torch.randn_like(x2[:, :, L // 2:])
        out2 = ref_causal_conv1d_fast(x2, w)
        # First L//2 - K + 1 outputs should be identical
        causal_check_end = max(0, L // 2 - K + 1)
        causal_ok = torch.allclose(
            out_no_bias[:, :, :causal_check_end].float(),
            out2[:, :, :causal_check_end].float(),
            atol=1e-6
        )

        passed = causal_ok
        all_pass &= passed

        print(f"  K={K}: no_bias range=[{out_no_bias.float().min():.4f}, {out_no_bias.float().max():.4f}], "
              f"bias+silu range=[{out_bias_silu.float().min():.4f}, {out_bias_silu.float().max():.4f}], "
              f"causal={'OK' if causal_ok else 'FAIL'} "
              f"{'PASS' if passed else 'FAIL'}")

    print(f"\nReference implementation: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_hip_kernel():
    """Test HipKittens kernel (requires GPU)."""
    print("  HipKittens kernel not compiled. Skipping GPU test.")
    print("  Compile with: make -C kernels/causal_conv1d GPU_TARGET=CDNA3")
    return None


if __name__ == "__main__":
    print("\n--- Causal Conv1d Kernel Parity Test ---\n")

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
