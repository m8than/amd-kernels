"""
Parity test for RoPE (Rotary Position Embeddings) HipKittens kernel.

Tests the NeoX-style rotary embedding:
  x_out[..., :D/2] = x[..., :D/2] * cos - x[..., D/2:] * sin
  x_out[..., D/2:] = x[..., D/2:] * cos + x[..., :D/2] * sin

Input shapes:
  x:   (B, H, N, D) bf16 — input tensor
  cos: (N, D/2) bf16 — cosine frequencies
  sin: (N, D/2) bf16 — sine frequencies

Output shapes:
  o:   (B, H, N, D) bf16 — rotary-embedded output

Compile command:
  make -C kernels/rope GPU_TARGET=CDNA3
"""

import torch
import subprocess
import os
import sys

# Test parameters
B = 4      # batch size
H = 32     # number of heads
N = 2048   # sequence length
D = 128    # head dimension
D_HALF = D // 2

ATOL = 1e-2
RTOL = 1e-2


def reference_rope(x, cos_freq, sin_freq):
    """PyTorch reference implementation of NeoX-style RoPE."""
    x1 = x[..., :D_HALF]       # first half
    x2 = x[..., D_HALF:]       # second half
    # Broadcast cos/sin: (N, D/2) -> (1, 1, N, D/2)
    cos_b = cos_freq.unsqueeze(0).unsqueeze(0)
    sin_b = sin_freq.unsqueeze(0).unsqueeze(0)
    out_first = x1 * cos_b - x2 * sin_b
    out_second = x2 * cos_b + x1 * sin_b
    return torch.cat([out_first, out_second], dim=-1)


def get_cos_sin(seq_len, dim, base=10000.0, dtype=torch.bfloat16, device='cpu'):
    """Generate rotary position embedding cos/sin tables."""
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(t, inv_freq)
    cos_freq = torch.cos(freqs).to(dtype)
    sin_freq = torch.sin(freqs).to(dtype)
    return cos_freq, sin_freq


def test_reference():
    """Test using PyTorch reference only (no GPU required)."""
    print(f"RoPE Parity Test: B={B}, H={H}, N={N}, D={D}")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cpu'

    x = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)
    cos_freq, sin_freq = get_cos_sin(N, D, device=device)

    out = reference_rope(x, cos_freq, sin_freq)

    # Verify basic properties
    assert out.shape == (B, H, N, D), f"Output shape mismatch: {out.shape}"
    assert out.dtype == torch.bfloat16, f"Output dtype mismatch: {out.dtype}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

    print(f"  Reference output shape: {out.shape}")
    print(f"  Reference output dtype: {out.dtype}")
    print(f"  Reference output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print("  Reference implementation: PASS")
    return True


def test_hip_kernel():
    """Test HipKittens kernel against PyTorch reference (requires GPU)."""
    try:
        import rope_kernel
    except ImportError:
        print("  HipKittens kernel not compiled. Skipping GPU test.")
        print("  Compile with: make -C kernels/rope GPU_TARGET=CDNA3")
        return None

    torch.manual_seed(42)
    device = 'cuda'

    x = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)
    cos_freq, sin_freq = get_cos_sin(N, D, device=device)

    ref_out = reference_rope(x, cos_freq, sin_freq)

    o = torch.zeros_like(x)
    rope_kernel.dispatch_rope(x, o, sin_freq, cos_freq)
    torch.cuda.synchronize()

    max_diff = (ref_out - o).abs().max().item()
    passed = torch.allclose(ref_out, o, atol=ATOL, rtol=RTOL)

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"  GPU parity test: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n--- RoPE Kernel Parity Test ---\n")

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
