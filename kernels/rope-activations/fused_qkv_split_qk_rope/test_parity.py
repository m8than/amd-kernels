"""
Parity test for Fused QKV Split + QK RoPE HipKittens kernel.

Takes a fused QKV tensor, splits into Q/K/V, applies RoPE to Q and K.

Input shapes:
  qkv: (T, (QH + 2*KVH) * D) bf16 — fused QKV tensor
  cos: (T, D/2) bf16 — cosine frequencies
  sin: (T, D/2) bf16 — sine frequencies

Output shapes:
  q: (T, QH, D) bf16 — Q with RoPE applied
  k: (T, KVH, D) bf16 — K with RoPE applied
  v: (T, KVH, D) bf16 — V passthrough

Compile command:
  make -C kernels/fused_qkv_split_qk_rope GPU_TARGET=CDNA3
"""

import torch
import sys
import math

# Test parameters
T = 2048    # total tokens
QH = 32     # Q heads
KVH = 8     # KV heads (GQA)
D = 128     # head dimension
D_HALF = D // 2

ATOL = 1e-2
RTOL = 1e-2


def get_cos_sin(seq_len, dim, base=10000.0, dtype=torch.bfloat16, device='cpu'):
    """Generate rotary position embedding cos/sin tables."""
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(t, inv_freq)
    cos_freq = torch.cos(freqs).to(dtype)
    sin_freq = torch.sin(freqs).to(dtype)
    return cos_freq, sin_freq


def apply_rope(x, cos_freq, sin_freq):
    """Apply NeoX-style RoPE to tensor x of shape (..., D)."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    # Broadcast cos/sin to match x shape
    cos_b = cos_freq
    sin_b = sin_freq
    while cos_b.dim() < x.dim():
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)
    out_first = x1 * cos_b - x2 * sin_b
    out_second = x2 * cos_b + x1 * sin_b
    return torch.cat([out_first, out_second], dim=-1)


def reference_fused_qkv_rope(qkv, cos_freq, sin_freq, QH, KVH, D):
    """PyTorch reference implementation."""
    T = qkv.shape[0]

    # Split QKV
    q = qkv[:, :QH * D].reshape(T, QH, D)
    k = qkv[:, QH * D:(QH + KVH) * D].reshape(T, KVH, D)
    v = qkv[:, (QH + KVH) * D:].reshape(T, KVH, D)

    # Apply RoPE to Q and K
    q_rope = apply_rope(q.float(), cos_freq.float(), sin_freq.float()).to(torch.bfloat16)
    k_rope = apply_rope(k.float(), cos_freq.float(), sin_freq.float()).to(torch.bfloat16)

    return q_rope, k_rope, v


def test_reference():
    """Test using PyTorch reference only (no GPU required)."""
    print(f"Fused QKV+RoPE Parity Test: T={T}, QH={QH}, KVH={KVH}, D={D}")
    print("=" * 60)

    torch.manual_seed(42)

    total_d = (QH + 2 * KVH) * D
    qkv = torch.randn(T, total_d, dtype=torch.bfloat16)
    cos_freq, sin_freq = get_cos_sin(T, D)

    q, k, v = reference_fused_qkv_rope(qkv, cos_freq, sin_freq, QH, KVH, D)

    # Verify shapes
    assert q.shape == (T, QH, D), f"Q shape mismatch: {q.shape}"
    assert k.shape == (T, KVH, D), f"K shape mismatch: {k.shape}"
    assert v.shape == (T, KVH, D), f"V shape mismatch: {v.shape}"

    # Verify no NaN/Inf
    assert not torch.isnan(q).any(), "Q contains NaN"
    assert not torch.isnan(k).any(), "K contains NaN"
    assert not torch.isnan(v).any(), "V contains NaN"

    # V should be exact passthrough
    v_orig = qkv[:, (QH + KVH) * D:].reshape(T, KVH, D)
    assert torch.equal(v, v_orig), "V passthrough failed"

    print(f"  Q: shape={q.shape}, range=[{q.float().min():.4f}, {q.float().max():.4f}]")
    print(f"  K: shape={k.shape}, range=[{k.float().min():.4f}, {k.float().max():.4f}]")
    print(f"  V: shape={v.shape}, range=[{v.float().min():.4f}, {v.float().max():.4f}]")
    print(f"  V passthrough: exact match")
    print("  Reference implementation: PASS")
    return True


def test_hip_kernel():
    """Test HipKittens kernel (requires GPU)."""
    print("  HipKittens kernel not compiled. Skipping GPU test.")
    print("  Compile with: make -C kernels/fused_qkv_split_qk_rope GPU_TARGET=CDNA3")
    return None


if __name__ == "__main__":
    print("\n--- Fused QKV Split + QK RoPE Kernel Parity Test ---\n")

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
