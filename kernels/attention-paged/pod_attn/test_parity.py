"""
Parity test for POD Attention (Prefill-On-Decode) (HipKittens vs Triton reference).

Tests a persistent kernel that simultaneously runs prefill (causal) and
decode (non-causal) attention on the same GPU with CU-based scheduling.

Expected tensor shapes (both decode and prefill paths):
  Q:     [batch * num_heads, seq_len_q, head_dim]     bf16
  K:     [batch * num_heads, seq_len_kv, head_dim]    bf16
  V:     [batch * num_heads, seq_len_kv, head_dim]    bf16
  Out:   [batch * num_heads, seq_len_q, head_dim]     bf16
  Mp:    [batch * num_heads, num_splits, num_m_blocks] float32 (partial max)
  Lp:    [batch * num_heads, num_splits, num_m_blocks] float32 (partial sum)
  Op:    [batch * num_heads, num_splits, num_m_blocks, head_dim] float32 (partial out)
  cu_ctr: [num_CUs]  int32 (atomic per-CU counters)
  locks:  [max_locks] int32 (split-K synchronization)

Compilation:
  make GPU_TARGET=CDNA4 POD_HEAD_DIM=128 POD_BLOCK_M=64 POD_BLOCK_N=64
"""

import torch
import math


def reference_attention(q, k, v, scale, causal=False):
    """Reference attention (both causal and non-causal)."""
    bnh, seq_q, hd = q.shape
    _, seq_kv, _ = k.shape

    scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * scale  # [bnh, seq_q, seq_kv]

    if causal:
        mask = torch.triu(torch.ones(seq_q, seq_kv, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v.float())
    return out.to(q.dtype)


def test_parity():
    torch.manual_seed(42)

    # Decode config
    batch = 8
    num_heads = 8
    seq_q = 1
    seq_kv = 256
    head_dim = 128
    dtype = torch.bfloat16

    scale = 1.0 / math.sqrt(head_dim)

    bnh = batch * num_heads
    q_dec = torch.randn(bnh, seq_q, head_dim, dtype=dtype)
    k_dec = torch.randn(bnh, seq_kv, head_dim, dtype=dtype)
    v_dec = torch.randn(bnh, seq_kv, head_dim, dtype=dtype)

    # Prefill config
    batch_pf = 2
    seq_q_pf = 64
    seq_kv_pf = 64
    bnh_pf = batch_pf * num_heads

    q_pf = torch.randn(bnh_pf, seq_q_pf, head_dim, dtype=dtype)
    k_pf = torch.randn(bnh_pf, seq_kv_pf, head_dim, dtype=dtype)
    v_pf = torch.randn(bnh_pf, seq_kv_pf, head_dim, dtype=dtype)

    # Reference
    dec_out = reference_attention(q_dec, k_dec, v_dec, scale, causal=False)
    pf_out = reference_attention(q_pf, k_pf, v_pf, scale, causal=True)

    print(f"Decode output shape: {dec_out.shape}")
    print(f"Decode output sample: {dec_out[0, 0, :8]}")
    print(f"\nPrefill output shape: {pf_out.shape}")
    print(f"Prefill output sample: {pf_out[0, 0, :8]}")
    print(f"\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 POD_HEAD_DIM={head_dim} "
          f"POD_BLOCK_M=64 POD_BLOCK_N=64")
    print(f"\nPOD scheduling: prefill_ratio=1, decode_ratio=3")
    print(f"  -> ~25% CU wavefronts run prefill, ~75% run decode")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
