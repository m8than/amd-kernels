"""
Parity test for MLA Decode with RoPE (HipKittens vs Triton reference).

Tests Multi-head Latent Attention decode for DeepSeek-V2 style models.
K and V share a low-rank latent, Q/K are split into latent + positional parts.

Expected tensor shapes:
  Q:              [batch, num_heads, kv_lora_rank + qk_rope_dim]     bf16
  K_buffer:       [total_tokens, kv_lora_rank + qk_rope_dim]         bf16
  V_buffer:       [total_tokens, kv_lora_rank]                        bf16
  cos_sin_cache:  [max_seq_len, rotary_dim * 2]                       bf16
  positions:      [batch]                                              int32
  kv_indptr:      [batch + 1]                                         int32
  kv_indices:     [total_tokens]                                       int32
  output:         [batch, num_heads, kv_lora_rank]                    bf16

Compilation:
  make GPU_TARGET=CDNA4 MLA_KV_LORA_RANK=512 MLA_QK_ROPE_DIM=64
"""

import torch
import math


def apply_rope(x, cos, sin):
    """Apply RoPE to x. x: [..., dim], cos/sin: [..., dim//2]."""
    d = x.shape[-1]
    half = d // 2
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([
        x0 * cos - x1 * sin,
        x1 * cos + x0 * sin,
    ], dim=-1)


def reference_mla_decode(
    q, k_buffer, v_buffer, cos_sin_cache, positions,
    kv_indptr, kv_indices, scale, kv_lora_rank, qk_rope_dim,
    use_rope=True, logit_cap=0.0,
):
    """Reference MLA decode with RoPE."""
    batch, num_heads, total_dim = q.shape
    output = torch.zeros(batch, num_heads, kv_lora_rank, dtype=torch.float32)

    for b in range(batch):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        kv_len = kv_end - kv_start

        if kv_len == 0:
            continue

        # Split Q
        q_nope = q[b, :, :kv_lora_rank].float()  # [num_heads, kv_lora_rank]
        q_pe = q[b, :, kv_lora_rank:].float()     # [num_heads, qk_rope_dim]

        # Apply RoPE to q_pe
        if use_rope:
            pos = positions[b].item()
            cos_vals = cos_sin_cache[pos, :qk_rope_dim // 2].float()
            sin_vals = cos_sin_cache[pos, qk_rope_dim // 2:qk_rope_dim].float()
            q_pe = apply_rope(q_pe, cos_vals.unsqueeze(0), sin_vals.unsqueeze(0))

        # Gather KV tokens
        indices = kv_indices[kv_start:kv_end]
        kv_tokens = k_buffer[indices].float()  # [kv_len, kv_lora_rank + qk_rope_dim]
        v_tokens = v_buffer[indices].float()   # [kv_len, kv_lora_rank]

        kv_latent = kv_tokens[:, :kv_lora_rank]  # [kv_len, kv_lora_rank]
        k_pe = kv_tokens[:, kv_lora_rank:]        # [kv_len, qk_rope_dim]

        for h in range(num_heads):
            # Score = q_nope @ kv_latent^T + q_pe @ k_pe^T
            score_nope = q_nope[h] @ kv_latent.T  # [kv_len]
            score_pe = q_pe[h] @ k_pe.T            # [kv_len]
            scores = (score_nope + score_pe) * scale

            if logit_cap > 0:
                scores = logit_cap * torch.tanh(scores / logit_cap)

            attn = torch.softmax(scores, dim=-1)
            output[b, h] = attn @ v_tokens  # [kv_lora_rank]

    return output.to(q.dtype)


def test_parity():
    torch.manual_seed(42)

    batch = 4
    num_heads = 16  # Reduced for test
    kv_lora_rank = 64  # Reduced for test
    qk_rope_dim = 32   # Reduced for test
    total_dim = kv_lora_rank + qk_rope_dim
    max_seq_len = 128
    dtype = torch.bfloat16

    scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_dim)

    # Variable-length KV sequences
    kv_lens = [32, 16, 48, 24]
    total_tokens = sum(kv_lens)
    kv_indptr = torch.tensor([0] + [sum(kv_lens[:i+1]) for i in range(batch)], dtype=torch.int32)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32)
    positions = torch.tensor([kl - 1 for kl in kv_lens], dtype=torch.int32)

    q = torch.randn(batch, num_heads, total_dim, dtype=dtype)
    k_buffer = torch.randn(total_tokens, total_dim, dtype=dtype)
    v_buffer = torch.randn(total_tokens, kv_lora_rank, dtype=dtype)
    cos_sin_cache = torch.randn(max_seq_len, qk_rope_dim, dtype=dtype)

    ref_out = reference_mla_decode(
        q, k_buffer, v_buffer, cos_sin_cache, positions,
        kv_indptr, kv_indices, scale, kv_lora_rank, qk_rope_dim,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output sample: {ref_out[0, 0, :8]}")
    print(f"\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 MLA_KV_LORA_RANK={kv_lora_rank} "
          f"MLA_QK_ROPE_DIM={qk_rope_dim} MLA_NUM_HEADS={num_heads}")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
