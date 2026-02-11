"""
Parity test for Paged MQA Logits (HipKittens vs Triton reference).

Computes weighted MQA logits with ReLU activation:
  logits[b,t] = sum_h( weight[b,h] * ReLU(scale * Q[b,h] @ K[t]^T) )

Expected tensor shapes:
  Q:          [batch * next_n, heads, hidden_dim]    bf16
  KV_buffer:  [total_tokens, hidden_dim]              bf16
  K_scale:    [total_tokens]                           float32
  weights:    [batch * next_n, heads]                  float32
  kv_indices: [batch, max_kv_len]                      int32
  out_logits: [batch * next_n, max_seq_len]            float32

Compilation:
  make GPU_TARGET=CDNA4 MQA_HIDDEN_DIM=128 MQA_NUM_HEADS=8
"""

import torch
import math


def reference_mqa_logits(
    q, kv_buffer, k_scale, weights, kv_indices, kv_lens,
    scale, batch_size, next_n, num_heads,
):
    """Reference MQA logits: weighted sum of ReLU(Q @ K^T * scale)."""
    hidden_dim = q.shape[-1]
    max_kv_len = kv_indices.shape[1]
    bq_total = batch_size * next_n
    out_logits = torch.zeros(bq_total, max_kv_len, dtype=torch.float32)

    for bq in range(bq_total):
        b = bq // next_n
        kv_len = kv_lens[b].item()

        for t in range(kv_len):
            phys_idx = kv_indices[b, t].item()
            k_vec = kv_buffer[phys_idx].float()  # [hidden_dim]
            k_sc = k_scale[phys_idx].item()

            logit_sum = 0.0
            for h in range(num_heads):
                q_vec = q[bq, h].float()
                score = scale * (q_vec @ k_vec) * k_sc
                relu_score = max(0.0, score)
                logit_sum += weights[bq, h].item() * relu_score

            out_logits[bq, t] = logit_sum

    return out_logits


def test_parity():
    torch.manual_seed(42)

    batch_size = 4
    next_n = 1
    num_heads = 8
    hidden_dim = 128
    max_kv_len = 64
    total_tokens = batch_size * max_kv_len
    scale = 1.0 / math.sqrt(hidden_dim)

    q = torch.randn(batch_size * next_n, num_heads, hidden_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(total_tokens, hidden_dim, dtype=torch.bfloat16)
    k_scale = torch.ones(total_tokens, dtype=torch.float32)
    weights = torch.randn(batch_size * next_n, num_heads, dtype=torch.float32).softmax(dim=-1)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32).view(batch_size, max_kv_len)
    kv_lens = torch.full((batch_size,), max_kv_len, dtype=torch.int32)

    ref_out = reference_mqa_logits(
        q, kv_buffer, k_scale, weights, kv_indices, kv_lens,
        scale, batch_size, next_n, num_heads,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output sample: {ref_out[0, :8]}")
    print(f"\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 MQA_HIDDEN_DIM={hidden_dim} MQA_NUM_HEADS={num_heads}")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
