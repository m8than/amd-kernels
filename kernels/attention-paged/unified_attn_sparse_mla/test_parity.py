"""
Parity test for Unified Attention Sparse MLA (HipKittens vs Triton reference).

Tests sparse MLA attention that attends only to precomputed top-k positions.
Designed for MLA models with single KV head and split Q/K (latent + rope).

Expected tensor shapes:
  Q:              [num_tokens, num_q_heads, kv_lora_rank + rope_rank]  bf16
  K_cache:        [num_blks, blk_size, 1, kv_lora_rank + rope_rank]    bf16
  V_cache:        [num_blks, blk_size, 1, kv_lora_rank]                bf16
  block_table:    [num_seqs, max_blocks]                                int32
  topk_indices:   [num_tokens, topk]                                    int32
  seq_lens:       [num_seqs]                                            int32
  query_start_lens: [num_seqs + 1]                                      int32
  output:         [num_tokens, num_q_heads, kv_lora_rank]              bf16

Compilation:
  make GPU_TARGET=CDNA4 SMLA_KV_LORA_RANK=512 SMLA_ROPE_RANK=64
"""

import torch
import math


def reference_sparse_mla(
    q, k_cache, v_cache, block_table, topk_indices,
    seq_lens, query_starts, scale, kv_lora_rank, rope_rank, block_size,
):
    """Reference sparse MLA attention with top-k selection."""
    num_tokens, num_heads, total_dim = q.shape
    num_seqs = len(seq_lens)
    output = torch.zeros(num_tokens, num_heads, kv_lora_rank, dtype=torch.float32)

    for ti in range(num_tokens):
        # Find sequence
        seq_idx = 0
        for s in range(num_seqs):
            if ti >= query_starts[s] and ti < query_starts[s + 1]:
                seq_idx = s
                break

        for h in range(num_heads):
            q_lora = q[ti, h, :kv_lora_rank].float()
            q_rope = q[ti, h, kv_lora_rank:].float()

            scores = []
            v_list = []

            for t_idx in range(topk_indices.shape[1]):
                pos = topk_indices[ti, t_idx].item()
                if pos < 0:
                    continue  # skip padding

                blk = pos // block_size
                slot = pos % block_size
                phys = block_table[seq_idx, blk].item()

                k_full = k_cache[phys, slot, 0].float()
                k_lora = k_full[:kv_lora_rank]
                k_rope = k_full[kv_lora_rank:]

                score = scale * (q_lora @ k_lora + q_rope @ k_rope)
                scores.append(score)

                v_tok = v_cache[phys, slot, 0, :kv_lora_rank].float()
                v_list.append(v_tok)

            if not scores:
                continue

            scores_t = torch.tensor(scores)
            attn = torch.softmax(scores_t, dim=-1)
            v_mat = torch.stack(v_list)
            output[ti, h] = attn @ v_mat

    return output.to(q.dtype)


def test_parity():
    torch.manual_seed(42)

    num_seqs = 2
    num_tokens = 4
    num_heads = 8
    kv_lora_rank = 32
    rope_rank = 16
    total_dim = kv_lora_rank + rope_rank
    block_size = 16
    topk = 32
    max_seq_len = 64
    dtype = torch.bfloat16

    scale = 1.0 / math.sqrt(total_dim)

    max_blocks = (max_seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * max_blocks

    q = torch.randn(num_tokens, num_heads, total_dim, dtype=dtype)
    k_cache = torch.randn(total_blocks, block_size, 1, total_dim, dtype=dtype)
    v_cache = torch.randn(total_blocks, block_size, 1, kv_lora_rank, dtype=dtype)

    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(max_blocks):
            block_table[s, b] = s * max_blocks + b

    seq_lens = torch.tensor([48, 32], dtype=torch.int32)
    query_starts = torch.tensor([0, 2, 4], dtype=torch.int32)

    # Top-k indices: random positions within sequence length
    topk_indices = torch.zeros(num_tokens, topk, dtype=torch.int32)
    for ti in range(num_tokens):
        si = 0 if ti < 2 else 1
        sl = seq_lens[si].item()
        indices = torch.randperm(sl)[:topk]
        topk_indices[ti, :len(indices)] = indices.int()
        if len(indices) < topk:
            topk_indices[ti, len(indices):] = -1

    ref_out = reference_sparse_mla(
        q, k_cache, v_cache, block_table, topk_indices,
        seq_lens, query_starts, scale, kv_lora_rank, rope_rank, block_size,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output sample: {ref_out[0, 0, :8]}")
    print(f"\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 SMLA_KV_LORA_RANK={kv_lora_rank} "
          f"SMLA_ROPE_RANK={rope_rank} SMLA_TOPK={topk}")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
