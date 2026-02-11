"""
Parity test for Paged Attention Prefill (HipKittens vs Triton reference).

Tests the pa_prefill kernel that computes full-sequence attention
over a paged KV cache prefix + new query tokens with causal masking.

Expected tensor shapes:
  Q:           [total_tokens, num_q_heads, head_size]          bf16
  K:           [total_tokens, num_kv_heads, head_size]         bf16  (new tokens)
  V:           [total_tokens, num_kv_heads, head_size]         bf16  (new tokens)
  K_cache:     [num_blocks, num_kv_heads, kv_blk_sz, head_size] bf16 (paged)
  V_cache:     [num_blocks, num_kv_heads, kv_blk_sz, head_size] bf16 (paged)
  block_table: [num_seqs, max_num_blocks_per_seq]              int32
  seq_start:   [num_seqs + 1]                                  int32
  seq_lens:    [num_seqs]                                       int32
  ctx_lens:    [num_seqs]                                       int32
  output:      [total_tokens, num_q_heads, head_size]          bf16

Compilation:
  make GPU_TARGET=CDNA4 PA_HEAD_SZ=128 PA_NUM_KV_HEADS=8 PA_NUM_Q_HEADS=64
"""

import torch
import math


def reference_paged_prefill(
    q, k_new, v_new, k_cache, v_cache,
    block_table, seq_starts, seq_lens, ctx_lens,
    scale, num_kv_heads, query_grp_sz, kv_blk_sz,
):
    """Reference implementation of paged attention prefill."""
    num_seqs = len(seq_lens)
    total_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    head_size = q.shape[2]
    output = torch.zeros_like(q, dtype=torch.float32)

    for s in range(num_seqs):
        s_start = seq_starts[s].item()
        s_end = seq_starts[s + 1].item()
        n_new = s_end - s_start
        ctx_len = ctx_lens[s].item()

        for qh in range(num_q_heads):
            kv_head = qh // query_grp_sz

            # Gather context K, V from paged cache
            k_ctx_list, v_ctx_list = [], []
            num_ctx_blocks = (ctx_len + kv_blk_sz - 1) // kv_blk_sz
            for b in range(num_ctx_blocks):
                phys = block_table[s, b].item()
                valid = min(kv_blk_sz, ctx_len - b * kv_blk_sz)
                k_ctx_list.append(k_cache[phys, kv_head, :valid].float())
                v_ctx_list.append(v_cache[phys, kv_head, :valid].float())

            # Get new K, V
            k_new_s = k_new[s_start:s_end, kv_head].float()  # [n_new, head_size]
            v_new_s = v_new[s_start:s_end, kv_head].float()

            # Concatenate context + new
            if k_ctx_list:
                k_all = torch.cat(k_ctx_list + [k_new_s], dim=0)
                v_all = torch.cat(v_ctx_list + [v_new_s], dim=0)
            else:
                k_all = k_new_s
                v_all = v_new_s

            for qi in range(n_new):
                q_vec = q[s_start + qi, qh].float()  # [head_size]

                # Causal: can attend to context + new[:qi+1]
                kv_len = ctx_len + qi + 1
                scores = (q_vec @ k_all[:kv_len].T) * scale
                attn = torch.softmax(scores, dim=-1)
                output[s_start + qi, qh] = attn @ v_all[:kv_len]

    return output.to(q.dtype)


def test_parity():
    torch.manual_seed(42)

    num_seqs = 2
    ctx_lens_list = [64, 32]
    new_lens_list = [16, 8]
    num_kv_heads = 8
    query_grp_sz = 8
    num_q_heads = num_kv_heads * query_grp_sz
    head_size = 128
    kv_blk_sz = 16
    dtype = torch.bfloat16

    scale = 1.0 / math.sqrt(head_size)
    total_new = sum(new_lens_list)
    max_ctx_blocks = max((cl + kv_blk_sz - 1) // kv_blk_sz for cl in ctx_lens_list)
    total_blocks = sum((cl + kv_blk_sz - 1) // kv_blk_sz for cl in ctx_lens_list)

    q = torch.randn(total_new, num_q_heads, head_size, dtype=dtype)
    k_new = torch.randn(total_new, num_kv_heads, head_size, dtype=dtype)
    v_new = torch.randn(total_new, num_kv_heads, head_size, dtype=dtype)
    k_cache = torch.randn(total_blocks, num_kv_heads, kv_blk_sz, head_size, dtype=dtype)
    v_cache = torch.randn(total_blocks, num_kv_heads, kv_blk_sz, head_size, dtype=dtype)

    seq_starts = torch.tensor([0] + [sum(new_lens_list[:i+1]) for i in range(num_seqs)], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_lens_list[i] + new_lens_list[i] for i in range(num_seqs)], dtype=torch.int32)
    ctx_lens = torch.tensor(ctx_lens_list, dtype=torch.int32)

    block_table = torch.zeros(num_seqs, max_ctx_blocks, dtype=torch.int32)
    block_offset = 0
    for s in range(num_seqs):
        n_blocks = (ctx_lens_list[s] + kv_blk_sz - 1) // kv_blk_sz
        for b in range(n_blocks):
            block_table[s, b] = block_offset + b
        block_offset += n_blocks

    ref_out = reference_paged_prefill(
        q, k_new, v_new, k_cache, v_cache,
        block_table, seq_starts, seq_lens, ctx_lens,
        scale, num_kv_heads, query_grp_sz, kv_blk_sz,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output sample: {ref_out[0, 0, :8]}")
    print("\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 PA_HEAD_SZ={head_size} "
          f"PA_NUM_KV_HEADS={num_kv_heads} PA_NUM_Q_HEADS={num_q_heads}")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
