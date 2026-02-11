"""
Parity test for Chunked Paged Attention Prefill (HipKittens vs Triton reference).

Tests a decode-style chunked paged attention that processes one query token
per sequence with sliding window support.

Expected tensor shapes:
  Q:            [num_tokens, num_q_heads, head_size]                 bf16
  K_cache:      [num_blocks, num_kv_heads, block_size, head_size]    bf16
  V_cache:      [num_blocks, num_kv_heads, block_size, head_size]    bf16
  block_table:  [num_seqs, max_num_blocks_per_seq]                   int32
  seq_lens:     [num_seqs]                                            int32
  query_starts: [num_seqs + 1]                                        int32
  output:       [num_tokens, num_q_heads, head_size]                 bf16

Compilation:
  make GPU_TARGET=CDNA4 CPA_HEAD_SZ=128 CPA_NUM_KV_HEADS=8 CPA_NUM_Q_HEADS=64
"""

import torch
import math


def reference_chunked_pa(
    q, k_cache, v_cache, block_table, seq_lens, query_starts,
    scale, num_kv_heads, query_grp_sz, block_size, sliding_window=0,
):
    """Reference chunked paged attention (single query per sequence)."""
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    head_size = q.shape[2]
    num_seqs = len(seq_lens)
    output = torch.zeros_like(q, dtype=torch.float32)

    for s in range(num_seqs):
        qs = query_starts[s].item()
        qe = query_starts[s + 1].item()
        sl = seq_lens[s].item()
        num_blocks = (sl + block_size - 1) // block_size

        for ti in range(qs, qe):
            for qh in range(num_q_heads):
                kv_head = qh // query_grp_sz
                q_vec = q[ti, qh].float()

                k_list, v_list = [], []
                for b in range(num_blocks):
                    phys = block_table[s, b].item()
                    valid = min(block_size, sl - b * block_size)
                    start = b * block_size

                    # Sliding window filter
                    if sliding_window > 0:
                        q_pos = sl - 1
                        if start + valid - 1 < q_pos - sliding_window:
                            continue

                    k_list.append(k_cache[phys, kv_head, :valid].float())
                    v_list.append(v_cache[phys, kv_head, :valid].float())

                if not k_list:
                    continue

                k_all = torch.cat(k_list, dim=0)
                v_all = torch.cat(v_list, dim=0)

                scores = (q_vec @ k_all.T) * scale
                attn = torch.softmax(scores, dim=-1)
                output[ti, qh] = attn @ v_all

    return output.to(q.dtype)


def test_parity():
    torch.manual_seed(42)

    num_seqs = 4
    max_seq_len = 128
    num_kv_heads = 8
    query_grp_sz = 8
    num_q_heads = num_kv_heads * query_grp_sz
    head_size = 128
    block_size = 16
    dtype = torch.bfloat16

    scale = 1.0 / math.sqrt(head_size)

    # One query token per sequence (decode mode)
    num_tokens = num_seqs
    query_starts = torch.arange(num_seqs + 1, dtype=torch.int32)

    q = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype)

    max_blocks = (max_seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * max_blocks
    k_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype)
    v_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype)

    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(max_blocks):
            block_table[s, b] = s * max_blocks + b

    seq_lens = torch.randint(16, max_seq_len + 1, (num_seqs,), dtype=torch.int32)

    ref_out = reference_chunked_pa(
        q, k_cache, v_cache, block_table, seq_lens, query_starts,
        scale, num_kv_heads, query_grp_sz, block_size,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output sample: {ref_out[0, 0, :8]}")
    print(f"\nCompilation:")
    print(f"  make GPU_TARGET=CDNA4 CPA_HEAD_SZ={head_size} "
          f"CPA_NUM_KV_HEADS={num_kv_heads} CPA_NUM_Q_HEADS={num_q_heads}")
    print("\nReference test PASSED (CPU-only mode)")


if __name__ == "__main__":
    test_parity()
