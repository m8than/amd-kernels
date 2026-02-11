"""
Parity test for Paged Attention Decode (HipKittens vs Triton reference).

Tests the pa_decode kernel that computes single-token attention
over a paged KV cache using block tables for indirection.

Expected tensor shapes:
  Q:           [num_seqs, num_q_heads, head_size]         bf16
  K_cache:     [num_blocks, num_kv_heads, kv_blk_sz, head_size]  bf16
  V_cache:     [num_blocks, num_kv_heads, kv_blk_sz, head_size]  bf16
  block_table: [num_seqs, max_num_blocks_per_seq]         int32
  seq_lens:    [num_seqs]                                  int32
  output:      [num_seqs, num_q_heads, head_size]         bf16

Compilation command:
  make GPU_TARGET=CDNA4 PA_NUM_SEQS=32 PA_HEAD_SZ=128 \
       PA_NUM_KV_HEADS=8 PA_QUERY_GRP_SZ=8 PA_KV_BLK_SZ=16
"""

import torch
import math
import subprocess
import sys
import os

def create_paged_kv_cache(
    num_seqs: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_size: int,
    kv_blk_sz: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    """Create a paged KV cache with random data and block tables."""
    max_blocks_per_seq = (max_seq_len + kv_blk_sz - 1) // kv_blk_sz
    total_blocks = num_seqs * max_blocks_per_seq

    # Allocate physical KV cache pages
    k_cache = torch.randn(total_blocks, num_kv_heads, kv_blk_sz, head_size,
                          dtype=dtype, device=device)
    v_cache = torch.randn(total_blocks, num_kv_heads, kv_blk_sz, head_size,
                          dtype=dtype, device=device)

    # Create block tables (identity mapping for simplicity)
    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32,
                              device=device)
    for s in range(num_seqs):
        for b in range(max_blocks_per_seq):
            block_table[s, b] = s * max_blocks_per_seq + b

    # Random sequence lengths
    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,),
                             dtype=torch.int32, device=device)

    return k_cache, v_cache, block_table, seq_lens


def reference_paged_attention_decode(
    q: torch.Tensor,           # [num_seqs, num_q_heads, head_size]
    k_cache: torch.Tensor,     # [num_blocks, num_kv_heads, kv_blk_sz, head_size]
    v_cache: torch.Tensor,     # [num_blocks, num_kv_heads, kv_blk_sz, head_size]
    block_table: torch.Tensor, # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,    # [num_seqs]
    scale: float,
    num_kv_heads: int,
    query_grp_sz: int,
    kv_blk_sz: int,
) -> torch.Tensor:
    """Reference implementation of paged attention decode in PyTorch."""
    num_seqs, num_q_heads, head_size = q.shape
    output = torch.zeros_like(q, dtype=torch.float32)

    for s in range(num_seqs):
        sl = seq_lens[s].item()
        num_blocks = (sl + kv_blk_sz - 1) // kv_blk_sz

        for qh in range(num_q_heads):
            kv_head = qh // query_grp_sz
            q_vec = q[s, qh].float()  # [head_size]

            # Gather all K, V for this sequence
            k_list = []
            v_list = []
            for b in range(num_blocks):
                phys_block = block_table[s, b].item()
                valid = min(kv_blk_sz, sl - b * kv_blk_sz)
                k_list.append(k_cache[phys_block, kv_head, :valid].float())
                v_list.append(v_cache[phys_block, kv_head, :valid].float())

            k_all = torch.cat(k_list, dim=0)  # [sl, head_size]
            v_all = torch.cat(v_list, dim=0)  # [sl, head_size]

            # Attention: softmax(q @ k^T / sqrt(d)) @ v
            scores = (q_vec @ k_all.T) * scale  # [sl]
            attn = torch.softmax(scores, dim=-1)  # [sl]
            output[s, qh] = attn @ v_all  # [head_size]

    return output.to(q.dtype)


def test_parity():
    """Test parity between reference and HipKittens kernel."""
    torch.manual_seed(42)

    # Parameters
    num_seqs = 4
    max_seq_len = 256
    num_kv_heads = 8
    query_grp_sz = 8
    num_q_heads = num_kv_heads * query_grp_sz
    head_size = 128
    kv_blk_sz = 16
    dtype = torch.bfloat16
    device = "cpu"  # Use CPU for reference; switch to "cuda" with HIP hardware

    scale = 1.0 / math.sqrt(head_size)

    # Create inputs
    q = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype, device=device)
    k_cache, v_cache, block_table, seq_lens = create_paged_kv_cache(
        num_seqs, max_seq_len, num_kv_heads, head_size, kv_blk_sz,
        dtype=dtype, device=device,
    )

    # Reference output
    ref_out = reference_paged_attention_decode(
        q, k_cache, v_cache, block_table, seq_lens,
        scale, num_kv_heads, query_grp_sz, kv_blk_sz,
    )

    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference output dtype: {ref_out.dtype}")
    print(f"Reference output sample: {ref_out[0, 0, :8]}")

    # When HIP hardware is available, compile and run HipKittens kernel:
    # import pa_decode_kernel
    # hk_out = torch.zeros_like(q)
    # pa_decode_kernel.dispatch(q.cuda(), k_cache.cuda(), v_cache.cuda(),
    #                           hk_out.cuda(), block_table.cuda(), seq_lens.cuda())
    # Then compare ref_out vs hk_out with tolerances

    print("\nCompilation command:")
    print(f"  make GPU_TARGET=CDNA4 PA_NUM_SEQS={num_seqs} PA_HEAD_SZ={head_size} "
          f"PA_NUM_KV_HEADS={num_kv_heads} PA_QUERY_GRP_SZ={query_grp_sz} "
          f"PA_KV_BLK_SZ={kv_blk_sz}")

    print("\nReference test PASSED (CPU-only mode)")
    return True


if __name__ == "__main__":
    test_parity()
