"""
Parity test for Unified Attention HipKittens kernel.
Tests the unified decode+prefill attention with paged KV cache.

The Triton kernel (kernel_unified_attention_2d) handles:
- GQA with interleaved query heads
- Paged KV cache via block_tables
- Causal masking
- Optional: sliding window, alibi slopes, softcap, FP8

Usage:
    python test_parity.py                 # Full test (requires HIP GPU)
    python test_parity.py --compile-only  # Just print compile command
    python test_parity.py --shapes-only   # Just print expected shapes
"""

import argparse
import math
import sys


def get_test_config():
    return {
        "num_seqs": 4,
        "query_lens": [1, 1, 16, 32],    # decode (1 token) + prefill (multi-token)
        "context_lens": [512, 256, 128, 64],
        "H": 32,
        "H_KV": 8,
        "D": 128,
        "block_size": 16,  # KV cache page size
        "dtype": "bfloat16",
    }


def print_shapes(cfg):
    total_queries = sum(cfg["query_lens"])
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    bs = cfg["block_size"]
    total_seq = sum(cl + ql for cl, ql in zip(cfg["context_lens"], cfg["query_lens"]))
    max_blocks = max((cl + ql + bs - 1) // bs for cl, ql in zip(cfg["context_lens"], cfg["query_lens"]))
    num_total_blocks = sum((cl + ql + bs - 1) // bs for cl, ql in zip(cfg["context_lens"], cfg["query_lens"]))
    print("Expected tensor shapes and dtypes:")
    print(f"  Q:             ({total_queries}, {H}, {D})         bf16")
    print(f"  O:             ({total_queries}, {H}, {D})         bf16")
    print(f"  K_cache:       ({num_total_blocks}, {bs}, {H_KV}, {D})  bf16")
    print(f"  V_cache:       ({num_total_blocks}, {bs}, {H_KV}, {D})  bf16")
    print(f"  block_tables:  ({cfg['num_seqs']}, {max_blocks})   int32")
    print(f"  seq_lens:      ({cfg['num_seqs']},)               int32")
    print(f"  query_start:   ({cfg['num_seqs']+1},)             int32")
    print(f"  Scale: {1.0 / math.sqrt(D):.6f}")


def print_compile_cmd(cfg):
    H, H_KV, D, bs = cfg["H"], cfg["H_KV"], cfg["D"], cfg["block_size"]
    print("Compile command:")
    print(f"  make NUM_Q_HEADS={H} NUM_KV_HEADS={H_KV} HEAD_SIZE={D} BLOCK_SIZE={bs}")


def robustness_check(ref, pred):
    import torch
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = diff > (0.001 + 0.05 * denom)
    error_count = mask.sum().item()
    numel = ref.numel()
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), pred.flatten(), dim=0
    ).item()
    return {
        "max_abs": diff.max().item(),
        "rel_error": error_count / numel,
        "l2_error": l2_error,
        "cosine_sim": cos,
        "errors": error_count,
        "total": numel,
    }


def run_parity_test():
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    cfg = get_test_config()
    H, H_KV, D = cfg["H"], cfg["H_KV"], cfg["D"]
    bs = cfg["block_size"]
    group_size = H // H_KV

    torch.manual_seed(42)

    total_queries = sum(cfg["query_lens"])

    # Create query tensor
    q = torch.randn(total_queries, H, D, dtype=torch.bfloat16, device="cuda")

    # Create paged KV cache
    num_blocks_per_seq = [(cl + ql + bs - 1) // bs
                          for cl, ql in zip(cfg["context_lens"], cfg["query_lens"])]
    total_blocks = sum(num_blocks_per_seq)

    k_cache = torch.randn(total_blocks, bs, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(total_blocks, bs, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # Build block tables (identity mapping for simplicity)
    max_blocks = max(num_blocks_per_seq)
    block_tables = torch.zeros(cfg["num_seqs"], max_blocks, dtype=torch.int32, device="cuda")
    block_offset = 0
    for i, nb in enumerate(num_blocks_per_seq):
        for j in range(nb):
            block_tables[i, j] = block_offset + j
        block_offset += nb

    seq_lens = torch.tensor(
        [cl + ql for cl, ql in zip(cfg["context_lens"], cfg["query_lens"])],
        dtype=torch.int32, device="cuda"
    )

    # Build query_start_len
    starts = [0]
    for ql in cfg["query_lens"]:
        starts.append(starts[-1] + ql)
    query_start_len = torch.tensor(starts, dtype=torch.int32, device="cuda")

    # PyTorch reference
    ref_out = torch.zeros(total_queries, H, D, dtype=torch.float32, device="cuda")
    q_off = 0
    blk_off = 0
    for seq in range(cfg["num_seqs"]):
        ql = cfg["query_lens"][seq]
        cl = cfg["context_lens"][seq]
        sl = cl + ql
        nb = num_blocks_per_seq[seq]

        # Reconstruct flat KV from paged cache
        k_flat = k_cache[blk_off:blk_off+nb].reshape(-1, H_KV, D)[:sl]
        v_flat = v_cache[blk_off:blk_off+nb].reshape(-1, H_KV, D)[:sl]

        q_b = q[q_off:q_off+ql].unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k_flat.unsqueeze(0).permute(0, 2, 1, 3)
        v_b = v_flat.unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k_b.repeat_interleave(group_size, dim=1)
        v_b = v_b.repeat_interleave(group_size, dim=1)

        # Causal: each query at position q_pos (within [cl, sl)) can see [0, q_pos]
        attn_mask = torch.zeros(ql, sl, dtype=torch.bool, device="cuda")
        for i in range(ql):
            attn_mask[i, :cl + i + 1] = True
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(-1, H, -1, -1)

        o_b = scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask)
        ref_out[q_off:q_off+ql] = o_b.squeeze(0).permute(1, 0, 2).float()

        q_off += ql
        blk_off += nb

    ref_out = ref_out.to(torch.bfloat16)

    try:
        import tk_unified_attn
    except ImportError:
        print("FAIL: tk_unified_attn not compiled. Run 'make' first.")
        print_compile_cmd(cfg)
        return False

    out = torch.zeros(total_queries, H, D, dtype=torch.bfloat16, device="cuda")
    tk_unified_attn.dispatch(q, out, k_cache, v_cache,
                             block_tables, seq_lens, query_start_len)
    torch.cuda.synchronize()

    result = robustness_check(ref_out, out)
    passed = result["cosine_sim"] > 0.999 and result["rel_error"] < 0.01

    print(f"Unified Attention Parity Test:")
    print(f"  query_lens={cfg['query_lens']} context_lens={cfg['context_lens']}")
    print(f"  max_abs_diff:  {result['max_abs']:.6f}")
    print(f"  rel_error:     {result['rel_error']:.6f}")
    print(f"  l2_error:      {result['l2_error']:.6f}")
    print(f"  cosine_sim:    {result['cosine_sim']:.6f}")
    print(f"  errors:        {result['errors']}/{result['total']}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Unified Attention parity test")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--shapes-only", action="store_true")
    args = parser.parse_args()

    cfg = get_test_config()
    if args.compile_only:
        print_compile_cmd(cfg)
        return
    if args.shapes_only:
        print_shapes(cfg)
        return

    passed = run_parity_test()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
