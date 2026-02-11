#!/usr/bin/env python3
"""
Parity test for Lean Attention with Paged KV Cache (HipKittens port).

This test:
1. Creates random Q, paged K/V cache, and a page table
2. Computes reference attention output with PyTorch
3. Optionally compiles and runs the HipKittens kernel
4. Compares outputs within tolerance

Tensor shapes:
  Q:          [batch, seqlen_q, num_heads, head_dim]       bf16
  K_cache:    [num_pages, page_size, num_kv_heads, head_dim] bf16
  V_cache:    [num_pages, page_size, num_kv_heads, head_dim] bf16
  page_table: [batch, max_num_pages]                        int32
  O:          [batch, seqlen_q, num_heads, head_dim]        bf16
  L:          [batch, num_heads, 1, seqlen_q]               float32

Usage:
  python test_parity.py                  # Reference-only mode (no GPU)
  python test_parity.py --run-kernel     # Full parity test (requires HIP GPU)
  python test_parity.py --compile-only   # Just print compilation commands
"""

import argparse
import math
import subprocess
import sys
import os

import torch
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

BATCH       = 4
SEQLEN_Q    = 64        # query sequence length
SEQLEN_KV   = 1024      # KV sequence length (total)
NUM_HEADS   = 32         # number of query heads
NUM_KV_HEADS = 8         # number of KV heads
HEAD_DIM    = 128        # head dimension
PAGE_SIZE   = 64         # paged cache page size
BLOCK_M     = 64         # Q block size
IS_CAUSAL   = False      # causal masking (typically False for decode)
DTYPE       = torch.bfloat16

GQA_GROUPS = NUM_HEADS // NUM_KV_HEADS
NUM_KV_PAGES = (SEQLEN_KV + PAGE_SIZE - 1) // PAGE_SIZE

# softmax scale in log2 space: (1/sqrt(D)) * log2(e)
SM_SCALE = (1.0 / math.sqrt(HEAD_DIM)) * math.log2(math.e)


# ============================================================================
# Reference implementation
# ============================================================================

def create_paged_kv_cache(batch, seqlen_kv, num_kv_heads, head_dim, page_size, dtype):
    """Create a paged KV cache with random data and a simple page table.

    Returns:
        k_cache: [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: [num_pages, page_size, num_kv_heads, head_dim]
        page_table: [batch, num_pages_per_seq] int32
        k_flat: [batch, seqlen_kv, num_kv_heads, head_dim] (linearized K for reference)
        v_flat: [batch, seqlen_kv, num_kv_heads, head_dim] (linearized V for reference)
    """
    num_pages_per_seq = (seqlen_kv + page_size - 1) // page_size
    total_pages = batch * num_pages_per_seq

    # Allocate cache pages
    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=dtype)
    v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=dtype)

    # Create page table: batch b, logical page p -> physical page b * num_pages_per_seq + p
    # (simple identity mapping, but shuffled to test indirection)
    page_table = torch.zeros(batch, num_pages_per_seq, dtype=torch.int32)
    for b in range(batch):
        # Shuffle pages for this batch to test indirection
        perm = torch.randperm(num_pages_per_seq)
        physical_pages = b * num_pages_per_seq + perm
        page_table[b] = physical_pages.to(torch.int32)

    # Build linearized K, V for reference computation
    k_flat = torch.zeros(batch, seqlen_kv, num_kv_heads, head_dim, dtype=dtype)
    v_flat = torch.zeros(batch, seqlen_kv, num_kv_heads, head_dim, dtype=dtype)

    for b in range(batch):
        for p in range(num_pages_per_seq):
            phys_page = page_table[b, p].item()
            start = p * page_size
            end = min(start + page_size, seqlen_kv)
            length = end - start
            k_flat[b, start:end] = k_cache[phys_page, :length]
            v_flat[b, start:end] = v_cache[phys_page, :length]

    return k_cache, v_cache, page_table, k_flat, v_flat


def reference_attention(q, k, v, is_causal=False, sm_scale_natural=None):
    """Compute reference attention output using PyTorch.

    Args:
        q: [batch, seqlen_q, num_heads, head_dim]
        k: [batch, seqlen_kv, num_kv_heads, head_dim]
        v: [batch, seqlen_kv, num_kv_heads, head_dim]
        is_causal: whether to apply causal masking
        sm_scale_natural: 1/sqrt(head_dim) in natural scale

    Returns:
        o: [batch, seqlen_q, num_heads, head_dim] bf16
        lse: [batch, num_heads, seqlen_q] float32 (logsumexp)
    """
    batch, sq, nh, d = q.shape
    _, skv, nkv, _ = k.shape
    groups = nh // nkv

    if sm_scale_natural is None:
        sm_scale_natural = 1.0 / math.sqrt(d)

    # Expand KV heads for GQA: [batch, seqlen_kv, num_heads, head_dim]
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, groups, -1).reshape(batch, skv, nh, d)
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, groups, -1).reshape(batch, skv, nh, d)

    # Transpose to [batch, num_heads, seqlen, head_dim]
    q_t = q.transpose(1, 2).float()
    k_t = k_expanded.transpose(1, 2).float()
    v_t = v_expanded.transpose(1, 2).float()

    # Attention scores
    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale_natural  # [B, H, sq, skv]

    # Causal mask
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(sq, skv, dtype=torch.bool, device=q.device),
            diagonal=skv - sq + 1
        )
        attn.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Logsumexp for comparison
    lse = torch.logsumexp(attn, dim=-1)  # [B, H, sq]

    # Softmax
    attn = F.softmax(attn, dim=-1)

    # Weighted sum
    o = torch.matmul(attn, v_t)  # [B, H, sq, d]
    o = o.transpose(1, 2)  # [B, sq, H, d]

    return o.to(DTYPE), lse


def robustness_check(ref, pred, name=""):
    """Compare reference and predicted tensors."""
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)

    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    # Relative error check: |diff| > atol + rtol * |ref|
    atol, rtol = 0.01, 0.05
    mask = diff > (atol + rtol * denom)
    error_count = mask.sum().item()
    total = ref.numel()
    error_rate = error_count / total if total > 0 else 0

    # L2 relative error
    l2_err = (diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().clamp_min(1e-12).sqrt()).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        ref_f.flatten().unsqueeze(0),
        pred_f.flatten().unsqueeze(0)
    ).item()

    print(f"  {name}:")
    print(f"    max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}")
    print(f"    l2_relative={l2_err:.6f}, cosine_sim={cos_sim:.8f}")
    print(f"    error_pixels={error_count}/{total} ({100*error_rate:.4f}%)")

    passed = cos_sim > 0.99 and error_rate < 0.05
    return passed


# ============================================================================
# Compilation helper
# ============================================================================

def get_compile_command():
    """Return the hipcc compilation command."""
    return (
        f"cd {os.path.dirname(os.path.abspath(__file__))} && "
        f"make ATTN_B={BATCH} ATTN_H={NUM_HEADS} ATTN_H_KV={NUM_KV_HEADS} ATTN_D={HEAD_DIM}"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parity test for paged attention HipKittens kernel")
    parser.add_argument("--run-kernel", action="store_true",
                        help="Compile and run the HipKittens kernel (requires HIP GPU)")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only print compilation commands")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seqlen-q", type=int, default=SEQLEN_Q)
    parser.add_argument("--seqlen-kv", type=int, default=SEQLEN_KV)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--causal", action="store_true", default=IS_CAUSAL)
    args = parser.parse_args()

    # Update globals from args
    global BATCH, SEQLEN_Q, SEQLEN_KV, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, IS_CAUSAL
    BATCH = args.batch
    SEQLEN_Q = args.seqlen_q
    SEQLEN_KV = args.seqlen_kv
    NUM_HEADS = args.num_heads
    NUM_KV_HEADS = args.num_kv_heads
    HEAD_DIM = args.head_dim
    IS_CAUSAL = args.causal

    num_kv_pages = (SEQLEN_KV + PAGE_SIZE - 1) // PAGE_SIZE
    sm_scale_natural = 1.0 / math.sqrt(HEAD_DIM)

    print("=" * 70)
    print("Lean Attention with Paged KV Cache - Parity Test")
    print("=" * 70)
    print(f"  batch={BATCH}, seqlen_q={SEQLEN_Q}, seqlen_kv={SEQLEN_KV}")
    print(f"  num_heads={NUM_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  page_size={PAGE_SIZE}, num_kv_pages={num_kv_pages}")
    print(f"  is_causal={IS_CAUSAL}")
    print(f"  GQA groups={NUM_HEADS // NUM_KV_HEADS}")
    print(f"  sm_scale (natural)={sm_scale_natural:.6f}")
    print(f"  sm_scale (log2)={sm_scale_natural * math.log2(math.e):.6f}")
    print()

    if args.compile_only:
        print("Compilation command:")
        print(f"  {get_compile_command()}")
        print()
        print("Expected tensor shapes for kernel dispatch:")
        print(f"  Q:          [{BATCH}, {SEQLEN_Q}, {NUM_HEADS}, {HEAD_DIM}] bf16")
        print(f"  K_cache:    [{BATCH * num_kv_pages}, {PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}] bf16")
        print(f"  V_cache:    [{BATCH * num_kv_pages}, {PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}] bf16")
        print(f"  O:          [{BATCH}, {SEQLEN_Q}, {NUM_HEADS}, {HEAD_DIM}] bf16")
        print(f"  L:          [{BATCH}, {NUM_HEADS}, 1, {SEQLEN_Q}] float32")
        print(f"  page_table: [{BATCH}, {num_kv_pages}, 1, 1] int32")
        return

    # ----------------------------------------------------------------
    # Generate test data
    # ----------------------------------------------------------------
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() and args.run_kernel else "cpu"

    print(f"Device: {device}")
    print()

    Q = torch.randn(BATCH, SEQLEN_Q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k_cache, v_cache, page_table, k_flat, v_flat = create_paged_kv_cache(
        BATCH, SEQLEN_KV, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, DTYPE
    )

    # Move to device
    k_cache = k_cache.to(device)
    v_cache = v_cache.to(device)
    page_table = page_table.to(device)
    k_flat = k_flat.to(device)
    v_flat = v_flat.to(device)

    # ----------------------------------------------------------------
    # Reference computation
    # ----------------------------------------------------------------
    print("Computing reference attention (PyTorch)...")
    o_ref, lse_ref = reference_attention(Q, k_flat, v_flat, IS_CAUSAL, sm_scale_natural)
    print(f"  O shape: {o_ref.shape}, dtype: {o_ref.dtype}")
    print(f"  LSE shape: {lse_ref.shape}, dtype: {lse_ref.dtype}")
    print(f"  O range: [{o_ref.float().min().item():.4f}, {o_ref.float().max().item():.4f}]")
    print(f"  LSE range: [{lse_ref.min().item():.4f}, {lse_ref.max().item():.4f}]")
    print()

    # ----------------------------------------------------------------
    # HipKittens kernel (if requested)
    # ----------------------------------------------------------------
    if args.run_kernel:
        print("Compiling HipKittens kernel...")
        compile_cmd = get_compile_command()
        print(f"  {compile_cmd}")
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  COMPILE FAILED:")
            print(result.stderr)
            sys.exit(1)
        print("  Compilation succeeded.")
        print()

        # Import the compiled module
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import lean_atten_paged as hk_module

        # Prepare output tensors
        O_hk = torch.zeros_like(Q)
        q_tiles = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
        L_hk = torch.zeros(BATCH, NUM_HEADS, 1, SEQLEN_Q, dtype=torch.float32, device=device)

        # Reshape page_table for gl<int, -1, -1, -1, -1>: [batch, num_pages, 1, 1]
        pt_4d = page_table.unsqueeze(-1).unsqueeze(-1).contiguous()

        qk_scale = sm_scale_natural * math.log2(math.e)

        print("Running HipKittens kernel...")
        hk_module.dispatch(
            Q.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            O_hk.contiguous(),
            L_hk.contiguous(),
            pt_4d.contiguous(),
            SEQLEN_Q,
            SEQLEN_KV,
            num_kv_pages,
            qk_scale,
            1 if IS_CAUSAL else 0
        )
        torch.cuda.synchronize()
        print("  Kernel completed.")
        print()

        # Compare
        print("Parity check:")
        o_pass = robustness_check(o_ref, O_hk, "Output O")
        # For LSE, the kernel stores per-tile values; compare what's available
        # lse_ref: [batch, num_heads, seqlen_q]
        # L_hk: [batch, num_heads, 1, seqlen_q]
        lse_hk = L_hk.squeeze(2)  # [batch, num_heads, seqlen_q]
        l_pass = robustness_check(lse_ref, lse_hk, "Logsumexp L")

        print()
        if o_pass and l_pass:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
            if not o_pass:
                print("  Output O failed parity check")
            if not l_pass:
                print("  Logsumexp L failed parity check")
    else:
        print("Skipping kernel execution (use --run-kernel to test on GPU).")
        print()
        print("Reference output summary:")
        print(f"  O[0,0,0,:8] = {o_ref[0,0,0,:8].float().tolist()}")
        print(f"  LSE[0,0,:8] = {lse_ref[0,0,:8].tolist()}")
        print()
        print("To compile the HipKittens kernel:")
        print(f"  {get_compile_command()}")
        print()
        print("RESULT: REFERENCE_ONLY (no GPU comparison)")


if __name__ == "__main__":
    main()
