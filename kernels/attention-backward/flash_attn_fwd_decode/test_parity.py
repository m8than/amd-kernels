"""
Parity test for Flash Attention Forward Decode (HipKittens Split-K)

Tests the HipKittens split-K decode attention kernel against a PyTorch
reference implementation. The decode kernel handles the case where
seqlen_q is small (typically 1) while seqlen_k can be large.

Usage:
    # Run with default parameters (needs AMD GPU with HipKittens compiled)
    python test_parity.py

    # Run with custom parameters
    python test_parity.py --batch 4 --seqlen_q 1 --seqlen_k 2048 --heads_q 32 --heads_kv 8 --dim 128

    # Dry run (no GPU needed, just prints expected shapes and commands)
    python test_parity.py --dry-run
"""

import argparse
import math
import sys
import os

import torch
import torch.nn.functional as F


# ============================================================================
# PyTorch Reference: standard attention (no flash, for correctness checking)
# ============================================================================

def attention_ref(q, k, v, sm_scale, causal=False):
    """
    Reference attention computation in PyTorch.

    Args:
        q: [B, seqlen_q, H_q, D] bf16
        k: [B, seqlen_k, H_kv, D] bf16
        v: [B, seqlen_k, H_kv, D] bf16
        sm_scale: softmax scale (typically 1/sqrt(D))
        causal: whether to apply causal masking

    Returns:
        out: [B, seqlen_q, H_q, D] bf16
        lse: [B*H_q, seqlen_q] float32  (log-sum-exp)
    """
    B, Sq, Hq, D = q.shape
    _, Sk, Hkv, _ = k.shape
    group_size = Hq // Hkv

    # Expand K, V for GQA: repeat each KV head `group_size` times
    # k: [B, Sk, Hkv, D] -> [B, Sk, Hq, D]
    # v: [B, Sk, Hkv, D] -> [B, Sk, Hq, D]
    k_expanded = k[:, :, :, None, :].expand(B, Sk, Hkv, group_size, D).reshape(B, Sk, Hq, D)
    v_expanded = v[:, :, :, None, :].expand(B, Sk, Hkv, group_size, D).reshape(B, Sk, Hq, D)

    # Transpose to [B, H, S, D] for matmul
    q_t = q.permute(0, 2, 1, 3).float()  # [B, Hq, Sq, D]
    k_t = k_expanded.permute(0, 2, 1, 3).float()  # [B, Hq, Sk, D]
    v_t = v_expanded.permute(0, 2, 1, 3).float()  # [B, Hq, Sk, D]

    # QK^T
    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale  # [B, Hq, Sq, Sk]

    if causal:
        # Create causal mask: q_pos >= k_pos (adjusting for different seqlens)
        q_idx = torch.arange(Sq, device=q.device).unsqueeze(1)  # [Sq, 1]
        k_idx = torch.arange(Sk, device=q.device).unsqueeze(0)  # [1, Sk]
        col_offset = Sk - Sq
        mask = q_idx >= (k_idx - col_offset)  # [Sq, Sk]
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    m = attn.max(dim=-1, keepdim=True).values  # [B, H, Sq, 1]
    attn_shifted = attn - m
    p = torch.exp(attn_shifted)
    l = p.sum(dim=-1, keepdim=True)  # [B, H, Sq, 1]

    # Output
    out_t = torch.matmul(p / l, v_t)  # [B, Hq, Sq, D]
    out = out_t.permute(0, 2, 1, 3).to(torch.bfloat16)  # [B, Sq, Hq, D]

    # LSE: log(sum(exp(attn - max))) + max = log(l) + m
    lse = (torch.log(l.squeeze(-1)) + m.squeeze(-1))  # [B, Hq, Sq]
    lse = lse.reshape(B * Hq, Sq).float()

    return out, lse


# ============================================================================
# HipKittens Split-K: get_split_k heuristic (mirrors Triton reference)
# ============================================================================

def get_split_k(B, G, H, Mk):
    """Heuristic for the number of splits (from Triton reference)."""
    bh = max(B * H, 1)
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k


# ============================================================================
# Robustness checking utilities
# ============================================================================

def robustness_check(ref, pred, name=""):
    """Compare two tensors and report statistics."""
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)

    # Count elements exceeding tolerance
    mask = diff > (0.001 + 0.05 * denom)
    error_count = mask.sum().item()
    numel = ref_f.numel()
    rel_error = error_count / numel if numel > 0 else 0.0

    # L2 relative error
    ref_norm = ref_f.pow(2).sum().sqrt().item()
    l2_error = (diff.pow(2).sum().sqrt() / max(ref_norm, 1e-8)).item()

    # Cosine similarity
    cos = F.cosine_similarity(ref_f.flatten(), pred_f.flatten(), dim=0).item()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    return {
        "name": name,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rel_error_frac": rel_error,
        "l2_rel_error": l2_error,
        "cosine_sim": cos,
        "error_count": error_count,
        "total": numel,
    }


def print_check(stats):
    """Print robustness check results."""
    name = stats["name"]
    print(f"  {name}: max_abs={stats['max_abs_diff']:.6f}, "
          f"mean_abs={stats['mean_abs_diff']:.6f}, "
          f"l2_rel={stats['l2_rel_error']:.6f}, "
          f"cos={stats['cosine_sim']:.6f}, "
          f"errors={stats['error_count']}/{stats['total']} "
          f"({100*stats['rel_error_frac']:.4f}%)")


# ============================================================================
# Test runner
# ============================================================================

def run_test(batch, seqlen_q, seqlen_k, heads_q, heads_kv, dim, causal=False,
             num_splits=None, verbose=True):
    """
    Run parity test: HipKittens split-K decode vs PyTorch reference.

    Returns True if PASS, False if FAIL.
    """
    device = 'cuda'
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(dim)

    if num_splits is None:
        num_splits = get_split_k(batch, 1, heads_q, seqlen_k)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Test: B={batch}, Sq={seqlen_q}, Sk={seqlen_k}, "
              f"Hq={heads_q}, Hkv={heads_kv}, D={dim}, "
              f"causal={causal}, splits={num_splits}")
        print(f"{'='*70}")

    BLOCK_M = 64
    M_ceil = ((seqlen_q + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    bh = batch * heads_q

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch, seqlen_q, heads_q, dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen_k, heads_kv, dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen_k, heads_kv, dim, dtype=dtype, device=device)

    # Reference output
    out_ref, lse_ref = attention_ref(q, k, v, sm_scale, causal=causal)

    if verbose:
        print(f"  Reference output shape: {out_ref.shape}")
        print(f"  Reference LSE shape:    {lse_ref.shape}")
        print(f"  Reference output range: [{out_ref.float().min():.4f}, {out_ref.float().max():.4f}]")
        print(f"  Reference LSE range:    [{lse_ref.min():.4f}, {lse_ref.max():.4f}]")

    # Try to import and run HipKittens kernel
    try:
        import flash_attn_fwd_decode as hk

        # Allocate intermediate tensors
        o_partial = torch.zeros(bh, num_splits, M_ceil, dim,
                                dtype=torch.float32, device=device)
        metadata = torch.zeros(bh, 2, num_splits, M_ceil,
                               dtype=torch.float32, device=device)

        # Allocate output tensors
        out_hk = torch.zeros(batch, seqlen_q, heads_q, dim,
                             dtype=dtype, device=device)
        lse_hk = torch.zeros(bh, seqlen_q, 1, 1,
                             dtype=torch.float32, device=device)

        # Run splitk kernel
        hk.dispatch_splitk(q, k, v, o_partial, metadata,
                           heads_q, heads_kv,
                           seqlen_q, seqlen_k, num_splits, sm_scale)

        # Run reduce kernel
        hk.dispatch_reduce(o_partial, metadata, out_hk, lse_hk,
                           heads_q, seqlen_q, num_splits)

        torch.cuda.synchronize()

        # Compare outputs
        if verbose:
            print(f"\n  HipKittens output shape: {out_hk.shape}")
            print(f"  HipKittens LSE shape:    {lse_hk.shape}")
            print(f"  HipKittens output range: [{out_hk.float().min():.4f}, {out_hk.float().max():.4f}]")

        # Check O
        o_stats = robustness_check(out_ref, out_hk, name="O")
        print_check(o_stats)

        # Check LSE
        lse_hk_flat = lse_hk.squeeze(-1).squeeze(-1)  # [B*H, Sq]
        lse_stats = robustness_check(lse_ref, lse_hk_flat, name="LSE")
        print_check(lse_stats)

        # Determine PASS/FAIL
        # Use generous tolerances for bf16 attention
        o_pass = o_stats["cosine_sim"] > 0.99 and o_stats["l2_rel_error"] < 0.05
        lse_pass = lse_stats["cosine_sim"] > 0.99 and lse_stats["l2_rel_error"] < 0.05

        passed = o_pass and lse_pass
        status = "PASS" if passed else "FAIL"
        print(f"\n  Result: {status}")

        return passed

    except ImportError:
        print("\n  [SKIP] flash_attn_fwd_decode module not found.")
        print("  This is expected if HipKittens has not been compiled.")
        print("  Reference output computed successfully.")

        # Print compilation instructions
        print("\n  To compile:")
        print("    cd kernels/flash_attn_fwd_decode")
        print("    make")
        print("  Then re-run this test.")

        return None  # Skip, not fail


def run_benchmark(batch, seqlen_q, seqlen_k, heads_q, heads_kv, dim,
                  num_splits=None, num_warmup=50, num_iters=100):
    """
    Run performance benchmark of the HipKittens kernel.
    """
    device = 'cuda'
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(dim)

    if num_splits is None:
        num_splits = get_split_k(batch, 1, heads_q, seqlen_k)

    BLOCK_M = 64
    M_ceil = ((seqlen_q + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    bh = batch * heads_q

    print(f"\nBenchmark: B={batch}, Sq={seqlen_q}, Sk={seqlen_k}, "
          f"Hq={heads_q}, Hkv={heads_kv}, D={dim}, splits={num_splits}")

    try:
        import flash_attn_fwd_decode as hk

        q = torch.randn(batch, seqlen_q, heads_q, dim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen_k, heads_kv, dim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen_k, heads_kv, dim, dtype=dtype, device=device)

        o_partial = torch.zeros(bh, num_splits, M_ceil, dim,
                                dtype=torch.float32, device=device)
        metadata = torch.zeros(bh, 2, num_splits, M_ceil,
                               dtype=torch.float32, device=device)
        out_hk = torch.zeros(batch, seqlen_q, heads_q, dim,
                             dtype=dtype, device=device)
        lse_hk = torch.zeros(bh, seqlen_q, 1, 1,
                             dtype=torch.float32, device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(num_warmup):
            hk.dispatch_splitk(q, k, v, o_partial, metadata,
                               heads_q, heads_kv,
                               seqlen_q, seqlen_k, num_splits, sm_scale)
            hk.dispatch_reduce(o_partial, metadata, out_hk, lse_hk,
                               heads_q, seqlen_q, num_splits)

        # Timed runs
        timings = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start.record()
            hk.dispatch_splitk(q, k, v, o_partial, metadata,
                               heads_q, heads_kv,
                               seqlen_q, seqlen_k, num_splits, sm_scale)
            hk.dispatch_reduce(o_partial, metadata, out_hk, lse_hk,
                               heads_q, seqlen_q, num_splits)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

        avg_ms = sum(timings) / len(timings)
        # FLOPs for decode attention: 2 * B * Sq * Sk * H * D (for QK and PV)
        flops = 4 * batch * seqlen_q * seqlen_k * heads_q * dim
        tflops = flops / (avg_ms * 1e-3) / 1e12
        print(f"  Average time: {avg_ms:.4f} ms, {tflops:.2f} TFLOPS")

    except ImportError:
        print("  [SKIP] Module not compiled")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parity test for Flash Attention Forward Decode (HipKittens)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen_q", type=int, default=1,
                        help="Query sequence length (typically 1 for decode)")
    parser.add_argument("--seqlen_k", type=int, default=2048,
                        help="Key/Value sequence length")
    parser.add_argument("--heads_q", type=int, default=32,
                        help="Number of query heads")
    parser.add_argument("--heads_kv", type=int, default=8,
                        help="Number of KV heads (for GQA)")
    parser.add_argument("--dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal masking")
    parser.add_argument("--num_splits", type=int, default=None,
                        help="Number of splits (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print expected shapes without running GPU")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    args = parser.parse_args()

    if args.dry_run:
        B, Sq, Sk = args.batch, args.seqlen_q, args.seqlen_k
        Hq, Hkv, D = args.heads_q, args.heads_kv, args.dim
        BLOCK_M = 64
        M_ceil = ((Sq + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        num_splits = args.num_splits or get_split_k(B, 1, Hq, Sk)
        bh = B * Hq

        print("Flash Attention Forward Decode - Dry Run")
        print(f"  Parameters: B={B}, Sq={Sq}, Sk={Sk}, Hq={Hq}, Hkv={Hkv}, D={D}")
        print(f"  num_splits={num_splits}, M_ceil={M_ceil}")
        print()
        print("  Input tensors:")
        print(f"    Q:         [{B}, {Sq}, {Hq}, {D}] bf16")
        print(f"    K:         [{B}, {Sk}, {Hkv}, {D}] bf16")
        print(f"    V:         [{B}, {Sk}, {Hkv}, {D}] bf16")
        print()
        print("  Intermediate tensors:")
        print(f"    O_partial: [{bh}, {num_splits}, {M_ceil}, {D}] float32")
        print(f"    Metadata:  [{bh}, 2, {num_splits}, {M_ceil}] float32")
        print()
        print("  Output tensors:")
        print(f"    O:         [{B}, {Sq}, {Hq}, {D}] bf16")
        print(f"    LSE:       [{bh}, {Sq}, 1, 1] float32")
        print()
        print("  SplitK kernel grid:")
        grid_m = (Sq + BLOCK_M - 1) // BLOCK_M
        print(f"    ({grid_m}, {bh}, {num_splits})")
        print("  Reduce kernel grid:")
        print(f"    ({grid_m}, {bh}, 1)")
        print()
        print("  Compilation command:")
        print("    cd kernels/flash_attn_fwd_decode && make")
        return

    # Run correctness tests
    all_pass = True
    skipped = False

    # Test 1: Basic decode (seqlen_q=1)
    result = run_test(
        batch=args.batch, seqlen_q=1, seqlen_k=args.seqlen_k,
        heads_q=args.heads_q, heads_kv=args.heads_kv, dim=args.dim,
        causal=args.causal, num_splits=args.num_splits)
    if result is None:
        skipped = True
    elif not result:
        all_pass = False

    # Test 2: Small seqlen_q
    result = run_test(
        batch=args.batch, seqlen_q=4, seqlen_k=args.seqlen_k,
        heads_q=args.heads_q, heads_kv=args.heads_kv, dim=args.dim,
        causal=args.causal, num_splits=args.num_splits)
    if result is None:
        skipped = True
    elif not result:
        all_pass = False

    # Test 3: Larger seqlen_q (full block)
    result = run_test(
        batch=args.batch, seqlen_q=64, seqlen_k=args.seqlen_k,
        heads_q=args.heads_q, heads_kv=args.heads_kv, dim=args.dim,
        causal=args.causal, num_splits=args.num_splits)
    if result is None:
        skipped = True
    elif not result:
        all_pass = False

    # Test 4: MHA (no GQA)
    result = run_test(
        batch=2, seqlen_q=1, seqlen_k=1024,
        heads_q=16, heads_kv=16, dim=args.dim,
        causal=args.causal)
    if result is None:
        skipped = True
    elif not result:
        all_pass = False

    # Test 5: Large seqlen_k
    result = run_test(
        batch=2, seqlen_q=1, seqlen_k=8192,
        heads_q=args.heads_q, heads_kv=args.heads_kv, dim=args.dim,
        causal=args.causal)
    if result is None:
        skipped = True
    elif not result:
        all_pass = False

    if args.benchmark:
        run_benchmark(
            batch=args.batch, seqlen_q=args.seqlen_q, seqlen_k=args.seqlen_k,
            heads_q=args.heads_q, heads_kv=args.heads_kv, dim=args.dim,
            num_splits=args.num_splits)

    # Summary
    print(f"\n{'='*70}")
    if skipped:
        print("SUMMARY: SKIPPED (module not compiled)")
    elif all_pass:
        print("SUMMARY: ALL TESTS PASSED")
    else:
        print("SUMMARY: SOME TESTS FAILED")
    print(f"{'='*70}")

    if not skipped and not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
