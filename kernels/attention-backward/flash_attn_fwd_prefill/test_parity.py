"""
Flash Attention Forward Prefill - Parity Test

Tests the HipKittens flash attention forward prefill kernel against a
PyTorch reference implementation (scaled_dot_product_attention).

Usage:
    # Full hardware test (requires AMD GPU with ROCm):
    python test_parity.py

    # With custom parameters:
    python test_parity.py --batch 4 --seqlen 1024 --nheads 32 --nheads_kv 8 --dim 128

    # Non-causal:
    python test_parity.py --no-causal

Expected tensor shapes:
    Q:   [batch, seqlen, num_heads_q, head_dim]   bf16
    K:   [batch, seqlen, num_heads_kv, head_dim]   bf16
    V:   [batch, seqlen, num_heads_kv, head_dim]   bf16
    O:   [batch, seqlen, num_heads_q, head_dim]    bf16
    LSE: [batch, num_heads_q, 1, seqlen]           float32

Compilation command (for reference):
    make ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=1024 ATTN_D=128 IS_CAUSAL=1
"""

import argparse
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Flash Attention Forward Prefill Parity Test")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seqlen", type=int, default=1024, help="Sequence length")
    parser.add_argument("--nheads", type=int, default=32, help="Number of Q heads")
    parser.add_argument("--nheads_kv", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--no-causal", action="store_true", help="Disable causal masking")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation step")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance")
    return parser.parse_args()


def pytorch_flash_attention_ref(q, k, v, causal=True, sm_scale=None):
    """
    Reference flash attention implementation using PyTorch.

    Args:
        q: [batch, seqlen_q, nheads_q, dim] bf16
        k: [batch, seqlen_k, nheads_kv, dim] bf16
        v: [batch, seqlen_k, nheads_kv, dim] bf16
        causal: whether to apply causal masking
        sm_scale: softmax scale (1/sqrt(dim) if None)

    Returns:
        o: [batch, seqlen_q, nheads_q, dim] bf16
        lse: [batch, nheads_q, seqlen_q] float32
    """
    batch, seqlen_q, nheads_q, dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    group_size = nheads_q // nheads_kv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim)

    # Expand K and V for GQA: repeat KV heads to match Q heads
    if group_size > 1:
        k = k.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(
            batch, seqlen_k, nheads_q, dim
        )
        v = v.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(
            batch, seqlen_k, nheads_q, dim
        )

    # Transpose to [batch, nheads, seqlen, dim] for matmul
    q_t = q.transpose(1, 2).float()  # [B, H, Sq, D]
    k_t = k.transpose(1, 2).float()  # [B, H, Sk, D]
    v_t = v.transpose(1, 2).float()  # [B, H, Sk, D]

    # Compute attention scores
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale  # [B, H, Sq, Sk]

    # Apply causal mask
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Compute LSE for numerical stability check
    max_scores = attn_scores.max(dim=-1, keepdim=True).values
    # Handle -inf rows (all masked)
    max_scores = torch.where(
        max_scores == float("-inf"),
        torch.zeros_like(max_scores),
        max_scores,
    )
    shifted = attn_scores - max_scores
    exp_shifted = torch.exp(shifted)
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))  # [B, H, Sq]

    # Compute softmax and output
    attn_weights = exp_shifted / sum_exp
    o = torch.matmul(attn_weights, v_t)  # [B, H, Sq, D]

    # Transpose back to [batch, seqlen, nheads, dim]
    o = o.transpose(1, 2).to(q.dtype)

    return o, lse


def robustness_check(ref, pred, atol=1e-2, rtol=5e-2):
    """Compute various error metrics between reference and prediction."""
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)
    mask = diff > (atol + rtol * denom)
    error_count = mask.sum().item()
    numel = ref_f.numel()
    rel_error = error_count / numel if numel > 0 else 0.0
    l2_error = (diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().clamp_min(1e-12).sqrt()).item()
    cos = F.cosine_similarity(ref_f.flatten(), pred_f.flatten(), dim=0).item()
    max_abs_err = diff.max().item()
    return {
        "max_abs_error": max_abs_err,
        "l2_error": l2_error,
        "cosine_similarity": cos,
        "error_count": error_count,
        "total_elements": numel,
        "error_fraction": rel_error,
    }


def compile_kernel(batch, seqlen, nheads, nheads_kv, dim, causal):
    """Compile the HipKittens kernel with the given parameters."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    is_causal = 1 if causal else 0

    cmd = [
        "make", "-C", kernel_dir,
        f"ATTN_B={batch}",
        f"ATTN_H={nheads}",
        f"ATTN_H_KV={nheads_kv}",
        f"ATTN_N={seqlen}",
        f"ATTN_D={dim}",
        f"IS_CAUSAL={is_causal}",
    ]

    print(f"Compiling kernel with: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Compilation failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    print("Compilation successful.")
    return True


def test_kernel_parity(batch, seqlen, nheads, nheads_kv, dim, causal, atol, rtol):
    """Run the HipKittens kernel and compare against PyTorch reference."""
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Create input tensors
    q = torch.randn(batch, seqlen, nheads, dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)

    # Allocate output tensors
    o_hk = torch.zeros(batch, seqlen, nheads, dim, dtype=dtype, device=device)
    lse_hk = torch.zeros(batch, nheads, 1, seqlen, dtype=torch.float32, device=device)

    # ---- PyTorch reference ----
    print("\nComputing PyTorch reference...")
    o_ref, lse_ref = pytorch_flash_attention_ref(q, k, v, causal=causal)
    print(f"  O shape: {o_ref.shape}, dtype: {o_ref.dtype}")
    print(f"  LSE shape: {lse_ref.shape}, dtype: {lse_ref.dtype}")

    # ---- HipKittens kernel ----
    print("\nRunning HipKittens kernel...")
    try:
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, kernel_dir)
        import flash_attn_fwd_prefill as hk_module

        hk_module.dispatch(q, k, v, o_hk, lse_hk)
        torch.cuda.synchronize()
        print(f"  O shape: {o_hk.shape}, dtype: {o_hk.dtype}")
        print(f"  LSE shape: {lse_hk.shape}, dtype: {lse_hk.dtype}")
    except ImportError as e:
        print(f"  Could not import kernel module: {e}")
        print("  Ensure the kernel is compiled (run 'make' first)")
        print("\n--- Reference-only results ---")
        print(f"  O[0,0,:4,0]: {o_ref[0, 0, :4, 0]}")
        print(f"  LSE[0,0,:4]: {lse_ref[0, 0, :4]}")
        return False
    except Exception as e:
        print(f"  Kernel execution failed: {e}")
        return False

    # ---- Compare outputs ----
    print("\n" + "=" * 60)
    print("PARITY CHECK")
    print("=" * 60)

    # Compare O
    print("\nOutput O comparison:")
    o_metrics = robustness_check(o_ref, o_hk, atol=atol, rtol=rtol)
    for key, val in o_metrics.items():
        print(f"  {key}: {val}")

    # Compare LSE
    # Reference LSE shape: [batch, nheads, seqlen]
    # HK LSE shape: [batch, nheads, 1, seqlen]
    lse_ref_compare = lse_ref.unsqueeze(2)  # -> [batch, nheads, 1, seqlen]
    print("\nLSE comparison:")
    lse_metrics = robustness_check(lse_ref_compare, lse_hk, atol=atol, rtol=rtol)
    for key, val in lse_metrics.items():
        print(f"  {key}: {val}")

    # ---- Print sample values ----
    num_print = min(8, seqlen)
    print(f"\nSample values (first {num_print} elements):")
    print(f"  O ref [0,0,:,0]:  {o_ref[0, 0, :num_print, 0]}")
    print(f"  O hk  [0,0,:,0]:  {o_hk[0, 0, :num_print, 0]}")
    print(f"  LSE ref [0,0,:]:  {lse_ref[0, 0, :num_print]}")
    print(f"  LSE hk  [0,0,0,:]: {lse_hk[0, 0, 0, :num_print]}")

    # ---- Pass/Fail ----
    o_pass = o_metrics["cosine_similarity"] > 0.99 and o_metrics["error_fraction"] < 0.05
    lse_pass = lse_metrics["cosine_similarity"] > 0.99 and lse_metrics["error_fraction"] < 0.05

    print("\n" + "=" * 60)
    print(f"O:   {'PASS' if o_pass else 'FAIL'}  (cosine={o_metrics['cosine_similarity']:.6f}, "
          f"err_frac={o_metrics['error_fraction']:.4f})")
    print(f"LSE: {'PASS' if lse_pass else 'FAIL'}  (cosine={lse_metrics['cosine_similarity']:.6f}, "
          f"err_frac={lse_metrics['error_fraction']:.4f})")

    overall = o_pass and lse_pass
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    return overall


def benchmark_kernel(batch, seqlen, nheads, nheads_kv, dim, causal,
                     num_warmup=100, num_iters=50):
    """Benchmark the HipKittens kernel."""
    device = "cuda"
    dtype = torch.bfloat16

    try:
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, kernel_dir)
        import flash_attn_fwd_prefill as hk_module
    except ImportError:
        print("Kernel not compiled, skipping benchmark.")
        return

    # Allocate
    o = torch.zeros(batch, seqlen, nheads, dim, dtype=dtype, device=device)
    lse = torch.zeros(batch, nheads, 1, seqlen, dtype=torch.float32, device=device)

    # Warmup
    print(f"\nBenchmark: B={batch}, N={seqlen}, H={nheads}, H_KV={nheads_kv}, D={dim}, causal={causal}")
    for _ in range(num_warmup):
        q = torch.randn(batch, seqlen, nheads, dim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)
        hk_module.dispatch(q, k, v, o, lse)

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(num_iters):
        q = torch.randn(batch, seqlen, nheads, dim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads_kv, dim, dtype=dtype, device=device)
        torch.cuda.synchronize()
        start_event.record()
        hk_module.dispatch(q, k, v, o, lse)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    avg_ms = sum(timings) / len(timings)
    min_ms = min(timings)
    max_ms = max(timings)

    # Compute TFLOPS
    flops = 4 * batch * seqlen * seqlen * nheads * dim
    if causal:
        flops //= 2
    tflops = (flops / 1e12) / (avg_ms / 1e3)

    print(f"  Average: {avg_ms:.4f} ms")
    print(f"  Min:     {min_ms:.4f} ms")
    print(f"  Max:     {max_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.2f}")


def print_compilation_command(batch, seqlen, nheads, nheads_kv, dim, causal):
    """Print the expected compilation command for reference."""
    is_causal = 1 if causal else 0
    print("\nExpected compilation command:")
    print(f"  make ATTN_B={batch} ATTN_H={nheads} ATTN_H_KV={nheads_kv} "
          f"ATTN_N={seqlen} ATTN_D={dim} IS_CAUSAL={is_causal}")
    print()
    print("Expected hipcc invocation:")
    print(f"  hipcc kernel.cpp -DKITTENS_CDNA4 --offload-arch=gfx950 "
          f"-DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
          f"-std=c++20 -w -shared -fPIC "
          f"-DATTN_B={batch} -DATTN_H={nheads} -DATTN_H_KV={nheads_kv} "
          f"-DATTN_N={seqlen} -DATTN_D={dim} "
          f"-DIS_CAUSAL={'true' if causal else 'false'} "
          f"-I$THUNDERKITTENS_ROOT/include "
          f"$(python3 -m pybind11 --includes) "
          f"-o flash_attn_fwd_prefill$(python3-config --extension-suffix)")


def main():
    args = parse_args()
    causal = not args.no_causal

    print("=" * 60)
    print("Flash Attention Forward Prefill - HipKittens Parity Test")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Batch:      {args.batch}")
    print(f"  Seqlen:     {args.seqlen}")
    print(f"  Heads (Q):  {args.nheads}")
    print(f"  Heads (KV): {args.nheads_kv}")
    print(f"  Dim:        {args.dim}")
    print(f"  Causal:     {causal}")
    print(f"  GQA ratio:  {args.nheads // args.nheads_kv}x")

    # Print compilation info
    print_compilation_command(args.batch, args.seqlen, args.nheads, args.nheads_kv, args.dim, causal)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("\nNo CUDA device available. Running reference-only mode.")
        print("To run full parity test, ensure AMD GPU with ROCm is available.")

        # Still run the reference to show expected outputs
        dtype = torch.bfloat16
        device = "cpu"
        torch.manual_seed(42)
        q = torch.randn(args.batch, args.seqlen, args.nheads, args.dim, dtype=dtype, device=device)
        k = torch.randn(args.batch, args.seqlen, args.nheads_kv, args.dim, dtype=dtype, device=device)
        v = torch.randn(args.batch, args.seqlen, args.nheads_kv, args.dim, dtype=dtype, device=device)

        o_ref, lse_ref = pytorch_flash_attention_ref(q, k, v, causal=causal)
        num_print = min(4, args.seqlen)
        print(f"\nReference output shapes:")
        print(f"  O:   {o_ref.shape}")
        print(f"  LSE: {lse_ref.shape}")
        print(f"\nSample O[0,0,:{num_print},0]:  {o_ref[0, 0, :num_print, 0]}")
        print(f"Sample LSE[0,0,:{num_print}]: {lse_ref[0, 0, :num_print]}")
        return

    # Compile if needed
    if not args.skip_compile:
        success = compile_kernel(args.batch, args.seqlen, args.nheads, args.nheads_kv, args.dim, causal)
        if not success:
            print("\nCompilation failed. Showing reference-only results.")

    # Run parity test
    passed = test_kernel_parity(
        args.batch, args.seqlen, args.nheads, args.nheads_kv, args.dim,
        causal, args.atol, args.rtol,
    )

    # Run benchmark
    if passed:
        benchmark_kernel(args.batch, args.seqlen, args.nheads, args.nheads_kv, args.dim, causal)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
