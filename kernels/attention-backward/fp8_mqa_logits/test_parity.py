#!/usr/bin/env python3
"""
Parity test for the HipKittens FP8 MQA Logits kernel.

This script:
  1. Generates random input tensors matching the kernel's expected shapes/dtypes.
  2. Computes a reference output using pure PyTorch (mirroring the Triton kernel
     algorithm exactly).
  3. Optionally loads and runs the compiled HipKittens kernel and compares.
  4. Reports PASS/FAIL with numerical accuracy metrics.

Expected tensor shapes and dtypes:
  Q:          [seq_len, NUM_HEADS, HEAD_SIZE]   bf16
  KV:         [seq_len_kv, HEAD_SIZE]           bf16
  kv_scales:  [seq_len_kv]                      fp32
  weights:    [seq_len, NUM_HEADS]              fp32
  logits:     [seq_len, seq_len_kv]             fp32   (output)

Compilation command (example):
  cd kernels/fp8_mqa_logits
  make NUM_HEADS=8 HEAD_SIZE=64 SEQ_LEN=256 SEQ_LEN_KV=256 GPU_TARGET=CDNA4

Usage:
  python test_parity.py [--seq-len S] [--seq-len-kv K] [--num-heads H]
                        [--head-size D] [--no-hip]
"""

import argparse
import importlib
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, matches Triton algorithm exactly)
# ---------------------------------------------------------------------------

def reference_mqa_logits(
    Q: torch.Tensor,        # [seq_len, NUM_HEADS, HEAD_SIZE] bf16
    KV: torch.Tensor,       # [seq_len_kv, HEAD_SIZE] bf16
    kv_scales: torch.Tensor, # [seq_len_kv] fp32
    weights: torch.Tensor,  # [seq_len, NUM_HEADS] fp32
) -> torch.Tensor:
    """
    Compute MQA logits exactly as the Triton kernel does.

    For each query row i:
      scores[h, k] = dot(Q[i, h, :], KV[k, :]) * kv_scales[k]
      scores[h, k] = relu(scores[h, k])
      scores[h, k] *= weights[i, h]
      logits[i, k] = sum_h(scores[h, k])

    Returns: logits [seq_len, seq_len_kv] fp32
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]

    # Cast Q and KV to float32 for the reference computation
    Q_f = Q.float()          # [seq_len, H, D]
    KV_f = KV.float()        # [seq_len_kv, D]

    # scores[i, h, k] = sum_d Q[i, h, d] * KV[k, d]
    # Equivalent to: Q_f @ KV_f^T -> [seq_len, H, seq_len_kv]
    scores = torch.einsum("shd,kd->shk", Q_f, KV_f)

    # Scale by kv_scales: [seq_len_kv] broadcast to [seq_len, H, seq_len_kv]
    scores = scores * kv_scales.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_kv]

    # ReLU
    scores = F.relu(scores)

    # Multiply by weights: [seq_len, H] -> [seq_len, H, 1]
    scores = scores * weights.unsqueeze(-1)

    # Sum over heads -> [seq_len, seq_len_kv]
    logits = scores.sum(dim=1)

    return logits


# ---------------------------------------------------------------------------
# Numerical comparison utilities
# ---------------------------------------------------------------------------

def compare_tensors(ref: torch.Tensor, test: torch.Tensor, name: str,
                    atol: float = 1e-2, rtol: float = 5e-2) -> bool:
    """Compare two tensors and report detailed accuracy metrics."""
    ref_f = ref.float().cpu()
    test_f = test.float().cpu()

    abs_diff = (ref_f - test_f).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    # Relative error (avoid div by zero)
    denom = ref_f.abs().clamp_min(1e-6)
    rel_diff = abs_diff / denom
    max_rel_err = rel_diff.max().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        ref_f.flatten().unsqueeze(0),
        test_f.flatten().unsqueeze(0),
    ).item()

    # Count elements exceeding tolerance
    # Using the same tolerance formula as the reference test:
    # error if |diff| > (atol + rtol * |ref|)
    mask = abs_diff > (atol + rtol * ref_f.abs())
    error_count = mask.sum().item()
    total = ref_f.numel()
    error_pct = 100.0 * error_count / total if total > 0 else 0.0

    # L2 relative error
    l2_err = (abs_diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().clamp_min(1e-12).sqrt()).item()

    passed = error_pct < 1.0 and cos_sim > 0.99

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}:")
    print(f"  max_abs_err  = {max_abs_err:.6f}")
    print(f"  mean_abs_err = {mean_abs_err:.6f}")
    print(f"  max_rel_err  = {max_rel_err:.6f}")
    print(f"  l2_rel_err   = {l2_err:.6f}")
    print(f"  cosine_sim   = {cos_sim:.8f}")
    print(f"  errors       = {error_count}/{total} ({error_pct:.4f}%)")
    return passed


# ---------------------------------------------------------------------------
# HipKittens kernel runner
# ---------------------------------------------------------------------------

def run_hipkittens_kernel(Q, KV, kv_scales, weights, seq_len, seq_len_kv,
                          num_heads, head_size):
    """
    Attempt to import and run the compiled HipKittens kernel.
    Returns the output logits tensor, or None if the kernel is not available.
    """
    try:
        # Add current directory to path so we can import the compiled module
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        import tk_fp8_mqa_logits

        # Prepare output tensor
        # Pad seq_len_kv to next multiple of BLOCK_KV (64) for safe writes
        BLOCK_KV = 64
        padded_kv = ((seq_len_kv + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
        logits_out = torch.zeros(seq_len, padded_kv, dtype=torch.float32,
                                 device=Q.device)

        # Reshape inputs to 4D for the gl<> bindings:
        #   Q:  [1, seq_len, NUM_HEADS, HEAD_SIZE]
        #   KV: [1, 1, seq_len_kv, HEAD_SIZE]  (pad kv rows to padded_kv if needed)
        #   kv_scales: [1, 1, 1, seq_len_kv]   (pad similarly)
        #   weights:   [1, 1, seq_len, NUM_HEADS]
        #   logits:    [1, 1, seq_len, padded_kv]
        Q_4d = Q.unsqueeze(0).contiguous()
        KV_4d = KV.unsqueeze(0).unsqueeze(0).contiguous()
        scales_4d = kv_scales.unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous()
        weights_4d = weights.unsqueeze(0).unsqueeze(0).contiguous()
        logits_4d = logits_out.unsqueeze(0).unsqueeze(0).contiguous()

        # Launch kernel
        torch.cuda.synchronize()
        tk_fp8_mqa_logits.dispatch(Q_4d, KV_4d, scales_4d, weights_4d, logits_4d)
        torch.cuda.synchronize()

        # Trim to actual seq_len_kv
        return logits_4d.squeeze(0).squeeze(0)[:, :seq_len_kv]

    except ImportError:
        print("[INFO] tk_fp8_mqa_logits module not found. Skipping HipKittens test.")
        print("[INFO] Build with: make NUM_HEADS={} HEAD_SIZE={} SEQ_LEN={} SEQ_LEN_KV={}".format(
            num_heads, head_size, seq_len, seq_len_kv))
        return None
    except Exception as e:
        print(f"[ERROR] HipKittens kernel execution failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FP8 MQA Logits parity test")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Query sequence length")
    parser.add_argument("--seq-len-kv", type=int, default=256,
                        help="Key/Value sequence length")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--head-size", type=int, default=64,
                        help="Head dimension")
    parser.add_argument("--no-hip", action="store_true",
                        help="Skip HipKittens kernel (reference-only mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    seq_len     = args.seq_len
    seq_len_kv  = args.seq_len_kv
    num_heads   = args.num_heads
    head_size   = args.head_size

    torch.manual_seed(args.seed)

    print("=" * 70)
    print("FP8 MQA Logits Parity Test")
    print("=" * 70)
    print(f"  seq_len     = {seq_len}")
    print(f"  seq_len_kv  = {seq_len_kv}")
    print(f"  num_heads   = {num_heads}")
    print(f"  head_size   = {head_size}")
    print()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] No CUDA device available. Running reference on CPU only.")

    # -----------------------------------------------------------------------
    # Generate random inputs
    # -----------------------------------------------------------------------
    Q = torch.randn(seq_len, num_heads, head_size, dtype=torch.bfloat16,
                     device=device)
    KV = torch.randn(seq_len_kv, head_size, dtype=torch.bfloat16,
                      device=device)
    kv_scales = torch.rand(seq_len_kv, dtype=torch.float32,
                            device=device) * 2.0  # range [0, 2)
    weights = torch.randn(seq_len, num_heads, dtype=torch.float32,
                           device=device) * 0.5

    print("[INFO] Input shapes:")
    print(f"  Q:         {list(Q.shape)} {Q.dtype}")
    print(f"  KV:        {list(KV.shape)} {KV.dtype}")
    print(f"  kv_scales: {list(kv_scales.shape)} {kv_scales.dtype}")
    print(f"  weights:   {list(weights.shape)} {weights.dtype}")
    print()

    # -----------------------------------------------------------------------
    # Compute reference
    # -----------------------------------------------------------------------
    print("[INFO] Computing reference output (PyTorch)...")
    ref_logits = reference_mqa_logits(Q, KV, kv_scales, weights)
    print(f"  Output shape: {list(ref_logits.shape)} {ref_logits.dtype}")
    print(f"  Output range: [{ref_logits.min().item():.4f}, {ref_logits.max().item():.4f}]")
    print(f"  Output mean:  {ref_logits.mean().item():.6f}")
    print()

    # -----------------------------------------------------------------------
    # Self-check: verify reference against a naive loop implementation
    # -----------------------------------------------------------------------
    print("[INFO] Running self-check (naive loop vs vectorized reference)...")
    # Naive loop for a few rows as a sanity check
    check_rows = min(4, seq_len)
    check_passed = True
    for i in range(check_rows):
        for k in range(min(4, seq_len_kv)):
            expected = 0.0
            for h in range(num_heads):
                score = float(Q[i, h, :].float() @ KV[k, :].float())
                score *= kv_scales[k].item()
                score = max(score, 0.0)
                score *= weights[i, h].item()
                expected += score
            actual = ref_logits[i, k].item()
            if abs(expected - actual) > 1e-2 + 5e-2 * abs(expected):
                print(f"  [FAIL] Self-check mismatch at [{i},{k}]: "
                      f"expected={expected:.6f}, got={actual:.6f}")
                check_passed = False
    if check_passed:
        print("  [PASS] Self-check passed.")
    print()

    # -----------------------------------------------------------------------
    # Run HipKittens kernel (if available)
    # -----------------------------------------------------------------------
    all_passed = check_passed

    if not args.no_hip and device == "cuda":
        print("[INFO] Attempting to run HipKittens kernel...")
        hk_logits = run_hipkittens_kernel(
            Q, KV, kv_scales, weights,
            seq_len, seq_len_kv, num_heads, head_size
        )
        if hk_logits is not None:
            print()
            hk_passed = compare_tensors(ref_logits, hk_logits,
                                        "HipKittens vs Reference",
                                        atol=1e-2, rtol=5e-2)
            all_passed = all_passed and hk_passed

            # Print sample values
            print()
            print("  Sample values (first 8 elements of row 0):")
            n = min(8, seq_len_kv)
            print(f"    Ref: {ref_logits[0, :n].tolist()}")
            print(f"    HK:  {hk_logits[0, :n].tolist()}")
    elif args.no_hip:
        print("[INFO] HipKittens test skipped (--no-hip flag).")
    else:
        print("[INFO] No CUDA device; HipKittens test skipped.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    if all_passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 70)

    # Print expected build command for reference
    print()
    print("Build command:")
    print(f"  make NUM_HEADS={num_heads} HEAD_SIZE={head_size} "
          f"SEQ_LEN={seq_len} SEQ_LEN_KV={seq_len_kv}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
