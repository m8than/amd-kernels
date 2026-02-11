#!/usr/bin/env python3
"""
Parity test for SAGE Attention (Fav3) HipKittens kernel.

This test validates the HipKittens port of the Triton SAGE attention forward kernel.

The SAGE attention algorithm:
  1. Quantize Q to INT8 with per-block scale, K to INT8 with per-block scale
  2. Quantize V to FP8 with per-channel scale
  3. Compute attention: qk = dot(q, k) * q_descale * k_descale
  4. Online softmax with exp2
  5. acc += dot(softmax(qk), v); at end acc = acc / l_i * v_descale
  6. Supports causal masking and GQA

In this simplified HipKittens port, we use bf16 inputs (not INT8/FP8) and
apply the descale factors as floating-point multipliers. The mathematical
behavior is identical.

Compilation:
  cd kernels/sage_attention
  make ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=1024 ATTN_D=128

Run:
  python3 test_parity.py [--batch 4] [--heads 32] [--heads_kv 8] [--seqlen 1024] [--headdim 128] [--causal]

Input shapes (BHND layout stored as [B, N, H, D]):
  Q:       [B, N, H_Q, D]    bf16
  K:       [B, N, H_KV, D]   bf16
  V:       [B, N, H_KV, D]   bf16
  Q_desc:  [B, H_Q, N_Q_BLOCKS]  f32  (per Q-block descale)
  K_desc:  [B, H_KV, N_K_BLOCKS] f32  (per K-block descale)
  V_desc:  [B, H_KV, D]          f32  (per-channel V descale)

Output shapes:
  O:       [B, N, H_Q, D]    bf16
  LSE:     [B, H_Q, 1, N]    f32   (log-sum-exp)
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="SAGE Attention parity test")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads_kv", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--use_kernel", action="store_true", default=False,
                        help="Import and run the HipKittens kernel (requires HIP hardware)")
    return parser.parse_args()


def generate_inputs(B, H_Q, H_KV, N, D, device="cuda", dtype=torch.bfloat16):
    """Generate random inputs matching SAGE attention shapes.

    Returns tensors in the memory layout expected by the HipKittens kernel:
      Q, K, V in [B, N, H, D] layout (BNHD)
      Descale factors as separate tensors
    """
    BLOCK_M = 128
    BLOCK_N = 64

    # Main tensors in BNHD layout
    Q = torch.randn(B, N, H_Q, D, dtype=dtype, device=device) * 0.1
    K = torch.randn(B, N, H_KV, D, dtype=dtype, device=device) * 0.1
    V = torch.randn(B, N, H_KV, D, dtype=dtype, device=device) * 0.1

    # Per-block Q descale: simulate quantization scale
    # In real SAGE, this is max(abs(q_block)) / 127 for INT8
    num_q_blocks = (N + BLOCK_M - 1) // BLOCK_M
    num_k_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # Descale factors are always positive, typically in [0.001, 0.1] range
    Q_desc = torch.rand(B, H_Q, num_q_blocks, dtype=torch.float32, device=device) * 0.05 + 0.01
    K_desc = torch.rand(B, H_KV, num_k_blocks, dtype=torch.float32, device=device) * 0.05 + 0.01
    V_desc = torch.rand(B, H_KV, D, dtype=torch.float32, device=device) * 0.1 + 0.01

    return Q, K, V, Q_desc, K_desc, V_desc


def sage_attention_reference(Q, K, V, Q_desc, K_desc, V_desc,
                              sm_scale=None, causal=False):
    """
    Reference SAGE attention implementation in PyTorch.

    This implements the same algorithm as the Triton kernel:
      1. For each Q block, apply q_descale
      2. For each K block, compute qk = Q @ K^T * q_descale * k_descale
      3. Online softmax using exp2
      4. acc += softmax(qk) @ V
      5. acc = acc / l_i * v_descale

    For simplicity, this reference does full materialization (not online softmax),
    which gives numerically identical results for non-causal, and very close for causal.

    Args:
        Q: [B, N, H_Q, D] bf16 query
        K: [B, N, H_KV, D] bf16 key
        V: [B, N, H_KV, D] bf16 value
        Q_desc: [B, H_Q, num_q_blocks] f32 per-Q-block descale
        K_desc: [B, H_KV, num_k_blocks] f32 per-K-block descale
        V_desc: [B, H_KV, D] f32 per-channel V descale
        sm_scale: softmax scale (default: 1.0, since scales are external)
        causal: whether to apply causal mask

    Returns:
        O: [B, N, H_Q, D] f32 output
        LSE: [B, H_Q, N] f32 log-sum-exp
    """
    BLOCK_M = 128
    BLOCK_N = 64

    B, N, H_Q, D = Q.shape
    _, _, H_KV, _ = K.shape
    group_size = H_Q // H_KV

    # Work in float32 for reference accuracy
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    # Transpose to BHND for easier matmul
    Q_bhnd = Q_f.permute(0, 2, 1, 3)  # [B, H_Q, N, D]
    K_bhnd = K_f.permute(0, 2, 1, 3)  # [B, H_KV, N, D]
    V_bhnd = V_f.permute(0, 2, 1, 3)  # [B, H_KV, N, D]

    # Expand K, V for GQA: [B, H_KV, N, D] -> [B, H_Q, N, D]
    K_expanded = K_bhnd.repeat_interleave(group_size, dim=1)
    V_expanded = V_bhnd.repeat_interleave(group_size, dim=1)

    # Build the full attention score matrix with per-block scaling
    # S[b, h, i, j] = Q[b,h,i,:] @ K[b,h,j,:]^T * q_desc[b,h,i//BLOCK_M] * k_desc[b,h_kv,j//BLOCK_N]
    S = torch.matmul(Q_bhnd, K_expanded.transpose(-2, -1))  # [B, H_Q, N, N]

    # Apply per-block Q descale: each row block of M rows shares a scale
    num_q_blocks = (N + BLOCK_M - 1) // BLOCK_M
    num_k_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min((qb + 1) * BLOCK_M, N)
        for kb in range(num_k_blocks):
            k_start = kb * BLOCK_N
            k_end = min((kb + 1) * BLOCK_N, N)
            # q_desc: [B, H_Q, num_q_blocks]
            # k_desc: [B, H_KV, num_k_blocks]
            q_scale = Q_desc[:, :, qb].unsqueeze(-1).unsqueeze(-1)  # [B, H_Q, 1, 1]

            # For GQA, k_desc uses KV head index
            # k_desc for head h_q maps to h_kv = h_q // group_size
            k_scale_kv = K_desc[:, :, kb]  # [B, H_KV]
            # Expand to H_Q
            k_scale = k_scale_kv.repeat_interleave(group_size, dim=1)  # [B, H_Q]
            k_scale = k_scale.unsqueeze(-1).unsqueeze(-1)  # [B, H_Q, 1, 1]

            S[:, :, q_start:q_end, k_start:k_end] *= (q_scale * k_scale)

    # Apply causal mask
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    LSE = torch.logsumexp(S, dim=-1)  # [B, H_Q, N]
    P = torch.softmax(S, dim=-1)

    # Compute output: O = P @ V
    O_bhnd = torch.matmul(P, V_expanded)  # [B, H_Q, N, D]

    # Apply per-channel V descale
    # V_desc: [B, H_KV, D] -> expand to [B, H_Q, 1, D]
    V_desc_expanded = V_desc.repeat_interleave(group_size, dim=1)  # [B, H_Q, D]
    V_desc_expanded = V_desc_expanded.unsqueeze(2)  # [B, H_Q, 1, D]
    O_bhnd = O_bhnd * V_desc_expanded

    # Transpose back to BNHD
    O = O_bhnd.permute(0, 2, 1, 3)  # [B, N, H_Q, D]

    return O, LSE


def sage_attention_reference_no_vdescale(Q, K, V, Q_desc, K_desc,
                                          causal=False):
    """
    Reference without V descale (matching the kernel which omits V descale
    in-kernel for simplicity -- the test can apply it externally).

    Returns:
        O: [B, N, H_Q, D] f32 output (before V descale)
        LSE: [B, H_Q, N] f32 log-sum-exp
    """
    BLOCK_M = 128
    BLOCK_N = 64

    B, N, H_Q, D = Q.shape
    _, _, H_KV, _ = K.shape
    group_size = H_Q // H_KV

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    Q_bhnd = Q_f.permute(0, 2, 1, 3)
    K_bhnd = K_f.permute(0, 2, 1, 3)
    V_bhnd = V_f.permute(0, 2, 1, 3)

    K_expanded = K_bhnd.repeat_interleave(group_size, dim=1)
    V_expanded = V_bhnd.repeat_interleave(group_size, dim=1)

    S = torch.matmul(Q_bhnd, K_expanded.transpose(-2, -1))

    num_q_blocks = (N + BLOCK_M - 1) // BLOCK_M
    num_k_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min((qb + 1) * BLOCK_M, N)
        for kb in range(num_k_blocks):
            k_start = kb * BLOCK_N
            k_end = min((kb + 1) * BLOCK_N, N)
            q_scale = Q_desc[:, :, qb].unsqueeze(-1).unsqueeze(-1)
            k_scale_kv = K_desc[:, :, kb]
            k_scale = k_scale_kv.repeat_interleave(group_size, dim=1)
            k_scale = k_scale.unsqueeze(-1).unsqueeze(-1)
            S[:, :, q_start:q_end, k_start:k_end] *= (q_scale * k_scale)

    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    LSE = torch.logsumexp(S, dim=-1)
    P = torch.softmax(S, dim=-1)
    O_bhnd = torch.matmul(P, V_expanded)
    O = O_bhnd.permute(0, 2, 1, 3)

    return O, LSE


def robustness_check(ref, pred, name=""):
    """Compare two tensors with relative and absolute tolerances."""
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)

    # Errors: points where |diff| > atol + rtol * |ref|
    atol, rtol = 0.01, 0.05
    mask = diff > (atol + rtol * denom)
    error_count = mask.sum().item()
    numel = ref_f.numel()
    rel_error = error_count / numel if numel > 0 else 0.0

    # L2 relative error
    l2_error = (diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().sqrt().clamp_min(1e-12)).item()

    # Cosine similarity
    cos = F.cosine_similarity(ref_f.flatten(), pred_f.flatten(), dim=0).item()

    max_abs = diff.max().item()

    print(f"  {name}: max_abs={max_abs:.6f}, rel_error={rel_error:.4f}, "
          f"l2={l2_error:.6f}, cos={cos:.6f}, "
          f"errors={error_count}/{numel} ({100*rel_error:.4f}%)")

    return l2_error, cos, error_count


def run_reference_test(args):
    """Run the reference PyTorch implementation and validate it against itself."""
    print("=" * 70)
    print("SAGE Attention Parity Test")
    print("=" * 70)
    print(f"Config: B={args.batch}, H_Q={args.heads}, H_KV={args.heads_kv}, "
          f"N={args.seqlen}, D={args.headdim}, causal={args.causal}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No CUDA device available. Running on CPU (slow).")

    torch.manual_seed(42)

    B, H_Q, H_KV, N, D = args.batch, args.heads, args.heads_kv, args.seqlen, args.headdim
    Q, K, V, Q_desc, K_desc, V_desc = generate_inputs(B, H_Q, H_KV, N, D, device=device)

    print("Input shapes:")
    print(f"  Q:      {list(Q.shape)}  dtype={Q.dtype}")
    print(f"  K:      {list(K.shape)}  dtype={K.dtype}")
    print(f"  V:      {list(V.shape)}  dtype={V.dtype}")
    print(f"  Q_desc: {list(Q_desc.shape)}  dtype={Q_desc.dtype}")
    print(f"  K_desc: {list(K_desc.shape)}  dtype={K_desc.dtype}")
    print(f"  V_desc: {list(V_desc.shape)}  dtype={V_desc.dtype}")
    print()

    # ---- Reference computation ----
    print("Computing reference (full SAGE with V descale)...")
    O_ref, LSE_ref = sage_attention_reference(Q, K, V, Q_desc, K_desc, V_desc,
                                               causal=args.causal)
    print(f"  O_ref shape:   {list(O_ref.shape)}")
    print(f"  LSE_ref shape: {list(LSE_ref.shape)}")
    print(f"  O_ref range:   [{O_ref.min().item():.4f}, {O_ref.max().item():.4f}]")
    print(f"  LSE_ref range: [{LSE_ref.min().item():.4f}, {LSE_ref.max().item():.4f}]")
    print()

    # ---- Reference without V descale (what the kernel actually computes) ----
    print("Computing reference (no V descale, matching kernel output)...")
    O_no_vd, LSE_no_vd = sage_attention_reference_no_vdescale(
        Q, K, V, Q_desc, K_desc, causal=args.causal
    )
    print(f"  O_no_vd shape: {list(O_no_vd.shape)}")
    print(f"  O_no_vd range: [{O_no_vd.min().item():.4f}, {O_no_vd.max().item():.4f}]")
    print()

    # ---- Self-consistency: O_ref should equal O_no_vd * V_desc ----
    group_size = H_Q // H_KV
    V_desc_expanded = V_desc.repeat_interleave(group_size, dim=1)  # [B, H_Q, D]
    O_reconstructed = O_no_vd * V_desc_expanded.unsqueeze(1)  # broadcast over N
    print("Self-consistency check (O_ref vs O_no_vd * V_desc):")
    robustness_check(O_ref, O_reconstructed, name="O_self_check")
    print()

    # ---- Standard softmax reference for sanity ----
    print("Comparison with standard scaled dot-product attention (no SAGE scales):")
    Q_bhnd = Q.float().permute(0, 2, 1, 3)
    K_bhnd = K.float().permute(0, 2, 1, 3)
    V_bhnd = V.float().permute(0, 2, 1, 3)
    K_expanded = K_bhnd.repeat_interleave(group_size, dim=1)
    V_expanded = V_bhnd.repeat_interleave(group_size, dim=1)
    scale = 1.0 / math.sqrt(D)
    S_std = torch.matmul(Q_bhnd, K_expanded.transpose(-2, -1)) * scale
    if args.causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S_std.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    O_std = torch.matmul(torch.softmax(S_std, dim=-1), V_expanded).permute(0, 2, 1, 3)
    print(f"  O_std range: [{O_std.min().item():.4f}, {O_std.max().item():.4f}]")
    print(f"  (This will differ from SAGE output due to different scale factors)")
    print()

    return O_no_vd, LSE_no_vd


def run_kernel_test(args, O_ref, LSE_ref):
    """Run the HipKittens kernel and compare against reference."""
    print("-" * 70)
    print("Running HipKittens SAGE Attention Kernel")
    print("-" * 70)

    try:
        import sage_attn_kernel
    except ImportError as e:
        print(f"[ERROR] Could not import sage_attn_kernel: {e}")
        print("Build the kernel first:")
        print(f"  cd kernels/sage_attention")
        print(f"  make ATTN_B={args.batch} ATTN_H={args.heads} "
              f"ATTN_H_KV={args.heads_kv} ATTN_N={args.seqlen} ATTN_D={args.headdim}")
        return False

    B, H_Q, H_KV, N, D = args.batch, args.heads, args.heads_kv, args.seqlen, args.headdim
    BLOCK_M, BLOCK_N = 128, 64

    torch.manual_seed(42)
    Q, K, V, Q_desc, K_desc, V_desc = generate_inputs(B, H_Q, H_KV, N, D, device="cuda")

    # Allocate output
    O_hk = torch.zeros(B, N, H_Q, D, dtype=torch.bfloat16, device="cuda")
    LSE_hk = torch.zeros(B, H_Q, 1, N, dtype=torch.float32, device="cuda")

    # Dispatch
    dispatch_fn = sage_attn_kernel.dispatch_sage_fwd_causal if args.causal \
                  else sage_attn_kernel.dispatch_sage_fwd

    dispatch_fn(Q, K, V, O_hk, LSE_hk, Q_desc, K_desc, V_desc)
    torch.cuda.synchronize()

    print(f"  O_hk shape:   {list(O_hk.shape)}")
    print(f"  O_hk range:   [{O_hk.float().min().item():.4f}, {O_hk.float().max().item():.4f}]")
    print()

    # Compare kernel output (no V descale) against reference
    print("Parity check (HipKittens vs Reference):")
    l2_o, cos_o, err_o = robustness_check(O_ref, O_hk.float(), name="O")

    # LSE comparison
    LSE_hk_flat = LSE_hk.squeeze(2)  # [B, H_Q, N]
    l2_lse, cos_lse, err_lse = robustness_check(LSE_ref, LSE_hk_flat, name="LSE")
    print()

    # Pass/fail
    passed = (cos_o > 0.99) and (l2_o < 0.05)
    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        print(f"  O cosine similarity {cos_o:.6f} (need > 0.99)")
        print(f"  O L2 relative error {l2_o:.6f} (need < 0.05)")

    return passed


def print_compilation_instructions(args):
    """Print how to compile and run this test."""
    print()
    print("=" * 70)
    print("Compilation and Run Instructions")
    print("=" * 70)
    print()
    print("1. Set environment variables:")
    print("   export ROCM_PATH=/opt/rocm")
    print("   export THUNDERKITTENS_ROOT=/path/to/HipKittens")
    print()
    print("2. Build the kernel:")
    print(f"   cd kernels/sage_attention")
    print(f"   make ATTN_B={args.batch} ATTN_H={args.heads} "
          f"ATTN_H_KV={args.heads_kv} ATTN_N={args.seqlen} ATTN_D={args.headdim}")
    print()
    print("3. Run with kernel:")
    print(f"   python3 test_parity.py --batch {args.batch} --heads {args.heads} "
          f"--heads_kv {args.heads_kv} --seqlen {args.seqlen} --headdim {args.headdim} "
          f"{'--causal ' if args.causal else ''}--use_kernel")
    print()
    print("4. Run reference only (no HIP hardware needed):")
    print(f"   python3 test_parity.py --batch {args.batch} --heads {args.heads} "
          f"--heads_kv {args.heads_kv} --seqlen {args.seqlen} --headdim {args.headdim} "
          f"{'--causal' if args.causal else ''}")
    print()


def main():
    args = parse_args()
    O_ref, LSE_ref = run_reference_test(args)

    if args.use_kernel:
        passed = run_kernel_test(args, O_ref, LSE_ref)
        if not passed:
            sys.exit(1)
    else:
        print("-" * 70)
        print("[INFO] Kernel test skipped (use --use_kernel to run with HIP hardware)")
        print("-" * 70)
        print_compilation_instructions(args)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
