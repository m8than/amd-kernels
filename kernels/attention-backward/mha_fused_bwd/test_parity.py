#!/usr/bin/env python3
"""
Parity test for MHA Fused Backward Pass HipKittens kernel.

Reference: PyTorch autograd backward through scaled_dot_product_attention.

This test:
  1. Creates random Q, K, V, dO tensors matching the kernel's expected shapes/dtypes
  2. Computes reference dQ, dK, dV using PyTorch autograd
  3. (When HIP hardware is available) Compiles and runs the HipKittens kernel
  4. Compares outputs with appropriate tolerances
  5. Reports PASS/FAIL

Usage:
  python test_parity.py                    # reference-only mode (no HIP hardware)
  python test_parity.py --run-kernel       # full parity check (requires HIP hardware + compiled kernel)

Expected tensor shapes and dtypes:
  Q:     (B, N, H_Q, D)   bf16
  K:     (B, N, H_KV, D)  bf16
  V:     (B, N, H_KV, D)  bf16
  O:     (B, N, H_Q, D)   bf16  (forward output, needed for prep kernel)
  dO:    (B, N, H_Q, D)   bf16
  dQ:    (B, N, H_Q, D)   float32  (accumulated via atomic add)
  dK:    (B, N, H_KV, D)  bf16
  dV:    (B, N, H_KV, D)  bf16
  L:     (B, H_Q, 1, N)   float32  (softmax log-sum-exp from forward)
  delta: (B, H_Q, 1, N)   float32  (rowsum(dO * O))

Build command:
  cd kernels/mha_fused_bwd
  make ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=512 GPU_TARGET=CDNA3
"""

import argparse
import math
import sys
import os
import subprocess

import torch
import torch.nn.functional as F

# ============================================================
# Configuration (must match kernel compile-time params)
# ============================================================
B     = 4      # batch size
N     = 512    # sequence length
H_Q   = 32     # number of query heads
H_KV  = 8      # number of key/value heads
D     = 128    # head dimension

GROUP_SIZE = H_Q // H_KV
SM_SCALE   = 1.0 / math.sqrt(D)

# Tolerances for bf16 backward pass
ATOL_DQ = 5e-2
RTOL_DQ = 1e-1
ATOL_DK = 5e-2
RTOL_DK = 1e-1
ATOL_DV = 5e-2
RTOL_DV = 1e-1
ATOL_DELTA = 1e-3
RTOL_DELTA = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expand_kv_for_gqa(K, V, h_q, h_kv):
    """Expand K,V from h_kv heads to h_q heads for GQA."""
    group_size = h_q // h_kv
    K_exp = K.repeat_interleave(group_size, dim=2)
    V_exp = V.repeat_interleave(group_size, dim=2)
    return K_exp, V_exp


def reference_forward_and_lse(Q, K, V, causal=True):
    """
    Manual forward pass returning output O and log-sum-exp L.
    Inputs: (B, N, H, D) in bf16
    Returns: O (B, N, H_Q, D) bf16, L (B, H_Q, 1, N) float32
    """
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    K_exp, V_exp = expand_kv_for_gqa(K_f, V_f, H_Q, H_KV)

    # Transpose to (B, H, N, D) for matmul
    Q_t = Q_f.transpose(1, 2)  # (B, H_Q, N, D)
    K_t = K_exp.transpose(1, 2)  # (B, H_Q, N, D)
    V_t = V_exp.transpose(1, 2)  # (B, H_Q, N, D)

    S = torch.matmul(Q_t, K_t.transpose(-2, -1)) * SM_SCALE  # (B, H_Q, N, N)

    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))

    # Softmax
    m = S.max(dim=-1, keepdim=True).values  # (B, H_Q, N, 1)
    S_shifted = S - m
    P = torch.exp(S_shifted)
    l = P.sum(dim=-1, keepdim=True)  # (B, H_Q, N, 1)
    P = P / l

    # Log-sum-exp
    L = m.squeeze(-1) + torch.log(l.squeeze(-1))  # (B, H_Q, N)

    O_t = torch.matmul(P, V_t)  # (B, H_Q, N, D)
    O = O_t.transpose(1, 2)  # (B, N, H_Q, D)

    # L shape for kernel: (B, H_Q, 1, N)
    L_out = L.unsqueeze(2)  # (B, H_Q, 1, N)

    return O.bfloat16(), L_out.float()


def reference_backward(Q, K, V, dO, causal=True):
    """
    Reference backward using PyTorch autograd.
    Inputs: (B, N, H, D) in bf16
    Returns: dQ, dK, dV in float32
    """
    # Move to float64 for maximum precision
    Q_f = Q.detach().to(torch.float64).requires_grad_(True)
    K_f = K.detach().to(torch.float64).requires_grad_(True)
    V_f = V.detach().to(torch.float64).requires_grad_(True)
    dO_f = dO.detach().to(torch.float64)

    K_exp, V_exp = expand_kv_for_gqa(K_f, V_f, H_Q, H_KV)

    # Transpose to (B, H, N, D)
    Q_t = Q_f.transpose(1, 2)
    K_t = K_exp.transpose(1, 2)
    V_t = V_exp.transpose(1, 2)

    S = torch.matmul(Q_t, K_t.transpose(-2, -1)) * SM_SCALE

    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))

    P = torch.softmax(S, dim=-1)
    O_t = torch.matmul(P, V_t)
    O = O_t.transpose(1, 2)  # (B, N, H_Q, D)

    # Backward
    O.backward(dO_f)

    dQ_ref = Q_f.grad.float()  # (B, N, H_Q, D)
    dK_ref = K_f.grad.float()  # (B, N, H_KV, D)
    dV_ref = V_f.grad.float()  # (B, N, H_KV, D)

    return dQ_ref, dK_ref, dV_ref


def reference_delta(O, dO):
    """Compute delta = rowsum(O * dO) per position."""
    # O, dO: (B, N, H_Q, D) bf16
    # delta: (B, H_Q, 1, N) float32
    O_f = O.float()
    dO_f = dO.float()
    prod = O_f * dO_f  # (B, N, H_Q, D)
    delta = prod.sum(dim=-1)  # (B, N, H_Q)
    delta = delta.transpose(1, 2).unsqueeze(2)  # (B, H_Q, 1, N)
    return delta


def cosine_similarity(a, b):
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def check_parity(name, ref, pred, atol, rtol):
    """Check if ref and pred match within tolerances."""
    ref_f = ref.float()
    pred_f = pred.float()

    abs_diff = (ref_f - pred_f).abs()
    max_abs_err = abs_diff.max().item()

    # Relative error check: |ref - pred| <= atol + rtol * |ref|
    within_tol = abs_diff <= (atol + rtol * ref_f.abs())
    pass_rate = within_tol.float().mean().item() * 100

    cos_sim = cosine_similarity(ref_f, pred_f)

    l2_rel = (abs_diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().sqrt()).item()

    passed = pass_rate > 95.0 and cos_sim > 0.99
    status = "PASS" if passed else "FAIL"

    print(f"  {name}: {status}")
    print(f"    max_abs_err={max_abs_err:.6f}, l2_rel={l2_rel:.6f}, "
          f"cos_sim={cos_sim:.6f}, within_tol={pass_rate:.2f}%")
    return passed


def run_reference_test():
    """Run reference-only test to validate test infrastructure."""
    print(f"\n{'='*60}")
    print(f"MHA Fused Backward - Reference Test")
    print(f"B={B}, N={N}, H_Q={H_Q}, H_KV={H_KV}, D={D}, GROUP_SIZE={GROUP_SIZE}")
    print(f"{'='*60}\n")

    torch.manual_seed(42)

    if DEVICE == "cuda":
        Q = torch.randn(B, N, H_Q, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(B, N, H_KV, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(B, N, H_KV, D, device=DEVICE, dtype=torch.bfloat16)
        dO = torch.randn(B, N, H_Q, D, device=DEVICE, dtype=torch.bfloat16)
    else:
        Q = torch.randn(B, N, H_Q, D, dtype=torch.bfloat16)
        K = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16)
        V = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16)
        dO = torch.randn(B, N, H_Q, D, dtype=torch.bfloat16)

    print("1. Computing reference forward (O, L)...")
    O, L = reference_forward_and_lse(Q, K, V, causal=True)
    print(f"   O shape: {O.shape}, L shape: {L.shape}")

    print("2. Computing reference delta...")
    delta = reference_delta(O, dO)
    print(f"   delta shape: {delta.shape}")

    print("3. Computing reference backward (dQ, dK, dV)...")
    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO, causal=True)
    print(f"   dQ shape: {dQ_ref.shape}, dK shape: {dK_ref.shape}, dV shape: {dV_ref.shape}")

    # Sanity checks
    print("\n4. Sanity checks:")
    print(f"   O range:     [{O.float().min():.4f}, {O.float().max():.4f}]")
    print(f"   dQ range:    [{dQ_ref.min():.4f}, {dQ_ref.max():.4f}]")
    print(f"   dK range:    [{dK_ref.min():.4f}, {dK_ref.max():.4f}]")
    print(f"   dV range:    [{dV_ref.min():.4f}, {dV_ref.max():.4f}]")
    print(f"   delta range: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"   L range:     [{L.min():.4f}, {L.max():.4f}]")

    # Verify that dQ/dK/dV are not all zeros
    assert dQ_ref.abs().max() > 0, "dQ is all zeros!"
    assert dK_ref.abs().max() > 0, "dK is all zeros!"
    assert dV_ref.abs().max() > 0, "dV is all zeros!"
    print("\n   All sanity checks passed.")

    # Self-consistency: compute backward again with float32 and check against float64
    print("\n5. Self-consistency check (float32 vs float64)...")
    Q32 = Q.detach().float().requires_grad_(True)
    K32 = K.detach().float().requires_grad_(True)
    V32 = V.detach().float().requires_grad_(True)
    dO32 = dO.float()

    K_exp32, V_exp32 = expand_kv_for_gqa(K32, V32, H_Q, H_KV)
    Q_t32 = Q32.transpose(1, 2)
    K_t32 = K_exp32.transpose(1, 2)
    V_t32 = V_exp32.transpose(1, 2)

    S32 = torch.matmul(Q_t32, K_t32.transpose(-2, -1)) * SM_SCALE
    mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S32.masked_fill_(mask, float('-inf'))
    P32 = torch.softmax(S32, dim=-1)
    O32 = torch.matmul(P32, V_t32).transpose(1, 2)
    O32.backward(dO32)

    dQ32 = Q32.grad
    dK32 = K32.grad
    dV32 = V32.grad

    check_parity("dQ (f32 vs f64)", dQ_ref, dQ32, 1e-3, 1e-2)
    check_parity("dK (f32 vs f64)", dK_ref, dK32, 1e-3, 1e-2)
    check_parity("dV (f32 vs f64)", dV_ref, dV32, 1e-3, 1e-2)

    print(f"\n{'='*60}")
    print("Reference test complete. Kernel shapes and expected values documented.")
    print(f"{'='*60}\n")
    return True


def run_kernel_test():
    """Run full parity test with HipKittens kernel."""
    print(f"\n{'='*60}")
    print(f"MHA Fused Backward - Kernel Parity Test")
    print(f"B={B}, N={N}, H_Q={H_Q}, H_KV={H_KV}, D={D}")
    print(f"{'='*60}\n")

    # Try to import the kernel module
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)
    try:
        import mha_fused_bwd
    except ImportError as e:
        print(f"ERROR: Could not import mha_fused_bwd kernel module: {e}")
        print(f"Build the kernel first: cd {kernel_dir} && make")
        return False

    torch.manual_seed(42)

    Q = torch.randn(B, N, H_Q, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_KV, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_KV, D, device="cuda", dtype=torch.bfloat16)
    dO = torch.randn(B, N, H_Q, D, device="cuda", dtype=torch.bfloat16)

    # Reference forward
    print("1. Computing reference forward and backward...")
    O, L = reference_forward_and_lse(Q, K, V, causal=True)
    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO, causal=True)
    delta_ref = reference_delta(O, dO)

    # Allocate output tensors
    delta_hk = torch.zeros(B, H_Q, 1, N, device="cuda", dtype=torch.float32)
    dQ_hk = torch.zeros(B, N, H_Q, D, device="cuda", dtype=torch.float32)
    dK_hk = torch.zeros(B, N, H_KV, D, device="cuda", dtype=torch.bfloat16)
    dV_hk = torch.zeros(B, N, H_KV, D, device="cuda", dtype=torch.bfloat16)

    # Run prep kernel
    print("2. Running HipKittens prep kernel (delta computation)...")
    mha_fused_bwd.dispatch_prep(O, dO, delta_hk)
    torch.cuda.synchronize()

    # Check delta
    print("3. Checking delta parity...")
    delta_pass = check_parity("delta", delta_ref, delta_hk, ATOL_DELTA, RTOL_DELTA)

    # Run backward kernel
    print("4. Running HipKittens backward kernel...")
    mha_fused_bwd.dispatch_bwd(Q, K, V, dO, dQ_hk, dK_hk, dV_hk, L, delta_hk)
    torch.cuda.synchronize()

    # Check dQ, dK, dV
    print("5. Checking gradient parity...")
    dQ_pass = check_parity("dQ", dQ_ref, dQ_hk, ATOL_DQ, RTOL_DQ)
    dK_pass = check_parity("dK", dK_ref, dK_hk, ATOL_DK, RTOL_DK)
    dV_pass = check_parity("dV", dV_ref, dV_hk, ATOL_DV, RTOL_DV)

    all_pass = delta_pass and dQ_pass and dK_pass and dV_pass

    print(f"\n{'='*60}")
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print(f"{'='*60}\n")
    return all_pass


def print_build_commands():
    """Print the build commands for the kernel."""
    print("\nExpected build commands:")
    print(f"  cd {os.path.dirname(os.path.abspath(__file__))}")
    print(f"  export THUNDERKITTENS_ROOT=/path/to/HipKittens")
    print(f"  export ROCM_PATH=/opt/rocm")
    print(f"  make ATTN_B={B} ATTN_H={H_Q} ATTN_H_KV={H_KV} ATTN_N={N} GPU_TARGET=CDNA3")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHA Fused Backward Parity Test")
    parser.add_argument("--run-kernel", action="store_true",
                        help="Run full kernel parity test (requires compiled kernel)")
    parser.add_argument("--batch", type=int, default=B, help="Batch size")
    parser.add_argument("--seqlen", type=int, default=N, help="Sequence length")
    parser.add_argument("--heads-q", type=int, default=H_Q, help="Number of Q heads")
    parser.add_argument("--heads-kv", type=int, default=H_KV, help="Number of KV heads")
    args = parser.parse_args()

    B = args.batch
    N = args.seqlen
    H_Q = args.heads_q
    H_KV = args.heads_kv
    GROUP_SIZE = H_Q // H_KV
    SM_SCALE = 1.0 / math.sqrt(D)

    print_build_commands()

    if args.run_kernel:
        success = run_kernel_test()
    else:
        success = run_reference_test()

    sys.exit(0 if success else 1)
