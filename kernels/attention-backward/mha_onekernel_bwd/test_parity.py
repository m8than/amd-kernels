#!/usr/bin/env python3
"""
Parity test for MHA One-Kernel Backward Pass (HipKittens port).

Tests the HipKittens kernel against a PyTorch reference implementation of
scaled dot-product attention backward pass with causal masking and GQA support.

Usage:
    # Reference-only mode (no HIP hardware needed):
    python test_parity.py --reference-only

    # Full parity test (requires compiled kernel):
    python test_parity.py

    # Custom parameters:
    python test_parity.py --batch 8 --seq 2048 --hq 32 --hkv 8
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

# =====================================================================
# Configuration
# =====================================================================

DEFAULT_B     = 4       # batch size
DEFAULT_H_Q   = 32      # query heads
DEFAULT_H_KV  = 8       # key/value heads (GQA)
DEFAULT_N     = 1024    # sequence length
DEFAULT_D     = 128     # head dimension
CAUSAL        = True    # causal masking
DTYPE         = torch.bfloat16

# Tolerances for parity checking
ATOL = 1e-2   # absolute tolerance
RTOL = 5e-2   # relative tolerance


# =====================================================================
# Reference implementation
# =====================================================================

def expand_kv_for_gqa(K, V, h_q, h_kv):
    """Expand K,V from h_kv heads to h_q heads for GQA by replicating each KV head."""
    group_size = h_q // h_kv
    K_expanded = K.repeat_interleave(group_size, dim=1)
    V_expanded = V.repeat_interleave(group_size, dim=1)
    return K_expanded, V_expanded


def reference_forward(Q, K, V, causal=True):
    """
    Reference forward pass in float64 for high-precision comparison.
    Layout: (B, H, N, D)
    Returns: O, L (logsumexp)
    """
    B, H_Q, N, D = Q.shape
    H_KV = K.shape[1]
    scale = 1.0 / math.sqrt(D)

    q = Q.detach().to(torch.float64)
    k = K.detach().to(torch.float64)
    v = V.detach().to(torch.float64)

    k_exp, v_exp = expand_kv_for_gqa(k, v, H_Q, H_KV)

    S = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))

    L = torch.logsumexp(S, dim=-1)           # (B, H_Q, N)
    P = torch.exp(S - L.unsqueeze(-1))       # softmax
    O = torch.matmul(P, v_exp)               # (B, H_Q, N, D)

    return O.to(Q.dtype), L.float(), P


def reference_backward(Q, K, V, dO, O, L, causal=True):
    """
    Reference backward pass computing dQ, dK, dV.
    All computations in float64 for numerical accuracy.
    Layout: (B, H, N, D)
    """
    B, H_Q, N, D = Q.shape
    H_KV = K.shape[1]
    group_size = H_Q // H_KV
    scale = 1.0 / math.sqrt(D)

    q = Q.detach().to(torch.float64)
    k = K.detach().to(torch.float64)
    v = V.detach().to(torch.float64)
    do = dO.detach().to(torch.float64)
    o = O.detach().to(torch.float64)
    l = L.detach().to(torch.float64)

    k_exp, v_exp = expand_kv_for_gqa(k, v, H_Q, H_KV)

    # Recompute S and P
    S = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.exp(S - l.unsqueeze(-1))

    # delta = rowsum(dO * O)
    delta = (do * o).sum(dim=-1)  # (B, H_Q, N)

    # dV (need to sum across grouped Q heads)
    dV_expanded = torch.matmul(P.transpose(-2, -1), do)  # (B, H_Q, N, D)
    dV = torch.zeros_like(v)
    for i in range(H_KV):
        start = i * group_size
        end = (i + 1) * group_size
        dV[:, i, :, :] = dV_expanded[:, start:end, :, :].sum(dim=1)

    # dS = P * (dO @ V^T - delta)
    dS = P * (torch.matmul(do, v_exp.transpose(-2, -1)) - delta.unsqueeze(-1))

    # dQ = dS @ K * scale
    dQ = torch.matmul(dS, k_exp) * scale

    # dK (need to sum across grouped Q heads)
    dK_expanded = torch.matmul(dS.transpose(-2, -1), q) * scale
    dK = torch.zeros_like(k)
    for i in range(H_KV):
        start = i * group_size
        end = (i + 1) * group_size
        dK[:, i, :, :] = dK_expanded[:, start:end, :, :].sum(dim=1)

    return (
        dQ.to(Q.dtype),
        dK.to(K.dtype),
        dV.to(V.dtype),
        delta.float(),
    )


# =====================================================================
# Comparison utilities
# =====================================================================

def robustness_check(ref, pred, name=""):
    """Compute comparison metrics between reference and prediction."""
    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)

    # Relative error mask: |diff| > atol + rtol * |ref|
    mask = diff > (ATOL + RTOL * denom)
    error_count = mask.sum().item()
    numel = ref_f.numel()
    rel_error = error_count / numel if numel > 0 else 0.0
    l2_error = (diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().clamp_min(1e-12).sqrt()).item()

    cos = F.cosine_similarity(ref_f.flatten(), pred_f.flatten(), dim=0).item()

    passed = rel_error < 0.05 and cos > 0.99  # <5% elements fail, cosine > 0.99

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: "
          f"max_abs={diff.max().item():.6f}, "
          f"rel_error={100*rel_error:.4f}%, "
          f"l2_error={l2_error:.6f}, "
          f"cosine={cos:.6f}, "
          f"errors={error_count}/{numel}")
    return passed


# =====================================================================
# Input generation
# =====================================================================

def generate_inputs(B, H_Q, H_KV, N, D, device='cuda'):
    """Generate random input tensors for the backward pass test."""
    torch.manual_seed(42)

    # Generate with controlled magnitude for numerical stability
    def gen(shape):
        t = torch.randn(shape, dtype=DTYPE, device=device)
        mag = torch.norm(t, dim=-1, keepdim=True).clamp_min(1e-6)
        mean_mag = 10.0
        std_mag = 0.1
        target = torch.randn(mag.shape, dtype=DTYPE, device=device) * std_mag + mean_mag
        return (t * target / mag).contiguous()

    Q  = gen((B, H_Q,  N, D))
    K  = gen((B, H_KV, N, D))
    V  = gen((B, H_KV, N, D))
    dO = gen((B, H_Q,  N, D))

    return Q, K, V, dO


# =====================================================================
# HipKittens kernel interface
# =====================================================================

def run_hipkittens(Q, K, V, dO, O, L, delta):
    """
    Run the HipKittens one-kernel backward pass.
    Requires the compiled tk_mha_onekernel_bwd Python module.

    Tensor layout expected by the kernel:
        Q, dO, dQ : (B, N_tiles, H_Q, D) where N_tiles = N / 16
        K, V, dK, dV : (B, N_tiles, H_KV, D)
        L_vec, delta_vec : (B, H, 1, N_tiles)
    """
    import tk_mha_onekernel_bwd

    B, H_Q, N, D = Q.shape
    H_KV = K.shape[1]
    TILE_ROWS = 16

    N_tiles = N // TILE_ROWS

    # Convert from BHND to B(N/16)HD layout
    # Q: (B, H_Q, N, D) -> (B, N/16, H_Q, D)
    # Reshape: split N into (N/16, 16) then merge 16 into the tile dimension
    def bhnd_to_bnhd_tiled(x, n_heads):
        # (B, H, N, D) -> (B, N_tiles, H, D)
        # Each tile is TILE_ROWS rows, the kernel indexes by tile
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n // TILE_ROWS, TILE_ROWS, h, d) \
                .reshape(b, n // TILE_ROWS * TILE_ROWS // TILE_ROWS, h, d) \
                .contiguous()

    # Actually simpler: just transpose dim 1 and 2, then ensure contiguity
    # From (B, H, N, D) to (B, N, H, D) then view as (B, N_tiles*TILE_ROWS/TILE_ROWS, H, D)
    # which is just (B, N_tiles, H, D) if we think of each N_tile as TILE_ROWS rows
    # But the kernel's gl<> maps to 4D: (batch, seq_tiles, head, dim)
    # So we need: (B, N/TILE_ROWS, H, D)

    # The simplest correct conversion:
    # Q (B,H,N,D) -> Q_k (B, N/16, H, D)
    # Each "row" in dim 1 of Q_k corresponds to 16 sequence positions
    Q_k  = Q.transpose(1, 2).contiguous().view(B, N, H_Q, D)  # (B,N,H_Q,D)
    K_k  = K.transpose(1, 2).contiguous().view(B, N, H_KV, D)
    V_k  = V.transpose(1, 2).contiguous().view(B, N, H_KV, D)
    dO_k = dO.transpose(1, 2).contiguous().view(B, N, H_Q, D)
    O_k  = O.transpose(1, 2).contiguous().view(B, N, H_Q, D)

    # L and delta: (B, H_Q, N) -> (B, H_Q, 1, N_tiles)
    # The kernel expects these tiled by TILE_ROWS
    # L has one value per 16-row tile
    # For the reference, L is (B, H_Q, N) -- one value per sequence position
    # The kernel expects (B, H_Q, 1, N_tiles) where each tile summarizes TILE_ROWS rows
    # In the reference backward prep, L is stored as logsumexp per-tile
    # Here we just reshape: the kernel uses L per 16-row tile
    L_k = L.view(B, H_Q, 1, N).contiguous()

    # delta: (B, H_Q, N) -> (B, H_Q, 1, N_tiles)
    delta_k = delta.view(B, H_Q, 1, N).contiguous()

    # Allocate outputs
    dQ_k = torch.zeros_like(Q_k)
    dK_k = torch.zeros_like(K_k)
    dV_k = torch.zeros_like(V_k)

    # Run preprocessing (delta computation)
    delta_computed = torch.zeros(B, H_Q, 1, N // TILE_ROWS, device=Q.device, dtype=torch.float32)
    tk_mha_onekernel_bwd.dispatch_prep(O_k, dO_k, delta_computed)
    torch.cuda.synchronize()

    # Run combined backward kernel
    tk_mha_onekernel_bwd.dispatch_bwd_combined(
        Q_k, K_k, V_k, dO_k,
        dQ_k, dK_k, dV_k,
        L_k, delta_computed
    )
    torch.cuda.synchronize()

    # Convert outputs back to BHND
    dQ = dQ_k.view(B, N, H_Q, D).transpose(1, 2).contiguous()
    dK = dK_k.view(B, N, H_KV, D).transpose(1, 2).contiguous()
    dV = dV_k.view(B, N, H_KV, D).transpose(1, 2).contiguous()

    return dQ, dK, dV


# =====================================================================
# Main test
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MHA OneKernel Backward Parity Test")
    parser.add_argument("--batch", type=int, default=DEFAULT_B, help="Batch size")
    parser.add_argument("--seq", type=int, default=DEFAULT_N, help="Sequence length")
    parser.add_argument("--hq", type=int, default=DEFAULT_H_Q, help="Query heads")
    parser.add_argument("--hkv", type=int, default=DEFAULT_H_KV, help="KV heads")
    parser.add_argument("--dim", type=int, default=DEFAULT_D, help="Head dimension")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only run reference (no kernel needed)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu for reference)")
    args = parser.parse_args()

    B, N, H_Q, H_KV, D = args.batch, args.seq, args.hq, args.hkv, args.dim
    device = args.device if not args.reference_only else "cpu"
    if not args.reference_only and not torch.cuda.is_available():
        print("CUDA not available. Running reference-only mode.")
        args.reference_only = True
        device = "cpu"

    print("=" * 70)
    print("MHA One-Kernel Backward Pass -- Parity Test")
    print("=" * 70)
    print(f"  B={B}, N={N}, H_Q={H_Q}, H_KV={H_KV}, D={D}")
    print(f"  Causal={CAUSAL}, dtype={DTYPE}")
    print(f"  Device={device}")
    print(f"  Group size (GQA)={H_Q // H_KV}")
    print()

    # Generate inputs
    print("Generating inputs...")
    Q, K, V, dO = generate_inputs(B, H_Q, H_KV, N, D, device=device)

    # Run reference forward
    print("Running reference forward...")
    O_ref, L_ref, P_ref = reference_forward(Q, K, V, causal=CAUSAL)

    # Run reference backward
    print("Running reference backward...")
    dQ_ref, dK_ref, dV_ref, delta_ref = reference_backward(
        Q, K, V, dO, O_ref, L_ref, causal=CAUSAL
    )

    print(f"\nReference output shapes:")
    print(f"  dQ: {dQ_ref.shape}")
    print(f"  dK: {dK_ref.shape}")
    print(f"  dV: {dV_ref.shape}")
    print(f"  delta: {delta_ref.shape}")
    print(f"  L: {L_ref.shape}")

    # Quick sanity checks on reference
    print(f"\nReference value ranges:")
    print(f"  dQ: [{dQ_ref.float().min().item():.4f}, {dQ_ref.float().max().item():.4f}]")
    print(f"  dK: [{dK_ref.float().min().item():.4f}, {dK_ref.float().max().item():.4f}]")
    print(f"  dV: [{dV_ref.float().min().item():.4f}, {dV_ref.float().max().item():.4f}]")

    if args.reference_only:
        print("\n--- Reference-only mode: skipping kernel comparison ---")
        print("\nExpected compilation command:")
        print(f"  cd kernels/mha_onekernel_bwd && make ATTN_B={B} ATTN_H={H_Q} "
              f"ATTN_H_KV={H_KV} ATTN_N={N}")
        print("\nInput tensor shapes (BHND layout):")
        print(f"  Q:  ({B}, {H_Q}, {N}, {D})  dtype={DTYPE}")
        print(f"  K:  ({B}, {H_KV}, {N}, {D})  dtype={DTYPE}")
        print(f"  V:  ({B}, {H_KV}, {N}, {D})  dtype={DTYPE}")
        print(f"  dO: ({B}, {H_Q}, {N}, {D})  dtype={DTYPE}")
        print(f"\nKernel tensor shapes (B, N_tiles, H, D layout):")
        print(f"  Q_k:  ({B}, {N}, {H_Q}, {D})  dtype={DTYPE}")
        print(f"  K_k:  ({B}, {N}, {H_KV}, {D})  dtype={DTYPE}")
        print(f"  V_k:  ({B}, {N}, {H_KV}, {D})  dtype={DTYPE}")
        print(f"  L:    ({B}, {H_Q}, 1, {N})  dtype=float32")
        print(f"  delta:({B}, {H_Q}, 1, {N // 16})  dtype=float32")
        print(f"\nOutput shapes (BHND):")
        print(f"  dQ: ({B}, {H_Q}, {N}, {D})  dtype={DTYPE}")
        print(f"  dK: ({B}, {H_KV}, {N}, {D})  dtype={DTYPE}")
        print(f"  dV: ({B}, {H_KV}, {N}, {D})  dtype={DTYPE}")
        print("\nPASS (reference-only)")
        return 0

    # Run HipKittens kernel
    print("\nRunning HipKittens kernel...")
    try:
        dQ_hk, dK_hk, dV_hk = run_hipkittens(Q, K, V, dO, O_ref, L_ref, delta_ref)
    except ImportError as e:
        print(f"ERROR: Could not import kernel module: {e}")
        print("Please build the kernel first: cd kernels/mha_onekernel_bwd && make")
        return 1
    except Exception as e:
        print(f"ERROR: Kernel execution failed: {e}")
        return 1

    # Parity comparison
    print("\nParity comparison (HipKittens vs Reference):")
    all_pass = True
    all_pass &= robustness_check(dQ_ref, dQ_hk, "dQ")
    all_pass &= robustness_check(dK_ref, dK_hk, "dK")
    all_pass &= robustness_check(dV_ref, dV_hk, "dV")

    print()
    if all_pass:
        print("OVERALL: PASS")
        return 0
    else:
        print("OVERALL: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
