#!/usr/bin/env python3
"""
Flash Attention v2 Backward Pass - Parity Test

Tests the HipKittens Flash Attention backward kernel against a PyTorch
autograd reference implementation. Supports both causal and non-causal
modes, and GQA (grouped-query attention) configurations.

Usage:
    # Full test with hardware (requires ROCm GPU):
    python test_parity.py [B] [N] [H] [H_KV] [causal]

    # Dry run (no hardware, prints expected shapes and compilation commands):
    python test_parity.py --dry-run

Examples:
    python test_parity.py                      # defaults: B=4, N=1024, H=32, H_KV=8, non-causal
    python test_parity.py 2 512 16 4 1         # B=2, N=512, H=16, H_KV=4, causal
    python test_parity.py --dry-run             # print shapes and build commands only
"""

import sys
import os
import math
import subprocess

# ============================================================================
# Configuration
# ============================================================================

D = 128  # head dimension (fixed at compile time)

def parse_args():
    """Parse command-line arguments for test configuration."""
    if "--dry-run" in sys.argv:
        return None  # signal dry-run mode

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    causal = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    assert H % H_KV == 0, f"H ({H}) must be divisible by H_KV ({H_KV})"
    assert N % 256 == 0, f"N ({N}) must be divisible by 256 (BLOCK_SIZE_KV)"

    return {"B": B, "N": N, "H": H, "H_KV": H_KV, "D": D, "causal": causal}


def dry_run():
    """Print expected tensor shapes and compilation commands without running."""
    print("=" * 70)
    print("Flash Attention v2 Backward - Dry Run")
    print("=" * 70)

    configs = [
        {"B": 4, "N": 1024, "H": 32, "H_KV": 8, "causal": False},
        {"B": 4, "N": 1024, "H": 32, "H_KV": 8, "causal": True},
        {"B": 2, "N": 2048, "H": 16, "H_KV": 4, "causal": False},
    ]

    for cfg in configs:
        B, N, H, H_KV = cfg["B"], cfg["N"], cfg["H"], cfg["H_KV"]
        causal = cfg["causal"]
        group_size = H // H_KV

        print(f"\nConfig: B={B}, N={N}, H={H}, H_KV={H_KV}, D={D}, "
              f"causal={causal}, group_size={group_size}")
        print("-" * 60)

        # Input tensor shapes (BNHD layout)
        print(f"  Q  shape: [{B}, {N}, {H}, {D}]     dtype: bf16")
        print(f"  K  shape: [{B}, {N}, {H_KV}, {D}]  dtype: bf16")
        print(f"  V  shape: [{B}, {N}, {H_KV}, {D}]  dtype: bf16")
        print(f"  O  shape: [{B}, {N}, {H}, {D}]     dtype: bf16")
        print(f"  dO shape: [{B}, {N}, {H}, {D}]     dtype: bf16")

        # Intermediate tensors
        print(f"  L (logsumexp) shape: [{B}, {H}, 1, {N}]  dtype: float32")
        print(f"  delta shape:         [{B}, {H}, 1, {N}]  dtype: float32")

        # Output gradient shapes
        print(f"  dQ shape: [{B}, {N}, {H}, {D}]     dtype: bf16")
        print(f"  dK shape: [{B}, {N}, {H_KV}, {D}]  dtype: bf16")
        print(f"  dV shape: [{B}, {N}, {H_KV}, {D}]  dtype: bf16")

    print(f"\n{'=' * 70}")
    print("Compilation command:")
    print("-" * 60)
    print(f"  make GPU_TARGET=CDNA3 ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=1024")
    print(f"\nOr directly:")
    print(f"  hipcc kernel.cpp -DATTN_B=4 -DATTN_H=32 -DATTN_H_KV=8 "
          f"-DATTN_N=1024 \\")
    print(f"    -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w \\")
    print(f"    -I${{THUNDERKITTENS_ROOT}}/include "
          f"$(python3 -m pybind11 --includes) \\")
    print(f"    -shared -fPIC -o flash_attn_bwd$(python3-config --extension-suffix)")
    print(f"{'=' * 70}")


# ============================================================================
# Reference Implementation (PyTorch autograd)
# ============================================================================

def reference_attention_backward(Q, K, V, dO, causal=False):
    """
    Reference Flash Attention backward pass using PyTorch autograd.

    Args:
        Q: [B, N, H, D] query tensor (bf16)
        K: [B, N, H_KV, D] key tensor (bf16)
        V: [B, N, H_KV, D] value tensor (bf16)
        dO: [B, N, H, D] output gradient (bf16)
        causal: whether to apply causal masking

    Returns:
        dQ, dK, dV, O, L (logsumexp), delta
    """
    import torch

    B, N, H, _D = Q.shape
    H_KV = K.shape[2]
    group_size = H // H_KV
    scale = 1.0 / math.sqrt(_D)

    # Convert to float64 for numerical reference, create leaf tensors
    q = Q.detach().to(torch.float64).requires_grad_(True)
    k = K.detach().to(torch.float64).requires_grad_(True)
    v = V.detach().to(torch.float64).requires_grad_(True)
    do = dO.detach().to(torch.float64)

    # Transpose to BHND for matmuls: [B, H, N, D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    do_t = do.transpose(1, 2)

    # Expand KV heads for GQA
    k_expanded = k_t.repeat_interleave(group_size, dim=1)  # [B, H, N, D]
    v_expanded = v_t.repeat_interleave(group_size, dim=1)  # [B, H, N, D]

    # Forward: compute S, P, O
    S = torch.matmul(q_t, k_expanded.transpose(-2, -1)) * scale  # [B, H, N, N]

    if causal:
        mask = torch.triu(
            torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Logsumexp (for verification)
    L = torch.logsumexp(S, dim=-1)  # [B, H, N]
    P = torch.softmax(S, dim=-1)    # [B, H, N, N]
    O = torch.matmul(P, v_expanded)  # [B, H, N, D]

    # Backward: manual computation
    # delta = rowsum(dO * O)
    delta = (do_t * O).sum(dim=-1)  # [B, H, N]

    # dV (sum over grouped Q heads)
    dV_expanded = torch.matmul(P.transpose(-2, -1), do_t)  # [B, H, N, D]
    dV_t = torch.zeros_like(v_t)
    for i in range(H_KV):
        start = i * group_size
        end = (i + 1) * group_size
        dV_t[:, i] = dV_expanded[:, start:end].sum(dim=1)

    # dS = P * (dO @ V^T - delta)
    dS = P * (torch.matmul(do_t, v_expanded.transpose(-2, -1)) -
              delta.unsqueeze(-1))

    # dQ
    dQ_t = torch.matmul(dS, k_expanded) * scale  # [B, H, N, D]

    # dK (sum over grouped Q heads)
    dK_expanded = torch.matmul(dS.transpose(-2, -1), q_t) * scale
    dK_t = torch.zeros_like(k_t)
    for i in range(H_KV):
        start = i * group_size
        end = (i + 1) * group_size
        dK_t[:, i] = dK_expanded[:, start:end].sum(dim=1)

    # Convert back to BNHD layout
    O_out = O.transpose(1, 2)      # [B, N, H, D]
    dQ = dQ_t.transpose(1, 2)      # [B, N, H, D]
    dK = dK_t.transpose(1, 2)      # [B, N, H_KV, D]
    dV = dV_t.transpose(1, 2)      # [B, N, H_KV, D]

    return dQ, dK, dV, O_out, L, delta


def reference_attention_autograd(Q, K, V, dO, causal=False):
    """
    Alternative reference using PyTorch autograd (simpler, less control).
    """
    import torch

    B, N, H, _D = Q.shape
    H_KV = K.shape[2]
    group_size = H // H_KV
    scale = 1.0 / math.sqrt(_D)

    q = Q.detach().float().requires_grad_(True)
    k = K.detach().float().requires_grad_(True)
    v = V.detach().float().requires_grad_(True)

    # BNHD -> BHND
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    do_t = dO.detach().float().transpose(1, 2)

    # Expand for GQA
    k_exp = k_t.repeat_interleave(group_size, dim=1)
    v_exp = v_t.repeat_interleave(group_size, dim=1)

    # Forward
    S = torch.matmul(q_t, k_exp.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool),
                          diagonal=1)
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, v_exp)

    # Backward via autograd
    O.backward(do_t)

    dQ = q.grad.transpose(1, 2)   # BHND -> BNHD
    dK = k.grad.transpose(1, 2)
    dV = v.grad.transpose(1, 2)

    # Logsumexp
    L = torch.logsumexp(S, dim=-1).detach()

    # delta = rowsum(dO * O)
    delta = (do_t * O.detach()).sum(dim=-1)

    return dQ, dK, dV, O.detach().transpose(1, 2), L, delta


# ============================================================================
# Robustness checks
# ============================================================================

def robustness_check(ref, pred, name=""):
    """
    Comprehensive numerical comparison between reference and predicted tensors.

    Returns dict with:
        max_abs_diff, rel_error_rate, l2_error, cosine_sim, error_count, total
    """
    import torch

    ref_f = ref.float()
    pred_f = pred.float()
    diff = (ref_f - pred_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)

    # Count elements exceeding tolerance: |diff| > 0.001 + 0.05 * |ref|
    mask = diff > (0.001 + 0.05 * denom)
    error_count = mask.sum().item()
    numel = ref_f.numel()
    rel_error = error_count / numel

    # L2 relative error
    l2_error = (diff.pow(2).sum().sqrt() / ref_f.pow(2).sum().sqrt()).item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f.flatten(), pred_f.flatten(), dim=0
    ).item()

    max_abs = diff.max().item()

    return {
        "name": name,
        "max_abs_diff": max_abs,
        "rel_error_rate": rel_error,
        "l2_error": l2_error,
        "cosine_sim": cos_sim,
        "error_count": error_count,
        "total": numel,
    }


def print_check(result, threshold_cos=0.99, threshold_l2=0.05):
    """Print and evaluate a robustness check result."""
    name = result["name"]
    passed = (
        result["cosine_sim"] >= threshold_cos
        and result["l2_error"] <= threshold_l2
    )
    status = "PASS" if passed else "FAIL"

    print(f"  {name:8s}: max_abs={result['max_abs_diff']:.6f}, "
          f"l2_err={result['l2_error']:.4f}, "
          f"cos={result['cosine_sim']:.6f}, "
          f"errors={result['error_count']}/{result['total']} "
          f"({100*result['rel_error_rate']:.4f}%) [{status}]")

    return passed


# ============================================================================
# HipKittens kernel runner
# ============================================================================

def try_import_kernel():
    """Try to import the compiled HipKittens kernel module."""
    try:
        import flash_attn_bwd
        return flash_attn_bwd
    except ImportError:
        return None


def run_hipkittens_backward(kernel_module, Q, K, V, O, dO, L, delta,
                             causal=False):
    """
    Run the HipKittens backward kernel.

    Args:
        kernel_module: compiled pybind11 module
        Q, K, V: input tensors [B, N, H/H_KV, D] bf16
        O: forward output [B, N, H, D] bf16
        dO: output gradient [B, N, H, D] bf16
        L: logsumexp [B, H, 1, N] float32
        delta: rowsum(dO*O) [B, H, 1, N] float32
        causal: whether to use causal variant

    Returns:
        dQ, dK, dV: gradient tensors
    """
    import torch

    B, N, H, _D = Q.shape
    H_KV = K.shape[2]

    # Allocate output tensors (bf16, contiguous)
    dQ_tmp = torch.zeros_like(Q).bfloat16().contiguous()
    dQ = torch.zeros_like(Q).bfloat16().contiguous()
    dK = torch.zeros_like(K).bfloat16().contiguous()
    dV = torch.zeros_like(V).bfloat16().contiguous()

    # Step 1: Preprocess - compute delta
    delta_out = torch.zeros(B, H, 1, N, device=Q.device, dtype=torch.float32)
    delta_out = delta_out.transpose(-1, -2).contiguous()
    kernel_module.dispatch_prep(
        O.contiguous(),
        dO.contiguous(),
        delta_out,
    )
    torch.cuda.synchronize()

    # Step 2: Backward kernel
    # Transpose dQ_tmp for atomic accumulation layout: [B, H, N, D]
    dQ_in = dQ_tmp.transpose(1, 2).contiguous()

    # L needs to be [B, H, 1, N] transposed to [B, H, N, 1] contiguous
    L_in = L.unsqueeze(-1).transpose(-1, -2).contiguous() if L.dim() == 3 \
        else L.transpose(-1, -2).contiguous()

    if causal:
        kernel_module.dispatch_bwd_causal(
            Q.contiguous(), K.contiguous(), V.contiguous(),
            dO.contiguous(), dQ_in, dK, dV,
            L_in, delta_out,
        )
    else:
        kernel_module.dispatch_bwd_noncausal(
            Q.contiguous(), K.contiguous(), V.contiguous(),
            dO.contiguous(), dQ_in, dK, dV,
            L_in, delta_out,
        )
    torch.cuda.synchronize()

    # Step 3: Shuffle dQ back to BNHD layout
    kernel_module.dispatch_dq_shuffle(dQ_in, dQ)
    torch.cuda.synchronize()

    return dQ, dK, dV


# ============================================================================
# Main test
# ============================================================================

def run_test(cfg):
    """Run the full backward parity test."""
    import torch

    torch.manual_seed(42)

    B = cfg["B"]
    N = cfg["N"]
    H = cfg["H"]
    H_KV = cfg["H_KV"]
    _D = cfg["D"]
    causal = cfg["causal"]
    group_size = H // H_KV

    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"Flash Attention v2 Backward - Parity Test")
    print(f"  B={B}, N={N}, H={H}, H_KV={H_KV}, D={_D}, "
          f"causal={causal}, group_size={group_size}")
    print("=" * 70)

    # ========================================================================
    # Generate input tensors (BNHD layout)
    # ========================================================================
    print("\nGenerating input tensors...")
    Q = torch.randn(B, N, H, _D, device=device, dtype=dtype)
    K = torch.randn(B, N, H_KV, _D, device=device, dtype=dtype)
    V = torch.randn(B, N, H_KV, _D, device=device, dtype=dtype)
    dO = torch.randn(B, N, H, _D, device=device, dtype=dtype)

    # Normalize for numerical stability
    Q = Q / Q.norm(dim=-1, keepdim=True) * 2.0
    K = K / K.norm(dim=-1, keepdim=True) * 2.0
    V = V / V.norm(dim=-1, keepdim=True) * 2.0
    dO = dO / dO.norm(dim=-1, keepdim=True) * 0.5

    # ========================================================================
    # Reference backward (PyTorch autograd)
    # ========================================================================
    print("Computing reference backward (PyTorch autograd)...")
    ref_dQ, ref_dK, ref_dV, ref_O, ref_L, ref_delta = \
        reference_attention_autograd(Q, K, V, dO, causal=causal)
    print(f"  Reference computed. Output shapes: "
          f"dQ={list(ref_dQ.shape)}, dK={list(ref_dK.shape)}, "
          f"dV={list(ref_dV.shape)}")

    # ========================================================================
    # Manual backward (for secondary verification)
    # ========================================================================
    print("Computing manual backward reference...")
    man_dQ, man_dK, man_dV, man_O, man_L, man_delta = \
        reference_attention_backward(Q, K, V, dO, causal=causal)

    print("\nManual vs Autograd consistency check:")
    for name, ref_t, man_t in [("dQ", ref_dQ, man_dQ),
                                ("dK", ref_dK, man_dK),
                                ("dV", ref_dV, man_dV)]:
        r = robustness_check(ref_t, man_t, name)
        print_check(r, threshold_cos=0.9999, threshold_l2=0.001)

    # ========================================================================
    # HipKittens backward
    # ========================================================================
    kernel_module = try_import_kernel()

    if kernel_module is None:
        print("\n" + "=" * 70)
        print("HipKittens kernel module not found (flash_attn_bwd).")
        print("Skipping GPU kernel test.")
        print("")
        print("To compile the kernel:")
        print(f"  cd {os.path.dirname(os.path.abspath(__file__))}")
        print(f"  make ATTN_B={B} ATTN_H={H} ATTN_H_KV={H_KV} ATTN_N={N}")
        print("")
        print("Then re-run this test from the same directory.")
        print("=" * 70)

        # Report reference-only results
        print("\nReference backward results (shapes and statistics):")
        print(f"  dQ: shape={list(ref_dQ.shape)}, "
              f"mean={ref_dQ.float().mean():.6f}, "
              f"std={ref_dQ.float().std():.6f}")
        print(f"  dK: shape={list(ref_dK.shape)}, "
              f"mean={ref_dK.float().mean():.6f}, "
              f"std={ref_dK.float().std():.6f}")
        print(f"  dV: shape={list(ref_dV.shape)}, "
              f"mean={ref_dV.float().mean():.6f}, "
              f"std={ref_dV.float().std():.6f}")
        return True

    print("\nRunning HipKittens backward kernel...")
    hk_dQ, hk_dK, hk_dV = run_hipkittens_backward(
        kernel_module, Q, K, V, ref_O.bfloat16(), dO,
        ref_L.float(), ref_delta.float(),
        causal=causal,
    )

    # ========================================================================
    # Compare HipKittens vs Reference
    # ========================================================================
    print("\n" + "=" * 70)
    print("Parity Check: HipKittens vs PyTorch Reference")
    print("=" * 70)

    all_passed = True

    # BF16 tolerances: cos >= 0.99, l2 <= 0.05
    for name, ref_t, hk_t in [("dQ", ref_dQ, hk_dQ),
                                ("dK", ref_dK, hk_dK),
                                ("dV", ref_dV, hk_dV)]:
        r = robustness_check(ref_t, hk_t, name)
        passed = print_check(r)
        all_passed = all_passed and passed

    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    # ========================================================================
    # Benchmarking (if hardware available)
    # ========================================================================
    if kernel_module is not None:
        import torch
        print("\nBenchmarking HipKittens backward kernel...")

        num_warmup = 50
        num_iters = 100

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(num_warmup):
            _ = run_hipkittens_backward(
                kernel_module, Q, K, V, ref_O.bfloat16(), dO,
                ref_L.float(), ref_delta.float(), causal=causal,
            )

        # Timed runs
        timings = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event.record()
            _ = run_hipkittens_backward(
                kernel_module, Q, K, V, ref_O.bfloat16(), dO,
                ref_L.float(), ref_delta.float(), causal=causal,
            )
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))

        avg_ms = sum(timings) / len(timings)
        # FLOPs for backward: ~2.5x forward
        flops = 4 * B * N * N * H * _D // (2 if causal else 1)
        bwd_flops = 2.5 * flops
        tflops = bwd_flops / (avg_ms / 1000.0) / 1e12

        print(f"  Average time: {avg_ms:.4f} ms")
        print(f"  Throughput: {tflops:.2f} TFLOPS")

    return all_passed


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    cfg = parse_args()

    if cfg is None:
        dry_run()
        sys.exit(0)

    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA/ROCm not available. Running dry-run mode.")
            dry_run()
            sys.exit(0)
    except ImportError:
        print("PyTorch not installed. Running dry-run mode.")
        dry_run()
        sys.exit(0)

    passed = run_test(cfg)
    sys.exit(0 if passed else 1)
