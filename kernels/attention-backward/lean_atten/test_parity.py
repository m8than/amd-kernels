"""
Lean Attention Forward - Parity Test
=====================================
Tests the HipKittens lean attention forward kernel against a PyTorch
reference implementation.

Input tensors (BF16):
    Q: [B, N, H, D]   - queries
    K: [B, N, H, D]   - keys
    V: [B, N, H, D]   - values

Output tensors:
    O: [B, N, H, D]   - attention output (BF16)
    L: [B, H, 1, N//Q_WG_SIZE] - log-sum-exp per Q tile (FP32)

Algorithm:
    S = softmax_scale * Q @ K^T
    P = softmax(S)       (with causal mask, online softmax via exp2)
    O = P @ V

Build command:
    cd kernels/lean_atten
    make ATTN_B=4 ATTN_H=32 ATTN_N=1024 ATTN_D=128 GPU_TARGET=CDNA4

Run:
    python test_parity.py [B] [N] [H] [D]
"""

import sys
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Configuration (matches kernel compile-time defaults)
# ---------------------------------------------------------------------------
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4       # batch
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024    # sequence length
H = int(sys.argv[3]) if len(sys.argv) > 3 else 32      # heads
D = int(sys.argv[4]) if len(sys.argv) > 4 else 128     # head dim
causal = True
dtype = torch.bfloat16

Q_WG_SIZE = 32  # per-warp Q rows in the HipKittens kernel

print(f"Lean Attention Parity Test: B={B}, N={N}, H={H}, D={D}, causal={causal}")
print(f"  dtype={dtype}, Q_WG_SIZE={Q_WG_SIZE}")
print()

# ---------------------------------------------------------------------------
# Generate random inputs
# ---------------------------------------------------------------------------
Q = torch.randn(B, N, H, D, dtype=dtype, device='cuda')
K = torch.randn(B, N, H, D, dtype=dtype, device='cuda')
V = torch.randn(B, N, H, D, dtype=dtype, device='cuda')

# ---------------------------------------------------------------------------
# PyTorch reference attention
# ---------------------------------------------------------------------------
def reference_attention(Q, K, V, causal=True):
    """
    Standard scaled dot-product attention with optional causal mask.
    Input shapes: [B, N, H, D]
    Output shape: [B, N, H, D]
    """
    B, N, H, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # Transpose to [B, H, N, D] for batch matmul
    q = Q.transpose(1, 2).float()  # [B, H, N, D]
    k = K.transpose(1, 2).float()  # [B, H, N, D]
    v = V.transpose(1, 2).float()  # [B, H, N, D]

    # S = Q @ K^T * scale
    S = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    P = torch.softmax(S, dim=-1)

    # O = P @ V
    O = torch.matmul(P, v)  # [B, H, N, D]

    # Transpose back to [B, N, H, D]
    return O.transpose(1, 2).to(dtype)


print("Computing PyTorch reference...")
ref_out = reference_attention(Q, K, V, causal=causal)
print(f"  ref_out shape: {ref_out.shape}, dtype: {ref_out.dtype}")
print(f"  ref_out range: [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")
print()

# ---------------------------------------------------------------------------
# Also test with torch.nn.functional.scaled_dot_product_attention
# ---------------------------------------------------------------------------
print("Computing F.scaled_dot_product_attention reference...")
q_sdpa = Q.transpose(1, 2).float()
k_sdpa = K.transpose(1, 2).float()
v_sdpa = V.transpose(1, 2).float()
sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)
sdpa_out = sdpa_out.transpose(1, 2).to(dtype)
print(f"  sdpa_out shape: {sdpa_out.shape}, dtype: {sdpa_out.dtype}")
print()

# Cross-check the two references
ref_vs_sdpa = (ref_out.float() - sdpa_out.float()).abs()
print(f"  Reference vs SDPA max diff: {ref_vs_sdpa.max().item():.6f}")
print()

# ---------------------------------------------------------------------------
# HipKittens kernel test (requires compiled kernel)
# ---------------------------------------------------------------------------
try:
    import lean_attn_kernel

    print("HipKittens kernel found. Running parity check...")

    # Allocate output tensors
    hk_out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda')
    num_q_tiles = N // Q_WG_SIZE
    hk_lse = torch.zeros(B, H, 1, num_q_tiles, dtype=torch.float32, device='cuda')

    # Run kernel
    lean_attn_kernel.dispatch(Q, K, V, hk_out, hk_lse)
    torch.cuda.synchronize()

    print(f"  hk_out shape: {hk_out.shape}, dtype: {hk_out.dtype}")
    print(f"  hk_out range: [{hk_out.min().item():.4f}, {hk_out.max().item():.4f}]")

    # Compare with reference
    diff = (hk_out.float() - ref_out.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    # Relative error
    denom = ref_out.float().abs().clamp_min(1e-6)
    rel_err = (diff / denom)
    max_rel_err = rel_err.max().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        hk_out.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()

    # Element-wise tolerance check
    rtol = 1e-2
    atol = 1e-2
    close_mask = torch.isclose(hk_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    pct_close = close_mask.float().mean().item() * 100

    print()
    print("  === Parity Results ===")
    print(f"  Max absolute error:  {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Max relative error:  {max_rel_err:.6f}")
    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print(f"  % elements within tol (rtol={rtol}, atol={atol}): {pct_close:.2f}%")
    print()

    # Pass/Fail
    passed = (max_abs_err < 0.05 and cos_sim > 0.99)
    if passed:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")
        print(f"  Criteria: max_abs_err < 0.05 ({max_abs_err:.6f}), "
              f"cos_sim > 0.99 ({cos_sim:.6f})")

    # Print first few elements for debugging
    num_show = 8
    print()
    print(f"  First {num_show} output elements [0,0,0,:]:")
    print(f"    HK:  {hk_out[0, 0, 0, :num_show].tolist()}")
    print(f"    Ref: {ref_out[0, 0, 0, :num_show].tolist()}")

except ImportError:
    print("HipKittens kernel module not found (lean_attn_kernel).")
    print("This is expected if not running on AMD GPU hardware.")
    print()
    print("To compile and run:")
    print("  1. Set environment variables:")
    print("     export ROCM_PATH=/opt/rocm")
    print("     export THUNDERKITTENS_ROOT=/path/to/hipkittens")
    print("  2. Build the kernel:")
    print(f"     make ATTN_B={B} ATTN_H={H} ATTN_N={N} ATTN_D={D}")
    print("  3. Run this test:")
    print(f"     python test_parity.py {B} {N} {H} {D}")
    print()
    print("Expected input shapes:")
    print(f"  Q: [{B}, {N}, {H}, {D}] bf16")
    print(f"  K: [{B}, {N}, {H}, {D}] bf16")
    print(f"  V: [{B}, {N}, {H}, {D}] bf16")
    print()
    print("Expected output shapes:")
    print(f"  O:   [{B}, {N}, {H}, {D}] bf16")
    print(f"  LSE: [{B}, {H}, 1, {N // Q_WG_SIZE}] float32")
    print()
    print("Reference output computed successfully.")
    print(f"  ref_out shape: {ref_out.shape}")
    print(f"  ref_out range: [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")
    print()
    print("RESULT: SKIP (no GPU hardware / kernel not compiled)")

except Exception as e:
    print(f"Error running HipKittens kernel: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("RESULT: ERROR")
