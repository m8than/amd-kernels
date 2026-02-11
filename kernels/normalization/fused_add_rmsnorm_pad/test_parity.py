"""
Parity test for HipKittens Fused Add + RMSNorm + Pad kernel vs PyTorch reference.

Tests two kernel variants:
  1. fused_add_rmsnorm_pad — residual add + RMSNorm + padded output
  2. rmsnorm_pad           — RMSNorm only + padded output

The padding feature writes to an output tensor that may be wider than
the input (N_OUT >= N), with the extra columns zero-filled.

When GPU hardware is available and the .so is compiled, set USE_HK=True
to compare HipKittens output against the PyTorch reference.
"""

import torch
import os
import sys

# Toggle this when HIP hardware + compiled .so are available
USE_HK = False
DEVICE = "cuda" if torch.cuda.is_available() and USE_HK else "cpu"

# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------

def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight"""
    x_f32 = x.float()
    mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(mean_sq + eps)
    return (x_f32 * norm_factor * weight.float()).to(x.dtype)


def fused_add_rmsnorm_pad_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    n_out: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Add + RMSNorm + Pad:
        res_out = x + residual
        normed = rmsnorm(res_out) * weight
        output[:, :N] = normed; output[:, N:N_OUT] = 0

    Returns (output, res_out).
    """
    n_rows, n = x.shape
    res_out = (x.float() + residual.float()).to(x.dtype)
    normed = rmsnorm_ref(res_out, weight, eps)

    # Pad if needed
    if n_out > n:
        output = torch.zeros(n_rows, n_out, dtype=x.dtype, device=x.device)
        output[:, :n] = normed
    else:
        output = normed

    return output, res_out


def rmsnorm_pad_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_out: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm + Pad (no residual):
        normed = rmsnorm(x) * weight
        output[:, :N] = normed; output[:, N:N_OUT] = 0
    """
    n_rows, n = x.shape
    normed = rmsnorm_ref(x, weight, eps)

    if n_out > n:
        output = torch.zeros(n_rows, n_out, dtype=x.dtype, device=x.device)
        output[:, :n] = normed
    else:
        output = normed

    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fused_add_rmsnorm_pad():
    """Test Fused Add + RMSNorm + Pad."""
    print("=" * 60)
    print("Test: fused_add_rmsnorm_pad")
    print("=" * 60)

    B, S, N = 4, 1024, 128
    N_OUT = 128  # Match compiled kernel; change for padded variant
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(B * S, N, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(B * S, N, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    ref_output, ref_res_out = fused_add_rmsnorm_pad_ref(x, residual, weight, N_OUT, eps)

    print(f"  Input shape:    {x.shape}, dtype: {x.dtype}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Weight shape:   {weight.shape}")
    print(f"  N={N}, N_OUT={N_OUT}")
    print(f"  Output shape:   {ref_output.shape}")
    print(f"  Res_out shape:  {ref_res_out.shape}")
    print(f"  Epsilon:        {eps}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import fused_add_rmsnorm_pad_tk

            output_hk = torch.zeros(B * S, N_OUT, dtype=torch.bfloat16, device=DEVICE)
            res_out_hk = torch.zeros_like(x)
            n_rows = x.shape[0]

            fused_add_rmsnorm_pad_tk.fused_add_rmsnorm_pad(
                x, residual, weight, output_hk, res_out_hk, eps, n_rows
            )
            torch.cuda.synchronize()

            max_diff_out = (ref_output - output_hk).abs().max().item()
            max_diff_res = (ref_res_out - res_out_hk).abs().max().item()
            print(f"  Max diff output:  {max_diff_out:.6f}")
            print(f"  Max diff res_out: {max_diff_res:.6f}")
            assert max_diff_out < 0.05, f"Output parity FAILED: {max_diff_out}"
            assert max_diff_res < 0.05, f"Res_out parity FAILED: {max_diff_res}"
            print("  PASS")
        except ImportError:
            print("  [SKIP] fused_add_rmsnorm_pad_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def test_rmsnorm_pad():
    """Test RMSNorm + Pad (no residual)."""
    print("=" * 60)
    print("Test: rmsnorm_pad")
    print("=" * 60)

    B, S, N = 4, 1024, 128
    N_OUT = 128
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(B * S, N, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    ref_output = rmsnorm_pad_ref(x, weight, N_OUT, eps)

    print(f"  Input shape:  {x.shape}, dtype: {x.dtype}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  N={N}, N_OUT={N_OUT}")
    print(f"  Output shape: {ref_output.shape}")
    print(f"  Epsilon:      {eps}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import fused_add_rmsnorm_pad_tk

            output_hk = torch.zeros(B * S, N_OUT, dtype=torch.bfloat16, device=DEVICE)
            n_rows = x.shape[0]

            fused_add_rmsnorm_pad_tk.rmsnorm_pad(x, weight, output_hk, eps, n_rows)
            torch.cuda.synchronize()

            max_diff = (ref_output - output_hk).abs().max().item()
            print(f"  Max diff vs HK: {max_diff:.6f}")
            assert max_diff < 0.05, f"Parity check FAILED: max_diff={max_diff}"
            print("  PASS")
        except ImportError:
            print("  [SKIP] fused_add_rmsnorm_pad_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def test_padded_output():
    """Test with actual padding (N_OUT > N) — reference only."""
    print("=" * 60)
    print("Test: padded output (reference only)")
    print("=" * 60)

    N, N_OUT = 128, 256
    n_rows = 32
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(n_rows, N, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(n_rows, N, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)

    output, res_out = fused_add_rmsnorm_pad_ref(x, residual, weight, N_OUT, eps)

    # Verify padding: columns N..N_OUT should be zero
    padding_region = output[:, N:]
    assert padding_region.abs().max().item() == 0.0, "Padding region is not zero!"

    print(f"  N={N}, N_OUT={N_OUT}")
    print(f"  Output shape: {output.shape}")
    print(f"  Active region (cols 0..{N}): non-zero")
    print(f"  Padding region (cols {N}..{N_OUT}): all zeros ✓")
    print("  REFERENCE COMPUTED SUCCESSFULLY")
    print()


def print_build_instructions():
    """Print compilation instructions."""
    print("=" * 60)
    print("Build Instructions")
    print("=" * 60)
    print("  cd kernels/fused_add_rmsnorm_pad && make")
    print("  # Then set USE_HK=True in this script and re-run")
    print("  # For padded variant (N_OUT > N), modify D_OUT in kernel.cpp")
    print()


if __name__ == "__main__":
    test_fused_add_rmsnorm_pad()
    test_rmsnorm_pad()
    test_padded_output()
    print_build_instructions()
