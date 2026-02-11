"""
Parity test for HipKittens RMSNorm kernels vs PyTorch reference.

Tests two kernel variants:
  1. rmsnorm_fwd       — basic RMS normalization
  2. fused_add_rmsnorm — fused residual add + RMS normalization

When GPU hardware is available and the .so is compiled, set USE_HK=True
to compare HipKittens output against the PyTorch reference.
"""

import torch
import os
import subprocess
import sys

# Toggle this when HIP hardware + compiled .so are available
USE_HK = False
DEVICE = "cuda" if torch.cuda.is_available() and USE_HK else "cpu"

# ---------------------------------------------------------------------------
# PyTorch reference implementations (matching the Triton kernel math)
# ---------------------------------------------------------------------------

def rmsnorm_fwd_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm forward: output = (x / sqrt(mean(x^2) + eps)) * weight"""
    x_f32 = x.float()
    mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(mean_sq + eps)
    return (x_f32 * norm_factor * weight.float()).to(x.dtype)


def fused_add_rmsnorm_fwd_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Add + RMSNorm:
        res_out = x + residual
        output  = rmsnorm(res_out) * weight
    Returns (output, res_out).
    """
    res_out = (x.float() + residual.float()).to(x.dtype)
    output = rmsnorm_fwd_ref(res_out, weight, eps)
    return output, res_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rmsnorm_fwd():
    """Test basic RMSNorm forward pass."""
    print("=" * 60)
    print("Test: rmsnorm_fwd")
    print("=" * 60)

    # Typical LLM dimensions
    B, S, D = 4, 1024, 128
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    # Reference output
    ref_output = rmsnorm_fwd_ref(x, weight, eps)

    print(f"  Input shape:  {x.shape}, dtype: {x.dtype}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Output shape: {ref_output.shape}")
    print(f"  Epsilon:      {eps}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import rmsnorm_tk

            output_hk = torch.zeros_like(ref_output)
            n_rows = x.shape[0]

            rmsnorm_tk.rmsnorm_fwd(x, weight, output_hk, eps, n_rows)
            torch.cuda.synchronize()

            max_diff = (ref_output - output_hk).abs().max().item()
            print(f"  Max diff vs HK: {max_diff:.6f}")
            assert max_diff < 0.05, f"Parity check FAILED: max_diff={max_diff}"
            print("  PASS")
        except ImportError:
            print("  [SKIP] rmsnorm_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def test_fused_add_rmsnorm_fwd():
    """Test Fused Add + RMSNorm forward pass."""
    print("=" * 60)
    print("Test: fused_add_rmsnorm_fwd")
    print("=" * 60)

    B, S, D = 4, 1024, 128
    eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    ref_output, ref_res_out = fused_add_rmsnorm_fwd_ref(x, residual, weight, eps)

    print(f"  Input shape:    {x.shape}, dtype: {x.dtype}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Weight shape:   {weight.shape}")
    print(f"  Output shape:   {ref_output.shape}")
    print(f"  Res_out shape:  {ref_res_out.shape}")
    print(f"  Epsilon:        {eps}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import rmsnorm_tk

            output_hk = torch.zeros_like(ref_output)
            res_out_hk = torch.zeros_like(ref_res_out)
            n_rows = x.shape[0]

            rmsnorm_tk.fused_add_rmsnorm_fwd(
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
            print("  [SKIP] rmsnorm_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def print_build_instructions():
    """Print compilation instructions."""
    print("=" * 60)
    print("Build Instructions")
    print("=" * 60)
    print("  cd kernels/rmsnorm && make")
    print("  # Then set USE_HK=True in this script and re-run")
    print()


if __name__ == "__main__":
    test_rmsnorm_fwd()
    test_fused_add_rmsnorm_fwd()
    print_build_instructions()
