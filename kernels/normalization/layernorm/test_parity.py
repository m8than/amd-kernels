"""
Parity test for HipKittens LayerNorm kernels vs PyTorch reference.

Tests two kernel variants:
  1. layernorm_fwd       — standard layer normalization
  2. fused_add_layernorm — fused residual add + layer normalization

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
# PyTorch reference implementations (matching the Triton kernel math)
# ---------------------------------------------------------------------------

def layernorm_fwd_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm forward: output = ((x - mean) / sqrt(var + eps)) * weight + bias"""
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    normalized = (x_f32 - mean) * rstd
    output = normalized * weight.float() + bias.float()
    return output.to(x.dtype)


def fused_add_layernorm_fwd_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Add + LayerNorm:
        res_out = x + residual
        output  = layernorm(res_out) * weight + bias
    Returns (output, res_out).
    """
    res_out = (x.float() + residual.float()).to(x.dtype)
    output = layernorm_fwd_ref(res_out, weight, bias, eps)
    return output, res_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_layernorm_fwd():
    """Test basic LayerNorm forward pass."""
    print("=" * 60)
    print("Test: layernorm_fwd")
    print("=" * 60)

    B, S, D = 4, 1024, 128
    eps = 1e-5

    torch.manual_seed(42)
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    ref_output = layernorm_fwd_ref(x, weight, bias, eps)

    # Also compare with torch.nn.functional.layer_norm
    ref_torch = torch.nn.functional.layer_norm(
        x.float(), [D], weight.float(), bias.float(), eps
    ).to(x.dtype)
    torch_diff = (ref_output - ref_torch).abs().max().item()

    print(f"  Input shape:  {x.shape}, dtype: {x.dtype}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Bias shape:   {bias.shape}")
    print(f"  Output shape: {ref_output.shape}")
    print(f"  Epsilon:      {eps}")
    print(f"  Diff vs torch.layer_norm: {torch_diff:.8f}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import layernorm_tk

            output_hk = torch.zeros_like(ref_output)
            n_rows = x.shape[0]

            layernorm_tk.layernorm_fwd(x, weight, bias, output_hk, eps, n_rows)
            torch.cuda.synchronize()

            max_diff = (ref_output - output_hk).abs().max().item()
            print(f"  Max diff vs HK: {max_diff:.6f}")
            assert max_diff < 0.05, f"Parity check FAILED: max_diff={max_diff}"
            print("  PASS")
        except ImportError:
            print("  [SKIP] layernorm_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def test_fused_add_layernorm_fwd():
    """Test Fused Add + LayerNorm forward pass."""
    print("=" * 60)
    print("Test: fused_add_layernorm_fwd")
    print("=" * 60)

    B, S, D = 4, 1024, 128
    eps = 1e-5

    torch.manual_seed(42)
    x = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(B * S, D, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn(D, dtype=torch.bfloat16, device=DEVICE)

    ref_output, ref_res_out = fused_add_layernorm_fwd_ref(x, residual, weight, bias, eps)

    print(f"  Input shape:    {x.shape}, dtype: {x.dtype}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Weight shape:   {weight.shape}")
    print(f"  Bias shape:     {bias.shape}")
    print(f"  Output shape:   {ref_output.shape}")
    print(f"  Res_out shape:  {ref_res_out.shape}")
    print(f"  Epsilon:        {eps}")

    if USE_HK:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import layernorm_tk

            output_hk = torch.zeros_like(ref_output)
            res_out_hk = torch.zeros_like(ref_res_out)
            n_rows = x.shape[0]

            layernorm_tk.fused_add_layernorm_fwd(
                x, residual, weight, bias, output_hk, res_out_hk, eps, n_rows
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
            print("  [SKIP] layernorm_tk.so not found — compile with `make`")
    else:
        print("  REFERENCE COMPUTED SUCCESSFULLY (HK comparison skipped)")

    print()


def print_build_instructions():
    """Print compilation instructions."""
    print("=" * 60)
    print("Build Instructions")
    print("=" * 60)
    print("  cd kernels/layernorm && make")
    print("  # Then set USE_HK=True in this script and re-run")
    print()


if __name__ == "__main__":
    test_layernorm_fwd()
    test_fused_add_layernorm_fwd()
    print_build_instructions()
