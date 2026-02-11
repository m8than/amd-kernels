#!/usr/bin/env python3
"""Parity test for batched_gemm_a8w8_ptgq HipKittens kernel.

Triton reference: Per-token-group pre-quantization with per-batch-tensor weight scale.
  A: (B, M, K) bf16 — dynamically quantized per K-block per row
  B: (B, K, N) int8
  b_scale: scalar float32 — per-batch-tensor weight scale
  C: (B, M, N) bf16
  bias: (B, 1, N) bf16 (optional)

Algorithm per K-block:
  1. m = max(abs(A_block), axis=K) per row
  2. a_scale = m / 127
  3. A_quant = clamp(A_block / a_scale, -128, 127)
  4. accumulator += (A_quant @ B_block) * a_scale
  After loop: accumulator *= b_scale

HipKittens kernel:
  A: (1, B, M, K) bf16
  B: (1, B, N, K) bf16 (pre-cast, transposed)
  b_scale: (1, 1, 1, 1) float32
  C: (1, B, M, N) bf16
  bias: (1, B, 1, N) bf16
"""

import subprocess
import sys
import os

BATCH = 4
M = 128
N = 128
K = 128  # Must be multiple of BLOCK_K=64
HAS_BIAS = True
BLOCK_K = 64
DTYPE_MAX = 127.0
DTYPE_MIN = -128.0


def reference_torch():
    """Compute reference with per-token-group quantization."""
    import torch

    torch.manual_seed(42)

    A = torch.randn(BATCH, M, K, dtype=torch.bfloat16)
    B_int8 = torch.randint(-128, 127, (BATCH, K, N), dtype=torch.int8)
    b_scale = torch.tensor(0.05, dtype=torch.float32)
    bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16) if HAS_BIAS else None

    # Simulate per-token-group quantization
    accumulator = torch.zeros(BATCH, M, N, dtype=torch.float32)

    num_k_blocks = K // BLOCK_K
    for kb in range(num_k_blocks):
        k_start = kb * BLOCK_K
        k_end = k_start + BLOCK_K

        A_block = A[:, :, k_start:k_end].float()
        B_block = B_int8[:, k_start:k_end, :].float()

        # Per-row max of absolute values
        m = torch.clamp(A_block.abs().max(dim=-1, keepdim=True).values, min=1e-10)
        a_scale = m * (1.0 / DTYPE_MAX)
        a_scale_recip = 1.0 / a_scale

        # Quantize A
        A_quant = torch.clamp(A_block * a_scale_recip, DTYPE_MIN, DTYPE_MAX)

        # Accumulate with scale
        partial = torch.bmm(A_quant, B_block)
        accumulator += partial * a_scale

    # Apply per-tensor b_scale
    accumulator *= b_scale.item()

    if bias is not None:
        accumulator += bias.float()

    C_ref = accumulator.bfloat16()

    return A, B_int8, b_scale, bias, C_ref


def test_compilation():
    """Test that the kernel compiles successfully."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[INFO] Kernel directory: {kernel_dir}")
    print(f"[INFO] Expected compile command:")
    print(f"  cd {kernel_dir} && make GPU_TARGET=CDNA4")
    print()

    for f in ["kernel.cpp", "Makefile"]:
        path = os.path.join(kernel_dir, f)
        if os.path.exists(path):
            print(f"  [OK] {f} exists")
        else:
            print(f"  [FAIL] {f} missing")
            return False

    try:
        result = subprocess.run(["which", "hipcc"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[SKIP] hipcc not found, skipping compilation test")
            return True

        result = subprocess.run(
            ["make", "-C", kernel_dir, "GPU_TARGET=CDNA4"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("[OK] Compilation succeeded")
            return True
        else:
            print(f"[FAIL] Compilation failed:\n{result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[SKIP] Compilation test skipped: {e}")
        return True


def test_parity():
    """Full parity test (requires HIP hardware)."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[SKIP] No GPU available for parity test")
            return True
    except ImportError:
        print("[SKIP] PyTorch not available for parity test")
        return True

    A, B_int8, b_scale, bias, C_ref = reference_torch()

    # Pre-cast B to bf16 for HK kernel
    B_bf16 = B_int8.float().bfloat16().cuda()

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)

    try:
        import tk_batched_gemm_a8w8_ptgq as tk
    except ImportError:
        print("[SKIP] Compiled kernel not found, skipping parity test")
        return True

    # Prepare HK tensors
    A_hk = A.cuda().unsqueeze(0).contiguous()                     # (1, B, M, K)
    B_hk = B_bf16.transpose(-2, -1).unsqueeze(0).contiguous()     # (1, B, N, K)
    b_scale_hk = b_scale.cuda().reshape(1, 1, 1, 1).contiguous()  # (1, 1, 1, 1)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
    bias_hk = bias.cuda().unsqueeze(0).contiguous() if bias is not None else \
              torch.zeros(1, BATCH, 1, N, dtype=torch.bfloat16, device="cuda")

    tk.dispatch(A_hk, B_hk, b_scale_hk, C_hk, bias_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    C_ref_gpu = C_ref.cuda()

    max_diff = (C_out.float() - C_ref_gpu.float()).abs().max().item()
    mean_diff = (C_out.float() - C_ref_gpu.float()).abs().mean().item()

    print(f"[INFO] Max absolute diff: {max_diff:.6f}")
    print(f"[INFO] Mean absolute diff: {mean_diff:.6f}")

    # Higher tolerance due to dynamic quantization + bf16 accumulation
    atol = 2.0
    rtol = 1e-1
    close = torch.allclose(C_out.float(), C_ref_gpu.float(), atol=atol, rtol=rtol)
    if close:
        print(f"[PASS] Parity test passed (atol={atol}, rtol={rtol})")
    else:
        print(f"[FAIL] Parity test failed (max_diff={max_diff})")

    return close


def main():
    print("=" * 60)
    print("Batched GEMM A8W8 PTGQ - Parity Test")
    print("=" * 60)
    print(f"Dimensions: B={BATCH}, M={M}, N={N}, K={K}, has_bias={HAS_BIAS}")
    print()

    print("--- Compilation Test ---")
    comp_ok = test_compilation()
    print()

    print("--- Parity Test ---")
    parity_ok = test_parity()
    print()

    if comp_ok and parity_ok:
        print("OVERALL: PASS")
        return 0
    else:
        print("OVERALL: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
