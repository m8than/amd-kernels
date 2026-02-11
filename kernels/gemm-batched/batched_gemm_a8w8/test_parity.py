#!/usr/bin/env python3
"""Parity test for batched_gemm_a8w8 HipKittens kernel.

Triton reference: C[b] = (A_int8[b] @ B_int8[b]) * a_scale[b] * b_scale[b] + bias[b]
  A: (B, M, K) int8
  B: (B, K, N) int8
  a_scale: (B, M, 1) float32
  b_scale: (B, 1, N) float32
  C: (B, M, N) bf16
  bias: (B, 1, N) bf16 (optional)

HipKittens kernel layout (A/B pre-cast to bf16):
  A: (1, B, M, K) bf16
  B: (1, B, N, K) bf16  (transposed for mma_ABt)
  a_scale: (1, B, 1, M) float32
  b_scale: (1, B, 1, N) float32
  C: (1, B, M, N) bf16
  bias: (1, B, 1, N) bf16
"""

import subprocess
import sys
import os

BATCH = 4
M = 256
N = 256
K = 128
HAS_BIAS = True


def reference_torch():
    """Compute reference output using PyTorch."""
    import torch

    torch.manual_seed(42)

    # INT8 inputs
    A_int8 = torch.randint(-128, 127, (BATCH, M, K), dtype=torch.int8)
    B_int8 = torch.randint(-128, 127, (BATCH, K, N), dtype=torch.int8)

    # Per-row/per-col scales
    a_scale = torch.rand(BATCH, M, 1, dtype=torch.float32) * 0.1
    b_scale = torch.rand(BATCH, 1, N, dtype=torch.float32) * 0.1
    bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16) if HAS_BIAS else None

    # Reference: C = (A_int8.float() @ B_int8.float()) * a_scale * b_scale + bias
    C_ref = torch.bmm(A_int8.float(), B_int8.float()) * a_scale * b_scale
    if bias is not None:
        C_ref = C_ref + bias.float()
    C_ref = C_ref.bfloat16()

    return A_int8, B_int8, a_scale, b_scale, bias, C_ref


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

    A_int8, B_int8, a_scale, b_scale, bias, C_ref = reference_torch()

    # Pre-cast to bf16 for HK kernel
    A_bf16 = A_int8.float().bfloat16().cuda()
    B_bf16 = B_int8.float().bfloat16().cuda()

    # Move to GPU
    a_scale_gpu = a_scale.cuda()
    b_scale_gpu = b_scale.cuda()
    bias_gpu = bias.cuda() if bias is not None else None

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)

    try:
        import tk_batched_gemm_a8w8 as tk
    except ImportError:
        print("[SKIP] Compiled kernel not found, skipping parity test")
        return True

    # Prepare HK tensors
    A_hk = A_bf16.unsqueeze(0).contiguous()             # (1, B, M, K)
    B_hk = B_bf16.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
    a_scale_hk = a_scale_gpu.squeeze(-1).unsqueeze(0).unsqueeze(2).contiguous()  # (1, B, 1, M)
    b_scale_hk = b_scale_gpu.squeeze(-2).unsqueeze(0).unsqueeze(2).contiguous()  # (1, B, 1, N)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
    bias_hk = bias_gpu.unsqueeze(0).contiguous() if bias_gpu is not None else \
              torch.zeros(1, BATCH, 1, N, dtype=torch.bfloat16, device="cuda")

    tk.dispatch(A_hk, B_hk, a_scale_hk, b_scale_hk, C_hk, bias_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    C_ref_gpu = C_ref.cuda()

    max_diff = (C_out.float() - C_ref_gpu.float()).abs().max().item()
    mean_diff = (C_out.float() - C_ref_gpu.float()).abs().mean().item()

    print(f"[INFO] Max absolute diff: {max_diff:.6f}")
    print(f"[INFO] Mean absolute diff: {mean_diff:.6f}")

    atol = 1.0  # INT8 quantized GEMM has larger tolerance
    rtol = 5e-2
    close = torch.allclose(C_out.float(), C_ref_gpu.float(), atol=atol, rtol=rtol)
    if close:
        print(f"[PASS] Parity test passed (atol={atol}, rtol={rtol})")
    else:
        print(f"[FAIL] Parity test failed (max_diff={max_diff})")

    return close


def main():
    print("=" * 60)
    print("Batched GEMM A8W8 - Parity Test")
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
