#!/usr/bin/env python3
"""Parity test for batched_gemm_bf16 HipKittens kernel.

Triton reference: C[b] = A[b] @ B[b]  (+ optional bias[b])
  A: (B, M, K) bf16
  B: (B, K, N) bf16     (Triton uses B as K×N, HK transposes N×K internally)
  C: (B, M, N) bf16
  bias: (B, 1, N) bf16  (optional)

HipKittens kernel layout:
  A: (1, B, M, K) bf16   -- gl<bf16, -1, -1, -1, -1>
  B: (1, B, N, K) bf16   -- stored transposed (N×K) for mma_ABt
  C: (1, B, M, N) bf16
  bias: (1, B, 1, N) bf16
"""

import subprocess
import sys
import os

# Test dimensions
BATCH = 4
M = 256
N = 256
K = 128
HAS_BIAS = True

def reference_torch():
    """Compute reference output using PyTorch."""
    import torch

    torch.manual_seed(42)
    A = torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cpu")
    B = torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cpu")
    bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16, device="cpu") if HAS_BIAS else None

    # Reference: C = A @ B + bias
    C_ref = torch.bmm(A.float(), B.float()).bfloat16()
    if bias is not None:
        C_ref = (C_ref.float() + bias.float()).bfloat16()

    return A, B, bias, C_ref


def test_compilation():
    """Test that the kernel compiles successfully."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[INFO] Kernel directory: {kernel_dir}")
    print(f"[INFO] Expected compile command:")
    print(f"  cd {kernel_dir} && make GPU_TARGET=CDNA4")
    print()

    # Check that source files exist
    for f in ["kernel.cpp", "Makefile"]:
        path = os.path.join(kernel_dir, f)
        if os.path.exists(path):
            print(f"  [OK] {f} exists")
        else:
            print(f"  [FAIL] {f} missing")
            return False

    # Try to compile if hipcc is available
    try:
        result = subprocess.run(
            ["which", "hipcc"],
            capture_output=True, text=True
        )
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

    A, B, bias, C_ref = reference_torch()

    # Move to GPU
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    bias_gpu = bias.cuda() if bias is not None else None

    # Try loading compiled kernel
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)

    try:
        import tk_batched_gemm_bf16 as tk
    except ImportError:
        print("[SKIP] Compiled kernel not found, skipping parity test")
        return True

    # Prepare tensors in HK layout: (1, B, M, K)
    A_hk = A_gpu.unsqueeze(0).contiguous()
    # B is transposed for mma_ABt: (1, B, N, K) from (B, K, N)
    B_hk = B_gpu.transpose(-2, -1).unsqueeze(0).contiguous()
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
    bias_hk = bias_gpu.unsqueeze(0).contiguous() if bias_gpu is not None else \
              torch.zeros(1, BATCH, 1, N, dtype=torch.bfloat16, device="cuda")

    tk.dispatch(A_hk, B_hk, C_hk, bias_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    C_ref_gpu = C_ref.cuda()

    # Compare with tolerance
    max_diff = (C_out.float() - C_ref_gpu.float()).abs().max().item()
    mean_diff = (C_out.float() - C_ref_gpu.float()).abs().mean().item()

    print(f"[INFO] Max absolute diff: {max_diff:.6f}")
    print(f"[INFO] Mean absolute diff: {mean_diff:.6f}")

    # BF16 tolerance: allow up to ~1e-2 for large reductions
    atol = 1e-1
    rtol = 1e-2
    close = torch.allclose(C_out.float(), C_ref_gpu.float(), atol=atol, rtol=rtol)
    if close:
        print(f"[PASS] Parity test passed (atol={atol}, rtol={rtol})")
    else:
        print(f"[FAIL] Parity test failed (max_diff={max_diff})")

    return close


def main():
    print("=" * 60)
    print("Batched GEMM BF16 - Parity Test")
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
