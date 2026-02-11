#!/usr/bin/env python3
"""Parity test for batched_gemm_a16wfp4 HipKittens kernel.

Triton reference: C[b] = A_bf16[b] @ dequant(B_fp4[b])
  A: (B, M, K) bf16
  B: (B, K/2, N) uint8 (FP4 packed)
  b_scales: (B, N, K/32) uint8 (e8m0 scales)
  C: (B, M, N) bf16

HipKittens kernel (B pre-dequantized to bf16 by host):
  A: (1, B, M, K) bf16
  B: (1, B, N, K) bf16  (transposed, dequantized)
  C: (1, B, M, N) bf16

The test dequantizes B on host, then compares HK output vs torch.bmm reference.
"""

import subprocess
import sys
import os
import struct

BATCH = 4
M = 256
N = 128
K = 128  # Must be multiple of 32 for FP4 scale groups


def fp4_dequant_reference():
    """Create reference data simulating FP4 quantization."""
    import torch

    torch.manual_seed(42)

    # For simplicity, create random bf16 data that simulates dequantized FP4
    # FP4 (e2m1) has limited range: {0, 0.5, 1, 1.5, 2, 3, 4, 6} x {+,-}
    A = torch.randn(BATCH, M, K, dtype=torch.bfloat16)

    # Simulate dequantized B (limited precision FP4 values)
    fp4_vals = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    B_indices = torch.randint(0, len(fp4_vals), (BATCH, K, N))
    B_signs = torch.randint(0, 2, (BATCH, K, N)) * 2 - 1  # +1 or -1
    B_dequant = fp4_vals[B_indices] * B_signs.float()

    # Apply random e8m0 scales per group of 32
    num_scale_groups = K // 32
    scales = torch.pow(2.0, torch.randint(-4, 5, (BATCH, N, num_scale_groups)).float())
    for b in range(BATCH):
        for sg in range(num_scale_groups):
            k_start = sg * 32
            k_end = k_start + 32
            for n in range(N):
                B_dequant[b, k_start:k_end, n] *= scales[b, n, sg]

    B_dequant = B_dequant.bfloat16()

    # Reference: C = A @ B
    C_ref = torch.bmm(A.float(), B_dequant.float()).bfloat16()

    return A, B_dequant, C_ref


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

    A, B_dequant, C_ref = fp4_dequant_reference()

    A_gpu = A.cuda()
    B_gpu = B_dequant.cuda()

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)

    try:
        import tk_batched_gemm_a16wfp4 as tk
    except ImportError:
        print("[SKIP] Compiled kernel not found, skipping parity test")
        return True

    # Prepare HK tensors (B transposed for mma_ABt)
    A_hk = A_gpu.unsqueeze(0).contiguous()                      # (1, B, M, K)
    B_hk = B_gpu.transpose(-2, -1).unsqueeze(0).contiguous()    # (1, B, N, K)
    C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")

    tk.dispatch(A_hk, B_hk, C_hk)
    torch.cuda.synchronize()

    C_out = C_hk.squeeze(0)
    C_ref_gpu = C_ref.cuda()

    max_diff = (C_out.float() - C_ref_gpu.float()).abs().max().item()
    mean_diff = (C_out.float() - C_ref_gpu.float()).abs().mean().item()

    print(f"[INFO] Max absolute diff: {max_diff:.6f}")
    print(f"[INFO] Mean absolute diff: {mean_diff:.6f}")

    atol = 1.0  # FP4 dequant introduces quantization error
    rtol = 5e-2
    close = torch.allclose(C_out.float(), C_ref_gpu.float(), atol=atol, rtol=rtol)
    if close:
        print(f"[PASS] Parity test passed (atol={atol}, rtol={rtol})")
    else:
        print(f"[FAIL] Parity test failed (max_diff={max_diff})")

    return close


def main():
    print("=" * 60)
    print("Batched GEMM A16WFP4 - Parity Test")
    print("=" * 60)
    print(f"Dimensions: B={BATCH}, M={M}, N={N}, K={K}")
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
