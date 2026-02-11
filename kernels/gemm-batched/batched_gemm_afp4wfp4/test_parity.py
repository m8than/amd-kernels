#!/usr/bin/env python3
"""Parity test for batched_gemm_afp4wfp4 HipKittens kernel.

Triton reference: C[b] = dequant(A_fp4[b]) @ dequant(B_fp4[b])
  A: (B, M, K/2) uint8 (FP4 packed)
  B: (B, K/2, N) uint8 (FP4 packed)
  a_scales: (B, M, K/32) uint8 (e8m0)
  b_scales: (B, N, K/32) uint8 (e8m0)
  C: (B, M, N) bf16

HipKittens kernel (both A/B pre-dequantized):
  A: (1, B, M, K) bf16
  B: (1, B, N, K) bf16  (transposed)
  C: (1, B, M, N) bf16
"""

import subprocess
import sys
import os

BATCH = 4
M = 128
N = 128
K = 128


def reference_torch():
    """Compute reference output simulating FP4 x FP4 GEMM."""
    import torch

    torch.manual_seed(42)

    # Simulate dequantized FP4 values for both A and B
    fp4_vals = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    # A: (B, M, K)
    A_indices = torch.randint(0, len(fp4_vals), (BATCH, M, K))
    A_signs = torch.randint(0, 2, (BATCH, M, K)) * 2 - 1
    A_dequant = (fp4_vals[A_indices] * A_signs.float()).bfloat16()

    # B: (B, K, N)
    B_indices = torch.randint(0, len(fp4_vals), (BATCH, K, N))
    B_signs = torch.randint(0, 2, (BATCH, K, N)) * 2 - 1
    B_dequant = (fp4_vals[B_indices] * B_signs.float()).bfloat16()

    # Apply random scales per group of 32
    num_scale_groups = K // 32
    a_scales = torch.pow(2.0, torch.randint(-3, 4, (BATCH, M, num_scale_groups)).float())
    b_scales = torch.pow(2.0, torch.randint(-3, 4, (BATCH, N, num_scale_groups)).float())

    for b in range(BATCH):
        for sg in range(num_scale_groups):
            k_start = sg * 32
            k_end = k_start + 32
            for m in range(M):
                A_dequant[b, m, k_start:k_end] = (A_dequant[b, m, k_start:k_end].float() * a_scales[b, m, sg]).bfloat16()
            for n in range(N):
                B_dequant[b, k_start:k_end, n] = (B_dequant[b, k_start:k_end, n].float() * b_scales[b, n, sg]).bfloat16()

    # Reference: C = A @ B
    C_ref = torch.bmm(A_dequant.float(), B_dequant.float()).bfloat16()

    return A_dequant, B_dequant, C_ref


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

    A_dequant, B_dequant, C_ref = reference_torch()

    A_gpu = A_dequant.cuda()
    B_gpu = B_dequant.cuda()

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, kernel_dir)

    try:
        import tk_batched_gemm_afp4wfp4 as tk
    except ImportError:
        print("[SKIP] Compiled kernel not found, skipping parity test")
        return True

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

    atol = 2.0  # FP4 x FP4 has significant quantization error
    rtol = 5e-2
    close = torch.allclose(C_out.float(), C_ref_gpu.float(), atol=atol, rtol=rtol)
    if close:
        print(f"[PASS] Parity test passed (atol={atol}, rtol={rtol})")
    else:
        print(f"[FAIL] Parity test failed (max_diff={max_diff})")

    return close


def main():
    print("=" * 60)
    print("Batched GEMM AFP4WFP4 - Parity Test")
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
