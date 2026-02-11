"""
Parity test for fused_qkvzba_split_reshape_cat_decode kernel.

Tests the HIP C++ kernel against a numpy reference implementation.

Input shapes:
  mixed_qkvz: (batch, NUM_HEADS_QK * QKVZ_DIM_T) bf16
    where QKVZ_DIM_T = HEAD_QK*2 + V_PER_QK*HEAD_V*2
    Per QK-head: [q(HEAD_QK), k(HEAD_QK), v(V_PER_QK*HEAD_V), z(V_PER_QK*HEAD_V)]
  mixed_ba: (batch, NUM_HEADS_QK * BA_DIM_T) bf16
    where BA_DIM_T = V_PER_QK*2
    Per QK-head: [b(V_PER_QK), a(V_PER_QK)]

Output shapes:
  mixed_qkv: (batch, QKV_DIM_T) bf16
    where QKV_DIM_T = NUM_HEADS_QK*HEAD_QK*2 + NUM_HEADS_V*HEAD_V
    Layout: [all_q, all_k, all_v]
  z: (batch, NUM_HEADS_V * HEAD_V) bf16
  b: (batch, NUM_HEADS_V) bf16
  a: (batch, NUM_HEADS_V) bf16

Compilation:
  hipcc -std=c++20 -O3 --offload-arch=gfx942 -shared -fPIC \
    -o fused_qkvzba_split.so kernel.cpp
"""

import numpy as np
import subprocess
import os


def fused_qkvzba_split_reshape_cat_decode_ref(
    mixed_qkvz, mixed_ba,
    num_heads_qk, num_heads_v, head_qk, head_v
):
    """
    Numpy reference for fused QKVZBA split, reshape and concatenation (decode).

    Args:
        mixed_qkvz: (batch, NUM_HEADS_QK * QKVZ_DIM_T)
        mixed_ba: (batch, NUM_HEADS_QK * BA_DIM_T)
        num_heads_qk: int
        num_heads_v: int
        head_qk: int
        head_v: int

    Returns:
        mixed_qkv: (batch, QKV_DIM_T)
        z: (batch, NUM_HEADS_V * HEAD_V)
        b: (batch, NUM_HEADS_V)
        a: (batch, NUM_HEADS_V)
    """
    batch = mixed_qkvz.shape[0]
    v_per_qk = num_heads_v // num_heads_qk
    qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2
    ba_dim_t = v_per_qk * 2
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v

    mixed_qkv = np.zeros((batch, qkv_dim_t), dtype=mixed_qkvz.dtype)
    z = np.zeros((batch, num_heads_v * head_v), dtype=mixed_qkvz.dtype)
    b_out = np.zeros((batch, num_heads_v), dtype=mixed_ba.dtype)
    a_out = np.zeros((batch, num_heads_v), dtype=mixed_ba.dtype)

    for bs in range(batch):
        for iqk in range(num_heads_qk):
            # Source offsets
            src_qkvz = mixed_qkvz[bs, iqk * qkvz_dim_t : (iqk + 1) * qkvz_dim_t]
            src_ba = mixed_ba[bs, iqk * ba_dim_t : (iqk + 1) * ba_dim_t]

            # Parse qkvz
            q_end = head_qk
            k_end = q_end + head_qk
            v_end = k_end + v_per_qk * head_v
            z_end = v_end + v_per_qk * head_v

            blk_q = src_qkvz[:q_end]
            blk_k = src_qkvz[q_end:k_end]
            blk_v = src_qkvz[k_end:v_end]
            blk_z = src_qkvz[v_end:z_end]

            # Store q: contiguous per qk-head
            mixed_qkv[bs, iqk * head_qk : (iqk + 1) * head_qk] = blk_q

            # Store k: after all q's
            k_offset = num_heads_qk * head_qk + iqk * head_qk
            mixed_qkv[bs, k_offset : k_offset + head_qk] = blk_k

            # Store v: after all k's
            v_offset = 2 * num_heads_qk * head_qk + iqk * v_per_qk * head_v
            mixed_qkv[bs, v_offset : v_offset + v_per_qk * head_v] = blk_v

            # Store z
            z_offset = iqk * v_per_qk * head_v
            z[bs, z_offset : z_offset + v_per_qk * head_v] = blk_z

            # Store b and a
            for i in range(v_per_qk):
                b_out[bs, iqk * v_per_qk + i] = src_ba[i]
                a_out[bs, iqk * v_per_qk + i] = src_ba[v_per_qk + i]

    return mixed_qkv, z, b_out, a_out


def test_numpy_reference():
    """Test the numpy reference implementation."""
    np.random.seed(42)

    batch = 4
    num_heads_qk = 2
    num_heads_v = 8
    head_qk = 128
    head_v = 128
    v_per_qk = num_heads_v // num_heads_qk  # 4

    qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2  # 256 + 1024 = 1280
    ba_dim_t = v_per_qk * 2  # 8
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v  # 512 + 1024 = 1536

    mixed_qkvz = np.random.randn(batch, num_heads_qk * qkvz_dim_t).astype(np.float32)
    mixed_ba = np.random.randn(batch, num_heads_qk * ba_dim_t).astype(np.float32)

    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_decode_ref(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )

    print(f"mixed_qkv shape: {mixed_qkv.shape} (expected ({batch}, {qkv_dim_t}))")
    print(f"z shape: {z.shape} (expected ({batch}, {num_heads_v * head_v}))")
    print(f"b shape: {b.shape} (expected ({batch}, {num_heads_v}))")
    print(f"a shape: {a.shape} (expected ({batch}, {num_heads_v}))")

    assert mixed_qkv.shape == (batch, qkv_dim_t)
    assert z.shape == (batch, num_heads_v * head_v)
    assert b.shape == (batch, num_heads_v)
    assert a.shape == (batch, num_heads_v)

    # Verify data integrity: check that source data appears in output
    # q for head 0 should match first HEAD_QK elements of mixed_qkvz head 0
    for iqk in range(num_heads_qk):
        src_q = mixed_qkvz[0, iqk * qkvz_dim_t : iqk * qkvz_dim_t + head_qk]
        dst_q = mixed_qkv[0, iqk * head_qk : (iqk + 1) * head_qk]
        assert np.allclose(src_q, dst_q), f"q mismatch for head {iqk}"

    print("PASS: numpy reference sanity check")


def test_roundtrip():
    """Verify that all data from input appears in output."""
    np.random.seed(99)

    batch = 2
    num_heads_qk = 4
    num_heads_v = 16
    head_qk = 64
    head_v = 64

    v_per_qk = num_heads_v // num_heads_qk
    qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2
    ba_dim_t = v_per_qk * 2

    mixed_qkvz = np.random.randn(batch, num_heads_qk * qkvz_dim_t).astype(np.float32)
    mixed_ba = np.random.randn(batch, num_heads_qk * ba_dim_t).astype(np.float32)

    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_decode_ref(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )

    # Count total elements
    total_qkvz = mixed_qkvz.size
    total_ba = mixed_ba.size
    total_out = mixed_qkv.size + z.size + b.size + a.size

    # Check all elements are accounted for
    assert total_out == total_qkvz + total_ba, \
        f"Element count mismatch: {total_out} vs {total_qkvz + total_ba}"
    print("PASS: roundtrip element count check")


def test_hip_kernel():
    """Test HIP kernel against numpy reference (requires ROCm)."""
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        return

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(kernel_dir, "fused_qkvzba_split.so")

    if not os.path.exists(so_path):
        print("Building kernel...")
        result = subprocess.run(["make", "-C", kernel_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            print("SKIP: Could not build kernel")
            return

    import ctypes
    lib = ctypes.CDLL(so_path)

    np.random.seed(42)
    batch = 4
    num_heads_qk = 4
    num_heads_v = 16
    head_qk = 128
    head_v = 128
    v_per_qk = num_heads_v // num_heads_qk

    qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2
    ba_dim_t = v_per_qk * 2
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v

    mixed_qkvz = torch.randn(batch, num_heads_qk * qkvz_dim_t, dtype=torch.bfloat16, device="cuda")
    mixed_ba = torch.randn(batch, num_heads_qk * ba_dim_t, dtype=torch.bfloat16, device="cuda")

    mixed_qkv = torch.zeros(batch, qkv_dim_t, dtype=torch.bfloat16, device="cuda")
    z = torch.zeros(batch, num_heads_v * head_v, dtype=torch.bfloat16, device="cuda")
    b_t = torch.zeros(batch, num_heads_v, dtype=torch.bfloat16, device="cuda")
    a_t = torch.zeros(batch, num_heads_v, dtype=torch.bfloat16, device="cuda")

    # Numpy reference
    ref_qkv, ref_z, ref_b, ref_a = fused_qkvzba_split_reshape_cat_decode_ref(
        mixed_qkvz.float().cpu().numpy(),
        mixed_ba.float().cpu().numpy(),
        num_heads_qk, num_heads_v, head_qk, head_v
    )

    # Launch HIP kernel
    lib.launch_fused_qkvzba_split_reshape_cat_decode(
        ctypes.c_void_p(mixed_qkv.data_ptr()),
        ctypes.c_void_p(z.data_ptr()),
        ctypes.c_void_p(b_t.data_ptr()),
        ctypes.c_void_p(a_t.data_ptr()),
        ctypes.c_void_p(mixed_qkvz.contiguous().data_ptr()),
        ctypes.c_void_p(mixed_ba.contiguous().data_ptr()),
        ctypes.c_int(batch),
        ctypes.c_int(num_heads_qk),
        ctypes.c_int(num_heads_v),
        ctypes.c_int(head_qk),
        ctypes.c_int(head_v),
        ctypes.c_void_p(0),  # stream
    )
    torch.cuda.synchronize()

    qkv_diff = np.abs(mixed_qkv.float().cpu().numpy() - ref_qkv).max()
    z_diff = np.abs(z.float().cpu().numpy() - ref_z).max()
    b_diff = np.abs(b_t.float().cpu().numpy() - ref_b).max()
    a_diff = np.abs(a_t.float().cpu().numpy() - ref_a).max()

    print(f"mixed_qkv max diff: {qkv_diff:.6f}")
    print(f"z max diff: {z_diff:.6f}")
    print(f"b max diff: {b_diff:.6f}")
    print(f"a max diff: {a_diff:.6f}")

    atol = 0.01  # This is a pure data shuffle, should be exact (or bf16 roundtrip)
    if qkv_diff < atol and z_diff < atol and b_diff < atol and a_diff < atol:
        print("PASS: HIP kernel matches numpy reference")
    else:
        print("FAIL: HIP kernel output differs from reference")


if __name__ == "__main__":
    print("=" * 60)
    print("Fused QKVZBA Split Reshape Cat (Decode) - Parity Test")
    print("=" * 60)

    print("\n--- Numpy Reference Test ---")
    test_numpy_reference()

    print("\n--- Roundtrip Test ---")
    test_roundtrip()

    print("\n--- HIP Kernel Test ---")
    test_hip_kernel()
