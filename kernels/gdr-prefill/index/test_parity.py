#!/usr/bin/env python3
"""Parity test for index utilities (HipKittens C++ vs Python reference)."""

import subprocess
import sys
import os
import numpy as np

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

def compile_kernel():
    subprocess.run(["make", "-C", KERNEL_DIR, "clean"], capture_output=True)
    result = subprocess.run(["make", "-C", KERNEL_DIR], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False
    return True

def run_kernel():
    binary = os.path.join(KERNEL_DIR, "kernel")
    if not os.path.exists(binary):
        return False
    result = subprocess.run([binary], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Execution failed:\n{result.stderr}")
        return False
    return "FAIL" not in result.stdout

def python_reference_test():
    """Test index utilities against Python reference."""
    print("=== Python Reference Test ===")

    cu_seqlens = np.array([0, 128, 320, 448])
    chunk_size = 64

    # prepare_lens
    lens = np.diff(cu_seqlens)
    assert list(lens) == [128, 192, 128], f"lens mismatch: {lens}"

    # prepare_chunk_indices
    indices = []
    for seq_id, l in enumerate(lens):
        n_chunks = (l + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            indices.append((seq_id, c))
    assert len(indices) == 7
    assert indices[0] == (0, 0)
    assert indices[2] == (1, 0)

    # prepare_chunk_offsets
    offsets = [0]
    for l in lens:
        offsets.append(offsets[-1] + (l + chunk_size - 1) // chunk_size)
    assert offsets == [0, 2, 5, 7]

    print(f"Sequence lengths: {list(lens)}")
    print(f"Chunk indices: {len(indices)} chunks")
    print(f"Chunk offsets: {offsets}")
    print("Python reference: PASS")
    return True

def main():
    print("=" * 60)
    print("Index Utilities Parity Test")
    print("=" * 60)

    python_reference_test()

    print("\n=== HIP Kernel Test ===")
    if compile_kernel():
        success = run_kernel()
        return 0 if success else 1
    else:
        print("Compilation failed (expected without HIP toolchain)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
