#!/bin/bash
# Build fixed GEMM kernels with correct LDS sizes for MI325X (64KB)
# The fix: change dynamic_shared_memory() from MAX_SHARED_MEMORY (160KB) to actual LDS needed

set +e

WORKSPACE="/root/aiter-hipkittens/amd-kernels/workspaces/ws-gemm/fixed_kernels"
KERNELS_DIR="/root/aiter-hipkittens/amd-kernels/kernels/gemm-basic"
HIPKITTENS="/root/aiter-hipkittens/HipKittens/include"
PYBIND_INCLUDES=$(/root/aiter-hipkittens/amd-kernels/.venv/bin/python -m pybind11 --includes)

HIPCC_FLAGS="-DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w -O3 $PYBIND_INCLUDES -shared -fPIC -I $HIPKITTENS -I /opt/rocm-7.2.0/include -I /opt/rocm-7.2.0/include/hip"

mkdir -p "$WORKSPACE"

compile_kernel() {
    local name="$1"
    local lds_bytes="$2"
    local src="$KERNELS_DIR/$name/kernel.cpp"
    local patched="$WORKSPACE/${name}_kernel_fixed.cpp"
    local output="$WORKSPACE/${name}_tk.cpython-312-x86_64-linux-gnu.so"

    echo "=== Building $name (LDS: $lds_bytes bytes) ==="

    # Copy and patch: replace MAX_SHARED_MEMORY with actual LDS size (clean replacement, no comment)
    sed "s/return MAX_SHARED_MEMORY;/return ${lds_bytes};/" \
        "$src" > "$patched"

    # Compile
    hipcc $HIPCC_FLAGS -o "$output" "$patched" 2>&1
    if [ -f "$output" ]; then
        echo "  SUCCESS: $output"
        return 0
    else
        echo "  FAILED: $name"
        return 1
    fi
}

SUCCESSES=0
FAILURES=0

for entry in \
    "gemm_a16w16_atomic:32768" \
    "gemm_a8w8:65536" \
    "gemm_a8w8_per_token_scale:65536" \
    "gemm_a16wfp4:65536" \
    "gemm_a8wfp4:65536" \
    "gemm_afp4wfp4:65536" \
    "gemm_a16w16_gated:65536"; do

    name="${entry%%:*}"
    lds="${entry##*:}"
    if compile_kernel "$name" "$lds"; then
        SUCCESSES=$((SUCCESSES + 1))
    else
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
echo "=== Results: $SUCCESSES succeeded, $FAILURES failed ==="
echo ""
echo "Built .so files:"
ls -la "$WORKSPACE"/*.so 2>/dev/null || echo "  (none)"
