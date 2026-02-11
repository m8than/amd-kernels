// fused_kv_cache: Simplified KV cache write kernel
// Writes K and V tensors to paged KV cache
//
// Input:
//   k: (B, H_kv, D) key tensor
//   v: (B, H_kv, D) value tensor
//   slot_mapping: (B,) slot indices for each batch element
//
// Output:
//   key_cache: (num_blocks, H_kv, block_size, D) paged key cache
//   value_cache: (num_blocks, H_kv, block_size, D) paged value cache
//
// NOTE: This is a simplified version. Full Triton kernel includes:
// - RoPE application (GPTJ/NeoX variants)
// - K/V scaling
// - Flash attention layout
// - Complex slot/block mapping

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 256;

template<typename T, int D, int BLOCK_SIZE_KV = 16>
struct kv_cache_globals {
    using _gl_k = gl<T, -1, -1, -1, D>;
    using _gl_v = gl<T, -1, -1, -1, D>;
    // Cache layout: (num_blocks, H_kv, block_size, D) -> gl with 4 dims
    // BLOCK_SIZE_KV is a tiling parameter, not a separate dimension
    using _gl_key_cache = gl<T, -1, -1, -1, D>;
    using _gl_value_cache = gl<T, -1, -1, -1, D>;
    using _gl_slot_mapping = gl<int, -1, -1, -1, -1>;

    _gl_k k;
    _gl_v v;
    _gl_key_cache key_cache;
    _gl_value_cache value_cache;
    _gl_slot_mapping slot_mapping;

    int B;          // Batch size
    int H_kv;       // KV heads
    float k_scale;  // K scaling factor
    float v_scale;  // V scaling factor
    hipStream_t stream;

    dim3 grid() {
        return dim3(B, H_kv);
    }
    dim3 block() { return dim3(BLOCK_SIZE); }
};

template<typename T, int D, int BLOCK_SIZE_KV>
__global__ void kv_cache_kernel(
    const kv_cache_globals<T, D, BLOCK_SIZE_KV> g
) {
    const int b = blockIdx.x;
    const int h_kv = blockIdx.y;
    const int tid = threadIdx.x;

    // Load slot mapping for this batch element
    const int* slot_mapping_base = (int*)&g.slot_mapping[{0, 0, 0, 0}];
    const int slot = slot_mapping_base[b];

    if (slot < 0) return;  // Skip invalid slots

    // Decompose slot into block and position within block
    const int block_idx = slot / BLOCK_SIZE_KV;
    const int pos_in_block = slot % BLOCK_SIZE_KV;

    // Read K and V
    const T* k_base = (T*)&g.k[{0, 0, 0, 0}];
    const T* v_base = (T*)&g.v[{0, 0, 0, 0}];

    const int k_offset = (b * g.H_kv + h_kv) * D;
    const int v_offset = (b * g.H_kv + h_kv) * D;

    // Write to cache with optional scaling
    T* key_cache_base = (T*)&g.key_cache[{0, 0, 0, 0}];
    T* value_cache_base = (T*)&g.value_cache[{0, 0, 0, 0}];

    const int cache_offset = ((block_idx * g.H_kv + h_kv) * BLOCK_SIZE_KV + pos_in_block) * D;

    const float k_scale_inv = 1.0f / g.k_scale;
    const float v_scale_inv = 1.0f / g.v_scale;

    // Copy and scale
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float k_val = float(k_base[k_offset + i]);
        float v_val = float(v_base[v_offset + i]);

        key_cache_base[cache_offset + i] = T(k_val * k_scale_inv);
        value_cache_base[cache_offset + i] = T(v_val * v_scale_inv);
    }
}

template<typename T, int D, int BLOCK_SIZE_KV>
void dispatch_kv_cache(kv_cache_globals<T, D, BLOCK_SIZE_KV>& g) {
    kv_cache_kernel<T, D, BLOCK_SIZE_KV><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// Explicit instantiations
template void dispatch_kv_cache<bf16, 128, 16>(kv_cache_globals<bf16, 128, 16>&);
template void dispatch_kv_cache<bf16, 128, 32>(kv_cache_globals<bf16, 128, 32>&);
template void dispatch_kv_cache<half, 128, 16>(kv_cache_globals<half, 128, 16>&);
template void dispatch_kv_cache<half, 128, 32>(kv_cache_globals<half, 128, 32>&);

PYBIND11_MODULE(fused_kv_cache_tk, m) {
    m.doc() = "HipKittens Fused KV Cache Write kernel (simplified)";

    kittens::py::bind_function<dispatch_kv_cache<bf16, 128, 16>>(m, "dispatch_bf16_128_16",
        &kv_cache_globals<bf16, 128, 16>::k,
        &kv_cache_globals<bf16, 128, 16>::v,
        &kv_cache_globals<bf16, 128, 16>::key_cache,
        &kv_cache_globals<bf16, 128, 16>::value_cache,
        &kv_cache_globals<bf16, 128, 16>::slot_mapping,
        &kv_cache_globals<bf16, 128, 16>::B,
        &kv_cache_globals<bf16, 128, 16>::H_kv,
        &kv_cache_globals<bf16, 128, 16>::k_scale,
        &kv_cache_globals<bf16, 128, 16>::v_scale);

    kittens::py::bind_function<dispatch_kv_cache<bf16, 128, 32>>(m, "dispatch_bf16_128_32",
        &kv_cache_globals<bf16, 128, 32>::k,
        &kv_cache_globals<bf16, 128, 32>::v,
        &kv_cache_globals<bf16, 128, 32>::key_cache,
        &kv_cache_globals<bf16, 128, 32>::value_cache,
        &kv_cache_globals<bf16, 128, 32>::slot_mapping,
        &kv_cache_globals<bf16, 128, 32>::B,
        &kv_cache_globals<bf16, 128, 32>::H_kv,
        &kv_cache_globals<bf16, 128, 32>::k_scale,
        &kv_cache_globals<bf16, 128, 32>::v_scale);

    kittens::py::bind_function<dispatch_kv_cache<half, 128, 16>>(m, "dispatch_fp16_128_16",
        &kv_cache_globals<half, 128, 16>::k,
        &kv_cache_globals<half, 128, 16>::v,
        &kv_cache_globals<half, 128, 16>::key_cache,
        &kv_cache_globals<half, 128, 16>::value_cache,
        &kv_cache_globals<half, 128, 16>::slot_mapping,
        &kv_cache_globals<half, 128, 16>::B,
        &kv_cache_globals<half, 128, 16>::H_kv,
        &kv_cache_globals<half, 128, 16>::k_scale,
        &kv_cache_globals<half, 128, 16>::v_scale);

    kittens::py::bind_function<dispatch_kv_cache<half, 128, 32>>(m, "dispatch_fp16_128_32",
        &kv_cache_globals<half, 128, 32>::k,
        &kv_cache_globals<half, 128, 32>::v,
        &kv_cache_globals<half, 128, 32>::key_cache,
        &kv_cache_globals<half, 128, 32>::value_cache,
        &kv_cache_globals<half, 128, 32>::slot_mapping,
        &kv_cache_globals<half, 128, 32>::B,
        &kv_cache_globals<half, 128, 32>::H_kv,
        &kv_cache_globals<half, 128, 32>::k_scale,
        &kv_cache_globals<half, 128, 32>::v_scale);
}
