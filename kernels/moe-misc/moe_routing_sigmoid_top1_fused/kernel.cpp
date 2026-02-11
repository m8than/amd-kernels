// MoE Routing Sigmoid Top-1 Fused Kernel
// Ported from reference/triton/moe_routing_sigmoid_top1_fused.py
//
// Computes:
//   1. acc = X @ W (matmul)
//   2. acc = sigmoid(acc)
//   3. topk_ids = argmax(acc, axis=1)
//   4. topk_weights = max(acc, axis=1)
//   5. If FUSED_SHARED_EXPERTS: append (N, 1.0) to topk results

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using namespace kittens;

// Configuration
#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// Tile sizes - chosen for MFMA efficiency
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// Global memory layout types
template<int M, int N, int K>
struct routing_sigmoid_globals {
    using input_gl = gl<bf16, -1, M, K, -1>;
    using weight_gl = gl<bf16, -1, K, N, -1>;
    using ids_gl = gl<int32_t, -1, M, -1, -1>;
    using weights_gl = gl<bf16, -1, M, -1, -1>;

    input_gl X;
    weight_gl W;
    ids_gl topk_ids;
    weights_gl topk_weights;

    int M_size;
    int N_size;
    int K_size;
    int TOPK;
    bool FUSED_SHARED_EXPERTS;

    hipStream_t stream;

    dim3 grid() {
        return dim3((M_size + TILE_M - 1) / TILE_M, 1, 1);
    }

    dim3 block() {
        return dim3(NUM_THREADS);
    }

    size_t dynamic_shared_memory() {
        return 0; // No shared memory needed for this simple kernel
    }
};

// Sigmoid helper (element-wise on register tile)
template<typename T, int rows, int cols, typename layout, typename shape>
__device__ void sigmoid_inplace(rt<T, rows, cols, layout, shape> &tile) {
    const int height = rt<T, rows, cols, layout, shape>::height;
    const int width = rt<T, rows, cols, layout, shape>::width;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            auto &base_tile = tile.tiles[i][j];
            constexpr int packed_size = sizeof(base_tile.data) / sizeof(base_tile.data[0]);

            for (int k = 0; k < packed_size; k++) {
                // Convert packed bf16 to float, apply sigmoid, convert back
                auto packed = base_tile.data[k];

                // Unpack two bf16 values
                __hip_bfloat162 bf16_pair = *reinterpret_cast<__hip_bfloat162*>(&packed);
                float2 f_pair = __bfloat1622float2(bf16_pair);

                // Sigmoid: 1 / (1 + exp(-x))
                f_pair.x = 1.0f / (1.0f + expf(-f_pair.x));
                f_pair.y = 1.0f / (1.0f + expf(-f_pair.y));

                // Pack back to bf16
                bf16_pair = __float22bfloat162_rn(f_pair);
                base_tile.data[k] = *reinterpret_cast<decltype(packed)*>(&bf16_pair);
            }
        }
    }
}

// Argmax helper - returns {max_index, max_value} for a row
template<typename T, int cols, typename layout, typename shape>
__device__ void row_argmax(__hip_bfloat16 &max_val, int &max_idx,
                          const rt<T, 16, cols, layout, shape> &row_tile) {
    max_val = __float2bfloat16(-INFINITY);
    max_idx = 0;

    const int width = rt<T, 16, cols, layout, shape>::width;

    for (int j = 0; j < width; j++) {
        auto &base_tile = row_tile.tiles[0][j];
        constexpr int packed_size = sizeof(base_tile.data) / sizeof(base_tile.data[0]);

        for (int k = 0; k < packed_size; k++) {
            auto packed = base_tile.data[k];
            __hip_bfloat162 bf16_pair = *reinterpret_cast<__hip_bfloat162*>(&packed);

            // Each packed element contains 2 bf16 values
            int global_idx_0 = (j * 32 + k * 2);
            int global_idx_1 = (j * 32 + k * 2 + 1);

            __hip_bfloat16 val_0 = bf16_pair.x;
            __hip_bfloat16 val_1 = bf16_pair.y;

            if (__hgt(val_0, max_val)) {
                max_val = val_0;
                max_idx = global_idx_0;
            }
            if (__hgt(val_1, max_val)) {
                max_val = val_1;
                max_idx = global_idx_1;
            }
        }
    }
}

// Main kernel
template<int M, int N, int K>
__launch_bounds__(NUM_THREADS, 2)
__global__ void routing_sigmoid_top1_kernel(const routing_sigmoid_globals<M, N, K> g) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int wid = tid / 64;

    int m_start = bid * TILE_M;

    // Each warp processes multiple rows
    int rows_per_warp = TILE_M / NUM_WARPS;
    int m_local = wid * rows_per_warp;
    int m_global = m_start + m_local;

    if (m_global >= g.M_size) return;

    // Process each row assigned to this warp
    for (int row_offset = 0; row_offset < rows_per_warp; row_offset++) {
        int m = m_global + row_offset;
        if (m >= g.M_size) continue;

        // Accumulator for this row (one row, N columns)
        rt_fl<16, TILE_N> acc;
        zero(acc);

        // K-dimension loop
        for (int k_tile = 0; k_tile < (g.K_size + TILE_K - 1) / TILE_K; k_tile++) {
            int k_start = k_tile * TILE_K;

            // For simplicity in this first version, we'll do a naive matmul
            // Load X[m, k_start:k_start+TILE_K] - a single row
            // Load W[k_start:k_start+TILE_K, :] - TILE_K rows

            // Use naive approach: scalar loads and accumulation
            // HipKittens is best for tiled matmul; for 1xN output we use simpler approach

            for (int k = k_start; k < k_start + TILE_K && k < g.K_size; k++) {
                // Load X[m, k]
                __hip_bfloat16 x_val = *reinterpret_cast<const __hip_bfloat16*>(
                    &g.X.raw_ptr[m * g.K_size + k]);

                // Load W[k, :] and accumulate into acc
                for (int n = tid % 64; n < g.N_size; n += 64) {
                    __hip_bfloat16 w_val = *reinterpret_cast<const __hip_bfloat16*>(
                        &g.W.raw_ptr[k * g.N_size + n]);

                    float x_f = __bfloat162float(x_val);
                    float w_f = __bfloat162float(w_val);

                    // Accumulate into acc tile
                    // Map n to tile position
                    int tile_col = n / 32;
                    int elem_in_tile = n % 32;
                    int packed_idx = elem_in_tile / 2;
                    int elem_in_packed = elem_in_tile % 2;

                    if (tile_col < acc.width) {
                        auto &base = acc.tiles[0][tile_col];
                        auto packed = base.data[packed_idx];
                        __hip_bfloat162 bf_pair = *reinterpret_cast<__hip_bfloat162*>(&packed);
                        float2 f_pair = __bfloat1622float2(bf_pair);

                        if (elem_in_packed == 0) {
                            f_pair.x += x_f * w_f;
                        } else {
                            f_pair.y += x_f * w_f;
                        }

                        bf_pair = __float22bfloat162_rn(f_pair);
                        base.data[packed_idx] = *reinterpret_cast<decltype(packed)*>(&bf_pair);
                    }
                }
            }
        }

        // Warp reduction to combine partial sums across threads
        // Each thread computed different n values, need to combine
        // Use warp shuffle to reduce

        // Apply sigmoid element-wise
        // Convert acc to bf16 first
        rt_bf<16, TILE_N> acc_bf;
        copy(acc_bf, acc);

        // Sigmoid
        sigmoid_inplace(acc_bf);

        // Find argmax and max value
        __hip_bfloat16 max_val;
        int max_idx;
        row_argmax(max_val, max_idx, acc_bf);

        // Warp-level reduction to find global max
        for (int offset = 32; offset > 0; offset /= 2) {
            int other_idx = __shfl_down(max_idx, offset, 64);
            __hip_bfloat16 other_val = __shfl_down(*reinterpret_cast<unsigned short*>(&max_val), offset, 64);
            __hip_bfloat16 other_val_bf = *reinterpret_cast<__hip_bfloat16*>(&other_val);

            if (__hgt(other_val_bf, max_val)) {
                max_val = other_val_bf;
                max_idx = other_idx;
            }
        }

        // Thread 0 in warp writes result
        if ((tid % 64) == 0) {
            int _TOPK = g.FUSED_SHARED_EXPERTS ? (g.TOPK + 1) : g.TOPK;

            // Write topk_ids and topk_weights
            for (int topk_idx = 0; topk_idx < _TOPK; topk_idx++) {
                if (g.FUSED_SHARED_EXPERTS && topk_idx == _TOPK - 1) {
                    // Last entry: (N, 1.0)
                    g.topk_ids.raw_ptr[m * _TOPK + topk_idx] = g.N_size;
                    g.topk_weights.raw_ptr[m * _TOPK + topk_idx] = __float2bfloat16(1.0f);
                } else {
                    // Regular entry
                    g.topk_ids.raw_ptr[m * _TOPK + topk_idx] = max_idx;
                    g.topk_weights.raw_ptr[m * _TOPK + topk_idx] = max_val;
                }
            }
        }
    }
}

// Dispatch function
template<int M, int N, int K>
void dispatch_routing_sigmoid_top1(routing_sigmoid_globals<M, N, K>& g) {
    unsigned long mem_size = g.dynamic_shared_memory();

    hipFuncSetAttribute(
        (void*)routing_sigmoid_top1_kernel<M, N, K>,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    routing_sigmoid_top1_kernel<M, N, K><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Explicit instantiations for common sizes
template void dispatch_routing_sigmoid_top1(routing_sigmoid_globals<-1, -1, -1>&);

extern "C" {
    void launch_routing_sigmoid_top1(
        const __hip_bfloat16* X,
        const __hip_bfloat16* W,
        int32_t* topk_ids,
        __hip_bfloat16* topk_weights,
        int M, int N, int K,
        int TOPK,
        bool FUSED_SHARED_EXPERTS,
        hipStream_t stream
    ) {
        using globals_t = routing_sigmoid_globals<-1, -1, -1>;
        using input_gl_t = typename globals_t::input_gl;
        using weight_gl_t = typename globals_t::weight_gl;
        using ids_gl_t = typename globals_t::ids_gl;
        using weights_gl_t = typename globals_t::weights_gl;

        int _TOPK = FUSED_SHARED_EXPERTS ? (TOPK + 1) : TOPK;

        globals_t g {
            input_gl_t(const_cast<bf16*>(X), (size_t)1, (size_t)M, (size_t)K, (size_t)1),
            weight_gl_t(const_cast<bf16*>(W), (size_t)1, (size_t)K, (size_t)N, (size_t)1),
            ids_gl_t(topk_ids, (size_t)1, (size_t)M, (size_t)_TOPK, (size_t)1),
            weights_gl_t(reinterpret_cast<bf16*>(topk_weights), (size_t)1, (size_t)M, (size_t)_TOPK, (size_t)1),
            M, N, K, TOPK, FUSED_SHARED_EXPERTS, stream
        };

        dispatch_routing_sigmoid_top1(g);
    }
}

#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void moe_routing_sigmoid_top1_fused_wrapper(
    pybind11::object X, pybind11::object W,
    pybind11::object topk_ids, pybind11::object topk_weights,
    int M, int N, int K, int TOPK, bool FUSED_SHARED_EXPERTS)
{
    launch_routing_sigmoid_top1(
        (const __hip_bfloat16*)_get_ptr(X),
        (const __hip_bfloat16*)_get_ptr(W),
        (int32_t*)_get_ptr(topk_ids),
        (__hip_bfloat16*)_get_ptr(topk_weights),
        M, N, K, TOPK, FUSED_SHARED_EXPERTS, 0);
}

PYBIND11_MODULE(moe_routing_sigmoid_top1_fused_tk, m) {
    m.doc() = "MoE routing sigmoid top-1 fused kernel";
    m.def("moe_routing_sigmoid_top1_fused", &moe_routing_sigmoid_top1_fused_wrapper,
          "Fused sigmoid routing with top-1 selection");
}
