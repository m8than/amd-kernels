// Top-K Selection HipKittens Kernel
// Ported from reference/triton/topk.py
//
// Finds the top-K largest values and their indices from each row.
// Used in MoE routing and sampling.
//
// Strategy:
//   - Small K (K <= 32): Iterative selection — find max K times, masking out
//     previously selected elements. Simple and efficient for small K.
//   - This is the 1-stage approach from the Triton reference (_topk_kernel).
//
// Input layout:
//   x: (B, N) bf16/f32 — input tensor, B rows of N elements each
//
// Output layout:
//   values: (B, K) f32 — top-K values (descending order)
//   indices: (B, K) int64 — indices of top-K values in original row

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <float.h>
#include <cmath>
#include <limits.h>

constexpr int WARP_SIZE = 64;

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, WARP_SIZE));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min_idx(float val, int idx, int* out_idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor(val, offset, WARP_SIZE);
        int other_idx = __shfl_xor(idx, offset, WARP_SIZE);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    *out_idx = idx;
    return val;
}

// Block-level max with index tracking using shared memory
__device__ void block_reduce_max_with_idx(
    float val, int idx, float* smem_val, int* smem_idx,
    int tid, int num_warps, float* out_val, int* out_idx
) {
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Warp-level reduce
    int widx;
    val = warp_reduce_min_idx(val, idx, &widx);
    idx = widx;

    if (lane_id == 0) {
        smem_val[warp_id] = val;
        smem_idx[warp_id] = idx;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float wval = (lane_id < num_warps) ? smem_val[lane_id] : -FLT_MAX;
        int widx2 = (lane_id < num_warps) ? smem_idx[lane_id] : 0;
        wval = warp_reduce_min_idx(wval, widx2, &widx2);
        if (lane_id == 0) {
            smem_val[0] = wval;
            smem_idx[0] = widx2;
        }
    }
    __syncthreads();

    *out_val = smem_val[0];
    *out_idx = smem_idx[0];
}

// Top-K kernel: iterative selection approach
// Each block handles one row, selects top-K elements one at a time
template<int BLOCK_SIZE>
__global__ void topk_kernel(
    const __hip_bfloat16* __restrict__ input,
    float* __restrict__ out_values,
    int64_t* __restrict__ out_indices,
    int N,      // number of elements per row
    int K       // number of top elements to select
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_warps = BLOCK_SIZE / WARP_SIZE;

    const __hip_bfloat16* row_in = input + row * N;
    float* row_out_val = out_values + row * K;
    int64_t* row_out_idx = out_indices + row * K;

    extern __shared__ char shared_mem[];
    float* smem_val = (float*)shared_mem;
    int* smem_idx = (int*)(smem_val + num_warps);
    // Use extra shared memory for marking selected indices
    // We'll use a different approach: set selected values to -FLT_MAX

    // Each thread loads its portion of values into a local buffer
    // We process values in a strided manner
    constexpr float FILL_VALUE = -FLT_MAX;

    for (int k_iter = 0; k_iter < K; k_iter++) {
        // Find maximum value and its index across the row
        float local_max = FILL_VALUE;
        int local_max_idx = 0;

        for (int col = tid; col < N; col += BLOCK_SIZE) {
            float val = __bfloat162float(row_in[col]);

            // Check if this index was already selected
            bool already_selected = false;
            for (int prev_k = 0; prev_k < k_iter; prev_k++) {
                if (row_out_idx[prev_k] == (int64_t)col) {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && (val > local_max || (val == local_max && col < local_max_idx))) {
                local_max = val;
                local_max_idx = col;
            }
        }

        // Block-level reduce to find global max
        float global_max;
        int global_max_idx;
        block_reduce_max_with_idx(local_max, local_max_idx, smem_val, smem_idx,
                                  tid, num_warps, &global_max, &global_max_idx);

        // Thread 0 writes the result
        if (tid == 0) {
            row_out_val[k_iter] = global_max;
            row_out_idx[k_iter] = (int64_t)global_max_idx;
        }
        __syncthreads();
    }
}

extern "C" {

void launch_topk(
    const __hip_bfloat16* input, float* out_values, int64_t* out_indices,
    int B, int N, int K, hipStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    size_t smem_size = (num_warps * sizeof(float)) + (num_warps * sizeof(int));

    topk_kernel<BLOCK_SIZE><<<B, BLOCK_SIZE, smem_size, stream>>>(
        input, out_values, out_indices, N, K);
}

void launch_topk_large(
    const __hip_bfloat16* input, float* out_values, int64_t* out_indices,
    int B, int N, int K, hipStream_t stream
) {
    constexpr int BLOCK_SIZE = 1024;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    size_t smem_size = (num_warps * sizeof(float)) + (num_warps * sizeof(int));

    topk_kernel<BLOCK_SIZE><<<B, BLOCK_SIZE, smem_size, stream>>>(
        input, out_values, out_indices, N, K);
}

} // extern "C"

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

// input: (B, N) bf16, out_values: (B, K) f32, out_indices: (B, K) int64
void topk_fwd(pybind11::object input, pybind11::object out_values, pybind11::object out_indices, int K) {
    int B = _get_dim(input, 0), N = _get_dim(input, 1);
    auto inp = (const __hip_bfloat16*)_get_ptr(input);
    auto vals = (float*)_get_ptr(out_values);
    auto idxs = (int64_t*)_get_ptr(out_indices);
    if (N <= 256)
        launch_topk(inp, vals, idxs, B, N, K, 0);
    else
        launch_topk_large(inp, vals, idxs, B, N, K, 0);
}

PYBIND11_MODULE(topk_tk, m) {
    m.doc() = "HipKittens top-k kernel";
    m.def("topk_fwd", &topk_fwd, "Top-K selection forward");
}
