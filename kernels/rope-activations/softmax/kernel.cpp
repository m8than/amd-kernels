// Online Softmax HipKittens Kernel
// Ported from reference/triton/softmax.py
//
// Implements numerically stable row-wise softmax using the online algorithm:
//   Pass 1: find row max and compute sum(exp(x - max))
//   Pass 2: output = exp(x - max) / sum
//
// This uses a single-pass online variant where max and sum are updated together
// as blocks are processed, matching the Triton _softmax_kernel_online pattern.
//
// Input: (M, N) bf16 — 2D tensor, softmax applied along dim=-1 (columns)
// Output: (M, N) bf16

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <float.h>
#include <cmath>

// Block size for column tiling — must be power of 2
// Each thread block handles one row, threads cooperate on columns
constexpr int WARP_SIZE = 64;

// Warp-level reduction helpers
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, WARP_SIZE));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, WARP_SIZE);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_max(float val, float* smem, int tid, int num_warps) {
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    // Broadcast result
    if (warp_id == 0 && lane_id == 0) {
        smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_sum(float val, float* smem, int tid, int num_warps) {
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

// Online softmax: each block handles one row
// BLOCK_N threads per block, each thread handles ceil(N/BLOCK_N) elements
template<int BLOCK_N>
__global__ void softmax_kernel(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_warps = BLOCK_N / WARP_SIZE;

    const __hip_bfloat16* row_in = input + row * N;
    __hip_bfloat16* row_out = output + row * N;

    extern __shared__ float smem[];

    // Pass 1: Online max + sum computation
    float m = -FLT_MAX;    // running max
    float row_sum = 0.0f;  // running sum of exp(x - max)

    for (int col = tid; col < N; col += BLOCK_N) {
        float val = __bfloat162float(row_in[col]);
        float old_m = m;
        m = fmaxf(m, val);
        // Adjust running sum for new max
        row_sum = row_sum * expf(old_m - m) + expf(val - m);
    }

    // Reduce max across block
    m = block_reduce_max(m, smem, tid, num_warps);

    // Each thread adjusts its partial sum to global max, then reduce
    // Need to recompute since block_reduce_max changed m
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += BLOCK_N) {
        float val = __bfloat162float(row_in[col]);
        local_sum += expf(val - m);
    }

    float total_sum = block_reduce_sum(local_sum, smem, tid, num_warps);
    float inv_sum = 1.0f / total_sum;

    // Pass 2: Compute softmax output
    for (int col = tid; col < N; col += BLOCK_N) {
        float val = __bfloat162float(row_in[col]);
        float result = expf(val - m) * inv_sum;
        row_out[col] = __float2bfloat16(result);
    }
}

extern "C" {

void launch_softmax(const __hip_bfloat16* input, __hip_bfloat16* output,
                    int M, int N, hipStream_t stream) {
    // Choose block size based on N
    constexpr int BLOCK_N = 256;
    int num_warps = BLOCK_N / WARP_SIZE;
    size_t smem_size = num_warps * sizeof(float);

    softmax_kernel<BLOCK_N><<<M, BLOCK_N, smem_size, stream>>>(input, output, N);
}

void launch_softmax_large(const __hip_bfloat16* input, __hip_bfloat16* output,
                          int M, int N, hipStream_t stream) {
    constexpr int BLOCK_N = 1024;
    int num_warps = BLOCK_N / WARP_SIZE;
    size_t smem_size = num_warps * sizeof(float);

    softmax_kernel<BLOCK_N><<<M, BLOCK_N, smem_size, stream>>>(input, output, N);
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

// input: (M, N), output: (M, N)
void softmax_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    auto inp = (const __hip_bfloat16*)_get_ptr(input);
    auto out = (__hip_bfloat16*)_get_ptr(output);
    if (N <= 256)
        launch_softmax(inp, out, M, N, 0);
    else
        launch_softmax_large(inp, out, M, N, 0);
}

PYBIND11_MODULE(softmax_tk, m) {
    m.doc() = "HipKittens softmax kernel";
    m.def("softmax_fwd", &softmax_fwd, "Row-wise softmax forward");
}
