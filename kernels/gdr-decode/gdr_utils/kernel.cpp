// Gated Delta Rule Utilities
// Ported from reference/triton/gated_delta_rule_utils.py
//
// Provides device-side utility functions used by the gated delta rule kernels:
// 1. Error computation (absolute error, error ratio) -- for validation
// 2. L2 normalization -- used in QK normalization
// 3. Softplus/Sigmoid -- used in gating
// 4. Contiguous memory helper -- ensures proper alignment
//
// These are provided as both device functions (callable from other kernels)
// and as standalone kernels for testing/validation.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cfloat>

// ============================================================
// Device utility functions (can be included by other kernels)
// ============================================================

// Absolute error: max |x - y|
__device__ __forceinline__ float device_abs_error(float x, float y) {
    return fabsf(x - y);
}

// SiLU / Swish activation: x * sigmoid(x)
__device__ __forceinline__ float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Sigmoid activation: 1 / (1 + exp(-x))
__device__ __forceinline__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softplus with numerical stability: (1/beta) * log(1 + exp(beta*x))
// Falls back to x when beta*x > threshold
__device__ __forceinline__ float device_softplus(float x, float beta, float threshold) {
    float bx = beta * x;
    if (bx <= threshold) {
        return (1.0f / beta) * logf(1.0f + expf(bx));
    }
    return x;
}

// L2 normalization of a vector element (needs warp reduction for full norm)
// Returns x / sqrt(sum(x^2) + eps)
// Caller must ensure all threads in the warp participate
__device__ __forceinline__ float device_l2norm_element(float x, int warp_size_k) {
    float x_sq = x * x;
    // Warp-level reduction for sum of squares
    for (int offset = warp_size_k / 2; offset > 0; offset >>= 1) {
        x_sq += __shfl_xor(x_sq, offset);
    }
    return x / sqrtf(x_sq + 1e-6f);
}

// ============================================================
// Standalone kernels for computing error metrics
// ============================================================

// Compute element-wise absolute difference: out[i] = |x[i] - y[i]|
__global__ void abs_diff_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = fabsf(x[idx] - y[idx]);
    }
}

// Compute element-wise squared difference: out[i] = (x[i] - y[i])^2
__global__ void squared_diff_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float d = x[idx] - y[idx];
        out[idx] = d * d;
    }
}

// Compute element-wise squared values: out[i] = x[i]^2
__global__ void squared_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] * x[idx];
    }
}

// Max reduction kernel (single block)
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Sum reduction kernel (single block)
__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// BF16 to Float conversion kernel (for validation)
__global__ void bf16_to_float_kernel(
    const __hip_bfloat16* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

// Float to BF16 conversion kernel
__global__ void float_to_bf16_kernel(
    const float* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

// L2 normalization kernel: normalizes vectors of length vec_len
// input: (N, vec_len), output: (N, vec_len)
__global__ void l2_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int vec_len
) {
    int vec_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    if (vec_idx >= N || elem_idx >= vec_len) return;

    float val = input[vec_idx * vec_len + elem_idx];

    // Compute sum of squares via shared memory reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = val * val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < vec_len) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float norm = sqrtf(sdata[0] + 1e-6f);
    output[vec_idx * vec_len + elem_idx] = val / norm;
}

extern "C" {

void launch_abs_diff(const float* x, const float* y, float* out, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    abs_diff_kernel<<<blocks, threads, 0, stream>>>(x, y, out, N);
}

void launch_squared_diff(const float* x, const float* y, float* out, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    squared_diff_kernel<<<blocks, threads, 0, stream>>>(x, y, out, N);
}

void launch_max_reduce(const float* input, float* output, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    max_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(input, output, N);
}

void launch_sum_reduce(const float* input, float* output, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(input, output, N);
}

void launch_bf16_to_float(const __hip_bfloat16* input, float* output, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    bf16_to_float_kernel<<<blocks, threads, 0, stream>>>(input, output, N);
}

void launch_float_to_bf16(const float* input, __hip_bfloat16* output, int N, hipStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float_to_bf16_kernel<<<blocks, threads, 0, stream>>>(input, output, N);
}

void launch_l2_normalize(const float* input, float* output, int N, int vec_len, hipStream_t stream) {
    int threads = ((vec_len + 63) / 64) * 64;  // round up to warp
    if (threads > 1024) threads = 1024;
    l2_normalize_kernel<<<N, threads, threads * sizeof(float), stream>>>(input, output, N, vec_len);
}

} // extern "C"

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void abs_diff_wrapper(pybind11::object x, pybind11::object y, pybind11::object out, int N) {
    launch_abs_diff((const float*)get_data_ptr(x), (const float*)get_data_ptr(y),
                    (float*)get_data_ptr(out), N, 0);
}
void squared_diff_wrapper(pybind11::object x, pybind11::object y, pybind11::object out, int N) {
    launch_squared_diff((const float*)get_data_ptr(x), (const float*)get_data_ptr(y),
                        (float*)get_data_ptr(out), N, 0);
}
void max_reduce_wrapper(pybind11::object input, pybind11::object output, int N) {
    launch_max_reduce((const float*)get_data_ptr(input), (float*)get_data_ptr(output), N, 0);
}
void sum_reduce_wrapper(pybind11::object input, pybind11::object output, int N) {
    launch_sum_reduce((const float*)get_data_ptr(input), (float*)get_data_ptr(output), N, 0);
}
void bf16_to_float_wrapper(pybind11::object input, pybind11::object output, int N) {
    launch_bf16_to_float((const __hip_bfloat16*)get_data_ptr(input),
                         (float*)get_data_ptr(output), N, 0);
}
void float_to_bf16_wrapper(pybind11::object input, pybind11::object output, int N) {
    launch_float_to_bf16((const float*)get_data_ptr(input),
                         (__hip_bfloat16*)get_data_ptr(output), N, 0);
}
void l2_normalize_wrapper(pybind11::object input, pybind11::object output, int N, int vec_len) {
    launch_l2_normalize((const float*)get_data_ptr(input),
                        (float*)get_data_ptr(output), N, vec_len, 0);
}

PYBIND11_MODULE(gdr_utils_tk, m) {
    m.doc() = "GDR utility functions: error metrics, type conversion, L2 norm";
    m.def("abs_diff", &abs_diff_wrapper, "Element-wise absolute difference");
    m.def("squared_diff", &squared_diff_wrapper, "Element-wise squared difference");
    m.def("max_reduce", &max_reduce_wrapper, "Max reduction");
    m.def("sum_reduce", &sum_reduce_wrapper, "Sum reduction");
    m.def("bf16_to_float", &bf16_to_float_wrapper, "BF16 to float conversion");
    m.def("float_to_bf16", &float_to_bf16_wrapper, "Float to BF16 conversion");
    m.def("l2_normalize", &l2_normalize_wrapper, "L2 normalization of vectors",
          pybind11::arg("input"), pybind11::arg("output"),
          pybind11::arg("N"), pybind11::arg("vec_len"));
}
