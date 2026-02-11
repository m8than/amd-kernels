// SPDX-License-Identifier: MIT
// L2 normalization kernel ported from Triton to HIP C++
// Forward: y = x / sqrt(sum(x^2) + eps), rstd = 1/sqrt(sum(x^2) + eps)
// Backward: dx = dy * rstd - sum(dy * y) * y * rstd

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// L2 Norm Forward Kernel
// Input:  x of shape [T, D]
// Output: y of shape [T, D], rstd of shape [T]
// Each block handles BT rows
// ============================================================================
template<int BD>
__global__ void l2norm_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ rstd,
    float eps,
    int T, int D
) {
    int i_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (i_t >= T) return;

    int tid = threadIdx.x;

    // Compute sum of squares for this row using thread-parallel reduction
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = x[i_t * D + d];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }

    float r = 1.0f / sqrtf(sum_sq + eps);

    // Store rstd
    if (tid == 0) {
        rstd[i_t] = r;
    }

    // Normalize and write output
    for (int d = tid; d < D; d += blockDim.x) {
        y[i_t * D + d] = x[i_t * D + d] * r;
    }
}

// ============================================================================
// L2 Norm Backward Kernel
// Input:  y [T, D], rstd [T], dy [T, D]
// Output: dx [T, D]
// dx = dy * rstd - sum(dy * y) * y * rstd
// ============================================================================
__global__ void l2norm_bwd_kernel(
    const float* __restrict__ y,
    const float* __restrict__ rstd,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    float eps,
    int T, int D
) {
    int i_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (i_t >= T) return;

    int tid = threadIdx.x;

    // Compute sum(dy * y) for this row
    float dot_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        dot_sum += dy[i_t * D + d] * y[i_t * D + d];
    }

    // Warp reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        dot_sum += __shfl_xor(dot_sum, offset);
    }

    float r = rstd[i_t];

    // Compute gradient
    for (int d = tid; d < D; d += blockDim.x) {
        dx[i_t * D + d] = dy[i_t * D + d] * r - dot_sum * y[i_t * D + d] * r;
    }
}

// ============================================================================
// Large-D version: uses shared memory reduction across warps
// ============================================================================
__global__ void l2norm_fwd_kernel_large(
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ rstd,
    float eps,
    int T, int D
) {
    extern __shared__ float smem[];

    int i_t = blockIdx.x;
    if (i_t >= T) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Compute partial sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += num_threads) {
        float val = x[i_t * D + d];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }

    // Write warp result to shared memory
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    if (lane_id == 0) {
        smem[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = num_threads / 64;
    if (warp_id == 0 && lane_id < num_warps) {
        sum_sq = smem[lane_id];
    } else {
        sum_sq = 0.0f;
    }

    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }

    float r = 1.0f / sqrtf(sum_sq + eps);

    if (tid == 0) {
        rstd[i_t] = r;
    }
    __syncthreads();

    r = rstd[i_t];  // broadcast to all threads

    // Normalize
    for (int d = tid; d < D; d += num_threads) {
        y[i_t * D + d] = x[i_t * D + d] * r;
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_l2norm_fwd(
    const float* x, float* y, float* rstd,
    int T, int D, float eps, hipStream_t stream
) {
    if (D <= 512) {
        // Use multi-row kernel: each warp handles one row
        int threads_per_row = 64;  // one warp
        int rows_per_block = 4;
        dim3 block(threads_per_row, rows_per_block);
        dim3 grid((T + rows_per_block - 1) / rows_per_block);
        hipLaunchKernelGGL(l2norm_fwd_kernel<512>, grid, block, 0, stream,
            x, y, rstd, eps, T, D);
    } else {
        // Use one-row-per-block kernel for large D
        int threads = 256;
        int smem_size = (threads / 64) * sizeof(float);
        hipLaunchKernelGGL(l2norm_fwd_kernel_large, dim3(T), dim3(threads),
            smem_size, stream, x, y, rstd, eps, T, D);
    }
}

void dispatch_l2norm_bwd(
    const float* y, const float* rstd, const float* dy, float* dx,
    int T, int D, float eps, hipStream_t stream
) {
    int threads_per_row = 64;
    int rows_per_block = 4;
    dim3 block(threads_per_row, rows_per_block);
    dim3 grid((T + rows_per_block - 1) / rows_per_block);
    hipLaunchKernelGGL(l2norm_bwd_kernel, grid, block, 0, stream,
        y, rstd, dy, dx, eps, T, D);
}

// ============================================================================
// Test harness
// ============================================================================
void reference_l2norm_fwd(
    const float* x, float* y, float* rstd,
    int T, int D, float eps
) {
    for (int t = 0; t < T; t++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            sum_sq += x[t * D + d] * x[t * D + d];
        }
        float r = 1.0f / sqrtf(sum_sq + eps);
        rstd[t] = r;
        for (int d = 0; d < D; d++) {
            y[t * D + d] = x[t * D + d] * r;
        }
    }
}

void reference_l2norm_bwd(
    const float* y, const float* rstd, const float* dy, float* dx,
    int T, int D, float eps
) {
    for (int t = 0; t < T; t++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += dy[t * D + d] * y[t * D + d];
        }
        float r = rstd[t];
        for (int d = 0; d < D; d++) {
            dx[t * D + d] = dy[t * D + d] * r - dot * y[t * D + d] * r;
        }
    }
}

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static std::array<int,4> get_tensor_shape(pybind11::object t) {
    std::array<int,4> s = {1,1,1,1};
    auto shape = t.attr("shape").cast<pybind11::tuple>();
    int nd = shape.size();
    for (int i = 0; i < nd && i < 4; i++)
        s[4-nd+i] = pybind11::cast<int>(shape[i]);
    return s;
}

void l2norm_fwd_wrapper(pybind11::object x, pybind11::object y, pybind11::object rstd, float eps) {
    auto xs = get_tensor_shape(x);
    int T = xs[2]; // [T, D]
    int D = xs[3];
    dispatch_l2norm_fwd(
        (const float*)get_data_ptr(x),
        (float*)get_data_ptr(y),
        (float*)get_data_ptr(rstd),
        T, D, eps, 0);
}

void l2norm_bwd_wrapper(pybind11::object y, pybind11::object rstd,
                        pybind11::object dy, pybind11::object dx, float eps) {
    auto ys = get_tensor_shape(y);
    int T = ys[2];
    int D = ys[3];
    dispatch_l2norm_bwd(
        (const float*)get_data_ptr(y),
        (const float*)get_data_ptr(rstd),
        (const float*)get_data_ptr(dy),
        (float*)get_data_ptr(dx),
        T, D, eps, 0);
}

PYBIND11_MODULE(l2norm_tk, m) {
    m.doc() = "L2 normalization forward and backward kernels";
    m.def("l2norm_fwd", &l2norm_fwd_wrapper, "L2 norm forward: y = x / ||x||, rstd = 1/||x||",
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("rstd"), pybind11::arg("eps") = 1e-6f);
    m.def("l2norm_bwd", &l2norm_bwd_wrapper, "L2 norm backward: dx = dy*rstd - sum(dy*y)*y*rstd",
          pybind11::arg("y"), pybind11::arg("rstd"), pybind11::arg("dy"), pybind11::arg("dx"), pybind11::arg("eps") = 1e-6f);
}
