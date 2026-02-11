// SPDX-License-Identifier: MIT
// Cumulative sum kernel ported from Triton to HipKittens
// Supports both scalar (3D) and vector (4D) chunk-local cumsum

#include "kittens.cuh"
using namespace kittens;

// Configuration
#define NUM_WARPS_CUMSUM 4
#define NUM_THREADS_CUMSUM (kittens::WARP_THREADS * NUM_WARPS_CUMSUM)

// ============================================================================
// Scalar cumsum: input [B, T, H], chunk along T dimension
// Each thread block handles one chunk of BT elements for one (batch, head) pair
// ============================================================================
template<int BT>
struct cumsum_scalar_globals {
    using input_gl  = gl<float, -1, -1, -1, 1>;  // [B, T, H, 1] or reinterpreted
    using output_gl = gl<float, -1, -1, -1, 1>;

    input_gl  input;
    output_gl output;
    int B, T, H;
    float scale;
    bool has_scale;
    bool reverse;
    hipStream_t stream;

    dim3 grid() {
        int NT = (T + BT - 1) / BT;
        return dim3(NT, B * H);
    }
    dim3 block() { return dim3(NUM_THREADS_CUMSUM); }
    size_t dynamic_shared_memory() { return 0; }
};

// Scalar cumsum kernel: each block processes BT elements
// Using raw pointer arithmetic since this is a 1D scan within a chunk
template<int BT>
__global__ void chunk_local_cumsum_scalar_kernel(
    const float* __restrict__ s,
    float* __restrict__ o,
    int T, int H, int B,
    float scale, bool has_scale, bool reverse
) {
    int i_t = blockIdx.x;   // chunk index
    int i_bh = blockIdx.y;  // batch*head index
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int chunk_start = i_t * BT;

    // Each thread loads one element within the chunk
    // We do a simple sequential cumsum since BT is small (typically 64)
    __shared__ float smem[BT];

    // Load chunk into shared memory
    // Layout: [B, T, H] -> s[bos*H + t*H + h] for non-head-first
    // But Triton uses strides, we'll use [B, T, H] layout: element at (b, t, h) = s[(b*T + t)*H + h]
    for (int i = tid; i < BT; i += blockDim.x) {
        int t_idx = chunk_start + i;
        if (t_idx < T) {
            smem[i] = s[(bos + t_idx) * H + i_h];
        } else {
            smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Thread 0 performs prefix sum (BT is small, typically 64)
    if (tid == 0) {
        if (reverse) {
            // Reverse cumsum: sum from end to start
            float total = 0.0f;
            for (int i = 0; i < BT; i++) {
                total += smem[i];
            }
            float running = 0.0f;
            for (int i = 0; i < BT; i++) {
                float val = smem[i];
                smem[i] = total - running;  // -cumsum + total + original
                // Actually: reverse cumsum[i] = sum(s[i:]) = total - cumsum_exclusive[i]
                // Triton does: b_o = -b_o + b_z + b_s where b_o=cumsum, b_z=total, b_s=original
                running += val;
            }
            // Recompute properly: reverse_cumsum[i] = sum(s[i:BT])
            // = total - cumsum_exclusive[i] = total - (cumsum[i] - s[i])
            // Triton: b_o = cumsum, then -b_o + b_z + b_s = -cumsum + total + s = total - cumsum + s
            // = total - (cumsum - s) = total - cumsum_exclusive
            // So reverse_cumsum[i] = total - cumsum[i] + s[i]
            // Let's just redo it simply:
            float rev_sum = 0.0f;
            for (int i = BT - 1; i >= 0; i--) {
                int t_idx = chunk_start + i;
                if (t_idx < T) {
                    rev_sum += s[(bos + t_idx) * H + i_h];
                }
                smem[i] = rev_sum;
            }
        } else {
            // Forward cumsum
            float running = 0.0f;
            for (int i = 0; i < BT; i++) {
                running += smem[i];
                smem[i] = running;
            }
        }

        if (has_scale) {
            for (int i = 0; i < BT; i++) {
                smem[i] *= scale;
            }
        }
    }
    __syncthreads();

    // Write back
    for (int i = tid; i < BT; i += blockDim.x) {
        int t_idx = chunk_start + i;
        if (t_idx < T) {
            o[(bos + t_idx) * H + i_h] = smem[i];
        }
    }
}

// ============================================================================
// Vector cumsum: input [B, T, H, S], chunk along T dimension
// Each thread block handles one chunk of BT rows for one (batch, head) pair
// ============================================================================
template<int BT>
__global__ void chunk_local_cumsum_vector_kernel(
    const float* __restrict__ s,
    float* __restrict__ o,
    int T, int H, int S, int B,
    float scale, bool has_scale, bool reverse
) {
    int i_t = blockIdx.x;   // chunk index
    int i_bh = blockIdx.y;  // batch*head index
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;
    int chunk_start = i_t * BT;

    // Process one column at a time per thread across the S dimension
    // Layout: [B, T, H, S] -> element at (b,t,h,s) = s[((b*T + t)*H + h)*S + s_idx]
    for (int s_idx = tid; s_idx < S; s_idx += blockDim.x) {
        if (!reverse) {
            float running = 0.0f;
            for (int i = 0; i < BT; i++) {
                int t_idx = chunk_start + i;
                float val = 0.0f;
                if (t_idx < T) {
                    val = s[((bos + t_idx) * H + i_h) * S + s_idx];
                }
                running += val;
                float out_val = has_scale ? running * scale : running;
                if (t_idx < T) {
                    o[((bos + t_idx) * H + i_h) * S + s_idx] = out_val;
                }
            }
        } else {
            float running = 0.0f;
            for (int i = BT - 1; i >= 0; i--) {
                int t_idx = chunk_start + i;
                float val = 0.0f;
                if (t_idx < T) {
                    val = s[((bos + t_idx) * H + i_h) * S + s_idx];
                }
                running += val;
                float out_val = has_scale ? running * scale : running;
                if (t_idx < T) {
                    o[((bos + t_idx) * H + i_h) * S + s_idx] = out_val;
                }
            }
        }
    }
}

// ============================================================================
// Host dispatch functions
// ============================================================================
template<int BT>
void dispatch_cumsum_scalar(
    const float* s, float* o,
    int B, int T, int H,
    float scale, bool has_scale, bool reverse,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(NUM_THREADS_CUMSUM);
    hipLaunchKernelGGL(
        chunk_local_cumsum_scalar_kernel<BT>,
        grid, block, 0, stream,
        s, o, T, H, B, scale, has_scale, reverse
    );
}

template<int BT>
void dispatch_cumsum_vector(
    const float* s, float* o,
    int B, int T, int H, int S,
    float scale, bool has_scale, bool reverse,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(NUM_THREADS_CUMSUM);
    hipLaunchKernelGGL(
        chunk_local_cumsum_vector_kernel<BT>,
        grid, block, 0, stream,
        s, o, T, H, S, B, scale, has_scale, reverse
    );
}

// ============================================================================
// Test harness
// ============================================================================
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

void reference_cumsum_scalar(
    const float* s, float* o,
    int B, int T, int H, int BT,
    float scale, bool has_scale, bool reverse
) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int NT = (T + BT - 1) / BT;
            for (int nt = 0; nt < NT; nt++) {
                int chunk_start = nt * BT;
                if (!reverse) {
                    float running = 0.0f;
                    for (int i = 0; i < BT && chunk_start + i < T; i++) {
                        int t = chunk_start + i;
                        running += s[(b * T + t) * H + h];
                        o[(b * T + t) * H + h] = has_scale ? running * scale : running;
                    }
                } else {
                    float running = 0.0f;
                    for (int i = BT - 1; i >= 0; i--) {
                        int t = chunk_start + i;
                        if (t < T) {
                            running += s[(b * T + t) * H + h];
                            o[(b * T + t) * H + h] = has_scale ? running * scale : running;
                        }
                    }
                }
            }
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

void cumsum_scalar_wrapper(pybind11::object s, pybind11::object o,
                           int B, int T, int H, int BT,
                           float scale, bool has_scale, bool reverse) {
    if (BT == 64) {
        dispatch_cumsum_scalar<64>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, scale, has_scale, reverse, 0);
    } else if (BT == 32) {
        dispatch_cumsum_scalar<32>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, scale, has_scale, reverse, 0);
    } else {
        dispatch_cumsum_scalar<64>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, scale, has_scale, reverse, 0);
    }
}

void cumsum_vector_wrapper(pybind11::object s, pybind11::object o,
                           int B, int T, int H, int S, int BT,
                           float scale, bool has_scale, bool reverse) {
    if (BT == 64) {
        dispatch_cumsum_vector<64>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, S, scale, has_scale, reverse, 0);
    } else if (BT == 32) {
        dispatch_cumsum_vector<32>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, S, scale, has_scale, reverse, 0);
    } else {
        dispatch_cumsum_vector<64>(
            (const float*)get_data_ptr(s), (float*)get_data_ptr(o),
            B, T, H, S, scale, has_scale, reverse, 0);
    }
}

PYBIND11_MODULE(cumsum_tk, m) {
    m.doc() = "Chunk-local cumulative sum kernels (scalar and vector)";
    m.def("cumsum_scalar", &cumsum_scalar_wrapper, "Chunk-local scalar cumsum",
          pybind11::arg("s"), pybind11::arg("o"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"), pybind11::arg("BT"),
          pybind11::arg("scale") = 1.0f, pybind11::arg("has_scale") = false,
          pybind11::arg("reverse") = false);
    m.def("cumsum_vector", &cumsum_vector_wrapper, "Chunk-local vector cumsum",
          pybind11::arg("s"), pybind11::arg("o"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"), pybind11::arg("S"), pybind11::arg("BT"),
          pybind11::arg("scale") = 1.0f, pybind11::arg("has_scale") = false,
          pybind11::arg("reverse") = false);
}
