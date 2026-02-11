// SPDX-License-Identifier: MIT
// Causal conv1d with fused split q/k/v output for prefill
// Ported from Triton causal_conv1d_fwd_split_qkv.py
//
// Performs 1D causal convolution (kernel_width = 2,3,4) on input x,
// optionally applies SiLU activation, then splits output into q, k, v:
//   out = conv1d(x, w) + bias
//   if silu: out = silu(out)
//   q = out[:, :k_dim]
//   k = out[:, k_dim:2*k_dim]
//   v = out[:, 2*k_dim:2*k_dim+v_dim]
//
// Input: x [dim, total_tokens], w [dim, kernel_width], bias [dim]
//        query_start_loc [num_seqs+1] (cumulative sequence starts)
// Output: q [total_tokens, k_dim], k [total_tokens, k_dim], v [total_tokens, v_dim]

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
// Causal conv1d + split kernel
// Grid: (num_seqs, ceil(max_seqlen/BLOCK_M), ceil(dim/BLOCK_N))
// Each thread handles one feature dimension
// ============================================================================
template<int KERNEL_WIDTH, bool HAS_BIAS, bool SILU_ACTIVATION>
__global__ void causal_conv1d_fwd_split_kernel(
    const float* __restrict__ x,              // [dim, total_tokens] (dim-major)
    const float* __restrict__ w,              // [dim, kernel_width]
    const float* __restrict__ bias,           // [dim] or nullptr
    const int* __restrict__ query_start_loc,  // [num_seqs + 1]
    float* __restrict__ q_out,                // [total_tokens, k_dim]
    float* __restrict__ k_out,                // [total_tokens, k_dim]
    float* __restrict__ v_out,                // [total_tokens, v_dim]
    int dim, int k_dim, int v_dim,
    int stride_x_dim,   // stride between features (typically total_tokens)
    int stride_x_token, // stride between tokens (typically 1)
    int BLOCK_M
) {
    int idx_seq = blockIdx.x;
    int chunk_offset = blockIdx.y;
    int feat_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (feat_idx >= dim) return;

    int seq_start = query_start_loc[idx_seq];
    int seq_end = query_start_loc[idx_seq + 1];
    int seqlen = seq_end - seq_start;

    int token_offset = BLOCK_M * chunk_offset;
    int segment_len = min(BLOCK_M, seqlen - token_offset);
    if (segment_len <= 0) return;

    // Load convolution weights for this feature
    float w_vals[4];  // max KERNEL_WIDTH = 4
    for (int i = 0; i < KERNEL_WIDTH; i++) {
        w_vals[i] = w[feat_idx * KERNEL_WIDTH + i];
    }

    // Load bias
    float bias_val = 0.0f;
    if (HAS_BIAS) {
        bias_val = bias[feat_idx];
    }

    // Initialize sliding window (cols for history)
    float cols[3] = {0.0f, 0.0f, 0.0f};  // col0, col1, col2 for KERNEL_WIDTH-1 history

    // Load initial state from prior tokens or zeros
    if (chunk_offset > 0) {
        // Load from preceding tokens in the sequence
        int base = seq_start + token_offset;
        if (KERNEL_WIDTH >= 2) {
            cols[0] = x[feat_idx * stride_x_dim + (base - 1) * stride_x_token];
        }
        if (KERNEL_WIDTH >= 3) {
            cols[1] = cols[0];
            cols[0] = x[feat_idx * stride_x_dim + (base - 2) * stride_x_token];
        }
        if (KERNEL_WIDTH >= 4) {
            cols[2] = cols[1];
            cols[1] = cols[0];
            cols[0] = x[feat_idx * stride_x_dim + (base - 3) * stride_x_token];
        }
    }
    // else: first chunk starts with zeros (causal, no history)

    // Process each token in the segment
    for (int t = 0; t < segment_len; t++) {
        int global_token = seq_start + token_offset + t;

        // Load current input
        float x_curr = x[feat_idx * stride_x_dim + global_token * stride_x_token];

        // Compute convolution
        float acc = bias_val;
        if (KERNEL_WIDTH == 2) {
            acc += cols[0] * w_vals[0] + x_curr * w_vals[1];
            cols[0] = x_curr;
        } else if (KERNEL_WIDTH == 3) {
            acc += cols[0] * w_vals[0] + cols[1] * w_vals[1] + x_curr * w_vals[2];
            cols[0] = cols[1];
            cols[1] = x_curr;
        } else if (KERNEL_WIDTH == 4) {
            acc += cols[0] * w_vals[0] + cols[1] * w_vals[1] + cols[2] * w_vals[2] + x_curr * w_vals[3];
            cols[0] = cols[1];
            cols[1] = cols[2];
            cols[2] = x_curr;
        }

        // SiLU activation
        if (SILU_ACTIVATION) {
            acc = acc / (1.0f + expf(-acc));
        }

        // Split into q, k, v based on feature index
        if (feat_idx < k_dim) {
            q_out[global_token * k_dim + feat_idx] = acc;
        } else if (feat_idx < 2 * k_dim) {
            k_out[global_token * k_dim + (feat_idx - k_dim)] = acc;
        } else if (feat_idx < 2 * k_dim + v_dim) {
            v_out[global_token * v_dim + (feat_idx - 2 * k_dim)] = acc;
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_causal_conv1d_split(
    const float* x, const float* w, const float* bias,
    const int* query_start_loc,
    float* q_out, float* k_out, float* v_out,
    int dim, int k_dim, int v_dim, int kernel_width,
    int total_tokens, int num_seqs, int max_seqlen,
    int stride_x_dim, int stride_x_token,
    bool has_bias, bool silu_activation,
    hipStream_t stream
) {
    int BLOCK_M = 8;
    int BLOCK_N = 256;
    int num_feat_blocks = (dim + BLOCK_N - 1) / BLOCK_N;
    int num_chunks = (max_seqlen + BLOCK_M - 1) / BLOCK_M;

    dim3 grid(num_seqs, num_chunks, num_feat_blocks);
    dim3 block(BLOCK_N);

    // Dispatch based on kernel_width and activation
    #define LAUNCH(KW, BIAS, SILU) \
        hipLaunchKernelGGL((causal_conv1d_fwd_split_kernel<KW, BIAS, SILU>), \
            grid, block, 0, stream, \
            x, w, bias, query_start_loc, q_out, k_out, v_out, \
            dim, k_dim, v_dim, stride_x_dim, stride_x_token, BLOCK_M)

    if (kernel_width == 4 && has_bias && silu_activation) { LAUNCH(4, true, true); }
    else if (kernel_width == 4 && has_bias && !silu_activation) { LAUNCH(4, true, false); }
    else if (kernel_width == 4 && !has_bias && silu_activation) { LAUNCH(4, false, true); }
    else if (kernel_width == 4 && !has_bias && !silu_activation) { LAUNCH(4, false, false); }
    else if (kernel_width == 3 && has_bias && silu_activation) { LAUNCH(3, true, true); }
    else if (kernel_width == 3 && has_bias && !silu_activation) { LAUNCH(3, true, false); }
    else if (kernel_width == 3 && !has_bias && silu_activation) { LAUNCH(3, false, true); }
    else if (kernel_width == 3 && !has_bias && !silu_activation) { LAUNCH(3, false, false); }
    else if (kernel_width == 2 && has_bias && silu_activation) { LAUNCH(2, true, true); }
    else if (kernel_width == 2 && has_bias && !silu_activation) { LAUNCH(2, true, false); }
    else if (kernel_width == 2 && !has_bias && silu_activation) { LAUNCH(2, false, true); }
    else if (kernel_width == 2 && !has_bias && !silu_activation) { LAUNCH(2, false, false); }

    #undef LAUNCH
}

// ============================================================================
// Test harness
// ============================================================================
// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void causal_conv1d_fwd_split_wrapper(
    pybind11::object x, pybind11::object w, pybind11::object bias,
    pybind11::object query_start_loc,
    pybind11::object q_out, pybind11::object k_out, pybind11::object v_out,
    int dim, int k_dim, int v_dim, int kernel_width,
    int total_tokens, int num_seqs, int max_seqlen,
    int stride_x_dim, int stride_x_token,
    bool has_bias, bool silu_activation) {
    dispatch_causal_conv1d_split(
        (const float*)get_data_ptr(x),
        (const float*)get_data_ptr(w),
        has_bias ? (const float*)get_data_ptr(bias) : nullptr,
        (const int*)get_data_ptr(query_start_loc),
        (float*)get_data_ptr(q_out),
        (float*)get_data_ptr(k_out),
        (float*)get_data_ptr(v_out),
        dim, k_dim, v_dim, kernel_width,
        total_tokens, num_seqs, max_seqlen,
        stride_x_dim, stride_x_token,
        has_bias, silu_activation, 0);
}

PYBIND11_MODULE(causal_conv1d_fwd_split_qkv_tk, m) {
    m.doc() = "Causal conv1d with fused split q/k/v for prefill";
    m.def("causal_conv1d_fwd_split", &causal_conv1d_fwd_split_wrapper,
          "Causal 1D convolution with split output into q, k, v",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("bias"),
          pybind11::arg("query_start_loc"),
          pybind11::arg("q_out"), pybind11::arg("k_out"), pybind11::arg("v_out"),
          pybind11::arg("dim"), pybind11::arg("k_dim"), pybind11::arg("v_dim"),
          pybind11::arg("kernel_width"),
          pybind11::arg("total_tokens"), pybind11::arg("num_seqs"), pybind11::arg("max_seqlen"),
          pybind11::arg("stride_x_dim"), pybind11::arg("stride_x_token"),
          pybind11::arg("has_bias") = true, pybind11::arg("silu_activation") = true);
}
