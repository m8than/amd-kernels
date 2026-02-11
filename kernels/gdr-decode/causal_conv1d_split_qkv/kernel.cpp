// Causal Conv1D Update with Split QKV Output
// Ported from reference/triton/decode/causal_conv1d_split_qkv.py
//
// Performs causal 1D convolution update and directly splits output into q, k, v.
// Used in gated delta rule decode where the conv1d output is the concatenated
// [query, key, value] along the feature dimension.
//
// Algorithm:
//   1. Load conv_state (sliding window of past inputs)
//   2. Update conv_state with new input tokens
//   3. For each token: compute conv1d = sum(weight[j] * window[j]) + bias
//   4. Optionally apply SiLU activation
//   5. Split output: features [0, key_dim) -> q, [key_dim, 2*key_dim) -> k,
//                    [2*key_dim, 2*key_dim+value_dim) -> v
//
// Input layout:
//   x:          (batch, dim, seqlen) bf16 -- dim = 2*key_dim + value_dim
//   weight:     (dim, width) bf16 -- conv weights
//   bias:       (dim,) bf16 or nullptr
//   conv_state: (num_cache_lines, dim, state_len) bf16 -- sliding window state
//   conv_state_indices: (batch,) int64 or nullptr -- for continuous batching
//
// Output layout:
//   q: (batch, key_dim, seqlen) bf16
//   k: (batch, key_dim, seqlen) bf16
//   v: (batch, value_dim, seqlen) bf16
//   conv_state: updated in-place
//
// Grid: (batch, ceil(dim / BLOCK_N))
// Each thread handles one feature dimension element across all seqlen tokens.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

constexpr int PAD_SLOT_ID = -1;

__device__ __forceinline__ float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

template<int KERNEL_WIDTH, bool HAS_BIAS, bool SILU_ACTIVATION,
         bool IS_CONTINUOUS_BATCHING, bool USE_PAD_SLOT>
__global__ void causal_conv1d_update_split_qkv_kernel(
    const __hip_bfloat16* __restrict__ x_ptr,           // (batch, dim, seqlen)
    const __hip_bfloat16* __restrict__ w_ptr,            // (dim, width)
    const __hip_bfloat16* __restrict__ bias_ptr,         // (dim,) or nullptr
    __hip_bfloat16* __restrict__ conv_state_ptr,         // (num_cache_lines, dim, state_len)
    const int64_t* __restrict__ conv_state_indices_ptr,  // (batch,) or nullptr
    __hip_bfloat16* __restrict__ q_ptr,                  // (batch, key_dim, seqlen)
    __hip_bfloat16* __restrict__ k_ptr,                  // (batch, key_dim, seqlen)
    __hip_bfloat16* __restrict__ v_ptr,                  // (batch, value_dim, seqlen)
    int key_dim,
    int value_dim,
    int batch,
    int dim,
    int seqlen,
    int state_len,       // = KERNEL_WIDTH - 1
    int num_cache_lines,
    // Strides for x: (batch, dim, seqlen) contiguous
    int stride_x_batch,   // dim * seqlen
    int stride_x_dim,     // seqlen
    int stride_x_token,   // 1
    // Strides for weight: (dim, width) contiguous
    int stride_w_dim,     // width
    int stride_w_width,   // 1
    // Strides for conv_state: (num_cache_lines, dim, state_len) contiguous
    int stride_cs_batch,  // dim * state_len
    int stride_cs_dim,    // state_len
    int stride_cs_tok,    // 1
    // Strides for q: (batch, key_dim, seqlen)
    int stride_q_batch,
    int stride_q_dim,
    int stride_q_token,
    // Strides for k: (batch, key_dim, seqlen)
    int stride_k_batch,
    int stride_k_dim,
    int stride_k_token,
    // Strides for v: (batch, value_dim, seqlen)
    int stride_v_batch,
    int stride_v_dim,
    int stride_v_token
) {
    int idx_batch = blockIdx.x;
    if (idx_batch >= batch) return;

    int idx_feat = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx_feat >= dim) return;

    // Resolve conv_state batch index
    int64_t cs_batch_idx;
    if (IS_CONTINUOUS_BATCHING) {
        cs_batch_idx = conv_state_indices_ptr[idx_batch];
    } else {
        cs_batch_idx = idx_batch;
    }

    if (USE_PAD_SLOT) {
        if (cs_batch_idx == PAD_SLOT_ID) return;
    }

    // Load conv_state (sliding window of past values) for this feature
    float state[KERNEL_WIDTH - 1];
    #pragma unroll
    for (int j = 0; j < KERNEL_WIDTH - 1; j++) {
        int cs_offset = cs_batch_idx * stride_cs_batch + idx_feat * stride_cs_dim + j * stride_cs_tok;
        state[j] = __bfloat162float(conv_state_ptr[cs_offset]);
    }

    // Load weights for this feature
    float w[KERNEL_WIDTH];
    #pragma unroll
    for (int j = 0; j < KERNEL_WIDTH; j++) {
        w[j] = __bfloat162float(w_ptr[idx_feat * stride_w_dim + j * stride_w_width]);
    }

    // Load bias
    float bias_val = 0.0f;
    if (HAS_BIAS) {
        bias_val = __bfloat162float(bias_ptr[idx_feat]);
    }

    // Base pointer for input x
    int x_base = idx_batch * stride_x_batch + idx_feat * stride_x_dim;

    // Process each token
    for (int t = 0; t < seqlen; t++) {
        float x_val = __bfloat162float(x_ptr[x_base + t * stride_x_token]);

        // Compute convolution
        float acc = bias_val;
        #pragma unroll
        for (int j = 0; j < KERNEL_WIDTH - 1; j++) {
            acc += w[j] * state[j];
        }
        acc += w[KERNEL_WIDTH - 1] * x_val;

        // Shift sliding window
        #pragma unroll
        for (int j = 0; j < KERNEL_WIDTH - 2; j++) {
            state[j] = state[j + 1];
        }
        if constexpr (KERNEL_WIDTH > 1) {
            state[KERNEL_WIDTH - 2] = x_val;
        }

        // Apply SiLU activation
        if (SILU_ACTIVATION) {
            acc = device_silu(acc);
        }

        // Split and store to q, k, v based on feature index
        if (idx_feat < key_dim) {
            // Query: features [0, key_dim)
            int q_offset = idx_batch * stride_q_batch + idx_feat * stride_q_dim + t * stride_q_token;
            q_ptr[q_offset] = __float2bfloat16(acc);
        } else if (idx_feat < 2 * key_dim) {
            // Key: features [key_dim, 2*key_dim)
            int k_feat = idx_feat - key_dim;
            int k_offset = idx_batch * stride_k_batch + k_feat * stride_k_dim + t * stride_k_token;
            k_ptr[k_offset] = __float2bfloat16(acc);
        } else if (idx_feat < 2 * key_dim + value_dim) {
            // Value: features [2*key_dim, 2*key_dim + value_dim)
            int v_feat = idx_feat - 2 * key_dim;
            int v_offset = idx_batch * stride_v_batch + v_feat * stride_v_dim + t * stride_v_token;
            v_ptr[v_offset] = __float2bfloat16(acc);
        }
    }

    // Update conv_state with new sliding window
    #pragma unroll
    for (int j = 0; j < KERNEL_WIDTH - 1; j++) {
        int cs_offset = cs_batch_idx * stride_cs_batch + idx_feat * stride_cs_dim + j * stride_cs_tok;
        conv_state_ptr[cs_offset] = __float2bfloat16(state[j]);
    }
}

extern "C" {

void launch_causal_conv1d_update_split_qkv(
    const __hip_bfloat16* x, const __hip_bfloat16* w, const __hip_bfloat16* bias,
    __hip_bfloat16* conv_state,
    const int64_t* conv_state_indices,
    __hip_bfloat16* q, __hip_bfloat16* k, __hip_bfloat16* v,
    int key_dim, int value_dim,
    int batch, int dim, int seqlen, int width,
    int num_cache_lines,
    bool has_bias, bool silu_activation,
    bool is_continuous_batching,
    hipStream_t stream
) {
    int state_len = width - 1;
    int threads = 256;
    dim3 grid(batch, (dim + threads - 1) / threads);

    // Compute strides assuming contiguous layout
    // x: (batch, dim, seqlen)
    int stride_x_batch = dim * seqlen;
    int stride_x_dim = seqlen;
    int stride_x_token = 1;
    // w: (dim, width)
    int stride_w_dim = width;
    int stride_w_width = 1;
    // conv_state: (num_cache_lines, dim, state_len)
    int stride_cs_batch = dim * state_len;
    int stride_cs_dim = state_len;
    int stride_cs_tok = 1;
    // q: (batch, key_dim, seqlen)
    int stride_q_batch = key_dim * seqlen;
    int stride_q_dim = seqlen;
    int stride_q_token = 1;
    // k: (batch, key_dim, seqlen)
    int stride_k_batch = key_dim * seqlen;
    int stride_k_dim = seqlen;
    int stride_k_token = 1;
    // v: (batch, value_dim, seqlen)
    int stride_v_batch = value_dim * seqlen;
    int stride_v_dim = seqlen;
    int stride_v_token = 1;

    bool use_pad_slot = (conv_state_indices != nullptr);

    #define LAUNCH(KW, HB, SA, CB, PS) \
        causal_conv1d_update_split_qkv_kernel<KW, HB, SA, CB, PS> \
            <<<grid, threads, 0, stream>>>( \
                x, w, bias, conv_state, conv_state_indices, q, k, v, \
                key_dim, value_dim, batch, dim, seqlen, state_len, num_cache_lines, \
                stride_x_batch, stride_x_dim, stride_x_token, \
                stride_w_dim, stride_w_width, \
                stride_cs_batch, stride_cs_dim, stride_cs_tok, \
                stride_q_batch, stride_q_dim, stride_q_token, \
                stride_k_batch, stride_k_dim, stride_k_token, \
                stride_v_batch, stride_v_dim, stride_v_token)

    // Dispatch based on kernel width and flags
    // Most common: width=4, silu, with bias, continuous batching
    if (width == 4) {
        if (has_bias && silu_activation) {
            if (is_continuous_batching && use_pad_slot)
                LAUNCH(4, true, true, true, true);
            else if (is_continuous_batching)
                LAUNCH(4, true, true, true, false);
            else
                LAUNCH(4, true, true, false, false);
        } else if (!has_bias && !silu_activation) {
            if (is_continuous_batching && use_pad_slot)
                LAUNCH(4, false, false, true, true);
            else
                LAUNCH(4, false, false, false, false);
        } else if (has_bias && !silu_activation) {
            LAUNCH(4, true, false, false, false);
        } else {
            LAUNCH(4, false, true, false, false);
        }
    } else if (width == 3) {
        if (has_bias && silu_activation) {
            if (is_continuous_batching && use_pad_slot)
                LAUNCH(3, true, true, true, true);
            else
                LAUNCH(3, true, true, false, false);
        } else {
            LAUNCH(3, false, false, false, false);
        }
    } else if (width == 2) {
        if (has_bias && silu_activation) {
            LAUNCH(2, true, true, false, false);
        } else {
            LAUNCH(2, false, false, false, false);
        }
    } else {
        // Fallback: width=4 as default
        LAUNCH(4, true, true, false, false);
    }

    #undef LAUNCH
}

} // extern "C"

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void causal_conv1d_update_split_qkv_wrapper(
    pybind11::object x, pybind11::object w, pybind11::object bias,
    pybind11::object conv_state, pybind11::object conv_state_indices,
    pybind11::object q, pybind11::object k, pybind11::object v,
    int key_dim, int value_dim,
    int batch, int dim, int seqlen, int width,
    int num_cache_lines,
    bool has_bias, bool silu_activation,
    bool is_continuous_batching) {
    launch_causal_conv1d_update_split_qkv(
        (const __hip_bfloat16*)get_data_ptr(x),
        (const __hip_bfloat16*)get_data_ptr(w),
        has_bias ? (const __hip_bfloat16*)get_data_ptr(bias) : nullptr,
        (__hip_bfloat16*)get_data_ptr(conv_state),
        is_continuous_batching ? (const int64_t*)get_data_ptr(conv_state_indices) : nullptr,
        (__hip_bfloat16*)get_data_ptr(q),
        (__hip_bfloat16*)get_data_ptr(k),
        (__hip_bfloat16*)get_data_ptr(v),
        key_dim, value_dim,
        batch, dim, seqlen, width,
        num_cache_lines,
        has_bias, silu_activation,
        is_continuous_batching, 0);
}

PYBIND11_MODULE(causal_conv1d_split_qkv_tk, m) {
    m.doc() = "Causal conv1d update with split QKV output (decode, bf16)";
    m.def("causal_conv1d_update_split_qkv", &causal_conv1d_update_split_qkv_wrapper,
          "Causal conv1d update with split q, k, v output",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("bias"),
          pybind11::arg("conv_state"), pybind11::arg("conv_state_indices"),
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("key_dim"), pybind11::arg("value_dim"),
          pybind11::arg("batch"), pybind11::arg("dim"), pybind11::arg("seqlen"),
          pybind11::arg("width"), pybind11::arg("num_cache_lines"),
          pybind11::arg("has_bias") = true, pybind11::arg("silu_activation") = true,
          pybind11::arg("is_continuous_batching") = false);
}
