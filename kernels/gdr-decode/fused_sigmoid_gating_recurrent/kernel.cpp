// Fused Sigmoid Gating Delta Rule Update Kernel
// Ported from reference/triton/decode/fused_sigmoid_gating_recurrent.py
//
// Combines sigmoid gating computation with recurrent delta rule update.
// For each timestep t:
//   1. Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
//   2. Compute beta = sigmoid(b)
//   3. Apply gate: h *= exp(g)
//   4. Delta rule: v' = beta * (v - sum(h * k[:, None], 0))
//   5. Update: h += k[:, None] * v'[None, :]
//   6. Output: o = sum(h * q[:, None], 0)
//
// Shapes:
//   q, k:    (B*T, H, K) bf16
//   v:       (B*T, HV, V) bf16
//   b:       (B*T, HV) bf16 -- pre-sigmoid beta input
//   a:       (B*T, HV) bf16 -- gating input
//   A_log:   (HV,) bf16 -- log-scale parameter
//   dt_bias: (HV,) bf16 -- bias for gating
//   o:       (NK, B*T, HV, V) bf16 -- output (NK=1 for K<=BK)
//   h0_source: (num_states, HV, K, V) float32 -- state bank (in-place updated)
//   h0_indices: (N,) int32 -- maps batch to state index (-1 = no state)
//
// Grid: (NK, ceil(V/BV), N*HV)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

__device__ __forceinline__ float device_softplus(float x, float beta_sp, float threshold) {
    float bx = beta_sp * x;
    if (bx <= threshold) {
        return (1.0f / beta_sp) * logf(1.0f + expf(bx));
    }
    return x;
}

__device__ __forceinline__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ void fused_sigmoid_gating_delta_rule_update_kernel(
    const __hip_bfloat16* __restrict__ A_log,    // (HV,)
    const __hip_bfloat16* __restrict__ a,        // (total_tokens, HV)
    const __hip_bfloat16* __restrict__ dt_bias,  // (HV,)
    float softplus_beta,
    float softplus_threshold,
    const __hip_bfloat16* __restrict__ q,        // (total_tokens, H, K)
    const __hip_bfloat16* __restrict__ k,        // (total_tokens, H, K)
    const __hip_bfloat16* __restrict__ v,        // (total_tokens, HV, V)
    const __hip_bfloat16* __restrict__ b,        // (total_tokens, HV)
    __hip_bfloat16* __restrict__ o,              // (NK, total_tokens, HV, V)
    float* __restrict__ h0_source,               // (num_states, HV, K, V)
    const int* __restrict__ h0_indices,          // (N,)
    const int* __restrict__ cu_seqlens,          // (N+1,) or nullptr
    float scale,
    int T, int B, int H, int HV, int K, int V_dim,
    int total_tokens  // B*T for fixed length
) {
    int i_k = blockIdx.x;    // key block index (should be 0 for NK=1)
    int i_v = blockIdx.y;    // value block index
    int i_nh = blockIdx.z;   // combined (batch, head_v) index
    int i_n = i_nh / HV;
    int i_hv = i_nh % HV;
    int i_h = i_hv / (HV / H);

    int tid = threadIdx.x;
    int k_idx = i_k * BK + tid;
    int v_idx = i_v * BV + (tid % BV);  // for output reduction

    // Sequence bounds
    int bos, eos, seq_len;
    if (cu_seqlens != nullptr) {
        bos = cu_seqlens[i_n];
        eos = cu_seqlens[i_n + 1];
        seq_len = eos - bos;
    } else {
        bos = i_n * T;
        eos = bos + T;
        seq_len = T;
    }

    // Each thread handles one k-element; reduction needed for output
    if (k_idx >= K) return;

    // For this kernel, each thread handles h[k_idx, v_idx] for one v-element
    // We need one thread per (k, v) pair; grid handles v-blocks
    // Actually, from the Triton code: grid = (NK, ceil(V/BV), N*HV)
    // and within each block, threads tile over k and v.
    // Simplification: each thread handles one k-element, one v-element
    // threads = BK, grid.y tiles over V

    // Re-interpret: each block handles BK k-elements for one v-element
    v_idx = i_v;  // one v-element per grid.y position
    if (v_idx >= V_dim) return;

    // Load gating parameters (constant per head)
    float b_A_log = __bfloat162float(A_log[i_hv]);
    float b_dt_bias = __bfloat162float(dt_bias[i_hv]);

    // Initialize hidden state
    float b_h = 0.0f;
    int state_idx = -1;
    if (USE_INITIAL_STATE) {
        state_idx = h0_indices[i_n];
        if (state_idx >= 0) {
            int h0_offset = state_idx * HV * K * V_dim + i_hv * K * V_dim + k_idx * V_dim + v_idx;
            b_h = h0_source[h0_offset];
        }
    }

    // Pointer strides
    int q_stride = H * K;
    int k_stride = H * K;
    int v_stride = HV * V_dim;
    int o_stride = HV * V_dim;

    int q_base = (bos * H + i_h) * K + k_idx;
    int k_base = (bos * H + i_h) * K + k_idx;
    int v_base = (bos * HV + i_hv) * V_dim + v_idx;
    int b_base = bos * HV + i_hv;
    int a_base = bos * HV + i_hv;
    int o_base = (i_k * total_tokens + bos) * HV * V_dim + i_hv * V_dim + v_idx;

    // Recurrent loop
    for (int t = 0; t < seq_len; t++) {
        float b_q = __bfloat162float(q[q_base + t * q_stride]);
        float b_k_val = __bfloat162float(k[k_base + t * k_stride]);
        float b_v_val = __bfloat162float(v[v_base + t * v_stride]);
        float b_b = __bfloat162float(b[b_base + t * HV]);
        float b_a = __bfloat162float(a[a_base + t * HV]);

        // L2 normalization (optional)
        if (USE_QK_L2NORM) {
            // Need cross-thread reduction for L2 norm of q and k
            float q_sq = b_q * b_q;
            float k_sq = b_k_val * b_k_val;

            // Warp reduction for L2 norms
            for (int offset = 32; offset > 0; offset >>= 1) {
                q_sq += __shfl_xor(q_sq, offset);
                k_sq += __shfl_xor(k_sq, offset);
            }
            b_q /= sqrtf(q_sq + 1e-6f);
            b_k_val /= sqrtf(k_sq + 1e-6f);
        }

        b_q *= scale;

        // Compute sigmoid gating
        // g = -exp(A_log) * softplus(a + dt_bias)
        float sp = device_softplus(b_a + b_dt_bias, softplus_beta, softplus_threshold);
        float b_g = -expf(b_A_log) * sp;

        // Compute beta = sigmoid(b)
        float b_beta = device_sigmoid(b_b);

        // Apply gate: h *= exp(g)
        b_h *= expf(b_g);

        // Delta rule: need sum(h * k, axis=k) -- cross-thread reduction
        float hk_product = b_h * b_k_val;
        float hk_sum = hk_product;
        for (int offset = BK / 2; offset > 0; offset >>= 1) {
            hk_sum += __shfl_xor(hk_sum, offset);
        }

        // v' = beta * (v - hk_sum)
        float b_v_prime = b_beta * (b_v_val - hk_sum);

        // h += k * v'
        b_h += b_k_val * b_v_prime;

        // Output: sum(h * q, axis=k)
        float hq_product = b_h * b_q;
        float hq_sum = hq_product;
        for (int offset = BK / 2; offset > 0; offset >>= 1) {
            hq_sum += __shfl_xor(hq_sum, offset);
        }

        // Write output (only k_idx == 0 writes)
        if (k_idx == 0) {
            o[o_base + t * o_stride] = __float2bfloat16(hq_sum);
        }
    }

    // Store final state back to h0_source (in-place)
    if (USE_INITIAL_STATE) {
        if (state_idx >= 0) {
            int h0_offset = state_idx * HV * K * V_dim + i_hv * K * V_dim + k_idx * V_dim + v_idx;
            h0_source[h0_offset] = b_h;
        }
    }
}

extern "C" {

void launch_fused_sigmoid_gating_delta_rule_update(
    const __hip_bfloat16* A_log,
    const __hip_bfloat16* a,
    const __hip_bfloat16* dt_bias,
    float softplus_beta,
    float softplus_threshold,
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const __hip_bfloat16* b,
    __hip_bfloat16* o,
    float* h0_source,
    const int* h0_indices,
    const int* cu_seqlens,
    float scale,
    int T, int B, int H, int HV, int K, int V_dim,
    bool use_initial_state,
    bool use_qk_l2norm,
    hipStream_t stream
) {
    constexpr int BK = 128;
    constexpr int BV = 64;

    int N = B;
    int total_tokens = B * T;
    int NK = (K + BK - 1) / BK;
    int threads = K < BK ? ((K + 63) / 64) * 64 : BK;  // round up to warp size

    dim3 grid(NK, V_dim, N * HV);

    // Common decode path: with initial state, no L2 norm
    if (use_initial_state && !use_qk_l2norm) {
        fused_sigmoid_gating_delta_rule_update_kernel<BK, BV, true, false>
            <<<grid, threads, 0, stream>>>(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
                scale, T, B, H, HV, K, V_dim, total_tokens);
    } else if (use_initial_state && use_qk_l2norm) {
        fused_sigmoid_gating_delta_rule_update_kernel<BK, BV, true, true>
            <<<grid, threads, 0, stream>>>(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
                scale, T, B, H, HV, K, V_dim, total_tokens);
    } else if (!use_initial_state && !use_qk_l2norm) {
        fused_sigmoid_gating_delta_rule_update_kernel<BK, BV, false, false>
            <<<grid, threads, 0, stream>>>(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
                scale, T, B, H, HV, K, V_dim, total_tokens);
    } else {
        fused_sigmoid_gating_delta_rule_update_kernel<BK, BV, false, true>
            <<<grid, threads, 0, stream>>>(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
                scale, T, B, H, HV, K, V_dim, total_tokens);
    }
}

} // extern "C"

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void fused_sigmoid_gating_recurrent_wrapper(
    pybind11::object A_log, pybind11::object a, pybind11::object dt_bias,
    float softplus_beta, float softplus_threshold,
    pybind11::object q, pybind11::object k, pybind11::object v,
    pybind11::object b, pybind11::object o,
    pybind11::object h0_source, pybind11::object h0_indices,
    pybind11::object cu_seqlens,
    float scale,
    int T, int B, int H, int HV, int K, int V_dim,
    bool use_initial_state, bool use_qk_l2norm) {
    launch_fused_sigmoid_gating_delta_rule_update(
        (const __hip_bfloat16*)get_data_ptr(A_log),
        (const __hip_bfloat16*)get_data_ptr(a),
        (const __hip_bfloat16*)get_data_ptr(dt_bias),
        softplus_beta, softplus_threshold,
        (const __hip_bfloat16*)get_data_ptr(q),
        (const __hip_bfloat16*)get_data_ptr(k),
        (const __hip_bfloat16*)get_data_ptr(v),
        (const __hip_bfloat16*)get_data_ptr(b),
        (__hip_bfloat16*)get_data_ptr(o),
        (float*)get_data_ptr(h0_source),
        (const int*)get_data_ptr(h0_indices),
        (const int*)get_data_ptr(cu_seqlens),
        scale, T, B, H, HV, K, V_dim,
        use_initial_state, use_qk_l2norm, 0);
}

PYBIND11_MODULE(fused_sigmoid_gating_recurrent_tk, m) {
    m.doc() = "Fused sigmoid gating + delta rule recurrent update (decode, bf16)";
    m.def("fused_sigmoid_gating_recurrent", &fused_sigmoid_gating_recurrent_wrapper,
          "Fused sigmoid gating + recurrent delta rule update",
          pybind11::arg("A_log"), pybind11::arg("a"), pybind11::arg("dt_bias"),
          pybind11::arg("softplus_beta"), pybind11::arg("softplus_threshold"),
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("b"), pybind11::arg("o"),
          pybind11::arg("h0_source"), pybind11::arg("h0_indices"),
          pybind11::arg("cu_seqlens"),
          pybind11::arg("scale"),
          pybind11::arg("T"), pybind11::arg("B"), pybind11::arg("H"),
          pybind11::arg("HV"), pybind11::arg("K"), pybind11::arg("V_dim"),
          pybind11::arg("use_initial_state") = true, pybind11::arg("use_qk_l2norm") = false);
}
