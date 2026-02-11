// Fused Recurrent Gated Delta Rule Forward Kernel
// Ported from reference/triton/decode/fused_recurrent.py
//
// Implements the recurrent state update:
//   For each timestep t:
//     if USE_G:  h *= exp(g)
//     if USE_GK: h *= exp(gk[:, None])
//     if USE_GV: h *= exp(gv[None, :])
//     v' = beta * (v - sum(h * k[:, None], axis=0))
//     h += k[:, None] * v'[None, :]
//     o = sum(h * q[:, None], axis=0)
//
// Shapes:
//   q, k: (B*T, H, K) -- query and key
//   v:    (B*T, HV, V) -- value
//   o:    (B*T, HV, V) -- output
//   g:    (B*T, HV) -- global gate (optional)
//   gk:   (B*T, HV, K) -- key gate (optional)
//   gv:   (B*T, HV, V) -- value gate (optional)
//   beta: (B*T, HV) or (B*T, HV, V) -- beta param
//   h0:   (B*HV, K, V) -- initial hidden state (optional)
//   ht:   (B*HV, K, V) -- final hidden state (optional)
//
// Grid: (ceil(V/BV), N*HV) where N = B (fixed len) or len(cu_seqlens)-1
// Each thread handles one (k_idx, v_idx) element of the hidden state h[BK, BV].

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

// Block sizes -- matching Triton BK/BV (must be powers of 2, >= K and V)
// Template parameters for flexibility
template<int BK, int BV, bool USE_G, bool USE_GK, bool USE_GV,
         bool IS_BETA_HEADWISE, bool USE_INITIAL_STATE, bool STORE_FINAL_STATE>
__global__ void fused_recurrent_gated_delta_rule_fwd_kernel(
    const __hip_bfloat16* __restrict__ q,      // (total_tokens, H, K)
    const __hip_bfloat16* __restrict__ k,      // (total_tokens, H, K)
    const __hip_bfloat16* __restrict__ v,      // (total_tokens, HV, V)
    const float* __restrict__ g,               // (total_tokens, HV) or nullptr
    const float* __restrict__ gk,              // (total_tokens, HV, K) or nullptr
    const float* __restrict__ gv,              // (total_tokens, HV, V) or nullptr
    const __hip_bfloat16* __restrict__ beta,   // (total_tokens, HV) or (total_tokens, HV, V)
    __hip_bfloat16* __restrict__ o,            // (total_tokens, HV, V)
    const float* __restrict__ h0,              // (N*HV, K, V) or nullptr
    float* __restrict__ ht,                    // (N*HV, K, V) or nullptr
    const int* __restrict__ cu_seqlens,        // (N+1,) or nullptr
    float scale,
    int T,       // sequence length (per-batch, fixed for non-varlen)
    int B,       // batch size
    int H,       // number of q/k heads
    int HV,      // number of v heads
    int K,       // key dimension
    int V_dim    // value dimension
) {
    // Grid mapping: (ceil(V/BV), N*HV)
    int i_v = blockIdx.x;    // value block index
    int i_nh = blockIdx.y;   // combined (batch, head_v) index
    int i_n = i_nh / HV;     // batch index
    int i_hv = i_nh % HV;    // value head index
    int i_h = i_hv / (HV / H); // q/k head index (GQA mapping)

    int tid = threadIdx.x;   // thread within block

    // Determine sequence bounds
    int bos, eos;
    int seq_len = T;
    if (cu_seqlens != nullptr) {
        bos = cu_seqlens[i_n];
        eos = cu_seqlens[i_n + 1];
        seq_len = eos - bos;
    } else {
        bos = i_n * T;
        eos = bos + T;
    }

    // Each thread handles one (k_idx, v_idx) pair
    // We tile K dimension across threads in a warp, V dimension across blocks
    int k_idx = tid % BK;
    int v_idx = i_v * BV + tid / BK;

    if (k_idx >= K || v_idx >= V_dim) return;

    // Initialize hidden state element h[k_idx, v_idx]
    float b_h = 0.0f;
    if (USE_INITIAL_STATE) {
        int h0_offset = i_nh * K * V_dim + k_idx * V_dim + v_idx;
        b_h = h0[h0_offset];
    }

    // Stride calculations
    // q, k: layout (total_tokens, H, K) => stride: (H*K, K, 1)
    // v, o: layout (total_tokens, HV, V) => stride: (HV*V, V, 1)
    int q_base = (bos * H + i_h) * K + k_idx;
    int k_base = (bos * H + i_h) * K + k_idx;
    int v_base = (bos * HV + i_hv) * V_dim + v_idx;
    int o_base = (bos * HV + i_hv) * V_dim + v_idx;

    int q_stride = H * K;
    int k_stride = H * K;
    int v_stride = HV * V_dim;
    int o_stride = HV * V_dim;

    // Gate pointers
    int g_base = bos * HV + i_hv;
    int gk_base = (bos * HV + i_hv) * K + k_idx;
    int gv_base = (bos * HV + i_hv) * V_dim + v_idx;
    int g_stride_t = HV;
    int gk_stride_t = HV * K;
    int gv_stride_t = HV * V_dim;

    // Beta pointer
    int beta_base, beta_stride_t;
    if (IS_BETA_HEADWISE) {
        beta_base = bos * HV + i_hv;
        beta_stride_t = HV;
    } else {
        beta_base = (bos * HV + i_hv) * V_dim + v_idx;
        beta_stride_t = HV * V_dim;
    }

    // Recurrent loop over timesteps
    for (int t = 0; t < seq_len; t++) {
        float b_q = __bfloat162float(q[q_base + t * q_stride]) * scale;
        float b_k = __bfloat162float(k[k_base + t * k_stride]);
        float b_v = __bfloat162float(v[v_base + t * v_stride]);

        float b_beta;
        if (IS_BETA_HEADWISE) {
            b_beta = __bfloat162float(beta[beta_base + t * beta_stride_t]);
        } else {
            b_beta = __bfloat162float(beta[beta_base + t * beta_stride_t]);
        }

        // Apply gates to hidden state
        if (USE_G) {
            float b_g = g[g_base + t * g_stride_t];
            b_h *= expf(b_g);
        }
        if (USE_GK) {
            float b_gk = gk[gk_base + t * gk_stride_t];
            b_h *= expf(b_gk);
        }
        if (USE_GV) {
            float b_gv = gv[gv_base + t * gv_stride_t];
            b_h *= expf(b_gv);
        }

        // Delta rule update:
        // v' = beta * (v - sum(h * k, axis=k_dim))
        // We need cross-thread reduction for sum(h * k[:, None], axis=0) at fixed v_idx
        // Each thread has h[k_idx, v_idx] and k[k_idx]
        // Need: sum over k_idx of h[k_idx, v_idx] * k[k_idx]

        float hk_product = b_h * b_k;

        // Warp-level reduction over k_idx dimension (threads with same v_idx)
        // Threads are laid out: tid = k_idx + v_local * BK
        // So threads tid, tid+1, ..., tid+BK-1 share the same v_idx? No:
        // k_idx = tid % BK, v_local = tid / BK
        // Threads with same v_local (same v_idx) are at tid = v_local*BK + 0..BK-1
        // These are consecutive threads, we can use warp shuffle

        // Reduction: sum hk_product across k_idx for fixed v_idx
        // Since k_idx = tid % BK, threads sharing v_idx are strided by 1
        // within a group of BK consecutive threads
        float hk_sum = hk_product;
        for (int offset = BK / 2; offset > 0; offset >>= 1) {
            // Only reduce within the BK-sized group
            int lane_in_group = tid % BK;
            float other = __shfl_xor(hk_sum, offset);
            if (lane_in_group < BK) {
                hk_sum += other;
            }
        }
        // Now hk_sum has the sum across k_idx dimension for this v_idx

        float b_v_prime = b_beta * (b_v - hk_sum);

        // Update hidden state: h[k_idx, v_idx] += k[k_idx] * v'[v_idx]
        b_h += b_k * b_v_prime;

        // Compute output: o[v_idx] = sum(h[k_idx, v_idx] * q[k_idx], axis=k_dim)
        float hq_product = b_h * b_q;

        // Warp-level reduction over k_idx
        float hq_sum = hq_product;
        for (int offset = BK / 2; offset > 0; offset >>= 1) {
            float other = __shfl_xor(hq_sum, offset);
            hq_sum += other;
        }

        // Only one thread per v_idx writes output (k_idx == 0)
        if (k_idx == 0) {
            o[o_base + t * o_stride] = __float2bfloat16(hq_sum);
        }
    }

    // Store final state
    if (STORE_FINAL_STATE) {
        if (k_idx < K && v_idx < V_dim) {
            int ht_offset = i_nh * K * V_dim + k_idx * V_dim + v_idx;
            ht[ht_offset] = b_h;
        }
    }
}

// Launch wrapper
extern "C" {

void launch_fused_recurrent_gated_delta_rule_fwd(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const float* g, const float* gk, const float* gv,
    const __hip_bfloat16* beta,
    __hip_bfloat16* o,
    const float* h0, float* ht,
    const int* cu_seqlens,
    float scale,
    int T, int B, int H, int HV, int K, int V_dim,
    bool use_g, bool use_gk, bool use_gv,
    bool is_beta_headwise,
    bool use_initial_state, bool store_final_state,
    hipStream_t stream
) {
    constexpr int BK = 128;
    constexpr int BV = 128;

    // threads_per_block = BK * (BV / elements_per_vblock)
    // Each thread handles one (k, v) element
    // But BK * (BV per block) might be too many threads
    // Limit: BK threads handle k-reduction, grid handles v-dimension
    int threads = BK;  // Each block has BK threads, each thread handles one k-element
                        // Grid.x tiles over V dimension, each block processes BK k-elements for one v-element

    // Actually: total hidden state per (head) is K*V elements
    // We launch K threads per block, grid over V
    int N = (cu_seqlens != nullptr) ? B : B;  // N = number of sequences
    // For varlen, N should be parsed from cu_seqlens, but we pass B for non-varlen
    dim3 grid((V_dim + 0) / 1, N * HV);  // one block per v-element per (batch, head)
    // Revisit: each block handles BK k-elements for ONE v-element
    // So grid.x = V_dim (one block per v-element)
    // threads = min(K, BK) (each thread handles one k-element)

    int actual_threads = K;  // K threads per block
    if (actual_threads < 64) actual_threads = 64; // min warp size on AMD

    grid = dim3(V_dim, N * HV);

    // Dispatch based on flags
    // Most common decode path: USE_G=true, no GK/GV, IS_BETA_HEADWISE=true
    #define LAUNCH_KERNEL(UG, UGK, UGV, IBH, UIS, SFS) \
        fused_recurrent_gated_delta_rule_fwd_kernel<BK, BV, UG, UGK, UGV, IBH, UIS, SFS> \
            <<<grid, actual_threads, 0, stream>>>( \
                q, k, v, g, gk, gv, beta, o, h0, ht, cu_seqlens, \
                scale, T, B, H, HV, K, V_dim)

    // Template dispatch (common configurations)
    if (use_g && !use_gk && !use_gv && is_beta_headwise && use_initial_state && store_final_state) {
        LAUNCH_KERNEL(true, false, false, true, true, true);
    } else if (use_g && !use_gk && !use_gv && is_beta_headwise && use_initial_state && !store_final_state) {
        LAUNCH_KERNEL(true, false, false, true, true, false);
    } else if (use_g && !use_gk && !use_gv && is_beta_headwise && !use_initial_state && !store_final_state) {
        LAUNCH_KERNEL(true, false, false, true, false, false);
    } else if (!use_g && !use_gk && !use_gv && is_beta_headwise && use_initial_state && store_final_state) {
        LAUNCH_KERNEL(false, false, false, true, true, true);
    } else if (!use_g && !use_gk && !use_gv && !is_beta_headwise && use_initial_state && store_final_state) {
        LAUNCH_KERNEL(false, false, false, false, true, true);
    } else if (use_g && !use_gk && !use_gv && !is_beta_headwise && use_initial_state && store_final_state) {
        LAUNCH_KERNEL(true, false, false, false, true, true);
    } else {
        // Fallback: most general case
        LAUNCH_KERNEL(true, true, true, false, true, true);
    }

    #undef LAUNCH_KERNEL
}

} // extern "C"

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void fused_recurrent_wrapper(
    pybind11::object q, pybind11::object k, pybind11::object v,
    pybind11::object g, pybind11::object gk, pybind11::object gv,
    pybind11::object beta, pybind11::object o,
    pybind11::object h0, pybind11::object ht,
    pybind11::object cu_seqlens,
    float scale,
    int T, int B, int H, int HV, int K, int V_dim,
    bool use_g, bool use_gk, bool use_gv,
    bool is_beta_headwise,
    bool use_initial_state, bool store_final_state) {
    launch_fused_recurrent_gated_delta_rule_fwd(
        (const __hip_bfloat16*)get_data_ptr(q),
        (const __hip_bfloat16*)get_data_ptr(k),
        (const __hip_bfloat16*)get_data_ptr(v),
        use_g ? (const float*)get_data_ptr(g) : nullptr,
        use_gk ? (const float*)get_data_ptr(gk) : nullptr,
        use_gv ? (const float*)get_data_ptr(gv) : nullptr,
        (const __hip_bfloat16*)get_data_ptr(beta),
        (__hip_bfloat16*)get_data_ptr(o),
        use_initial_state ? (const float*)get_data_ptr(h0) : nullptr,
        store_final_state ? (float*)get_data_ptr(ht) : nullptr,
        (const int*)get_data_ptr(cu_seqlens),
        scale, T, B, H, HV, K, V_dim,
        use_g, use_gk, use_gv,
        is_beta_headwise,
        use_initial_state, store_final_state, 0);
}

PYBIND11_MODULE(fused_recurrent_tk, m) {
    m.doc() = "Fused recurrent gated delta rule forward (decode, bf16)";
    m.def("fused_recurrent_fwd", &fused_recurrent_wrapper,
          "Fused recurrent gated delta rule forward pass",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("g"), pybind11::arg("gk"), pybind11::arg("gv"),
          pybind11::arg("beta"), pybind11::arg("o"),
          pybind11::arg("h0"), pybind11::arg("ht"),
          pybind11::arg("cu_seqlens"),
          pybind11::arg("scale"),
          pybind11::arg("T"), pybind11::arg("B"), pybind11::arg("H"),
          pybind11::arg("HV"), pybind11::arg("K"), pybind11::arg("V_dim"),
          pybind11::arg("use_g") = true, pybind11::arg("use_gk") = false,
          pybind11::arg("use_gv") = false, pybind11::arg("is_beta_headwise") = true,
          pybind11::arg("use_initial_state") = true, pybind11::arg("store_final_state") = true);
}
