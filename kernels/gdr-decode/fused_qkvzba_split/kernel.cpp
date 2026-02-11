// Fused QKVZBA Split, Reshape and Concatenation Kernel (Decode Stage)
// Ported from reference/triton/fused_qkvzba_split.py
//
// Reshuffles mixed_qkvz and mixed_ba tensors into separate q, k, v, z, b, a
// tensors with the correct layout for the gated delta rule decode.
//
// The input mixed_qkvz tensor has layout:
//   (batch, NUM_HEADS_QK, QKVZ_DIM_T) where
//   QKVZ_DIM_T = HEAD_QK*2 + (NUM_HEADS_V/NUM_HEADS_QK)*HEAD_V*2
//   Within each head group: [q(HEAD_QK), k(HEAD_QK), v(V_PER_QK*HEAD_V), z(V_PER_QK*HEAD_V)]
//
// The input mixed_ba tensor has layout:
//   (batch, NUM_HEADS_QK, BA_DIM_T) where BA_DIM_T = (NUM_HEADS_V/NUM_HEADS_QK)*2
//   Within each head group: [b(V_PER_QK), a(V_PER_QK)]
//
// Output layout:
//   mixed_qkv: (batch, QKV_DIM_T) where QKV_DIM_T = NUM_HEADS_QK*HEAD_QK*2 + NUM_HEADS_V*HEAD_V
//     Layout: [all_q(NUM_HEADS_QK*HEAD_QK), all_k(NUM_HEADS_QK*HEAD_QK), all_v(NUM_HEADS_V*HEAD_V)]
//   z: (batch, NUM_HEADS_V, HEAD_V)
//   b: (batch, NUM_HEADS_V)
//   a: (batch, NUM_HEADS_V)
//
// Grid: (batch, NUM_HEADS_QK) -- one thread-block per (batch, qk_head) pair

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

template<int NUM_HEADS_QK, int NUM_HEADS_V, int HEAD_QK, int HEAD_V>
__global__ void fused_qkvzba_split_reshape_cat_decode_kernel(
    __hip_bfloat16* __restrict__ mixed_qkv,   // output: (batch, QKV_DIM_T)
    __hip_bfloat16* __restrict__ z,            // output: (batch, NUM_HEADS_V*HEAD_V)
    __hip_bfloat16* __restrict__ b_out,        // output: (batch, NUM_HEADS_V)
    __hip_bfloat16* __restrict__ a_out,        // output: (batch, NUM_HEADS_V)
    const __hip_bfloat16* __restrict__ mixed_qkvz,  // input: (batch, NUM_HEADS_QK*QKVZ_DIM_T)
    const __hip_bfloat16* __restrict__ mixed_ba,     // input: (batch, NUM_HEADS_QK*BA_DIM_T)
    int batch_size
) {
    constexpr int V_PER_QK = NUM_HEADS_V / NUM_HEADS_QK;
    constexpr int QKVZ_DIM_T = HEAD_QK * 2 + V_PER_QK * HEAD_V * 2;
    constexpr int BA_DIM_T = V_PER_QK * 2;
    constexpr int QKV_DIM_T = HEAD_QK * 2 * NUM_HEADS_QK + NUM_HEADS_V * HEAD_V;

    int i_bs = blockIdx.x;   // batch index
    int i_qk = blockIdx.y;   // qk head index
    int tid = threadIdx.x;

    if (i_bs >= batch_size) return;

    // Source pointers for this (batch, qk_head)
    const __hip_bfloat16* src_qkvz = mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T;
    const __hip_bfloat16* src_ba = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T;

    // Destination base pointers
    __hip_bfloat16* dst_qkv = mixed_qkv + i_bs * QKV_DIM_T;
    __hip_bfloat16* dst_z = z + i_bs * NUM_HEADS_V * HEAD_V;

    // Copy q: src[0..HEAD_QK) -> dst[i_qk*HEAD_QK..(i_qk+1)*HEAD_QK)
    for (int i = tid; i < HEAD_QK; i += blockDim.x) {
        dst_qkv[i_qk * HEAD_QK + i] = src_qkvz[i];
    }

    // Copy k: src[HEAD_QK..2*HEAD_QK) -> dst[NUM_HEADS_QK*HEAD_QK + i_qk*HEAD_QK..]
    for (int i = tid; i < HEAD_QK; i += blockDim.x) {
        dst_qkv[NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK + i] = src_qkvz[HEAD_QK + i];
    }

    // Copy v: src[2*HEAD_QK..2*HEAD_QK+V_PER_QK*HEAD_V)
    //       -> dst[2*NUM_HEADS_QK*HEAD_QK + i_qk*V_PER_QK*HEAD_V..]
    for (int i = tid; i < V_PER_QK * HEAD_V; i += blockDim.x) {
        dst_qkv[2 * NUM_HEADS_QK * HEAD_QK + i_qk * V_PER_QK * HEAD_V + i] =
            src_qkvz[2 * HEAD_QK + i];
    }

    // Copy z: src[2*HEAD_QK+V_PER_QK*HEAD_V..2*HEAD_QK+2*V_PER_QK*HEAD_V)
    //       -> z[i_qk*V_PER_QK*HEAD_V..]
    for (int i = tid; i < V_PER_QK * HEAD_V; i += blockDim.x) {
        dst_z[i_qk * V_PER_QK * HEAD_V + i] =
            src_qkvz[2 * HEAD_QK + V_PER_QK * HEAD_V + i];
    }

    // Copy b and a from mixed_ba
    // b: src_ba[0..V_PER_QK) -> b_out[i_bs*NUM_HEADS_V + i_qk*V_PER_QK..]
    // a: src_ba[V_PER_QK..2*V_PER_QK) -> a_out[i_bs*NUM_HEADS_V + i_qk*V_PER_QK..]
    if (tid < V_PER_QK) {
        b_out[i_bs * NUM_HEADS_V + i_qk * V_PER_QK + tid] = src_ba[tid];
        a_out[i_bs * NUM_HEADS_V + i_qk * V_PER_QK + tid] = src_ba[V_PER_QK + tid];
    }
}

// Non-templated version that dispatches based on runtime parameters
__global__ void fused_qkvzba_split_reshape_cat_decode_generic(
    __hip_bfloat16* __restrict__ mixed_qkv,
    __hip_bfloat16* __restrict__ z,
    __hip_bfloat16* __restrict__ b_out,
    __hip_bfloat16* __restrict__ a_out,
    const __hip_bfloat16* __restrict__ mixed_qkvz,
    const __hip_bfloat16* __restrict__ mixed_ba,
    int batch_size,
    int num_heads_qk,
    int num_heads_v,
    int head_qk,
    int head_v
) {
    int v_per_qk = num_heads_v / num_heads_qk;
    int qkvz_dim_t = head_qk * 2 + v_per_qk * head_v * 2;
    int ba_dim_t = v_per_qk * 2;
    int qkv_dim_t = head_qk * 2 * num_heads_qk + num_heads_v * head_v;

    int i_bs = blockIdx.x;
    int i_qk = blockIdx.y;
    int tid = threadIdx.x;

    if (i_bs >= batch_size) return;

    const __hip_bfloat16* src_qkvz = mixed_qkvz + i_bs * num_heads_qk * qkvz_dim_t + i_qk * qkvz_dim_t;
    const __hip_bfloat16* src_ba = mixed_ba + i_bs * num_heads_qk * ba_dim_t + i_qk * ba_dim_t;

    __hip_bfloat16* dst_qkv = mixed_qkv + i_bs * qkv_dim_t;
    __hip_bfloat16* dst_z = z + i_bs * num_heads_v * head_v;

    // Copy q
    for (int i = tid; i < head_qk; i += blockDim.x) {
        dst_qkv[i_qk * head_qk + i] = src_qkvz[i];
    }

    // Copy k
    for (int i = tid; i < head_qk; i += blockDim.x) {
        dst_qkv[num_heads_qk * head_qk + i_qk * head_qk + i] = src_qkvz[head_qk + i];
    }

    // Copy v
    for (int i = tid; i < v_per_qk * head_v; i += blockDim.x) {
        dst_qkv[2 * num_heads_qk * head_qk + i_qk * v_per_qk * head_v + i] =
            src_qkvz[2 * head_qk + i];
    }

    // Copy z
    for (int i = tid; i < v_per_qk * head_v; i += blockDim.x) {
        dst_z[i_qk * v_per_qk * head_v + i] =
            src_qkvz[2 * head_qk + v_per_qk * head_v + i];
    }

    // Copy b and a
    for (int i = tid; i < v_per_qk; i += blockDim.x) {
        b_out[i_bs * num_heads_v + i_qk * v_per_qk + i] = src_ba[i];
        a_out[i_bs * num_heads_v + i_qk * v_per_qk + i] = src_ba[v_per_qk + i];
    }
}

extern "C" {

void launch_fused_qkvzba_split_reshape_cat_decode(
    __hip_bfloat16* mixed_qkv,
    __hip_bfloat16* z,
    __hip_bfloat16* b_out,
    __hip_bfloat16* a_out,
    const __hip_bfloat16* mixed_qkvz,
    const __hip_bfloat16* mixed_ba,
    int batch_size,
    int num_heads_qk,
    int num_heads_v,
    int head_qk,
    int head_v,
    hipStream_t stream
) {
    dim3 grid(batch_size, num_heads_qk);
    int threads = 64;  // Small blocks since data per thread-block is small

    // Dispatch specialized templates for common configurations
    if (num_heads_qk == 1 && num_heads_v == 4 && head_qk == 256 && head_v == 512) {
        fused_qkvzba_split_reshape_cat_decode_kernel<1, 4, 256, 512>
            <<<grid, threads, 0, stream>>>(
                mixed_qkv, z, b_out, a_out, mixed_qkvz, mixed_ba, batch_size);
    } else if (num_heads_qk == 2 && num_heads_v == 8 && head_qk == 256 && head_v == 512) {
        fused_qkvzba_split_reshape_cat_decode_kernel<2, 8, 256, 512>
            <<<grid, threads, 0, stream>>>(
                mixed_qkv, z, b_out, a_out, mixed_qkvz, mixed_ba, batch_size);
    } else if (num_heads_qk == 4 && num_heads_v == 16 && head_qk == 128 && head_v == 128) {
        fused_qkvzba_split_reshape_cat_decode_kernel<4, 16, 128, 128>
            <<<grid, threads, 0, stream>>>(
                mixed_qkv, z, b_out, a_out, mixed_qkvz, mixed_ba, batch_size);
    } else if (num_heads_qk == 8 && num_heads_v == 32 && head_qk == 128 && head_v == 128) {
        fused_qkvzba_split_reshape_cat_decode_kernel<8, 32, 128, 128>
            <<<grid, threads, 0, stream>>>(
                mixed_qkv, z, b_out, a_out, mixed_qkvz, mixed_ba, batch_size);
    } else {
        // Generic fallback
        fused_qkvzba_split_reshape_cat_decode_generic
            <<<grid, threads, 0, stream>>>(
                mixed_qkv, z, b_out, a_out, mixed_qkvz, mixed_ba,
                batch_size, num_heads_qk, num_heads_v, head_qk, head_v);
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

void fused_qkvzba_split_wrapper(
    pybind11::object mixed_qkv, pybind11::object z,
    pybind11::object b_out, pybind11::object a_out,
    pybind11::object mixed_qkvz, pybind11::object mixed_ba,
    int batch_size, int num_heads_qk, int num_heads_v,
    int head_qk, int head_v) {
    launch_fused_qkvzba_split_reshape_cat_decode(
        (__hip_bfloat16*)get_data_ptr(mixed_qkv),
        (__hip_bfloat16*)get_data_ptr(z),
        (__hip_bfloat16*)get_data_ptr(b_out),
        (__hip_bfloat16*)get_data_ptr(a_out),
        (const __hip_bfloat16*)get_data_ptr(mixed_qkvz),
        (const __hip_bfloat16*)get_data_ptr(mixed_ba),
        batch_size, num_heads_qk, num_heads_v, head_qk, head_v, 0);
}

PYBIND11_MODULE(fused_qkvzba_split_tk, m) {
    m.doc() = "Fused QKVZBA split, reshape and concatenation (decode, bf16)";
    m.def("fused_qkvzba_split", &fused_qkvzba_split_wrapper,
          "Reshuffle mixed_qkvz and mixed_ba into separate q, k, v, z, b, a",
          pybind11::arg("mixed_qkv"), pybind11::arg("z"),
          pybind11::arg("b_out"), pybind11::arg("a_out"),
          pybind11::arg("mixed_qkvz"), pybind11::arg("mixed_ba"),
          pybind11::arg("batch_size"), pybind11::arg("num_heads_qk"),
          pybind11::arg("num_heads_v"), pybind11::arg("head_qk"), pybind11::arg("head_v"));
}
