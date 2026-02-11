// fused_qk_concat: Fused Q/K concatenation kernel
// Concatenates Q1 with Q2, and K1 with K2 along the head dimension.
//
// Input:
//   q1: (B, H, D1) or (B, H, S, D1)
//   q2: (B, H, D2) or (B, H, S, D2)
//   k1: (B, H_kv, D1) or (B, H_kv, S, D1)
//   k2: (B, H_kv, D2) or (B, H_kv, S, D2)
//
// Output:
//   q_out: (B, H, D1+D2) or (B, H, S, D1+D2)
//   k_out: (B, H_kv, D1+D2) or (B, H_kv, S, D1+D2)
//
// Supports Multi-Query/Group-Query Attention where H = QH_PER_KH * H_kv

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 256;
constexpr int VEC_SIZE = 128;

template<typename T, int D1, int D2>
struct qk_concat_globals {
    using _gl_q1 = gl<T, -1, -1, -1, D1>;
    using _gl_q2 = gl<T, -1, -1, -1, D2>;
    using _gl_k1 = gl<T, -1, -1, -1, D1>;
    using _gl_k2 = gl<T, -1, -1, -1, D2>;
    using _gl_q_out = gl<T, -1, -1, -1, D1 + D2>;
    using _gl_k_out = gl<T, -1, -1, -1, D1 + D2>;

    _gl_q1 q1;
    _gl_q2 q2;
    _gl_k1 k1;
    _gl_k2 k2;
    _gl_q_out q_out;
    _gl_k_out k_out;

    int B;          // Batch size
    int H_q;        // Query heads
    int H_kv;       // Key/Value heads
    int QH_PER_KH;  // Query heads per KV head (for GQA/MQA)
    hipStream_t stream;

    dim3 grid() {
        return dim3(B, H_q);
    }
    dim3 block() { return dim3(BLOCK_SIZE); }
};

template<typename T, int D1, int D2>
__global__ void qk_concat_kernel(
    const qk_concat_globals<T, D1, D2> g
) {
    const int b = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    // Each block handles one (batch, head_q) combination
    // Concatenate q1[b, h_q, :] with q2[b, h_q, :]

    const T* q1_base = (T*)&g.q1[{0, 0, 0, 0}];
    const T* q2_base = (T*)&g.q2[{0, 0, 0, 0}];
    T* q_out_base = (T*)&g.q_out[{0, 0, 0, 0}];

    const int q1_offset = (b * g.H_q + h_q) * D1;
    const int q2_offset = (b * g.H_q + h_q) * D2;
    const int q_out_offset = (b * g.H_q + h_q) * (D1 + D2);

    // Copy q1 to first D1 elements of q_out
    for (int i = tid; i < D1; i += BLOCK_SIZE) {
        q_out_base[q_out_offset + i] = q1_base[q1_offset + i];
    }

    // Copy q2 to next D2 elements of q_out
    for (int i = tid; i < D2; i += BLOCK_SIZE) {
        q_out_base[q_out_offset + D1 + i] = q2_base[q2_offset + i];
    }

    // Handle K concatenation (only for certain heads based on QH_PER_KH)
    if (h_q % g.QH_PER_KH == 0) {
        const int h_kv = h_q / g.QH_PER_KH;

        const T* k1_base = (T*)&g.k1[{0, 0, 0, 0}];
        const T* k2_base = (T*)&g.k2[{0, 0, 0, 0}];
        T* k_out_base = (T*)&g.k_out[{0, 0, 0, 0}];

        const int k1_offset = (b * g.H_kv + h_kv) * D1;
        const int k2_offset = (b * g.H_kv + h_kv) * D2;
        const int k_out_offset = (b * g.H_kv + h_kv) * (D1 + D2);

        // Copy k1 to first D1 elements of k_out
        for (int i = tid; i < D1; i += BLOCK_SIZE) {
            k_out_base[k_out_offset + i] = k1_base[k1_offset + i];
        }

        // Copy k2 to next D2 elements of k_out
        for (int i = tid; i < D2; i += BLOCK_SIZE) {
            k_out_base[k_out_offset + D1 + i] = k2_base[k2_offset + i];
        }
    }
}

template<typename T, int D1, int D2>
void dispatch_qk_concat(qk_concat_globals<T, D1, D2>& g) {
    qk_concat_kernel<T, D1, D2><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// Explicit instantiations for common dimension combinations
template void dispatch_qk_concat<bf16, 64, 64>(qk_concat_globals<bf16, 64, 64>&);
template void dispatch_qk_concat<bf16, 128, 128>(qk_concat_globals<bf16, 128, 128>&);
template void dispatch_qk_concat<bf16, 64, 128>(qk_concat_globals<bf16, 64, 128>&);
template void dispatch_qk_concat<half, 64, 64>(qk_concat_globals<half, 64, 64>&);
template void dispatch_qk_concat<half, 128, 128>(qk_concat_globals<half, 128, 128>&);

PYBIND11_MODULE(fused_qk_concat_tk, m) {
    m.doc() = "HipKittens Fused Q/K Concatenation kernel";

    kittens::py::bind_function<dispatch_qk_concat<bf16, 64, 64>>(m, "dispatch_bf16_64_64",
        &qk_concat_globals<bf16, 64, 64>::q1,
        &qk_concat_globals<bf16, 64, 64>::q2,
        &qk_concat_globals<bf16, 64, 64>::k1,
        &qk_concat_globals<bf16, 64, 64>::k2,
        &qk_concat_globals<bf16, 64, 64>::q_out,
        &qk_concat_globals<bf16, 64, 64>::k_out,
        &qk_concat_globals<bf16, 64, 64>::B,
        &qk_concat_globals<bf16, 64, 64>::H_q,
        &qk_concat_globals<bf16, 64, 64>::H_kv,
        &qk_concat_globals<bf16, 64, 64>::QH_PER_KH);

    kittens::py::bind_function<dispatch_qk_concat<bf16, 128, 128>>(m, "dispatch_bf16_128_128",
        &qk_concat_globals<bf16, 128, 128>::q1,
        &qk_concat_globals<bf16, 128, 128>::q2,
        &qk_concat_globals<bf16, 128, 128>::k1,
        &qk_concat_globals<bf16, 128, 128>::k2,
        &qk_concat_globals<bf16, 128, 128>::q_out,
        &qk_concat_globals<bf16, 128, 128>::k_out,
        &qk_concat_globals<bf16, 128, 128>::B,
        &qk_concat_globals<bf16, 128, 128>::H_q,
        &qk_concat_globals<bf16, 128, 128>::H_kv,
        &qk_concat_globals<bf16, 128, 128>::QH_PER_KH);

    kittens::py::bind_function<dispatch_qk_concat<bf16, 64, 128>>(m, "dispatch_bf16_64_128",
        &qk_concat_globals<bf16, 64, 128>::q1,
        &qk_concat_globals<bf16, 64, 128>::q2,
        &qk_concat_globals<bf16, 64, 128>::k1,
        &qk_concat_globals<bf16, 64, 128>::k2,
        &qk_concat_globals<bf16, 64, 128>::q_out,
        &qk_concat_globals<bf16, 64, 128>::k_out,
        &qk_concat_globals<bf16, 64, 128>::B,
        &qk_concat_globals<bf16, 64, 128>::H_q,
        &qk_concat_globals<bf16, 64, 128>::H_kv,
        &qk_concat_globals<bf16, 64, 128>::QH_PER_KH);

    kittens::py::bind_function<dispatch_qk_concat<half, 64, 64>>(m, "dispatch_fp16_64_64",
        &qk_concat_globals<half, 64, 64>::q1,
        &qk_concat_globals<half, 64, 64>::q2,
        &qk_concat_globals<half, 64, 64>::k1,
        &qk_concat_globals<half, 64, 64>::k2,
        &qk_concat_globals<half, 64, 64>::q_out,
        &qk_concat_globals<half, 64, 64>::k_out,
        &qk_concat_globals<half, 64, 64>::B,
        &qk_concat_globals<half, 64, 64>::H_q,
        &qk_concat_globals<half, 64, 64>::H_kv,
        &qk_concat_globals<half, 64, 64>::QH_PER_KH);

    kittens::py::bind_function<dispatch_qk_concat<half, 128, 128>>(m, "dispatch_fp16_128_128",
        &qk_concat_globals<half, 128, 128>::q1,
        &qk_concat_globals<half, 128, 128>::q2,
        &qk_concat_globals<half, 128, 128>::k1,
        &qk_concat_globals<half, 128, 128>::k2,
        &qk_concat_globals<half, 128, 128>::q_out,
        &qk_concat_globals<half, 128, 128>::k_out,
        &qk_concat_globals<half, 128, 128>::B,
        &qk_concat_globals<half, 128, 128>::H_q,
        &qk_concat_globals<half, 128, 128>::H_kv,
        &qk_concat_globals<half, 128, 128>::QH_PER_KH);
}
