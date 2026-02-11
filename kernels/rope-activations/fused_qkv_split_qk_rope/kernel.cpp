// Fused QKV Split + QK RoPE HipKittens Kernel
// Ported from reference/triton/rope/fused_qkv_split_qk_rope.py
//
// Takes a fused QKV tensor and in one kernel:
//   1. Splits into separate Q, K, V tensors
//   2. Applies NeoX-style RoPE to Q and K
//   3. Passes V through unchanged
//
// This avoids multiple memory passes over the QKV tensor.
//
// Input layout:
//   qkv: (T, (QH + 2*KVH) * D) bf16 — fused packed QKV, T = total tokens
//   cos: (T, D/2) bf16 — cosine frequencies
//   sin: (T, D/2) bf16 — sine frequencies
//
// Output layout:
//   q: (T, QH, D) bf16
//   k: (T, KVH, D) bf16
//   v: (T, KVH, D) bf16
//
// Uses HipKittens register tiles for efficient bf16 RoPE computation.

#include "kittens.cuh"

#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

using namespace kittens;

constexpr int BLOCK_T = 32;  // tokens per block

using shape = kittens::rt_32x32_8_s;

template<int HEAD_DIM, int QH, int KVH>
struct fused_qkv_rope_globals {
    static constexpr int D = HEAD_DIM;
    static constexpr int HALF_DIM = HEAD_DIM / 2;
    static constexpr int TOTAL_HEADS = QH + 2 * KVH;

    using qkv_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using q_gl = gl<bf16, -1, -1, -1, -1>;
    using k_gl = gl<bf16, -1, -1, -1, -1>;
    using v_gl = gl<bf16, -1, -1, -1, -1>;

    qkv_gl qkv;
    cos_gl cos_freq;
    sin_gl sin_freq;
    q_gl q;
    k_gl k;
    v_gl v;

    int T;  // total number of tokens

    hipStream_t stream;

    dim3 grid() {
        // grid: (token_blocks, max(QH, KVH))
        return dim3((T + BLOCK_T - 1) / BLOCK_T, QH);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// The kernel processes one head per thread block (y dim = head index)
// and a chunk of tokens per block (x dim = token chunk)
template<int HEAD_DIM, int QH, int KVH>
__global__ void fused_qkv_split_qk_rope_kernel(
    const fused_qkv_rope_globals<HEAD_DIM, QH, KVH> g
) {
    constexpr int D = HEAD_DIM;
    constexpr int HALF_DIM = D / 2;
    constexpr int half_dim_tiles = HALF_DIM / BLOCK_T;

    using tile_t = rt<bf16, BLOCK_T, BLOCK_T, row_l, shape>;
    constexpr int PT = tile_t::packed_per_thread;

    int pid_t = blockIdx.x;  // token block index
    int hq = blockIdx.y;     // Q head index

    int t_start = pid_t * BLOCK_T;
    if (t_start >= g.T) return;

    // Load cos/sin for this token block
    rt<bf16, BLOCK_T, HALF_DIM, row_l, shape> cos_reg, sin_reg;
    load<2>(cos_reg, g.cos_freq, {0, 0, pid_t, 0});
    load<2>(sin_reg, g.sin_freq, {0, 0, pid_t, 0});

    // Process Q head: load from QKV, apply RoPE, store to Q
    {
        rt<bf16, BLOCK_T, D, row_l, shape> q_reg;
        // In the fused QKV, Q heads come first: offset = hq * D
        load<2>(q_reg, g.qkv, {0, 0, pid_t, hq * D / BLOCK_T});

        asm volatile("s_waitcnt lgkmcnt(0)");

        // Apply NeoX RoPE to Q
        #pragma unroll
        for (int i = 0; i < half_dim_tiles; ++i) {
            #pragma unroll
            for (int j = 0; j < PT; ++j) {
                const auto x1 = q_reg.tiles[0][i].data[j];
                const auto x2 = q_reg.tiles[0][i + half_dim_tiles].data[j];
                q_reg.tiles[0][i].data[j] = __hsub2(
                    __hmul2(x1, cos_reg.tiles[0][i].data[j]),
                    __hmul2(x2, sin_reg.tiles[0][i].data[j])
                );
                q_reg.tiles[0][i + half_dim_tiles].data[j] = __hadd2(
                    __hmul2(x2, cos_reg.tiles[0][i].data[j]),
                    __hmul2(x1, sin_reg.tiles[0][i].data[j])
                );
            }
        }

        // Store Q output: (T, QH, D) -> index by (0, 0, pid_t*BLOCK_T, hq*D)
        store(g.q, q_reg, {0, hq, pid_t, 0});
    }

    // Process K and V (only if hq < KVH)
    if (hq < KVH) {
        // K offset in fused QKV: QH * D + hq * D
        constexpr int Q_SIZE = QH * D;
        {
            rt<bf16, BLOCK_T, D, row_l, shape> k_reg;
            load<2>(k_reg, g.qkv, {0, 0, pid_t, (Q_SIZE + hq * D) / BLOCK_T});

            asm volatile("s_waitcnt lgkmcnt(0)");

            // Apply NeoX RoPE to K
            #pragma unroll
            for (int i = 0; i < half_dim_tiles; ++i) {
                #pragma unroll
                for (int j = 0; j < PT; ++j) {
                    const auto x1 = k_reg.tiles[0][i].data[j];
                    const auto x2 = k_reg.tiles[0][i + half_dim_tiles].data[j];
                    k_reg.tiles[0][i].data[j] = __hsub2(
                        __hmul2(x1, cos_reg.tiles[0][i].data[j]),
                        __hmul2(x2, sin_reg.tiles[0][i].data[j])
                    );
                    k_reg.tiles[0][i + half_dim_tiles].data[j] = __hadd2(
                        __hmul2(x2, cos_reg.tiles[0][i].data[j]),
                        __hmul2(x1, sin_reg.tiles[0][i].data[j])
                    );
                }
            }

            store(g.k, k_reg, {0, hq, pid_t, 0});
        }

        // V: passthrough (no RoPE), offset = QH*D + KVH*D + hq*D
        {
            constexpr int KV_SIZE = KVH * D;
            rt<bf16, BLOCK_T, D, row_l, shape> v_reg;
            load<2>(v_reg, g.qkv, {0, 0, pid_t, (Q_SIZE + KV_SIZE + hq * D) / BLOCK_T});

            asm volatile("s_waitcnt lgkmcnt(0)");

            store(g.v, v_reg, {0, hq, pid_t, 0});
        }
    }
}

template<int HEAD_DIM, int QH, int KVH>
void dispatch_fused_qkv_rope(fused_qkv_rope_globals<HEAD_DIM, QH, KVH>& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fused_qkv_split_qk_rope_kernel<HEAD_DIM, QH, KVH>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fused_qkv_split_qk_rope_kernel<HEAD_DIM, QH, KVH>
        <<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Common configurations
template void dispatch_fused_qkv_rope<128, 32, 8>(fused_qkv_rope_globals<128, 32, 8>&);
template void dispatch_fused_qkv_rope<128, 64, 8>(fused_qkv_rope_globals<128, 64, 8>&);
template void dispatch_fused_qkv_rope<128, 32, 32>(fused_qkv_rope_globals<128, 32, 32>&);

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

// qkv: (T, (QH+2*KVH)*D), cos: (T, D/2), sin: (T, D/2)
// q: (T, QH, D), k: (T, KVH, D), v: (T, KVH, D)
template<int HEAD_DIM, int QH, int KVH>
void fused_qkv_rope_impl(pybind11::object qkv, pybind11::object cos_freq, pybind11::object sin_freq,
                          pybind11::object q, pybind11::object k, pybind11::object v) {
    int T = _get_dim(qkv, 0);
    constexpr int D = HEAD_DIM;
    constexpr int TOTAL_D = (QH + 2*KVH) * D;

    using globals = fused_qkv_rope_globals<HEAD_DIM, QH, KVH>;
    globals g{
        make_gl<typename globals::qkv_gl>(_get_ptr(qkv), 1, 1, T, TOTAL_D),
        make_gl<typename globals::cos_gl>(_get_ptr(cos_freq), 1, 1, T, D/2),
        make_gl<typename globals::sin_gl>(_get_ptr(sin_freq), 1, 1, T, D/2),
        make_gl<typename globals::q_gl>(_get_ptr(q), 1, QH, T, D),
        make_gl<typename globals::k_gl>(_get_ptr(k), 1, KVH, T, D),
        make_gl<typename globals::v_gl>(_get_ptr(v), 1, KVH, T, D),
        T,
        0  // default HIP stream
    };

    dispatch_fused_qkv_rope<HEAD_DIM, QH, KVH>(g);
}

// Dispatch based on QH/KVH configuration
void fused_qkv_split_qk_rope_fwd(pybind11::object qkv, pybind11::object cos_freq, pybind11::object sin_freq,
                                   pybind11::object q, pybind11::object k, pybind11::object v,
                                   int num_q_heads, int num_kv_heads) {
    if (num_q_heads == 32 && num_kv_heads == 8)
        fused_qkv_rope_impl<128, 32, 8>(qkv, cos_freq, sin_freq, q, k, v);
    else if (num_q_heads == 64 && num_kv_heads == 8)
        fused_qkv_rope_impl<128, 64, 8>(qkv, cos_freq, sin_freq, q, k, v);
    else if (num_q_heads == 32 && num_kv_heads == 32)
        fused_qkv_rope_impl<128, 32, 32>(qkv, cos_freq, sin_freq, q, k, v);
    else
        throw std::runtime_error("Unsupported QH/KVH configuration: QH=" +
                                 std::to_string(num_q_heads) + ", KVH=" + std::to_string(num_kv_heads));
}

PYBIND11_MODULE(fused_qkv_split_qk_rope_tk, m) {
    m.doc() = "HipKittens fused QKV split + QK RoPE kernel";
    m.def("fused_qkv_split_qk_rope_fwd", &fused_qkv_split_qk_rope_fwd,
          "Fused QKV split + QK RoPE forward pass");
}
