/**
 * Lean Attention Forward Kernel (HipKittens)
 * ==========================================
 * A HipKittens port inspired by the Lean Attention algorithm from
 * https://arxiv.org/abs/2405.10480
 *
 * Core algorithm (online softmax with exp2):
 *   1. For each K/V tile j:
 *      - Compute S = scale * Q @ K_j^T  (via transposed mma)
 *      - Apply causal mask (if enabled)
 *      - Update running max:  m_new = max(m_old, colmax(S))
 *      - Rescale accumulators: O *= exp2(m_old - m_new), l *= exp2(m_old - m_new)
 *      - P = exp2(S - m_new)
 *      - Update sum: l += colsum(P)
 *      - Accumulate: O += V_j^T @ P  (via transposed mma)
 *   2. Finalize: O = O / l
 *
 * Tile configuration:
 *   - Q_WG_SIZE = 32 per-warp Q rows (matching 32x32 MFMA base tiles)
 *   - KV_BLOCK_SIZE = 64 K/V rows per iteration step
 *   - NUM_WARPS = 4 warps per workgroup
 *   - BLOCK_M = 128 total Q rows per threadblock (4 * 32)
 *   - BF16 inputs, FP32 accumulator
 *   - Template on head dimension D (default 128)
 *   - Double-buffered K/V shared memory with group-cooperative loads
 *
 * Grid: (num_heads, cdiv(seqlen, BLOCK_M), batch)
 * Each warp independently computes its 32-row output tile.
 * Warps collaborate only on K/V group loads to shared memory.
 *
 * Note: The full Triton lean attention uses stream-K persistent
 * work-stealing with lock-based partial result gathering. This port
 * uses standard tiled forward with the same online softmax numerics.
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// ---------------------------------------------------------------------------
// Compile-time configuration (overridable via -D flags)
// ---------------------------------------------------------------------------

#ifndef ATTN_B
constexpr int ATTN_B = 4;
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 32;
#endif

#ifndef ATTN_N
constexpr int ATTN_N = 1024;
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 128;
#endif

// Block sizes
constexpr int Q_WG_SIZE     = 32;  // Per-warp Q rows
constexpr int KV_BLOCK_SIZE = 64;  // K/V rows per step
constexpr bool IS_CAUSAL    = true;

#define NUM_WARPS  4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// Total Q rows per threadblock
constexpr int BLOCK_M = Q_WG_SIZE * NUM_WARPS;  // 128

// Global memory layout descriptors (all dynamic dims)
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

// ---------------------------------------------------------------------------
// Tile type aliases (closely following reference GQA causal kernels)
// ---------------------------------------------------------------------------

// Q/O per warp: Q_WG_SIZE(32) x D, row-major, rt_32x16_s base shape
template<int D, typename T = bf16, typename L = row_l, typename S = rt_32x16_s>
using qo_tile = rt<T, Q_WG_SIZE, D, L, S>;

// Q transposed: D x Q_WG_SIZE(32), col-major
template<int D, typename T = bf16, typename L = col_l, typename S = rt_16x32_s>
using qo_tile_t = rt<T, D, Q_WG_SIZE, L, S>;

// K tile: KV_BLOCK_SIZE(64) x D, row-major
template<int D, typename T = bf16, typename L = row_l, typename S = rt_32x16_s>
using k_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;

// K transposed: D x KV_BLOCK_SIZE(64), col-major
template<int D, typename T = bf16, typename L = col_l, typename S = rt_16x32_s>
using k_tile_t = rt<T, D, KV_BLOCK_SIZE, L, S>;

// V tile: KV_BLOCK_SIZE(64) x D, col-major with rt_16x32_4_s for mma
template<int D, typename T = bf16, typename L = col_l, typename S = rt_16x32_4_s>
using v_tile_t = rt<T, KV_BLOCK_SIZE, D, L, S>;

// Attention score tile: KV_BLOCK_SIZE(64) x Q_WG_SIZE(32), col-major
// col operations (col_max, col_sum) reduce along KV dim -> per-Q-row vector
template<int D, typename T = float, typename L = col_l, typename S = rt_16x32_4_s>
using att_tile = rt<T, KV_BLOCK_SIZE, Q_WG_SIZE, L, S>;

// ---------------------------------------------------------------------------
// exp2 for individual base tiles
// ---------------------------------------------------------------------------
template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void tile_exp2(rt_base<T, layout, shape> &dst,
                                 const rt_base<T, layout, shape> &src) {
    #pragma unroll
    for (int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }
}

// ---------------------------------------------------------------------------
// Causal mask: set S[kv, q] = -inf where kv_pos > q_pos
// Uses the same VGPR-based masking approach as the reference
// ---------------------------------------------------------------------------
template<ducks::rt::col_layout RT>
__device__ inline void apply_causal_mask(RT &dst, int q_tile_abs, int kv_tile_abs,
                                          uint32_t neg_inf_bits, int lane) {
    const int col = lane & 31;  // Q position within 32-wide col tile

    const int q_base = q_tile_abs * Q_WG_SIZE;
    const int k_base = kv_tile_abs * KV_BLOCK_SIZE;
    const int q_pos  = q_base + col;

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int row_base = (i * 32) + ((lane >> 5) << 2);
        const int rel0 = q_pos - (k_base + row_base);
        const uint32_t rel = static_cast<uint32_t>(rel0);

        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.tiles[i][j].packed_per_thread; k++) {
                // Row offset within base tile for float2-packed element k
                int row_off = (k & 3) + ((k >> 2) << 3);
                int kv_pos = k_base + row_base + row_off;
                if (kv_pos > q_pos) {
                    reinterpret_cast<uint32_t*>(
                        &dst.tiles[i][j].data[k]
                    )[0] = neg_inf_bits;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Globals struct
// ---------------------------------------------------------------------------
template<int D>
struct lean_attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;  // [B, H, 1, num_q_tiles]
    hipStream_t stream;

    dim3 grid() {
        // Each block handles NUM_WARPS warp-tiles of Q
        int total_q_tiles = ATTN_N / Q_WG_SIZE;
        int num_blocks_y  = (total_q_tiles + NUM_WARPS - 1) / NUM_WARPS;
        return dim3(ATTN_H, num_blocks_y, ATTN_B);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ---------------------------------------------------------------------------
// Lean Attention Forward Kernel
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(NUM_THREADS, 2)
__global__ void lean_attn_fwd(const lean_attn_globals<D> g) {

    // ---- Shared memory allocation (double-buffered) ----
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K uses st_32x32_s layout (for row-major register loads)
    st_bf<KV_BLOCK_SIZE, D, st_32x32_s> (&k_smem)[2] =
        al.allocate<st_bf<KV_BLOCK_SIZE, D, st_32x32_s>, 2>();
    // V uses st_8x32_s layout (for col-major rt_16x32_4_s register loads)
    st_bf<KV_BLOCK_SIZE, D, st_8x32_s> (&v_smem)[2] =
        al.allocate<st_bf<KV_BLOCK_SIZE, D, st_8x32_s>, 2>();

    // ---- Indices ----
    const int head_idx     = blockIdx.x;
    const int block_tile   = blockIdx.y;
    const int batch_idx    = blockIdx.z;
    const int wid          = warpid();
    const int tile_idx     = block_tile * NUM_WARPS + wid;
    const int lane         = laneid();

    const int num_kv_tiles = ATTN_N / KV_BLOCK_SIZE;
    const int q_start_pos  = tile_idx * Q_WG_SIZE;

    // Causal limit
    int max_kv;
    if constexpr (IS_CAUSAL) {
        max_kv = min(num_kv_tiles,
                     (q_start_pos + Q_WG_SIZE + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE);
    } else {
        max_kv = num_kv_tiles;
    }

    // Softmax scale: 1/sqrt(D) * log2(e)
    constexpr float SM_SCALE = (D == 128) ? 0.08838834764f * 1.44269504089f
                                          : 0.125f * 1.44269504089f;
    uint32_t neg_inf_bits = 0xff800000;

    // ---- Register tiles ----
    qo_tile<D, bf16> q_reg;
    qo_tile_t<D, bf16> q_reg_t;
    k_tile<D, bf16> k_reg;
    k_tile_t<D, bf16> k_reg_t;
    v_tile_t<D, bf16> v_reg;

    // Output accumulator: D x Q_WG_SIZE, col-major, float
    qo_tile_t<D, float, col_l, rt_32x32_s> o_reg;

    // Attention tiles
    att_tile<D, float, col_l, rt_32x32_s> att_block;
    att_tile<D, bf16, col_l, rt_32x32_s> att_bf16;
    att_tile<D, bf16, col_l, rt_16x32_4_s> att_bf16_in;

    // Online softmax state (per Q position = row_vec of att tile)
    typename att_tile<D, float, col_l, rt_32x32_s>::row_vec max_vec, max_prev,
                                                             norm_vec, scale_vec;

    kittens::zero(o_reg);
    kittens::neg_infty(max_vec);
    kittens::zero(norm_vec);
    kittens::ones(scale_vec);

    // ---- Swizzled offsets for group loads ----
    // K offsets (st_32x32_s)
    using T_K = typename st_bf<KV_BLOCK_SIZE, D, st_32x32_s>::dtype;
    constexpr int k_bpt = st_32x32_s::template bytes_per_thread<T_K>();
    constexpr int k_bpm = k_bpt * NUM_THREADS;
    constexpr int k_mpt = KV_BLOCK_SIZE * D * sizeof(T_K) / k_bpm;
    uint32_t sw_K[k_mpt];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, sw_K);

    // V offsets (st_8x32_s)
    using T_V = typename st_bf<KV_BLOCK_SIZE, D, st_8x32_s>::dtype;
    constexpr int v_bpt = st_8x32_s::template bytes_per_thread<T_V>();
    constexpr int v_bpm = v_bpt * NUM_THREADS;
    constexpr int v_mpt = KV_BLOCK_SIZE * D * sizeof(T_V) / v_bpm;
    uint32_t sw_V[v_mpt];
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, sw_V);

    // ---- Load Q (per-warp, global -> registers) ----
    qo_tile<D, float> q_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_fl, q_fl, SM_SCALE);
    copy(q_reg, q_fl);
    transpose(q_reg_t, q_reg);

    // ---- Pre-load first K tile ----
    if (max_kv > 0) {
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx, 0}, sw_K);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // ===== MAIN LOOP over K/V tiles =====
    for (int j = 0; j < max_kv; j++) {
        int buf = j & 1;

        // K: shared -> registers
        load(k_reg, k_smem[buf]);

        // Prefetch next K tile
        if (j + 1 < max_kv) {
            G::load<1, false>(k_smem[1 - buf], g.Kg,
                              {batch_idx, j + 1, head_idx, 0}, sw_K);
        }

        // Start V load for current tile
        G::load<1, false>(v_smem[buf], g.Vg,
                          {batch_idx, j, head_idx, 0}, sw_V);

        // ---- S = K^T @ Q  via mma_AtB (all col_l) ----
        // k_reg_t: D(128) x KV(64), col_l
        // q_reg_t: D(128) x Q_WG(32), col_l
        // att_block: KV(64) x Q_WG(32), col_l
        // mma_AtB: att = k_t^T @ q_t -> (KV x D)(D x Q_WG) = KV x Q_WG
        kittens::zero(att_block);
        transpose(k_reg_t, k_reg);
        mma_AtB(att_block, k_reg_t, q_reg_t, att_block);
        __builtin_amdgcn_sched_barrier(0);

        // ---- Causal mask ----
        if constexpr (IS_CAUSAL) {
            const int kv_end = (j + 1) * KV_BLOCK_SIZE;
            if (q_start_pos < kv_end) {
                apply_causal_mask(att_block, tile_idx, j, neg_inf_bits, lane);
            }
        }

        // ---- Online softmax ----
        // 1. m_new = max(m_old, colmax(S))
        copy(max_prev, max_vec);
        col_max(max_vec, att_block, max_prev);

        // 2. scale = exp2(m_old - m_new)
        sub(scale_vec, max_prev, max_vec);
        exp2(scale_vec, scale_vec);

        // 3. Rescale accumulators
        mul_col(o_reg, o_reg, scale_vec);
        mul(norm_vec, norm_vec, scale_vec);

        // 4. P = exp2(S - m_new)
        sub_col(att_block, att_block, max_vec);
        #pragma unroll
        for (int ti = 0; ti < att_block.height; ti++) {
            #pragma unroll
            for (int tj = 0; tj < att_block.width; tj++) {
                tile_exp2(att_block.tiles[ti][tj], att_block.tiles[ti][tj]);
            }
        }

        // 5. l += colsum(P)
        col_sum(norm_vec, att_block, norm_vec);

        // 6. Convert P to bf16 for MMA
        copy(att_bf16, att_block);
        att_bf16_in = *reinterpret_cast<att_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_bf16);

        // Wait for V load to finish
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // V: shared -> registers
        load(v_reg, v_smem[buf]);

        // 7. O += V^T @ P  via mma_AtB
        // v_reg: KV(64) x D(128), col_l, rt_16x32_4_s
        // att_bf16_in: KV(64) x Q_WG(32), col_l, rt_16x32_4_s
        // o_reg: D(128) x Q_WG(32), col_l, rt_32x32_s
        // mma_AtB: o += v^T @ att -> (D x KV)(KV x Q_WG) = D x Q_WG
        mma_AtB(o_reg, v_reg, att_bf16_in, o_reg);

        // Sync before reusing shared memory buffers
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // ---- Finalize: O = O / l ----
    div_col(o_reg, o_reg, norm_vec);

    // ---- Store output ----
    // o_reg is D(128) x Q_WG(32) col_l; transpose to Q_WG(32) x D(128) row_l
    qo_tile<D, float, row_l, rt_32x32_s> o_row;
    transpose(o_row, o_reg);
    store<1>(g.Og, o_row, {batch_idx, tile_idx, head_idx, 0});

    // ---- Store LSE = max * ln(2) + log(norm) ----
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
template<int D>
void dispatch_lean_attn(lean_attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)lean_attn_fwd<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size
    );
    lean_attn_fwd<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ---------------------------------------------------------------------------
// PyBind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(lean_attn_kernel, m) {
    m.doc() = "Lean Attention Forward (HipKittens) - port of lean_atten.py";
    py::bind_function<dispatch_lean_attn<ATTN_D>>(m, "dispatch",
        &lean_attn_globals<ATTN_D>::Qg,
        &lean_attn_globals<ATTN_D>::Kg,
        &lean_attn_globals<ATTN_D>::Vg,
        &lean_attn_globals<ATTN_D>::Og,
        &lean_attn_globals<ATTN_D>::L_vec
    );
}
