/**
 * Flash Attention Forward Prefill - HipKittens Port
 *
 * Implements Flash Attention v2 forward pass for prefill (full sequence processing).
 *
 * Algorithm (Online Softmax):
 *   1. Grid: (batch, num_heads_q, cdiv(seqlen_q, BLOCK_M))
 *   2. Load Q block [BLOCK_M, D]
 *   3. Initialize: m_i = -inf, l_i = 0, acc = 0
 *   4. For each K/V block (with causal masking where applicable):
 *      - Load K [KV_BLOCK_SIZE, D] -> shared, V [KV_BLOCK_SIZE, D] -> shared
 *      - qk = Q @ K^T  (-> [BLOCK_M, KV_BLOCK_SIZE])
 *      - Apply causal mask if needed
 *      - m_new = max(m_i, rowmax(qk * sm_scale))
 *      - p = exp2((qk * sm_scale - m_new) * log2(e))
 *      - alpha = exp2((m_i - m_new) * log2(e))
 *      - acc = acc * alpha + p @ V
 *      - l_i = l_i * alpha + rowsum(p)
 *      - m_i = m_new
 *   5. acc = acc / l_i
 *   6. Store O, optionally store LSE = m_i + log(l_i)
 *
 * Configuration:
 *   - BLOCK_M = 128 (Q block size = 4 warps * 32 rows/warp)
 *   - BLOCK_N = 64  (KV block size)
 *   - D = 128       (head dimension)
 *   - NUM_WARPS = 4
 *   - Uses exp2 (base-2 exponent) with log2(e) scaling for hardware efficiency
 *   - GQA/MQA supported via GROUP_SIZE = ATTN_H / ATTN_H_KV
 *   - Causal masking supported
 *
 * Memory layout: Q, K, V, O are [batch, seqlen, num_heads, head_dim] (BSHD)
 *                LSE is [batch, num_heads, 1, seqlen]
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

// ============================================================================
// Compile-time parameters (overridable via -D flags)
// ============================================================================

#ifndef ATTN_B
constexpr int ATTN_B = 4;
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 32;
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8;
#endif

#ifndef ATTN_N
constexpr int ATTN_N = 1024;
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 128;
#endif

#ifndef IS_CAUSAL
constexpr bool IS_CAUSAL = true;
#endif

// ============================================================================
// Derived constants
// ============================================================================

constexpr int GROUP_SIZE   = ATTN_H / ATTN_H_KV;  // GQA group size
constexpr int Q_BLOCK_SIZE = 32;    // rows per warp (each warp processes 32 Q rows)
constexpr int KV_BLOCK_SIZE = 64;   // K/V block size (BLOCK_N)
constexpr int BLOCK_M      = 128;   // total Q rows per threadblock = NUM_WARPS * Q_BLOCK_SIZE

#define NUM_WARPS    4
#define NUM_THREADS  (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

// ============================================================================
// Global layout types
// ============================================================================

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;   // Q, K, V, O: [batch, seq, heads, dim]
using _gl_f32  = gl<float, -1, -1, -1, -1>;  // LSE: [batch, heads, 1, seq]

// Group of warps for collaborative loads
using G = kittens::group<NUM_WARPS>;

// ============================================================================
// Tile type aliases
// ============================================================================

// Q/O tile: [Q_BLOCK_SIZE=32, D=128] per warp
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;

// Transposed Q/O tile: [D=128, Q_BLOCK_SIZE=32]
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s>
using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;

// K/V tile: [KV_BLOCK_SIZE=64, D=128]
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;

// Transposed K/V tile: [D=128, KV_BLOCK_SIZE=64]
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s>
using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L, S>;

// Attention score tile: [KV_BLOCK_SIZE=64, Q_BLOCK_SIZE=32] (col-major for MMA)
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s>
using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

// ============================================================================
// exp2 helper for base tiles
// ============================================================================

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2_tile(rt_base<T, layout, shape> &dst,
                                 const rt_base<T, layout, shape> &src) {
    #pragma unroll
    for (int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }
}

// ============================================================================
// Causal masking helper
// ============================================================================

// Mask attention scores: set future tokens to -inf
// att_block is col-major [KV_BLOCK_SIZE, Q_BLOCK_SIZE]:
//   rows = K positions, cols = Q positions
// For causal: mask where k_pos > q_pos, i.e. set att[k, q] = -inf when k > q
template<ducks::rt::col_layout RT>
__device__ inline void apply_causal_mask(RT &dst, int q_block_start, int kv_block_start,
                                         uint32_t neg_inf_v, int lane) {
    const int col = lane & 31;   // column within 32-wide col tile = Q position offset

    // Absolute Q position for this lane's column
    const int q_pos = q_block_start + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        // Row base within this subtile
        const int row_base = (i * 32) + ((lane >> 5) << 2);

        // Signed relative index: q_pos - k_pos for the first element in this row chunk
        // If rel < 0, means k_pos > q_pos -> mask these
        // If rel >= 0 for elements at offsets 0,1,..., means causal is ok for those
        const int k_pos_base = kv_block_start + row_base;
        const int rel = q_pos - k_pos_base;
        const uint32_t rel_u = static_cast<uint32_t>(rel);

        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            // Each rt_base tile has 8 data elements in col-major MFMA layout
            // The 8 elements cover 4 rows (row_base + {0,1,2,3}) x 2 sub-columns
            // But in the MFMA 32x32 col layout, the elements are:
            //   data[0].x, data[0].y -> rows row_base+0, row_base+1
            //   data[1].x, data[1].y -> rows row_base+2, row_base+3
            //   ...pattern for row offsets 0..3, 8..11, 16..19, 24..27
            auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].x);
            auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].y);
            auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].x);
            auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].y);
            auto& d2x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].x);
            auto& d2y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].y);
            auto& d3x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].x);
            auto& d3y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].y);
            auto& d4x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].x);
            auto& d4y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].y);
            auto& d5x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].x);
            auto& d5y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].y);
            auto& d6x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].x);
            auto& d6y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].y);
            auto& d7x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].x);
            auto& d7y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].y);

            // Row offsets 0,1 (threshold 0,1)
            if (rel < 0) { d0x = neg_inf_v; d0y = neg_inf_v; }
            else if (rel < 1) { d0y = neg_inf_v; }
            // Row offsets 2,3
            if (rel < 2) { d1x = neg_inf_v; d1y = neg_inf_v; }
            else if (rel < 3) { d1y = neg_inf_v; }
            // Row offsets 8,9
            if (rel < 8) { d2x = neg_inf_v; d2y = neg_inf_v; }
            else if (rel < 9) { d2y = neg_inf_v; }
            // Row offsets 10,11
            if (rel < 10) { d3x = neg_inf_v; d3y = neg_inf_v; }
            else if (rel < 11) { d3y = neg_inf_v; }
            // Row offsets 16,17
            if (rel < 16) { d4x = neg_inf_v; d4y = neg_inf_v; }
            else if (rel < 17) { d4y = neg_inf_v; }
            // Row offsets 18,19
            if (rel < 18) { d5x = neg_inf_v; d5y = neg_inf_v; }
            else if (rel < 19) { d5y = neg_inf_v; }
            // Row offsets 24,25
            if (rel < 24) { d6x = neg_inf_v; d6y = neg_inf_v; }
            else if (rel < 25) { d6y = neg_inf_v; }
            // Row offsets 26,27
            if (rel < 26) { d7x = neg_inf_v; d7y = neg_inf_v; }
            else if (rel < 27) { d7y = neg_inf_v; }
        }
    }
}

// ============================================================================
// Globals struct
// ============================================================================

template<int D>
struct flash_attn_fwd_globals {
    _gl_bf16 Qg;   // [batch, seqlen, num_heads_q, D]
    _gl_bf16 Kg;   // [batch, seqlen, num_heads_kv, D]
    _gl_bf16 Vg;   // [batch, seqlen, num_heads_kv, D]
    _gl_bf16 Og;   // [batch, seqlen, num_heads_q, D]
    _gl_f32  Lg;   // [batch, num_heads_q, 1, seqlen]  (LSE output)
    hipStream_t stream;

    // Grid: (num_heads_q, cdiv(seqlen, BLOCK_M), batch)
    // - x: head index
    // - y: Q block index (each block covers BLOCK_M=128 rows = 4 warps * 32)
    // - z: batch index
    dim3 grid() {
        int num_q_blocks = (ATTN_N + BLOCK_M - 1) / BLOCK_M;
        return dim3(ATTN_H, num_q_blocks, ATTN_B);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// ============================================================================
// Main kernel
// ============================================================================

template<int D>
__launch_bounds__(NUM_THREADS, 2)
__global__ void flash_attn_fwd_kernel(const flash_attn_fwd_globals<D> g) {

    // ---- Shared memory allocation ----
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Double-buffered K and V in shared memory
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s> (&k_smem)[2] =
        al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s> (&v_smem)[2] =
        al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s>, 2>();

    // ---- Index computation ----
    const int head_idx_q = blockIdx.x;                     // Q head index
    const int q_block_idx = blockIdx.y;                    // which BLOCK_M chunk of Q
    const int batch_idx = blockIdx.z;

    // GQA: map Q head to KV head
    const int head_idx_kv = head_idx_q / GROUP_SIZE;

    // Each warp handles Q_BLOCK_SIZE=32 rows within the BLOCK_M=128 block
    const int warp_id = warpid();
    const int tile_idx = q_block_idx * NUM_WARPS + warp_id;  // absolute Q tile index
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;         // absolute Q row start
    const int lane = laneid();

    // ---- Compute how many KV blocks to iterate ----
    const int num_kv_blocks = ATTN_N / KV_BLOCK_SIZE;
    int max_kv_blocks;
    if constexpr (IS_CAUSAL) {
        // For causal: only attend to K positions <= max Q position in this block
        const int max_q_pos_in_block = (q_block_idx + 1) * BLOCK_M - 1;
        max_kv_blocks = min((max_q_pos_in_block / KV_BLOCK_SIZE) + 1, num_kv_blocks);
    } else {
        max_kv_blocks = num_kv_blocks;
    }

    // ---- Constants ----
    // Scale = 1/sqrt(D) * log2(e) for use with exp2
    constexpr float LOG2E = 1.44269504089f;
    constexpr float LN2   = 0.69314718056f;
    constexpr float SM_SCALE = (D == 128) ? 0.08838834764f : 0.125f;  // 1/sqrt(D)
    constexpr float TEMPERATURE_SCALE = SM_SCALE * LOG2E;
    uint32_t neg_inf_v = 0xff800000;  // IEEE -inf as uint32

    // ---- Declare register tiles ----
    qo_tile<D, bf16>                q_reg;
    qo_tile_transposed<D, bf16>     q_reg_t;
    kv_tile<D, bf16>                k_reg;
    kv_tile_transposed<D, bf16>     k_reg_t;
    kv_tile<D, bf16, col_l, rt_16x32_4_s> v_reg;
    qo_tile_transposed<D, float, col_l, rt_32x32_s> o_reg;  // accumulator (float)
    attn_tile<D, float, col_l, rt_32x32_s> att_block;        // attention scores (float)
    attn_tile<D, bf16, col_l, rt_32x32_s>  att_block_bf16;
    attn_tile<D, bf16, col_l, rt_16x32_4_s> att_block_mma;   // for MMA input

    // Softmax state vectors (per-row of this warp's Q tile)
    typename attn_tile<D, float, col_l, rt_32x32_s>::row_vec max_vec, max_vec_prev, norm_vec, scale_vec;

    // ---- Initialize ----
    kittens::zero(o_reg);
    kittens::zero(norm_vec);
    // max_vec starts at -inf (will be set by first col_max)
    // scale_vec starts at 1.0
    kittens::ones(scale_vec);

    // ---- Precompute swizzled offsets for group loads ----
    using KSmemType = st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>;
    using VSmemType = st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s>;
    using T_K = typename KSmemType::dtype;
    constexpr int bytes_per_thread_k = st_32x32_s::template bytes_per_thread<T_K>();
    constexpr int bytes_per_memcpy_k = bytes_per_thread_k * NUM_THREADS;
    constexpr int memcpy_per_tile_k = KV_BLOCK_SIZE * ATTN_D * sizeof(T_K) / bytes_per_memcpy_k;

    uint32_t swizzled_offsets_K[memcpy_per_tile_k];
    uint32_t swizzled_offsets_V[memcpy_per_tile_k];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    // ---- Create buffer resource descriptors for fast loads ----
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
    i32x4 k_srsrc = make_srsrc(k_base, k_row_stride * ATTN_N, k_row_stride);
    i32x4 v_srsrc = make_srsrc(v_base, v_row_stride * ATTN_N, v_row_stride);

    // ---- Compute LDS base addresses for each buffer ----
    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t k_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));

    // ---- Load Q into registers and apply temperature scaling ----
    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_bf16>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx_q, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);
    copy(q_reg, q_reg_fl);
    transpose(q_reg_t, q_reg);

    // ========================================================================
    // Main K/V loop with double-buffered shared memory
    // ========================================================================

    // Load first K block into smem[0]
    if (max_kv_blocks > 0) {
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0},
                          swizzled_offsets_K, k_srsrc, k_base, k_lds_base_0);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    for (int kv_idx = 0; kv_idx < max_kv_blocks; kv_idx++) {
        const int buf_cur = kv_idx & 1;
        const int buf_nxt = 1 - buf_cur;

        // ---- Load K from current shared memory buffer into registers ----
        if (buf_cur == 0) {
            load(k_reg, k_smem[0]);
        } else {
            load(k_reg, k_smem[1]);
        }

        // ---- Prefetch next K block (if any) ----
        if (kv_idx + 1 < max_kv_blocks) {
            if (buf_nxt == 0) {
                G::load<1, false>(k_smem[0], g.Kg, {batch_idx, kv_idx + 1, head_idx_kv, 0},
                                  swizzled_offsets_K, k_srsrc, k_base, k_lds_base_0);
            } else {
                G::load<1, false>(k_smem[1], g.Kg, {batch_idx, kv_idx + 1, head_idx_kv, 0},
                                  swizzled_offsets_K, k_srsrc, k_base, k_lds_base_1);
            }
        }

        // ---- Load V block into shared memory (other buffer) ----
        if (buf_nxt == 0) {
            G::load<1, false>(v_smem[0], g.Vg, {batch_idx, kv_idx, head_idx_kv, 0},
                              swizzled_offsets_V, v_srsrc, v_base, v_lds_base_0);
        } else {
            G::load<1, false>(v_smem[1], g.Vg, {batch_idx, kv_idx, head_idx_kv, 0},
                              swizzled_offsets_V, v_srsrc, v_base, v_lds_base_1);
        }

        // ---- Compute QK^T ----
        kittens::zero(att_block);
        transpose(k_reg_t, k_reg);
        mma_AtB(att_block, k_reg_t, q_reg_t, att_block);

        // ---- Apply causal mask ----
        if constexpr (IS_CAUSAL) {
            const int kv_end_pos = (kv_idx + 1) * KV_BLOCK_SIZE;
            if (q_start_pos < kv_end_pos) {
                apply_causal_mask(att_block, q_start_pos, kv_idx * KV_BLOCK_SIZE, neg_inf_v, lane);
            }
        }

        // ---- Online softmax: compute new max ----
        if (kv_idx == 0) {
            col_max(max_vec, att_block);
        } else {
            col_max(max_vec, att_block, max_vec_prev);
        }

        // ---- Rescale accumulator if max changed ----
        if (kv_idx > 0) {
            sub(scale_vec, max_vec_prev, max_vec);
            exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            mul(norm_vec, norm_vec, scale_vec);
        }
        copy(max_vec_prev, max_vec);

        // ---- Compute p = exp2(att - max) ----
        sub_col(att_block, att_block, max_vec);
        #pragma unroll
        for (int ti = 0; ti < att_block.height; ti++) {
            #pragma unroll
            for (int tj = 0; tj < att_block.width; tj++) {
                exp2_tile(att_block.tiles[ti][tj], att_block.tiles[ti][tj]);
            }
        }

        // ---- Update norm vector: l_i += rowsum(p) ----
        col_sum(norm_vec, att_block, norm_vec);

        // ---- Convert attention scores to bf16 for MMA ----
        copy(att_block_bf16, att_block);
        att_block_mma = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);

        // ---- Wait for V to arrive in shared memory ----
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

        // ---- Load V from shared memory ----
        if (buf_nxt == 0) {
            load(v_reg, v_smem[0]);
        } else {
            load(v_reg, v_smem[1]);
        }

        // ---- Accumulate: O += P @ V (using subtile MMA for efficiency) ----
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_mma, 0), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_mma, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_mma, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_mma, 3), o_reg);

        __builtin_amdgcn_sched_barrier(0);

        // ---- Synchronize before next iteration (K prefetch must complete) ----
        if (kv_idx + 1 < max_kv_blocks) {
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
        }
    }

    // ========================================================================
    // Epilogue: normalize and store
    // ========================================================================

    // O = O / l_i
    div_col(o_reg, o_reg, norm_vec);

    // Synchronize before store
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Transpose O from col-major accumulator to row-major for store
    qo_tile<D, float, row_l, rt_32x32_s> o_reg_row;
    transpose(o_reg_row, o_reg);
    store<1>(g.Og, o_reg_row, {batch_idx, tile_idx, head_idx_q, 0});

    // ---- Compute and store LSE = m_i * ln(2) + ln(l_i) ----
    // Since we used exp2 (base-2), max_vec is in log2 scale.
    // LSE = max_vec * ln(2) + ln(norm_vec)
    mul(max_vec, max_vec, LN2);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.Lg, norm_vec, {batch_idx, head_idx_q, 0, tile_idx});
}

// ============================================================================
// Host dispatch
// ============================================================================

template<int D>
void dispatch_flash_attn_fwd(flash_attn_fwd_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)flash_attn_fwd_kernel<D>,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    flash_attn_fwd_kernel<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ============================================================================
// Python binding
// ============================================================================

PYBIND11_MODULE(flash_attn_fwd_prefill, m) {
    m.doc() = "Flash Attention Forward Prefill - HipKittens";
    py::bind_function<dispatch_flash_attn_fwd<ATTN_D>>(m, "dispatch",
        &flash_attn_fwd_globals<ATTN_D>::Qg,
        &flash_attn_fwd_globals<ATTN_D>::Kg,
        &flash_attn_fwd_globals<ATTN_D>::Vg,
        &flash_attn_fwd_globals<ATTN_D>::Og,
        &flash_attn_fwd_globals<ATTN_D>::Lg
    );
}
