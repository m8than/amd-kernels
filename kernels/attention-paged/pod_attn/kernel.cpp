/**
 * POD Attention Kernel (Prefill-On-Decode) (HipKittens)
 *
 * Ported from AITER Triton: pod_attention.py
 *
 * Simultaneously runs prefill and decode attention on the same GPU
 * using a persistent kernel with CU (compute unit) based scheduling.
 *
 * Key concept: Each wavefront uses atomic counters on a per-CU basis
 * to decide whether to execute prefill or decode work. The ratio is
 * controlled by prefill_ratio and decode_ratio parameters.
 *
 * Architecture:
 *   - Persistent kernel: launched once, processes all work items
 *   - CU-based scheduling: each CU has a counter, wavefronts claim work
 *   - Prefill path: causal flash attention over full sequences
 *   - Decode path: non-causal flash attention (single token)
 *   - Split-K with locks for synchronization of partial results
 *
 * Grid: (max_output_tiles, 1, 1) - persistent
 */

#include "kittens.cuh"

using namespace kittens;

#ifndef POD_HEAD_DIM
#define POD_HEAD_DIM 128
#endif

#ifndef POD_BLOCK_M
#define POD_BLOCK_M 64
#endif

#ifndef POD_BLOCK_N
#define POD_BLOCK_N 64
#endif

#ifndef POD_BLOCK_M_PF
#define POD_BLOCK_M_PF 64
#endif

#ifndef POD_BLOCK_N_PF
#define POD_BLOCK_N_PF 64
#endif

constexpr int HEAD_DIM = POD_HEAD_DIM;
constexpr int BLOCK_M = POD_BLOCK_M;     // Q tile for decode
constexpr int BLOCK_N = POD_BLOCK_N;     // KV tile for decode
constexpr int BLOCK_M_PF = POD_BLOCK_M_PF; // Q tile for prefill
constexpr int BLOCK_N_PF = POD_BLOCK_N_PF; // KV tile for prefill

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

// Global memory types
using qkvo_gl = gl<bf16, -1, -1, -1, POD_HEAD_DIM>;
using mid_gl = gl<float, -1, -1, -1, -1>;
using lock_gl = gl<int, -1, -1, -1, -1>;

struct pod_attn_globals {
    // Decode tensors
    qkvo_gl Q;           // [batch * num_heads, seq_len_q, 1, head_dim]
    qkvo_gl K;           // [batch * num_heads, seq_len_kv, 1, head_dim]
    qkvo_gl V;           // [batch * num_heads, seq_len_kv, 1, head_dim]
    qkvo_gl Out;         // [batch * num_heads, seq_len_q, 1, head_dim]
    mid_gl   Mp;         // [batch * num_heads, num_splits, num_m_blocks, 1] max partial
    mid_gl   Lp;         // [batch * num_heads, num_splits, num_m_blocks, 1] sum partial
    mid_gl   Op;         // [batch * num_heads, num_splits, num_m_blocks, head_dim] output partial

    // Prefill tensors
    qkvo_gl Q_pf;        // [batch_pf * num_heads, seq_len_q_pf, 1, head_dim]
    qkvo_gl K_pf;        // [batch_pf * num_heads, seq_len_kv_pf, 1, head_dim]
    qkvo_gl V_pf;        // [batch_pf * num_heads, seq_len_kv_pf, 1, head_dim]
    qkvo_gl Out_pf;      // [batch_pf * num_heads, seq_len_q_pf, 1, head_dim]
    mid_gl   Mp_pf;
    mid_gl   Lp_pf;
    mid_gl   Op_pf;

    // Scheduling
    lock_gl  locks;      // [max_locks, 1, 1, 1] split-K synchronization
    lock_gl  locks_pf;
    gl<int, -1, -1, -1, -1> cu_ctr; // [num_CUs, 1, 1, 1] atomic per-CU counters

    // Configuration
    int batch_size;
    int batch_size_pf;
    int num_heads;
    int seq_len_q;
    int seq_len_kv;
    int seq_len_q_pf;
    int seq_len_kv_pf;
    int num_splits;
    int num_splits_pf;
    int prefill_ratio;   // e.g., 1
    int decode_ratio;    // e.g., 3 -> 25% prefill, 75% decode
    int max_tiles;       // total work items to process
    float scale;
    hipStream_t stream;

    dim3 grid() {
        return dim3(max_tiles);
    }
    dim3 block() {
        return dim3(NUM_THREADS);
    }
    size_t dynamic_shared_memory() {
        return MAX_SHARED_MEMORY;
    }
};

/**
 * Get the hardware CU (Compute Unit) ID.
 * On AMD, this maps to the shader engine / array / CU.
 */
__device__ int get_cu_id() {
    int cu_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(cu_id));
    // Extract CU ID from HW_REG_HW_ID bits
    return (cu_id >> 8) & 0xFF; // CU index within shader engine
}

/**
 * Flash attention core computation (used by both prefill and decode paths).
 *
 * Computes tiled attention with online softmax:
 *   For each Q tile:
 *     For each KV tile:
 *       S = Q @ K^T * scale
 *       Apply causal mask if needed
 *       Online softmax accumulation
 *       O += P @ V
 *     Final normalization
 */
template<int BM, int BN, bool CAUSAL>
__device__ void lean_attention_core(
    const qkvo_gl &Q_gl, const qkvo_gl &K_gl, const qkvo_gl &V_gl, const qkvo_gl &O_gl,
    const mid_gl &Mp_gl, const mid_gl &Lp_gl, const mid_gl &Op_gl,
    int batch_head_idx, int m_block_idx, int kv_start, int kv_end,
    int split_idx, float temperature
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BN, POD_HEAD_DIM, st_32x32_s> (&kv_smem)[2] = al.allocate<st_bf<BN, POD_HEAD_DIM, st_32x32_s>, 2>();

    // Load Q tile
    rt<bf16, BM, POD_HEAD_DIM, row_l, rt_32x16_s> q_reg;
    rt<float, BM, POD_HEAD_DIM, row_l, rt_32x32_s> q_fl;
    load<1>(q_fl, Q_gl, {batch_head_idx, m_block_idx * BM, 0, 0});
    asm volatile("s_waitcnt vmcnt(0)");
    mul(q_fl, q_fl, temperature);
    copy(q_reg, q_fl);

    // No need for transposed Q - we'll use mma_ABt

    // Output accumulator
    rt<float, POD_HEAD_DIM, BM, col_l, rt_32x32_s> o_reg;
    zero(o_reg);

    typename rt<float, BN, BM, col_l, rt_16x32_4_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;
    #pragma unroll
    for (int o = 0; o < max_vec.outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < max_vec.inner_dim; ++i) {
            max_vec.data[o][i] = -INFINITY;
        }
    }
    zero(norm_vec);

    // Loop over KV tiles
    for (int kv_blk = kv_start; kv_blk < kv_end; kv_blk += BN) {
        const int buf = (kv_blk / BN) & 1;

        // Load K
        G::load<1, false>(kv_smem[buf], K_gl, {batch_head_idx, kv_blk, 0, 0});
        __builtin_amdgcn_s_barrier();

        rt<bf16, BN, POD_HEAD_DIM, row_l, rt_32x16_s> k_reg;
        load(k_reg, kv_smem[buf]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // S = Q @ K^T
        rt<float, BN, BM, col_l, rt_32x32_s> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);

        // Causal mask
        if constexpr (CAUSAL) {
            if (kv_blk >= m_block_idx * BM) {
                make_causal(att, -1e10f);
            }
        }

        // Online softmax
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att, max_vec_prev);
        sub(scale_vec, max_vec_prev, max_vec);
        exp2(scale_vec, scale_vec);
        mul_col(o_reg, o_reg, scale_vec);
        mul(norm_vec, norm_vec, scale_vec);
        sub_col(att, att, max_vec);

        #pragma unroll
        for (int i = 0; i < att.height; i++) {
            #pragma unroll
            for (int j = 0; j < att.width; j++) {
                #pragma unroll
                for (int k = 0; k < att.tiles[i][j].packed_per_thread; k++) {
                    att.tiles[i][j].data[k] = base_ops::exp2::op(att.tiles[i][j].data[k]);
                }
            }
        }
        col_sum(norm_vec, att, norm_vec);

        // Load V
        G::load<1, false>(kv_smem[buf ^ 1], V_gl, {batch_head_idx, kv_blk, 0, 0});
        __builtin_amdgcn_s_barrier();

        rt<bf16, BN, POD_HEAD_DIM, col_l, rt_16x32_4_s> v_reg;
        load(v_reg, kv_smem[buf ^ 1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        rt<bf16, BM, BN, row_l, rt_16x32_4_s> att_bf16;
        copy(att_bf16, att);
        mma_AB(o_reg, att_bf16, v_reg, o_reg);

        __builtin_amdgcn_s_barrier();
    }

    // Normalize
    div_col(o_reg, o_reg, norm_vec);

    // Store partial results - o_reg is already col_l, copy to row_l tile for store
    rt<float, BM, POD_HEAD_DIM, row_l, rt_32x32_s> o_row;
    copy(o_row, o_reg);
    rt<bf16, BM, POD_HEAD_DIM, row_l, rt_32x16_s> o_out;
    copy(o_out, o_row);

    store<1>(O_gl, o_out, {batch_head_idx, m_block_idx * BM, 0, 0});
}

/**
 * POD persistent kernel.
 *
 * Each wavefront:
 *   1. Get CU ID
 *   2. Atomically increment per-CU counter
 *   3. Based on counter % (prefill_ratio + decode_ratio):
 *      - If < prefill_ratio: execute prefill attention (causal)
 *      - Else: execute decode attention (non-causal)
 *   4. Claim next work item from global work queue
 */
__launch_bounds__(NUM_THREADS, 2)
__global__ void pod_persistent_kernel(const pod_attn_globals g) {
    const int wg_idx = blockIdx.x;

    // Get CU ID for scheduling
    const int cu_id = get_cu_id();
    const int total_ratio = g.prefill_ratio + g.decode_ratio;

    // Determine if this workgroup does prefill or decode
    // Atomic increment on per-CU counter
    int ctr = atomicAdd((int*)&g.cu_ctr[{cu_id, 0, 0, 0}], 1);
    const bool do_prefill = (ctr % total_ratio) < g.prefill_ratio;

    const float temperature = g.scale * 1.44269504089f;

    if (do_prefill && g.batch_size_pf > 0) {
        // Prefill path: causal attention
        const int total_pf_heads = g.batch_size_pf * g.num_heads;
        const int num_m_blocks_pf = (g.seq_len_q_pf + BLOCK_M_PF - 1) / BLOCK_M_PF;
        const int total_pf_tiles = total_pf_heads * num_m_blocks_pf;

        // Simple round-robin work assignment
        int tile_idx = wg_idx;
        if (tile_idx < total_pf_tiles) {
            int bh = tile_idx / num_m_blocks_pf;
            int mb = tile_idx % num_m_blocks_pf;

            lean_attention_core<BLOCK_M_PF, BLOCK_N_PF, true>(
                g.Q_pf, g.K_pf, g.V_pf, g.Out_pf,
                g.Mp_pf, g.Lp_pf, g.Op_pf,
                bh, mb, 0, g.seq_len_kv_pf, 0, temperature
            );
        }
    } else {
        // Decode path: non-causal attention
        const int total_dec_heads = g.batch_size * g.num_heads;
        const int num_m_blocks = (g.seq_len_q + BLOCK_M - 1) / BLOCK_M;
        const int total_dec_tiles = total_dec_heads * num_m_blocks;

        int tile_idx = wg_idx;
        if (tile_idx < total_dec_tiles) {
            int bh = tile_idx / num_m_blocks;
            int mb = tile_idx % num_m_blocks;

            lean_attention_core<BLOCK_M, BLOCK_N, false>(
                g.Q, g.K, g.V, g.Out,
                g.Mp, g.Lp, g.Op,
                bh, mb, 0, g.seq_len_kv, 0, temperature
            );
        }
    }
}

void dispatch_pod_attn(pod_attn_globals &g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)pod_persistent_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    pod_persistent_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}
