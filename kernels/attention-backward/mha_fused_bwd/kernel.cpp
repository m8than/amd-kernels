// MHA Fused Backward Pass - HipKittens Port
// Ported from: reference/triton/mha_fused_bwd.py
//
// Kernels:
//   1. prep_kernel: delta[b,h,n] = sum_d( O[b,n,h,d] * dO[b,n,h,d] )
//   2. bwd_kernel:  compute dK, dV per K-block; accumulate dQ via atomic add
//
// Tensor layouts (BNHD):
//   Q, O, dO:  (B, N, H_Q, D)
//   K, V:      (B, N, H_KV, D)
//   dQ:        (B, N, H_Q, D) as float for atomic accumulation
//   dK, dV:    (B, N, H_KV, D) as bf16
//   L:         (B, H_Q, 1, N) -- softmax log-sum-exp from forward
//   delta:     (B, H_Q, 1, N)
//
// bwd_kernel grid: (H_KV, cdiv(N, BLOCK_N), B)
// Each block owns BLOCK_N rows of K/V, iterates over Q blocks and GQA heads.
// dK, dV stored directly. dQ accumulated via atomic add (multiple K blocks).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// Helper: reinterpret a register tile's layout tag without changing data.
template<typename T, int R, int C, ducks::rt_layout::all from_layout,
         ducks::rt_layout::all to_layout = col_l,
         ducks::rt_shape::all shape = ducks::rt_shape::rt_16x16>
__device__ inline rt<T, R, C, to_layout, shape>&
as_layout(rt<T, R, C, from_layout, shape>& t) {
    return reinterpret_cast<rt<T, R, C, to_layout, shape>&>(t);
}

// ============================================================
// Compile-time parameters
// ============================================================
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
constexpr int ATTN_N = 512;
#endif

constexpr int ATTN_D     = 128;
constexpr int GROUP_SIZE  = ATTN_H / ATTN_H_KV;
constexpr int BLOCK_M     = 64;
constexpr int BLOCK_N     = 64;
constexpr int NUM_KV_BLKS = ATTN_N / BLOCK_N;
constexpr int NUM_Q_BLKS  = ATTN_N / BLOCK_M;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

constexpr int WARP_KV_ROWS = BLOCK_N / NUM_WARPS; // 16
constexpr float SM_SCALE = 0.08838834764f; // 1/sqrt(128)

using G = kittens::group<NUM_WARPS>;

// ============================================================
// KERNEL 1: Preprocess -- delta = rowsum(dO * O)
// ============================================================
constexpr int PREP_TILE = 16;

template<int D>
struct prep_globals {
    gl<bf16, -1, -1, -1, -1> O;
    gl<bf16, -1, -1, -1, -1> dO;
    gl<float, -1, -1, -1, -1> delta;
    hipStream_t stream;

    dim3 grid() {
        int seq_blocks = (ATTN_N + PREP_TILE * NUM_WARPS - 1) / (PREP_TILE * NUM_WARPS);
        return dim3(seq_blocks, ATTN_H, ATTN_B);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int D>
__launch_bounds__(NUM_THREADS, 1)
__global__ void prep_kernel(const prep_globals<D> g) {
    const int seq_block = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int warpid    = kittens::warpid();

    const int tile_idx = seq_block * NUM_WARPS + warpid;
    if (tile_idx * PREP_TILE >= ATTN_N) return;

    rt_bf<PREP_TILE, D> O_reg, dO_reg;
    rt_fl<PREP_TILE, D> O_f, dO_f, prod;

    load<1>(O_reg,  g.O,  {batch_idx, tile_idx, head_idx, 0});
    load<1>(dO_reg, g.dO, {batch_idx, tile_idx, head_idx, 0});

    copy(O_f, O_reg);
    copy(dO_f, dO_reg);
    mul(prod, O_f, dO_f);

    typename rt_fl<PREP_TILE, D>::col_vec delta_vec;
    row_sum(delta_vec, prod);

    store(g.delta, delta_vec, {batch_idx, head_idx, 0, tile_idx});
}

template<int D>
void dispatch_prep(prep_globals<D> g) {
    prep_kernel<D><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ============================================================
// KERNEL 2: Main backward
// ============================================================
template<int D>
struct bwd_globals {
    gl<bf16, -1, -1, -1, -1> Q;
    gl<bf16, -1, -1, -1, -1> K;
    gl<bf16, -1, -1, -1, -1> V;
    gl<bf16, -1, -1, -1, -1> dO;
    gl<float, -1, -1, -1, -1> dQ; // float for atomic accumulation
    gl<bf16, -1, -1, -1, -1> dK;
    gl<bf16, -1, -1, -1, -1> dV;
    gl<float, -1, -1, -1, -1> L;
    gl<float, -1, -1, -1, -1> delta;
    hipStream_t stream;

    dim3 grid() { return dim3(ATTN_H_KV, NUM_KV_BLKS, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D>
__launch_bounds__(NUM_THREADS, 1)
__global__ void bwd_kernel(const bwd_globals<D> g) {
    const int kv_head_idx  = blockIdx.x;
    const int kv_blk_idx   = blockIdx.y;
    const int batch_idx    = blockIdx.z;
    const int warpid       = kittens::warpid();
    const int laneid       = kittens::laneid();
    const int tid          = threadIdx.x;

    const int kv_seq_start = kv_blk_idx * BLOCK_N;

    // ---- Shared memory ----
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared tiles for K/V loading (reused)
    st_bf<BLOCK_N, ATTN_D, st_16x16_s> (&KV_smem)  = al.allocate<st_bf<BLOCK_N, ATTN_D, st_16x16_s>>();
    // Shared tiles for Q and dO (reloaded each Q iteration)
    st_bf<BLOCK_M, ATTN_D, st_16x16_s> (&Q_smem)   = al.allocate<st_bf<BLOCK_M, ATTN_D, st_16x16_s>>();
    st_bf<BLOCK_M, ATTN_D, st_16x16_s> (&dO_smem)  = al.allocate<st_bf<BLOCK_M, ATTN_D, st_16x16_s>>();
    // L and delta vectors
    sv_fl<BLOCK_M> (&L_sv)     = al.allocate<sv_fl<BLOCK_M>>();
    sv_fl<BLOCK_M> (&delta_sv) = al.allocate<sv_fl<BLOCK_M>>();

    // ---- Load K into registers ----
    G::load<1, false>(KV_smem, g.K, {batch_idx, kv_blk_idx, kv_head_idx, 0});
    __syncthreads();
    rt_bf<WARP_KV_ROWS, ATTN_D> K_reg;
    load(K_reg, subtile_inplace<WARP_KV_ROWS, ATTN_D>(KV_smem, {warpid, 0}));

    // ---- Load V into registers (reuse KV_smem) ----
    __syncthreads();
    G::load<1, false>(KV_smem, g.V, {batch_idx, kv_blk_idx, kv_head_idx, 0});
    __syncthreads();
    rt_bf<WARP_KV_ROWS, ATTN_D> V_reg;
    load(V_reg, subtile_inplace<WARP_KV_ROWS, ATTN_D>(KV_smem, {warpid, 0}));

    // ---- Accumulators ----
    rt_fl<WARP_KV_ROWS, ATTN_D> dK_acc, dV_acc;
    zero(dK_acc);
    zero(dV_acc);

    // ---- Main loop ----
    for (int gqa = 0; gqa < GROUP_SIZE; gqa++) {
        const int q_head_idx = kv_head_idx * GROUP_SIZE + gqa;

        for (int q_blk = 0; q_blk < NUM_Q_BLKS; q_blk++) {
            const int q_seq_start = q_blk * BLOCK_M;

            // Causal skip
            if (q_seq_start + BLOCK_M - 1 < kv_seq_start) continue;

            // Load Q, dO, L, delta
            __syncthreads();
            G::load<1, false>(Q_smem,  g.Q,  {batch_idx, q_blk, q_head_idx, 0});
            G::load<1, false>(dO_smem, g.dO, {batch_idx, q_blk, q_head_idx, 0});
            load(L_sv,     g.L,     {batch_idx, q_head_idx, 0, q_blk});
            load(delta_sv, g.delta, {batch_idx, q_head_idx, 0, q_blk});
            __syncthreads();

            rt_bf<BLOCK_M, ATTN_D> Q_reg, dO_reg;
            load(Q_reg,  Q_smem);
            load(dO_reg, dO_smem);

            // ---- S^T = K @ Q^T: (16 x 64) ----
            // mma_ABt requires D,C = col_l; use as_layout to reinterpret
            rt_fl<WARP_KV_ROWS, BLOCK_M> ST;
            zero(ST);
            mma_ABt(as_layout(ST), K_reg, Q_reg, as_layout(ST));

            // Scale
            mul(ST, ST, SM_SCALE);

            // Subtract L (per Q position = column of ST)
            {
                rv_fl<BLOCK_M> L_vec;
                load(L_vec, L_sv);
                sub_col(ST, ST, L_vec);
            }

            // Causal mask
            {
                const int k_base = kv_seq_start + warpid * WARP_KV_ROWS;
                if (k_base > q_seq_start + BLOCK_M - 1) {
                    neg_infty(ST);
                } else if (k_base + WARP_KV_ROWS - 1 > q_seq_start) {
                    make_causal(ST, ST, k_base - q_seq_start);
                }
            }

            // P^T = exp(ST)
            exp(ST, ST);

            // dV += P^T @ dO
            {
                rt_bf<WARP_KV_ROWS, BLOCK_M> PT_bf;
                copy(PT_bf, ST);
                mma_AB(as_layout(dV_acc), PT_bf, as_layout(dO_reg), as_layout(dV_acc));
            }

            // dP^T = V @ dO^T
            rt_fl<WARP_KV_ROWS, BLOCK_M> dPT;
            zero(dPT);
            mma_ABt(as_layout(dPT), V_reg, dO_reg, as_layout(dPT));

            // dS^T = P^T * (dP^T - delta)
            {
                rv_fl<BLOCK_M> delta_vec;
                load(delta_vec, delta_sv);
                sub_col(dPT, dPT, delta_vec);
            }
            mul(dPT, dPT, ST);

            // dK += dS^T @ Q
            {
                rt_bf<WARP_KV_ROWS, BLOCK_M> dST_bf;
                copy(dST_bf, dPT);
                mma_AB(as_layout(dK_acc), dST_bf, as_layout(Q_reg), as_layout(dK_acc));
            }

            // dQ_partial = dS^T^T @ K * sm_scale = (64 x D)
            // Using mma_AtB: C += A^T @ B
            //   A = dST_bf (WARP_KV_ROWS x BLOCK_M), B = K_reg (WARP_KV_ROWS x D)
            //   C = (BLOCK_M x D)
            {
                rt_bf<WARP_KV_ROWS, BLOCK_M> dST_bf;
                copy(dST_bf, dPT);

                rt_fl<BLOCK_M, ATTN_D> dQ_partial;
                zero(dQ_partial);
                mma_AtB(as_layout(dQ_partial), as_layout(dST_bf), as_layout(K_reg), as_layout(dQ_partial));
                mul(dQ_partial, dQ_partial, SM_SCALE);

                // Convert to bf16 and store to Q_smem for staging
                rt_bf<BLOCK_M, ATTN_D> dQ_bf;
                copy(dQ_bf, dQ_partial);

                // Each warp stores its dQ contribution to shared memory,
                // then cooperatively atomicAdds to global float dQ.
                // We serialize warps to avoid contention on shared memory.
                float* dQ_base = (float*)&g.dQ[{batch_idx, q_seq_start, q_head_idx, 0}];
                const int dQ_stride = g.dQ.template stride<1>();

                for (int w = 0; w < NUM_WARPS; w++) {
                    if (warpid == w) {
                        // Store this warp's dQ_bf to Q_smem
                        store(Q_smem, dQ_bf);
                    }
                    __syncthreads();

                    if (warpid == w) {
                        // Read from Q_smem and atomicAdd to global dQ
                        // Q_smem uses st_16x16_s (non-swizzled), so data is at
                        // byte offset = sizeof(bf16) * (row * subtile_cols + col)
                        // within each subtile. But the full tile is rows*cols elements
                        // laid out as data[rows*cols] with subtile interleaving.
                        //
                        // For st_16x16 (non-swizzled), the byte offset for (r,c) is
                        // simply sizeof(T)*(r*16 + c%16) within each 16x16 block,
                        // plus offsets for which 16x16 block we're in.
                        //
                        // The full tile stores subtiles in row-major order:
                        // subtile (sr, sc) starts at data[sr * subtile_rows * cols + sc * subtile_cols]
                        // Then within subtile, element (lr, lc) is at
                        //   data[sr * subtile_rows * cols + sc * subtile_cols + lr * subtile_cols + lc]
                        // = data[(sr * subtile_rows + lr) * cols + sc * subtile_cols + lc]
                        // Wait, that's not right either for the internal layout.
                        //
                        // Actually, for st_16x16_s (non-swizzled), the swizzle function
                        // returns sizeof(T) * (r * 16 + c). This is the byte offset within
                        // the subtile. But the data array is flat: data[rows*cols].
                        // The overall offset for row r, col c in the big tile:
                        //   subtile_row = r / 16, local_row = r % 16
                        //   subtile_col = c / 16, local_col = c % 16
                        //   offset = (subtile_row * subtiles_per_row + subtile_col) * 16*16
                        //          + local_row * 16 + local_col
                        //
                        // This is the standard blocked (tiled) layout.
                        // For BLOCK_M=64, ATTN_D=128 with 16x16 subtiles:
                        //   subtiles_per_row = 128/16 = 8
                        //   subtiles_per_col = 64/16 = 4

                        constexpr int ST_R = 16; // subtile rows
                        constexpr int ST_C = 16; // subtile cols
                        constexpr int SPR = ATTN_D / ST_C; // subtiles per row = 8

                        // Each thread handles multiple elements
                        for (int idx = laneid; idx < BLOCK_M * ATTN_D; idx += kittens::WARP_THREADS) {
                            int row = idx / ATTN_D;
                            int col = idx % ATTN_D;

                            int sr = row / ST_R;
                            int lr = row % ST_R;
                            int sc = col / ST_C;
                            int lc = col % ST_C;

                            int flat_idx = (sr * SPR + sc) * (ST_R * ST_C) + lr * ST_C + lc;
                            float val = static_cast<float>(Q_smem.data[flat_idx]);
                            if (val != 0.0f) {
                                atomicAdd(&dQ_base[row * dQ_stride + col], val);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        } // Q block loop
    } // GQA loop

    // ---- Store dK, dV ----
    mul(dK_acc, dK_acc, SM_SCALE);

    rt_bf<WARP_KV_ROWS, ATTN_D> dK_bf, dV_bf;
    copy(dK_bf, dK_acc);
    copy(dV_bf, dV_acc);

    const int warp_tile = kv_blk_idx * NUM_WARPS + warpid;
    store<1>(g.dK, dK_bf, {batch_idx, warp_tile, kv_head_idx, 0});
    store<1>(g.dV, dV_bf, {batch_idx, warp_tile, kv_head_idx, 0});
}

template<int D>
void dispatch_bwd(bwd_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)bwd_kernel<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    bwd_kernel<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ============================================================
// Python bindings
// ============================================================
PYBIND11_MODULE(mha_fused_bwd, m) {
    m.doc() = "MHA Fused Backward Pass - HipKittens kernel";

    py::bind_function<dispatch_prep<ATTN_D>>(m, "dispatch_prep",
        &prep_globals<ATTN_D>::O,
        &prep_globals<ATTN_D>::dO,
        &prep_globals<ATTN_D>::delta
    );

    py::bind_function<dispatch_bwd<ATTN_D>>(m, "dispatch_bwd",
        &bwd_globals<ATTN_D>::Q,
        &bwd_globals<ATTN_D>::K,
        &bwd_globals<ATTN_D>::V,
        &bwd_globals<ATTN_D>::dO,
        &bwd_globals<ATTN_D>::dQ,
        &bwd_globals<ATTN_D>::dK,
        &bwd_globals<ATTN_D>::dV,
        &bwd_globals<ATTN_D>::L,
        &bwd_globals<ATTN_D>::delta
    );
}
