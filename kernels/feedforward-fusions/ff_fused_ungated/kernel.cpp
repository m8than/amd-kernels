// ff_fused_ungated: Fused Ungated Feed Forward BF16 GEMM
// Implements: out = activation(X @ W1) @ W2
//
// X: (M, K) bf16 input
// W1: (N, K) bf16 weights for first projection (transposed)
// W2: (N, K) bf16 weights for second projection (transposed)
// out: (M, K) bf16 output
//
// This is a 2-stage fused kernel:
// 1. First GEMM: X @ W1^T with optional activation -> (M, N)
// 2. Second GEMM: intermediate @ W2 -> (M, K) output
//
// NOTE: This simplified implementation includes stage 1 only.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M       = 256;
constexpr int BLOCK_N       = 256;
constexpr int BLOCK_K       = 64;
constexpr int HALF_BLOCK_M  = BLOCK_M / 2;
constexpr int HALF_BLOCK_N  = BLOCK_N / 2;
constexpr int WARPS_M       = 2;
constexpr int WARPS_N       = 4;
constexpr int REG_BLOCK_M   = BLOCK_M / WARPS_M;
constexpr int REG_BLOCK_N   = BLOCK_N / WARPS_N;
constexpr int HALF_REG_M    = REG_BLOCK_M / 2;
constexpr int HALF_REG_N    = REG_BLOCK_N / 2;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

template<int K_dim>
struct ff_fused_ungated_globals {
    using _gl_X = gl<bf16, -1, -1, -1, K_dim>;
    using _gl_W1 = gl<bf16, -1, -1, -1, K_dim>;
    using _gl_W2 = gl<bf16, -1, -1, -1, K_dim>;
    using _gl_Out = gl<bf16, -1, -1, -1, K_dim>;

    _gl_X x;
    _gl_W1 w1;
    _gl_W2 w2;
    _gl_Out out;
    int M;
    int N;
    int K;
    hipStream_t stream;

    dim3 grid() {
        return dim3(ceil_div(N, BLOCK_N) * ceil_div(M, BLOCK_M));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ReLU activation
__device__ __forceinline__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

using G = kittens::group<NUM_WARPS>;

template<int K_dim, bool use_activation = true>
__global__ __launch_bounds__(NUM_THREADS, 2)
void ff_fused_ungated_kernel(const ff_fused_ungated_globals<K_dim> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const int M = g.M;
    const int N = g.N;
    const int K = g.K;

    // Shared memory: double buffered, split into halves along M and N
    using ST_X = st_bf<HALF_BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_W1 = st_bf<HALF_BLOCK_N, BLOCK_K, st_16x32_s>;
    ST_X (&Xs)[2][2] = al.allocate<ST_X, 2, 2>();
    ST_W1 (&W1s)[2][2] = al.allocate<ST_W1, 2, 2>();

    // Register tiles
    rt_bf<HALF_REG_M, BLOCK_K, row_l, rt_16x32_s> X_tile;
    rt_bf<HALF_REG_N, BLOCK_K, row_l, rt_16x32_s> W1_tile_0;
    rt_bf<HALF_REG_N, BLOCK_K, row_l, rt_16x32_s> W1_tile_1;

    // FP32 accumulators: 2x2 grid covering full output tile per warp
    rt_fl<HALF_REG_M, HALF_REG_N, col_l, rt_16x16_s> acc[2][2];
    zero(acc[0][0]);
    zero(acc[0][1]);
    zero(acc[1][0]);
    zero(acc[1][1]);

    // Chiplet-aware scheduling
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 8;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);

    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    int row = pid_m;
    int col = pid_n;

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int num_tiles = K / BLOCK_K;

    // Buffer setup
    const bf16* x_base = (bf16*)&g.x[{0, 0, 0, 0}];
    const bf16* w1_base = (bf16*)&g.w1[{0, 0, 0, 0}];
    const int x_row_stride = g.x.template stride<2>() * sizeof(bf16);
    const int w1_row_stride = g.w1.template stride<2>() * sizeof(bf16);
    i32x4 x_srsrc = make_srsrc(x_base, M * x_row_stride, x_row_stride);
    i32x4 w1_srsrc = make_srsrc(w1_base, N * w1_row_stride, w1_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;

    uint32_t x_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Xs[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t x_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Xs[0][1].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t x_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Xs[1][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t x_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Xs[1][1].data[0]) + wid * elem_per_warp * sizeof(bf16)));

    uint32_t w1_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&W1s[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t w1_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&W1s[0][1].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t w1_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&W1s[1][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t w1_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&W1s[1][1].data[0]) + wid * elem_per_warp * sizeof(bf16)));

    using T = typename ST_X::dtype;
    constexpr int bytes_per_thread_x = ST_X::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy_x = bytes_per_thread_x * NUM_THREADS;
    constexpr int memcpy_per_tile_x = BLOCK_M * BLOCK_K * sizeof(T) / bytes_per_memcpy_x;
    uint32_t swizzled_X[memcpy_per_tile_x / 2];
    uint32_t swizzled_W1[memcpy_per_tile_x / 2];
    G::prefill_swizzled_offsets(Xs[0][0], g.x, swizzled_X);
    G::prefill_swizzled_offsets(W1s[0][0], g.w1, swizzled_W1);

    int tic = 0, toc = 1;

    // Prefetch first tile
    G::load(W1s[tic][0], g.w1, {0, 0, col*2, 0}, swizzled_W1, w1_srsrc, w1_base, w1_lds_00);
    G::load(Xs[tic][0], g.x, {0, 0, row*2, 0}, swizzled_X, x_srsrc, x_base, x_lds_00);
    G::load(W1s[tic][1], g.w1, {0, 0, col*2 + 1, 0}, swizzled_W1, w1_srsrc, w1_base, w1_lds_01);
    G::load(Xs[tic][1], g.x, {0, 0, row*2 + 1, 0}, swizzled_X, x_srsrc, x_base, x_lds_01);

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    // Prefetch second tile
    G::load(W1s[toc][0], g.w1, {0, 0, col*2, 1}, swizzled_W1, w1_srsrc, w1_base, w1_lds_10);
    G::load(Xs[toc][0], g.x, {0, 0, row*2, 1}, swizzled_X, x_srsrc, x_base, x_lds_10);
    G::load(W1s[toc][1], g.w1, {0, 0, col*2 + 1, 1}, swizzled_W1, w1_srsrc, w1_base, w1_lds_11);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // Main K-loop (simplified - processes 2 K-tiles per iteration)
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile += 2) {
        auto st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[0][0], {warp_col, 0});
        load(W1_tile_0, st_subtile_w1);
        auto st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[0][0], {warp_row, 0});
        load(X_tile, st_subtile_x);
        G::load(Xs[1][1], g.x, {0, 0, row*2 + 1, tile + 1}, swizzled_X, x_srsrc, x_base, x_lds_11);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][0], X_tile, W1_tile_0, acc[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[0][1], {warp_col, 0});
        load(W1_tile_1, st_subtile_w1);
        G::load(W1s[0][0], g.w1, {0, 0, col*2, tile + 2}, swizzled_W1, w1_srsrc, w1_base, w1_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][1], X_tile, W1_tile_1, acc[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[0][1], {warp_row, 0});
        load(X_tile, st_subtile_x);
        G::load(Xs[0][0], g.x, {0, 0, row*2, tile + 2}, swizzled_X, x_srsrc, x_base, x_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][0], X_tile, W1_tile_0, acc[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[1][0], {warp_col, 0});
        load(W1_tile_0, st_subtile_w1);
        G::load(W1s[0][1], g.w1, {0, 0, col*2 + 1, tile + 2}, swizzled_W1, w1_srsrc, w1_base, w1_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][1], X_tile, W1_tile_1, acc[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[1][0], {warp_row, 0});
        load(X_tile, st_subtile_x);
        G::load(Xs[0][1], g.x, {0, 0, row*2 + 1, tile + 2}, swizzled_X, x_srsrc, x_base, x_lds_01);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][0], X_tile, W1_tile_0, acc[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[1][1], {warp_col, 0});
        load(W1_tile_1, st_subtile_w1);
        G::load(W1s[1][0], g.w1, {0, 0, col*2, tile + 3}, swizzled_W1, w1_srsrc, w1_base, w1_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][1], X_tile, W1_tile_1, acc[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[1][1], {warp_row, 0});
        load(X_tile, st_subtile_x);
        G::load(Xs[1][0], g.x, {0, 0, row*2, tile + 3}, swizzled_X, x_srsrc, x_base, x_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][0], X_tile, W1_tile_0, acc[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(W1s[1][1], g.w1, {0, 0, col*2 + 1, tile + 3}, swizzled_W1, w1_srsrc, w1_base, w1_lds_11);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][1], X_tile, W1_tile_1, acc[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Tail: second-to-last K tile
    {
        int tile = num_tiles - 2;
        auto st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[tic][0], {warp_col, 0});
        load(W1_tile_0, st_subtile_w1);
        auto st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[tic][0], {warp_row, 0});
        load(X_tile, st_subtile_x);
        G::load(Xs[toc][1], g.x, {0, 0, row*2 + 1, tile + 1}, swizzled_X, x_srsrc, x_base, x_lds_11);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][0], X_tile, W1_tile_0, acc[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[tic][1], {warp_col, 0});
        load(W1_tile_1, st_subtile_w1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][1], X_tile, W1_tile_1, acc[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[tic][1], {warp_row, 0});
        load(X_tile, st_subtile_x);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][0], X_tile, W1_tile_0, acc[1][0]);
        mma_ABt(acc[1][1], X_tile, W1_tile_1, acc[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1; toc ^= 1;
    }

    // Tail: last K tile
    {
        auto st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[tic][0], {warp_col, 0});
        load(W1_tile_0, st_subtile_w1);
        auto st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[tic][0], {warp_row, 0});
        load(X_tile, st_subtile_x);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][0], X_tile, W1_tile_0, acc[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_w1 = subtile_inplace<HALF_REG_N, BLOCK_K>(W1s[tic][1], {warp_col, 0});
        load(W1_tile_1, st_subtile_w1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[0][1], X_tile, W1_tile_1, acc[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_x = subtile_inplace<HALF_REG_M, BLOCK_K>(Xs[tic][1], {warp_row, 0});
        load(X_tile, st_subtile_x);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc[1][0], X_tile, W1_tile_0, acc[1][0]);
        mma_ABt(acc[1][1], X_tile, W1_tile_1, acc[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Apply activation (SiLU) if requested
    if (use_activation) {
        constexpr int PT = rt_fl<HALF_REG_M, HALF_REG_N, col_l, rt_16x16_s>::packed_per_thread;
        constexpr int HT = acc[0][0].height;
        constexpr int WT = acc[0][0].width;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                #pragma unroll
                for (int h = 0; h < HT; h++) {
                    #pragma unroll
                    for (int w = 0; w < WT; w++) {
                        #pragma unroll
                        for (int p = 0; p < PT; p++) {
                            // data[p] is float2 (packed type); process .x and .y
                            acc[i][j].tiles[h][w].data[p].x = silu(acc[i][j].tiles[h][w].data[p].x);
                            acc[i][j].tiles[h][w].data[p].y = silu(acc[i][j].tiles[h][w].data[p].y);
                        }
                    }
                }
            }
        }
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    // Store results
    store(g.out, acc[0][0], {0, 0,
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col});
    store(g.out, acc[0][1], {0, 0,
        (row * 2) * WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col});
    store(g.out, acc[1][0], {0, 0,
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + warp_col});
    store(g.out, acc[1][1], {0, 0,
        (row * 2) * WARPS_M + WARPS_M + warp_row,
        col * 2 * WARPS_N + WARPS_N + warp_col});
}

template<int K_dim>
void dispatch_ff_fused_ungated(ff_fused_ungated_globals<K_dim>& g, bool use_activation = true) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ff_fused_ungated_kernel<K_dim, true>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipFuncSetAttribute((void*)ff_fused_ungated_kernel<K_dim, false>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    if (use_activation) {
        ff_fused_ungated_kernel<K_dim, true><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {
        ff_fused_ungated_kernel<K_dim, false><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    }
}

// Template instantiation
template void dispatch_ff_fused_ungated<4096>(ff_fused_ungated_globals<4096>&, bool);
template void dispatch_ff_fused_ungated<8192>(ff_fused_ungated_globals<8192>&, bool);

// Wrappers for bind_function (which requires single-globals-arg dispatch)
template<int K_dim>
void dispatch_ff_fused_ungated_activated(ff_fused_ungated_globals<K_dim>& g) {
    dispatch_ff_fused_ungated<K_dim>(g, true);
}
template<int K_dim>
void dispatch_ff_fused_ungated_linear(ff_fused_ungated_globals<K_dim>& g) {
    dispatch_ff_fused_ungated<K_dim>(g, false);
}

PYBIND11_MODULE(ff_fused_ungated_tk, m) {
    m.doc() = "HipKittens Fused Ungated Feed Forward kernel";

    kittens::py::bind_function<dispatch_ff_fused_ungated_activated<4096>>(m, "dispatch_4096",
        &ff_fused_ungated_globals<4096>::x,
        &ff_fused_ungated_globals<4096>::w1,
        &ff_fused_ungated_globals<4096>::w2,
        &ff_fused_ungated_globals<4096>::out,
        &ff_fused_ungated_globals<4096>::M,
        &ff_fused_ungated_globals<4096>::N,
        &ff_fused_ungated_globals<4096>::K);

    kittens::py::bind_function<dispatch_ff_fused_ungated_linear<4096>>(m, "dispatch_4096_no_activation",
        &ff_fused_ungated_globals<4096>::x,
        &ff_fused_ungated_globals<4096>::w1,
        &ff_fused_ungated_globals<4096>::w2,
        &ff_fused_ungated_globals<4096>::out,
        &ff_fused_ungated_globals<4096>::M,
        &ff_fused_ungated_globals<4096>::N,
        &ff_fused_ungated_globals<4096>::K);

    kittens::py::bind_function<dispatch_ff_fused_ungated_activated<8192>>(m, "dispatch_8192",
        &ff_fused_ungated_globals<8192>::x,
        &ff_fused_ungated_globals<8192>::w1,
        &ff_fused_ungated_globals<8192>::w2,
        &ff_fused_ungated_globals<8192>::out,
        &ff_fused_ungated_globals<8192>::M,
        &ff_fused_ungated_globals<8192>::N,
        &ff_fused_ungated_globals<8192>::K);

    kittens::py::bind_function<dispatch_ff_fused_ungated_linear<8192>>(m, "dispatch_8192_no_activation",
        &ff_fused_ungated_globals<8192>::x,
        &ff_fused_ungated_globals<8192>::w1,
        &ff_fused_ungated_globals<8192>::w2,
        &ff_fused_ungated_globals<8192>::out,
        &ff_fused_ungated_globals<8192>::M,
        &ff_fused_ungated_globals<8192>::N,
        &ff_fused_ungated_globals<8192>::K);
}
