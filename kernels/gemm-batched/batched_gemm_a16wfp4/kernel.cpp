// Batched BF16 x FP4 GEMM: C[b] = A_bf16[b] @ dequant(B_fp4[b])^T
//
// A: (B, M, K) bf16
// B: (B, K/2, N) uint8 (FP4 packed: 2 FP4 values per byte)
//    -> Pre-dequantized to (B, N, K) bf16 by host for mma_ABt
// b_scales: (B, N, K/32) uint8 (e8m0 format)
//    -> Folded into pre-dequantized B by host
// C: (B, M, N) bf16
//
// Strategy: B is pre-dequantized to bf16 by host (scales already applied).
// Kernel performs standard bf16 MMA with batch dimension.
// Ported from Triton batched_gemm_a16wfp4.py + wave1-patterns/gemm_a16wfp4_kernel.cpp

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 64;
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 4;
constexpr int REG_BLOCK_M = BLOCK_M / WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_N / WARPS_N;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct batched_gemm_a16wfp4_globals {
    _gl_bf16 a;    // (1, B, M, K) bf16
    _gl_bf16 b;    // (1, B, N, K) bf16 (pre-dequantized, transposed)
    _gl_bf16 c;    // (1, B, M, N) bf16
    hipStream_t stream;
    int M;
    int N;
    int K;
    int BATCH;

    dim3 grid()  {
        return dim3(
            BATCH,
            ceil_div(M, BLOCK_M) * ceil_div(N, BLOCK_N)
        );
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void batched_gemm_a16wfp4_kernel(const batched_gemm_a16wfp4_globals g,
                                  int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st<bf16, BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_B = st<bf16, BLOCK_N, BLOCK_K, st_16x32_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt<bf16, REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_tile;
    rt<bf16, REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s> B_tile;
    rt<float, REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum;
    zero(C_accum);

    // Batch and spatial indices
    const int batch_id = blockIdx.x;
    int wgid = blockIdx.y;
    const int NUM_WGS = gridDim.y;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);

    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    const int WGM = 8;
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int num_tiles = K / BLOCK_K;

    // Buffer descriptors with batch offset
    const bf16* a_base = (bf16*)&g.a[{0, batch_id, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, batch_id, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    i32x4 a_srsrc = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srsrc = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t a_lds[2], b_lds[2];
    for (int i = 0; i < 2; i++) {
        a_lds[i] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i].data[0]) + wid * elem_per_warp * sizeof(bf16)));
        b_lds[i] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&Bs[i].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    }

    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile_a = BLOCK_M * BLOCK_K * sizeof(bf16) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_b = BLOCK_N * BLOCK_K * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_A[memcpy_per_tile_a];
    uint32_t swizzled_B[memcpy_per_tile_b];
    G::prefill_swizzled_offsets(As[0], g.a, swizzled_A);
    G::prefill_swizzled_offsets(Bs[0], g.b, swizzled_B);

    // Load first tiles
    G::load(As[0], g.a, {0, batch_id, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
    G::load(Bs[0], g.b, {0, batch_id, pid_n, 0}, swizzled_B, b_srsrc, b_base, b_lds[0]);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // K-loop with double buffering
    for (int kt = 0; kt < num_tiles; kt++) {
        int buf = kt & 1;
        int next_buf = 1 - buf;

        if (kt + 1 < num_tiles) {
            G::load(As[next_buf], g.a, {0, batch_id, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
            G::load(Bs[next_buf], g.b, {0, batch_id, pid_n, kt + 1}, swizzled_B, b_srsrc, b_base, b_lds[next_buf]);
        }

        auto a_sub = subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[buf], {warp_row, 0});
        load(A_tile, a_sub);
        auto b_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[buf], {warp_col, 0});
        load(B_tile, b_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile, B_tile, C_accum);
        __builtin_amdgcn_s_setprio(0);

        if (kt + 1 < num_tiles) {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        __builtin_amdgcn_s_barrier();
    }

    // Store result (scales folded into pre-dequantized B)
    store(g.c, C_accum, {0, batch_id,
        pid_m * WARPS_M + warp_row,
        pid_n * WARPS_N + warp_col});
}

void dispatch_batched_gemm_a16wfp4(batched_gemm_a16wfp4_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)batched_gemm_a16wfp4_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    batched_gemm_a16wfp4_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M, g.N, g.K);
}

PYBIND11_MODULE(batched_gemm_a16wfp4_tk, m) {
    m.doc() = "HipKittens Batched BF16 x FP4 GEMM kernel";
    kittens::py::bind_function<dispatch_batched_gemm_a16wfp4>(m, "dispatch",
        &batched_gemm_a16wfp4_globals::a,
        &batched_gemm_a16wfp4_globals::b,
        &batched_gemm_a16wfp4_globals::c);
}
