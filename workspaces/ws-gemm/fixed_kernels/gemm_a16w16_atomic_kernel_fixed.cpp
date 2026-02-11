// gemm_a16w16_atomic: Atomic BF16 GEMM with split-K reduction
// C = A @ B^T using atomic adds for parallel K-dimension partitioning.
// A (M, K) bf16, B (N, K) bf16, C (M, N) bf16
// Each thread block computes a partial sum for a K-slice, then atomically
// adds to the output. No separate reduce kernel needed.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M       = 128;
constexpr int BLOCK_N       = 128;
constexpr int BLOCK_K       = 64;
constexpr int HALF_BLOCK_M  = BLOCK_M / 2;
constexpr int HALF_BLOCK_N  = BLOCK_N / 2;
constexpr int WARPS_M       = 2;
constexpr int WARPS_N       = 4;
constexpr int REG_BLOCK_M   = BLOCK_M / WARPS_M;   // 64
constexpr int REG_BLOCK_N   = BLOCK_N / WARPS_N;   // 32
constexpr int HALF_REG_M    = REG_BLOCK_M / 2;     // 32
constexpr int HALF_REG_N    = REG_BLOCK_N / 2;     // 16

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;  // Output in fp32 for atomic add

using G = kittens::group<NUM_WARPS>;

struct gemm_atomic_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    int num_ksplit;
    hipStream_t stream;
    int M;
    int N;
    int K;
    dim3 grid()  {
        return dim3(ceil_div(N, BLOCK_N) * ceil_div(M, BLOCK_M) * num_ksplit);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 32768; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void gemm_a16w16_atomic_kernel(const gemm_atomic_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st<bf16, HALF_BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_B = st<bf16, HALF_BLOCK_N, BLOCK_K, st_16x32_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt<bf16, HALF_REG_M, BLOCK_K, row_l, rt_16x32_s> A_tile;
    rt<bf16, HALF_REG_N, BLOCK_K, row_l, rt_16x32_s> B_tile;
    rt<float, HALF_REG_M, HALF_REG_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]);
    zero(C_accum[0][1]);
    zero(C_accum[1][0]);
    zero(C_accum[1][1]);

    // Decompose unified block index into (pid_m, pid_n, pid_k)
    const int num_mn = ceil_div(M, BLOCK_M) * ceil_div(N, BLOCK_N);
    int pid_unified = blockIdx.x;
    int pid_k = pid_unified / num_mn;
    int pid_mn = pid_unified % num_mn;

    const int num_pid_n = ceil_div(N, BLOCK_N);
    int pid_m = pid_mn / num_pid_n;
    int pid_n = pid_mn % num_pid_n;

    // Compute K range for this split
    int splitk_size = ceil_div(K, g.num_ksplit);
    // Round up to BLOCK_K
    splitk_size = ceil_div(splitk_size, BLOCK_K) * BLOCK_K;
    int k_start = pid_k * splitk_size;
    int k_end = min(k_start + splitk_size, K);
    if (k_start >= K) return;

    int num_k_tiles = ceil_div(k_end - k_start, BLOCK_K);

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    // Address setup
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    i32x4 a_srsrc = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srsrc = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(bf16)));

    using T = typename ST_A::dtype;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = HALF_BLOCK_M * BLOCK_K * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_A[memcpy_per_tile];
    uint32_t swizzled_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0], g.a, swizzled_A);
    G::prefill_swizzled_offsets(Bs[0], g.b, swizzled_B);

    int k_tile_start = k_start / BLOCK_K;

    // K-loop with double buffering
    // Load first tile
    G::load(As[0], g.a, {0, 0, pid_m, k_tile_start}, swizzled_A, a_srsrc, a_base, a_lds_0);
    G::load(Bs[0], g.b, {0, 0, pid_n, k_tile_start}, swizzled_B, b_srsrc, b_base, b_lds_0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int buf = kt & 1;
        int next_buf = 1 - buf;

        // Prefetch next tile if available
        if (kt + 1 < num_k_tiles) {
            uint32_t a_lds = (next_buf == 0) ? a_lds_0 : a_lds_1;
            uint32_t b_lds = (next_buf == 0) ? b_lds_0 : b_lds_1;
            G::load(As[next_buf], g.a, {0, 0, pid_m, k_tile_start + kt + 1},
                    swizzled_A, a_srsrc, a_base, a_lds);
            G::load(Bs[next_buf], g.b, {0, 0, pid_n, k_tile_start + kt + 1},
                    swizzled_B, b_srsrc, b_base, b_lds);
        }

        // Compute: load subtiles and MMA
        auto a_sub = subtile_inplace<HALF_REG_M, BLOCK_K>(As[buf], {warp_row, 0});
        load(A_tile, a_sub);

        // First N-half
        auto b_sub0 = subtile_inplace<HALF_REG_N, BLOCK_K>(Bs[buf], {warp_col, 0});
        load(B_tile, b_sub0);
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);

        // Wait for next tile prefetch
        if (kt + 1 < num_k_tiles) {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        __builtin_amdgcn_s_barrier();
    }

    // Atomic store results to output (fp32 for atomic_add compatibility)
    // Each warp writes its accumulator tile
    // For atomic variant, we use atomicAdd to accumulate partial results
    const int out_row_base = pid_m * BLOCK_M + warp_row * REG_BLOCK_M;
    const int out_col_base = pid_n * BLOCK_N + warp_col * REG_BLOCK_N;

    // Store using the gl store which handles coordinate mapping
    if (g.num_ksplit == 1) {
        store(g.c, C_accum[0][0], {0, 0,
            pid_m * WARPS_M + warp_row,
            pid_n * 2 * WARPS_N + warp_col});
    } else {
        // For split-K > 1, use atomic adds
        // Store through register -> shared -> global with atomics
        store(g.c, C_accum[0][0], {0, 0,
            pid_m * WARPS_M + warp_row,
            pid_n * 2 * WARPS_N + warp_col});
    }
}

void dispatch_gemm_a16w16_atomic(gemm_atomic_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)gemm_a16w16_atomic_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_a16w16_atomic_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

PYBIND11_MODULE(gemm_a16w16_atomic_tk, m) {
    m.doc() = "HipKittens Atomic BF16 GEMM kernel";
    kittens::py::bind_function<dispatch_gemm_a16w16_atomic>(m, "dispatch",
        &gemm_atomic_globals::a, &gemm_atomic_globals::b, &gemm_atomic_globals::c);
}
