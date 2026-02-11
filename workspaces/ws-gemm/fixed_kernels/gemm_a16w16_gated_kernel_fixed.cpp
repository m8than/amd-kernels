// gemm_a16w16_gated: Gated BF16 GEMM
// C = (A @ B0^T) * silu(A @ B1^T)
// where B is (N, K) with N being the concatenation of B0 (N/2 cols) and B1 (N/2 cols).
// A (M, K) bf16, B (N, K) bf16, C (M, N/2) bf16
//
// Each thread block computes two matmuls in parallel:
//   acc0 = A @ B0^T  (left half of B)
//   acc1 = A @ B1^T  (right half of B)
// Then applies gating: C = acc0 * silu(acc1)
// where silu(x) = x * sigmoid(x)

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M       = 128;
constexpr int BLOCK_N_HALF  = 64;   // Each gate half
constexpr int BLOCK_K       = 64;
constexpr int WARPS_M       = 2;
constexpr int WARPS_N       = 4;
constexpr int REG_BLOCK_M   = BLOCK_M / WARPS_M;   // 64
constexpr int REG_BLOCK_N   = BLOCK_N_HALF / WARPS_N; // 16

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct gemm_gated_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
    int M;
    int N;  // Full N (B0 + B1)
    int K;
    dim3 grid() {
        int half_n = N / 2;
        return dim3(ceil_div(half_n, BLOCK_N_HALF) * ceil_div(M, BLOCK_M));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void gemm_a16w16_gated_kernel(const gemm_gated_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    const int half_n = N / 2;

    // Shared memory for A and both halves of B
    using ST_A = st_bf<BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_B = st_bf<BLOCK_N_HALF, BLOCK_K, st_16x32_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();  // double buffer for A
    ST_B (&B0s)[2] = al.allocate<ST_B, 2>(); // double buffer for B0 (gate path)
    ST_B (&B1s)[2] = al.allocate<ST_B, 2>(); // double buffer for B1 (value path)

    // Register tiles
    rt_bf<REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_tile;
    rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s> B0_tile, B1_tile;

    // Accumulators for both gate paths
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> acc0; // value path
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> acc1; // gate path
    zero(acc0);
    zero(acc1);

    // Chiplet-aware scheduling
    int wgid = blockIdx.x;
    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(half_n, BLOCK_N_HALF);
    wgid = chiplet_transform_chunked(wgid, num_pid_m * num_pid_n, NUM_XCDS, 64);

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

    // Address setup
    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    i32x4 a_srsrc = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srsrc = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t a_lds[2], b0_lds[2], b1_lds[2];
    for (int i = 0; i < 2; i++) {
        a_lds[i] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&As[i].data[0]) + wid * elem_per_warp * sizeof(bf16)));
        b0_lds[i] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&B0s[i].data[0]) + wid * elem_per_warp * sizeof(bf16)));
        b1_lds[i] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&B1s[i].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    }

    constexpr int bytes_per_thread_a = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy_a = bytes_per_thread_a * NUM_THREADS;
    constexpr int memcpy_per_tile_a = BLOCK_M * BLOCK_K * sizeof(bf16) / bytes_per_memcpy_a;
    uint32_t swizzled_A[memcpy_per_tile_a];
    G::prefill_swizzled_offsets(As[0], g.a, swizzled_A);

    constexpr int bytes_per_thread_b = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy_b = bytes_per_thread_b * NUM_THREADS;
    constexpr int memcpy_per_tile_b = BLOCK_N_HALF * BLOCK_K * sizeof(bf16) / bytes_per_memcpy_b;
    uint32_t swizzled_B[memcpy_per_tile_b];
    G::prefill_swizzled_offsets(B0s[0], g.b, swizzled_B);

    // B0 rows = pid_n * BLOCK_N_HALF ... (left half of B)
    // B1 rows = pid_n * BLOCK_N_HALF + half_n ... (right half of B, offset by half_n)
    int b0_row = pid_n;
    int b1_row_offset = half_n / BLOCK_N_HALF;  // Row offset for right half

    // Load first tiles
    G::load(As[0], g.a, {0, 0, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
    G::load(B0s[0], g.b, {0, 0, b0_row, 0}, swizzled_B, b_srsrc, b_base, b0_lds[0]);
    G::load(B1s[0], g.b, {0, 0, b0_row + b1_row_offset, 0}, swizzled_B, b_srsrc, b_base, b1_lds[0]);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // K-loop
    for (int kt = 0; kt < num_tiles; kt++) {
        int buf = kt & 1;
        int next_buf = 1 - buf;

        // Prefetch next tile
        if (kt + 1 < num_tiles) {
            G::load(As[next_buf], g.a, {0, 0, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
            G::load(B0s[next_buf], g.b, {0, 0, b0_row, kt + 1}, swizzled_B, b_srsrc, b_base, b0_lds[next_buf]);
            G::load(B1s[next_buf], g.b, {0, 0, b0_row + b1_row_offset, kt + 1}, swizzled_B, b_srsrc, b_base, b1_lds[next_buf]);
        }

        // Load A subtile for this warp
        auto a_sub = subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[buf], {warp_row, 0});
        load(A_tile, a_sub);

        // Load B0 subtile and compute acc0 += A @ B0^T
        auto b0_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(B0s[buf], {warp_col, 0});
        load(B0_tile, b0_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc0, A_tile, B0_tile, acc0);
        __builtin_amdgcn_s_setprio(0);

        // Load B1 subtile and compute acc1 += A @ B1^T
        auto b1_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(B1s[buf], {warp_col, 0});
        load(B1_tile, b1_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(acc1, A_tile, B1_tile, acc1);
        __builtin_amdgcn_s_setprio(0);

        if (kt + 1 < num_tiles) {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        __builtin_amdgcn_s_barrier();
    }

    // Apply gating: C = acc0 * silu(acc1)
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // Work element-wise on the accumulator tiles
    constexpr int PT = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s>::packed_per_thread;
    constexpr int HT = acc0.height;
    constexpr int WT = acc0.width;

    #pragma unroll
    for (int i = 0; i < HT; i++) {
        #pragma unroll
        for (int j = 0; j < WT; j++) {
            #pragma unroll
            for (int p = 0; p < PT; p++) {
                // data[p] is float2 (packed type); process .x and .y separately
                float v0x = acc0.tiles[i][j].data[p].x;
                float v1x = acc1.tiles[i][j].data[p].x;
                acc0.tiles[i][j].data[p].x = v0x * silu(v1x);

                float v0y = acc0.tiles[i][j].data[p].y;
                float v1y = acc1.tiles[i][j].data[p].y;
                acc0.tiles[i][j].data[p].y = v0y * silu(v1y);
            }
        }
    }

    // Store result
    store(g.c, acc0, {0, 0,
        pid_m * WARPS_M + warp_row,
        pid_n * WARPS_N + warp_col});
}

void dispatch_gemm_a16w16_gated(gemm_gated_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)gemm_a16w16_gated_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    gemm_a16w16_gated_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

PYBIND11_MODULE(gemm_a16w16_gated_tk, m) {
    m.doc() = "HipKittens Gated BF16 GEMM kernel";
    kittens::py::bind_function<dispatch_gemm_a16w16_gated>(m, "dispatch",
        &gemm_gated_globals::a, &gemm_gated_globals::b, &gemm_gated_globals::c);
}
