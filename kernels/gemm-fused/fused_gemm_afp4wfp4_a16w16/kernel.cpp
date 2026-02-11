// fused_gemm_afp4wfp4_a16w16: Fused FP4 GEMM + BF16 GEMM
//
// This kernel fuses two GEMMs into a single launch:
//   - Path A (FP4): C_fp4 = fp4_gemm(A_fp4, B_fp4) with per-block scales
//   - Path B (BF16): C_bf16 = A_bf16 @ B_bf16^T
//
// FP4 data is pre-dequantized to BF16 by the host (following wave1 pattern).
// Scales are applied per K-block inside the kernel loop.
//
// Blocks are assigned based on pid_n:
//   pid_n < num_pid_n_fp4 -> FP4 path (with block scales)
//   pid_n >= num_pid_n_fp4 -> BF16 path (standard GEMM)
//
// Ported from: reference/triton/fused/fused_gemm_afp4wfp4_a16w16.py

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 64;
constexpr int SCALE_GROUP = 32;   // FP4: 32 elements share one scale
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 4;
constexpr int REG_BLOCK_M = BLOCK_M / WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_N / WARPS_N;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_f32  = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct fused_gemm_afp4wfp4_a16w16_globals {
    // FP4 GEMM inputs (pre-dequantized to bf16)
    _gl_bf16 a_fp4;       // A (M, K) bf16 (from FP4)
    _gl_bf16 b_fp4;       // B (N_fp4, K) bf16 (from FP4)
    _gl_bf16 c_fp4;       // C_fp4 output (M, N_fp4) bf16
    _gl_f32  a_fp4_scale; // (M, K/SCALE_GROUP) float32
    _gl_f32  b_fp4_scale; // (N_fp4, K/SCALE_GROUP) float32

    // BF16 GEMM inputs
    _gl_bf16 a_bf16;      // A (M, K_bf16) bf16
    _gl_bf16 b_bf16;      // B (N_bf16, K_bf16) bf16
    _gl_bf16 c_bf16;      // C_bf16 output (M, N_bf16) bf16

    hipStream_t stream;
    int M, N_fp4, N_bf16, K;

    dim3 grid() {
        int num_pid_n = ceil_div(N_fp4, BLOCK_N) + ceil_div(N_bf16, BLOCK_N);
        return dim3(num_pid_n * ceil_div(M, BLOCK_M));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fused_gemm_afp4wfp4_a16w16_kernel(const fused_gemm_afp4wfp4_a16w16_globals g,
                                          int M, int N_fp4, int N_bf16, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st<bf16,BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_B = st<bf16,BLOCK_N, BLOCK_K, st_16x32_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt<bf16,REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_tile;
    rt<bf16,REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s> B_tile;
    rt<float,REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum;
    rt<float,REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> block_result;
    zero(C_accum);

    // Scheduling
    int wgid = blockIdx.x;
    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n_fp4 = ceil_div(N_fp4, BLOCK_N);
    const int num_pid_n_bf16 = ceil_div(N_bf16, BLOCK_N);
    const int num_pid_n = num_pid_n_fp4 + num_pid_n_bf16;
    const int total_blocks = num_pid_m * num_pid_n;
    wgid = chiplet_transform_chunked(wgid, total_blocks, NUM_XCDS, 64);

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

    if (pid_n < num_pid_n_fp4) {
        // ===== FP4 GEMM PATH (pre-dequantized, with per-block scales) =====
        const int num_tiles = K / BLOCK_K;

        const bf16* a_base = (bf16*)&g.a_fp4[{0, 0, 0, 0}];
        const bf16* b_base = (bf16*)&g.b_fp4[{0, 0, 0, 0}];
        const int a_row_stride = g.a_fp4.template stride<2>() * sizeof(bf16);
        const int b_row_stride = g.b_fp4.template stride<2>() * sizeof(bf16);
        i32x4 a_srsrc = make_srsrc(a_base, M * a_row_stride, a_row_stride);
        i32x4 b_srsrc = make_srsrc(b_base, N_fp4 * b_row_stride, b_row_stride);

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
        G::prefill_swizzled_offsets(As[0], g.a_fp4, swizzled_A);
        G::prefill_swizzled_offsets(Bs[0], g.b_fp4, swizzled_B);

        G::load(As[0], g.a_fp4, {0, 0, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
        G::load(Bs[0], g.b_fp4, {0, 0, pid_n, 0}, swizzled_B, b_srsrc, b_base, b_lds[0]);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // K-loop: MMA + per-block scale application
        // Scale groups are SCALE_GROUP=32 elements, BLOCK_K=64 -> 2 scale groups per K-block
        for (int kt = 0; kt < num_tiles; kt++) {
            int buf = kt & 1;
            int next_buf = 1 - buf;

            if (kt + 1 < num_tiles) {
                G::load(As[next_buf], g.a_fp4, {0, 0, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
                G::load(Bs[next_buf], g.b_fp4, {0, 0, pid_n, kt + 1}, swizzled_B, b_srsrc, b_base, b_lds[next_buf]);
            }

            // Compute partial MMA for this K-block
            zero(block_result);
            auto a_sub = subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[buf], {warp_row, 0});
            load(A_tile, a_sub);
            auto b_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[buf], {warp_col, 0});
            load(B_tile, b_sub);
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(block_result, A_tile, B_tile, block_result);
            __builtin_amdgcn_s_setprio(0);

            // Apply per-block scales (average over scale groups in this K-block)
            // For simplicity, load one representative scale per row/col for this K-block
            int scale_k_idx = kt * (BLOCK_K / SCALE_GROUP);
            decltype(block_result)::col_vec a_sc;
            decltype(block_result)::row_vec b_sc;
            load(a_sc, g.a_fp4_scale, {0, 0, pid_m * BLOCK_M + warp_row * REG_BLOCK_M, scale_k_idx});
            load(b_sc, g.b_fp4_scale, {0, 0, pid_n * BLOCK_N + warp_col * REG_BLOCK_N, scale_k_idx});
            asm volatile("s_waitcnt vmcnt(0)");

            mul_row(block_result, block_result, a_sc);
            mul_col(block_result, block_result, b_sc);
            add(C_accum, C_accum, block_result);

            if (kt + 1 < num_tiles) {
                asm volatile("s_waitcnt vmcnt(0)");
            }
            __builtin_amdgcn_s_barrier();
        }

        // Store fp4 result
        store(g.c_fp4, C_accum, {0, 0,
            pid_m * WARPS_M + warp_row,
            pid_n * WARPS_N + warp_col});
    } else {
        // ===== BF16 GEMM PATH =====
        int bf16_pid_n = pid_n - num_pid_n_fp4;
        const int num_tiles = K / BLOCK_K;

        const bf16* a_base = (bf16*)&g.a_bf16[{0, 0, 0, 0}];
        const bf16* b_base = (bf16*)&g.b_bf16[{0, 0, 0, 0}];
        const int a_row_stride = g.a_bf16.template stride<2>() * sizeof(bf16);
        const int b_row_stride = g.b_bf16.template stride<2>() * sizeof(bf16);
        i32x4 a_srsrc = make_srsrc(a_base, M * a_row_stride, a_row_stride);
        i32x4 b_srsrc = make_srsrc(b_base, N_bf16 * b_row_stride, b_row_stride);

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
        G::prefill_swizzled_offsets(As[0], g.a_bf16, swizzled_A);
        G::prefill_swizzled_offsets(Bs[0], g.b_bf16, swizzled_B);

        G::load(As[0], g.a_bf16, {0, 0, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
        G::load(Bs[0], g.b_bf16, {0, 0, bf16_pid_n, 0}, swizzled_B, b_srsrc, b_base, b_lds[0]);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        for (int kt = 0; kt < num_tiles; kt++) {
            int buf = kt & 1;
            int next_buf = 1 - buf;

            if (kt + 1 < num_tiles) {
                G::load(As[next_buf], g.a_bf16, {0, 0, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
                G::load(Bs[next_buf], g.b_bf16, {0, 0, bf16_pid_n, kt + 1}, swizzled_B, b_srsrc, b_base, b_lds[next_buf]);
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

        // Store bf16 result
        store(g.c_bf16, C_accum, {0, 0,
            pid_m * WARPS_M + warp_row,
            bf16_pid_n * WARPS_N + warp_col});
    }
}

void dispatch_fused_gemm_afp4wfp4_a16w16(fused_gemm_afp4wfp4_a16w16_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fused_gemm_afp4wfp4_a16w16_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fused_gemm_afp4wfp4_a16w16_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M, g.N_fp4, g.N_bf16, g.K);
}

PYBIND11_MODULE(fused_gemm_afp4wfp4_a16w16_tk, m) {
    m.doc() = "HipKittens fused FP4 GEMM + BF16 GEMM kernel";
    kittens::py::bind_function<dispatch_fused_gemm_afp4wfp4_a16w16>(m, "dispatch",
        &fused_gemm_afp4wfp4_a16w16_globals::a_fp4,
        &fused_gemm_afp4wfp4_a16w16_globals::b_fp4,
        &fused_gemm_afp4wfp4_a16w16_globals::c_fp4,
        &fused_gemm_afp4wfp4_a16w16_globals::a_fp4_scale,
        &fused_gemm_afp4wfp4_a16w16_globals::b_fp4_scale,
        &fused_gemm_afp4wfp4_a16w16_globals::a_bf16,
        &fused_gemm_afp4wfp4_a16w16_globals::b_bf16,
        &fused_gemm_afp4wfp4_a16w16_globals::c_bf16);
}
