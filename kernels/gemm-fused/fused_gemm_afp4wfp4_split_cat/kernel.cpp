// fused_gemm_afp4wfp4_split_cat: FP4 GEMM with split-concatenate epilogue
//
// GEMM: result = fp4_gemm(A, B, a_scale, b_scale)  shape (M, N) where N = D*(S1+S2)
//
// Split-cat epilogue:
//   c1[:, d, :S1] = result columns for S1 per D-group
//   c2[:, d, :S2] = result columns for S2 per D-group
//   c1[:, d, S1:S1+S3] = y[:, d, :S3]  (concatenation)
//
// FP4 data pre-dequantized to BF16 by host. Scales applied per K-block.
//
// Ported from: reference/triton/fused/fused_gemm_afp4wfp4_split_cat.py

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 64;
constexpr int SCALE_GROUP = 32;
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 4;
constexpr int REG_BLOCK_M = BLOCK_M / WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_N / WARPS_N;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_f32  = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct fused_gemm_afp4wfp4_split_cat_globals {
    _gl_bf16 a;           // A (M, K) bf16 (from FP4)
    _gl_bf16 b;           // B (N, K) bf16 (from FP4)
    _gl_f32  a_scale;     // (M, K/SCALE_GROUP) float32
    _gl_f32  b_scale;     // (N, K/SCALE_GROUP) float32
    _gl_bf16 y;           // Y tensor for concatenation (M, D, S3) bf16
    _gl_bf16 c1;          // Output 1 (M, D, S1+S3) bf16
    _gl_bf16 c2;          // Output 2 (M, D, S2) bf16
    hipStream_t stream;
    int M, N, K, D, S1, S2, S3;

    dim3 grid()  { return dim3(ceil_div(N, BLOCK_N) * ceil_div(M, BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fused_gemm_afp4wfp4_split_cat_kernel(const fused_gemm_afp4wfp4_split_cat_globals g,
                                             int M, int N, int K,
                                             int D, int S1, int S2, int S3) {
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
    const int num_pid_n = ceil_div(N, BLOCK_N);
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

    G::load(As[0], g.a, {0, 0, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
    G::load(Bs[0], g.b, {0, 0, pid_n, 0}, swizzled_B, b_srsrc, b_base, b_lds[0]);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // K-loop with per-block scale application
    for (int kt = 0; kt < num_tiles; kt++) {
        int buf = kt & 1;
        int next_buf = 1 - buf;

        if (kt + 1 < num_tiles) {
            G::load(As[next_buf], g.a, {0, 0, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
            G::load(Bs[next_buf], g.b, {0, 0, pid_n, kt + 1}, swizzled_B, b_srsrc, b_base, b_lds[next_buf]);
        }

        zero(block_result);
        auto a_sub = subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[buf], {warp_row, 0});
        load(A_tile, a_sub);
        auto b_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[buf], {warp_col, 0});
        load(B_tile, b_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(block_result, A_tile, B_tile, block_result);
        __builtin_amdgcn_s_setprio(0);

        int scale_k_idx = kt * (BLOCK_K / SCALE_GROUP);
        rv_naive<float,REG_BLOCK_M> a_sc;
        rv_naive<float,REG_BLOCK_N> b_sc;
        load(a_sc, g.a_scale, {0, 0, pid_m * BLOCK_M + warp_row * REG_BLOCK_M, scale_k_idx});
        load(b_sc, g.b_scale, {0, 0, pid_n * BLOCK_N + warp_col * REG_BLOCK_N, scale_k_idx});
        asm volatile("s_waitcnt vmcnt(0)");

        mul_row(block_result, block_result, a_sc);
        mul_col(block_result, block_result, b_sc);
        add(C_accum, C_accum, block_result);

        if (kt + 1 < num_tiles) {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        __builtin_amdgcn_s_barrier();
    }

    // ===== FUSED SPLIT-CAT EPILOGUE =====
    // Convert accumulator to bf16 and store to shared for scatter
    using ST_Out = st<bf16,REG_BLOCK_M, REG_BLOCK_N>;
    ST_Out& scratch = *reinterpret_cast<ST_Out*>(&As[0]);

    rt<bf16,REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_bf16;
    copy(C_bf16, C_accum);
    store(scratch, C_bf16);
    __builtin_amdgcn_s_barrier();

    // Scatter from shared to global: split into c1 and c2
    const int lane_id = threadIdx.x % kittens::WARP_THREADS;
    const int m_base = pid_m * BLOCK_M + warp_row * REG_BLOCK_M;
    const int n_base = pid_n * BLOCK_N + warp_col * REG_BLOCK_N;
    const int stride_per_s = S1 + S2;

    const int total_elems = REG_BLOCK_M * REG_BLOCK_N;
    const int elems_per_thread = total_elems / kittens::WARP_THREADS;

    for (int e = 0; e < elems_per_thread; e++) {
        int flat_idx = lane_id * elems_per_thread + e;
        int local_m = flat_idx / REG_BLOCK_N;
        int local_n = flat_idx % REG_BLOCK_N;
        int global_m = m_base + local_m;
        int global_n = n_base + local_n;

        if (global_m < M && global_n < N) {
            bf16 val = scratch.data[flat_idx];

            int d = global_n / stride_per_s;
            int s = global_n % stride_per_s;

            if (d < D) {
                if (s < S1) {
                    bf16* c1_ptr = (bf16*)&g.c1[{0, 0, 0, 0}];
                    int c1_stride_m = g.c1.template stride<2>();
                    int c1_stride_d = g.c1.template stride<3>();
                    c1_ptr[global_m * c1_stride_m + d * c1_stride_d + s] = val;
                } else if (s < S1 + S2) {
                    bf16* c2_ptr = (bf16*)&g.c2[{0, 0, 0, 0}];
                    int c2_stride_m = g.c2.template stride<2>();
                    int c2_stride_d = g.c2.template stride<3>();
                    c2_ptr[global_m * c2_stride_m + d * c2_stride_d + (s - S1)] = val;
                }
            }
        }
    }

    // Copy y values and concatenate to c1
    for (int e = 0; e < elems_per_thread; e++) {
        int flat_idx = lane_id * elems_per_thread + e;
        int local_m = flat_idx / REG_BLOCK_N;
        int local_n = flat_idx % REG_BLOCK_N;
        int global_m = m_base + local_m;

        if (local_n < S3 && global_m < M) {
            int d_idx = n_base / stride_per_s;
            if (d_idx < D) {
                bf16* y_ptr = (bf16*)&g.y[{0, 0, 0, 0}];
                int y_stride_m = g.y.template stride<2>();
                int y_stride_d = g.y.template stride<3>();
                bf16 y_val = y_ptr[global_m * y_stride_m + d_idx * y_stride_d + local_n];

                bf16* c1_ptr = (bf16*)&g.c1[{0, 0, 0, 0}];
                int c1_stride_m = g.c1.template stride<2>();
                int c1_stride_d = g.c1.template stride<3>();
                c1_ptr[global_m * c1_stride_m + d_idx * c1_stride_d + S1 + local_n] = y_val;
            }
        }
    }
}

void dispatch_fused_gemm_afp4wfp4_split_cat(fused_gemm_afp4wfp4_split_cat_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fused_gemm_afp4wfp4_split_cat_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fused_gemm_afp4wfp4_split_cat_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M, g.N, g.K, g.D, g.S1, g.S2, g.S3);
}
