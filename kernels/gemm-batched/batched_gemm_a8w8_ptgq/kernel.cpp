// Batched INT8 GEMM with per-token-group pre-quantization and per-batch-tensor weight scale
//
// C[b] = (quant_per_block(A_bf16[b]) @ B_int8[b]) * a_block_scale * b_scale[b] + bias[b]
//
// A: (B, M, K) bf16 — dynamically quantized per K-block per row
// B: (B, K, N) int8 -> pre-cast to bf16 by host (B transposed to (B, N, K) for mma_ABt)
// b_scale: scalar float32 — per-batch-tensor weight scale
// C: (B, M, N) bf16
// bias: (B, 1, N) bf16 (optional)
//
// Per-token-group quantization within K-loop:
//   For each K-block of A:
//     1. m = max(abs(A_block), axis=K_dim) per row  (row-wise max)
//     2. a_scale = m / DTYPE_MAX
//     3. A_quant = clamp(A_block / a_scale, DTYPE_MIN, DTYPE_MAX)
//     4. accumulator += (A_quant @ B_block) * a_scale
//   After K-loop: accumulator *= b_scale
//
// In HipKittens, we approximate this by doing the scale computation in fp32
// on the register tiles and applying it as a row-wise multiply after each MMA.
//
// Ported from Triton batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py

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
constexpr float DTYPE_MAX = 127.0f;
constexpr float DTYPE_MIN = -128.0f;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_bf16  = gl<bf16, -1, -1, -1, -1>;
using _gl_float = gl<float, -1, -1, -1, -1>;
using _gl_bias  = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct batched_gemm_a8w8_ptgq_globals {
    _gl_bf16  a;       // (1, B, M, K) bf16
    _gl_bf16  b;       // (1, B, N, K) bf16 (pre-cast from int8, transposed)
    _gl_float b_scale; // (1, 1, 1, 1) float32 — scalar per-tensor scale
    _gl_bf16  c;       // (1, B, M, N) bf16
    _gl_bias  bias;    // (1, B, 1, N) bf16
    hipStream_t stream;
    int M;
    int N;
    int K;
    int BATCH;
    int has_bias;

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
void batched_gemm_a8w8_ptgq_kernel(const batched_gemm_a8w8_ptgq_globals g,
                                     int M, int N, int K, int has_bias) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st<bf16,BLOCK_M, BLOCK_K, st_16x32_s>;
    using ST_B = st<bf16,BLOCK_N, BLOCK_K, st_16x32_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt<bf16,REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_tile;
    rt<bf16,REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s> B_tile;
    rt<float,REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum;
    zero(C_accum);

    // Temporary fp32 tile for A processing
    rt<float,REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_fl;

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

    // Load b_scale (scalar)
    const float b_scale = *(const float*)&g.b_scale[{0, 0, 0, 0}];

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

    // Precompute reciprocal of DTYPE_MAX
    const float one_over_dtype_max = 1.0f / DTYPE_MAX;

    // Load first tiles
    G::load(As[0], g.a, {0, batch_id, pid_m, 0}, swizzled_A, a_srsrc, a_base, a_lds[0]);
    G::load(Bs[0], g.b, {0, batch_id, pid_n, 0}, swizzled_B, b_srsrc, b_base, b_lds[0]);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // K-loop with double buffering and per-block A quantization
    for (int kt = 0; kt < num_tiles; kt++) {
        int buf = kt & 1;
        int next_buf = 1 - buf;

        if (kt + 1 < num_tiles) {
            G::load(As[next_buf], g.a, {0, batch_id, pid_m, kt + 1}, swizzled_A, a_srsrc, a_base, a_lds[next_buf]);
            G::load(Bs[next_buf], g.b, {0, batch_id, pid_n, kt + 1}, swizzled_B, b_srsrc, b_base, b_lds[next_buf]);
        }

        // Load A tile (bf16) from shared -> registers
        auto a_sub = subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[buf], {warp_row, 0});
        load(A_tile, a_sub);

        // Load B tile
        auto b_sub = subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[buf], {warp_col, 0});
        load(B_tile, b_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Per-token-group quantization of A:
        // 1. Convert A to fp32
        copy(A_fl, A_tile);

        // 2. Compute per-row absolute max
        rt<float,REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s> A_abs;
        abs(A_abs, A_fl);
        rv_naive<float,REG_BLOCK_M> row_max;
        row_max_reduce(row_max, A_abs);

        // 3. Compute a_scale = max(row_max, 1e-10) * one_over_dtype_max
        //    and scale_recip = 1.0 / a_scale
        rv_naive<float,REG_BLOCK_M> a_scale;
        rv_naive<float,REG_BLOCK_M> scale_recip;
        // Clamp minimum to avoid division by zero
        max(a_scale, row_max, 1e-10f);
        mul(a_scale, a_scale, one_over_dtype_max);
        div(scale_recip, 1.0f, a_scale);

        // 4. Scale A: A_quant = clamp(A * scale_recip, DTYPE_MIN, DTYPE_MAX)
        mul_row(A_fl, A_fl, scale_recip);
        // Clamp to INT8 range
        clamp(A_fl, A_fl, DTYPE_MIN, DTYPE_MAX);

        // Convert back to bf16 for MMA
        copy(A_tile, A_fl);

        // 5. MMA: partial = A_quant @ B^T
        rt<float,REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> partial;
        zero(partial);
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial, A_tile, B_tile, partial);
        __builtin_amdgcn_s_setprio(0);

        // 6. Apply per-row a_scale to partial result: partial *= a_scale[:, None]
        mul_row(partial, partial, a_scale);

        // 7. Accumulate
        add(C_accum, C_accum, partial);

        if (kt + 1 < num_tiles) {
            asm volatile("s_waitcnt vmcnt(0)");
        }
        __builtin_amdgcn_s_barrier();
    }

    // Apply per-tensor b_scale
    mul(C_accum, C_accum, b_scale);

    // Add bias if present
    if (has_bias) {
        rv_naive<bf16,REG_BLOCK_N> bias_vec;
        load(bias_vec, g.bias, {0, batch_id, 0, pid_n * BLOCK_N + warp_col * REG_BLOCK_N});
        asm volatile("s_waitcnt vmcnt(0)");
        rv_naive<float,REG_BLOCK_N> bias_fl;
        copy(bias_fl, bias_vec);
        add_col(C_accum, C_accum, bias_fl);
    }

    // Store result
    store(g.c, C_accum, {0, batch_id,
        pid_m * WARPS_M + warp_row,
        pid_n * WARPS_N + warp_col});
}

void dispatch_batched_gemm_a8w8_ptgq(batched_gemm_a8w8_ptgq_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)batched_gemm_a8w8_ptgq_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    batched_gemm_a8w8_ptgq_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M, g.N, g.K, g.has_bias);
}

PYBIND11_MODULE(tk_batched_gemm_a8w8_ptgq, m) {
    m.doc() = "HipKittens Batched INT8 GEMM with per-token-group quant";
    py::bind_function<dispatch_batched_gemm_a8w8_ptgq>(m, "dispatch",
        &batched_gemm_a8w8_ptgq_globals::a,
        &batched_gemm_a8w8_ptgq_globals::b,
        &batched_gemm_a8w8_ptgq_globals::b_scale,
        &batched_gemm_a8w8_ptgq_globals::c,
        &batched_gemm_a8w8_ptgq_globals::bias);
}
