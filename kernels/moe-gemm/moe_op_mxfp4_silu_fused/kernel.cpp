// MoE MXFP4 GEMM + Fused SiLU Kernel (moe_op_mxfp4_silu_fused)
// Ported from reference/triton/moe_op_mxfp4_silu_fused.py
//
// Microscaled FP4 (MXFP4) MoE GEMM with fused SwiGLU activation:
//   - A: FP4 packed as uint8 (2 values per byte), with MX microscales
//   - B: FP4 packed as uint8 (2 values per byte), with MX microscales
//       B has interleaved gate/up columns: [gate_0, up_0, gate_1, up_1, ...]
//   - Output: SiLU(gate) * up, shape [num_padded, N/2] bf16
//
// The B weight matrix has N columns where the first N/2 are "gate" weights
// and the second N/2 are "up" weights, interleaved pairwise.
// After GEMM, the kernel applies: output[n] = SiLU(gate[n]) * up[n]
//
// Uses sorted-token routing pattern (sorted_token_ids, expert_ids).

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;   // Full N block (gate+up interleaved)
constexpr int BLOCK_K = 128;   // In FP4 elements
constexpr int MX_GROUP_SIZE = 32;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// E2M1 FP4 lookup table
__device__ __constant__ float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

struct moe_mxfp4_silu_globals {
    const uint8_t* __restrict__ A;          // [total_tokens, K/2] packed FP4
    const uint8_t* __restrict__ B;          // [E, K/2, N] packed FP4
    bf16* __restrict__ C;                   // [num_padded, N/2] bf16
    const uint8_t* __restrict__ A_mx_scale; // [total_tokens, K/32] MX scales
    const uint8_t* __restrict__ B_mx_scale; // [E, K/32, N] MX scales
    const float* __restrict__ A_scale;      // [1] per-tensor scale
    const float* __restrict__ B_scale;      // [E] per-expert scale

    const float* __restrict__ topk_weights;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ expert_ids;

    int N, K;   // N is full width (gate+up), output is N/2
    int num_valid_tokens;
    int top_k;
    bool mul_routed_weight;

    int stride_am, stride_ak;
    int stride_be, stride_bk, stride_bn;
    int stride_cm, stride_cn;
    int stride_amxm, stride_amxk;
    int stride_bmxe, stride_bmxk, stride_bmxn;

    int grid_m, grid_n;

    hipStream_t stream;

    dim3 grid() { return dim3(grid_m * grid_n); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

__device__ __forceinline__ float unpack_fp4(uint8_t packed, int idx) {
    int shift = (idx & 1) * 4;
    int nibble = (packed >> shift) & 0xF;
    return fp4_lut[nibble];
}

__device__ __forceinline__ float decode_mx_scale(uint8_t val) {
    if (val == 0) return 0.0f;
    int exp_bits = (int)val - 127 + 127;
    if (exp_bits <= 0) return 0.0f;
    if (exp_bits >= 255) return __int_as_float(0x7f800000);
    uint32_t fbits = (uint32_t)exp_bits << 23;
    return __uint_as_float(fbits);
}

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// Using exp2 approximation: exp(-x) = exp2(-x * log2(e))
__device__ __forceinline__ float silu_exp2(float x) {
    return x / (1.0f + exp2f(-1.44269504089f * x));
}

__global__ __launch_bounds__(NUM_THREADS)
void moe_mxfp4_silu_kernel(const moe_mxfp4_silu_globals g) {
    const int pid = blockIdx.x;
    if (pid >= g.grid_m * g.grid_n) return;

    const int pid_n = pid % g.grid_n;
    const int pid_m = pid / g.grid_n;

    int expert_id = g.expert_ids[pid_m];
    if (expert_id == -1) {
        // Write zeros for N/2 output columns
        int N_half = g.N / 2;
        for (int i = threadIdx.x; i < BLOCK_M * (BLOCK_N / 2); i += NUM_THREADS) {
            int lm = i / (BLOCK_N / 2), ln = i % (BLOCK_N / 2);
            int gm = pid_m * BLOCK_M + lm;
            int gn = pid_n * (BLOCK_N / 2) + ln;
            if (gn < N_half) {
                int token_id = g.sorted_token_ids[gm];
                if (token_id < g.num_valid_tokens) {
                    g.C[token_id * g.stride_cm + gn * g.stride_cn] = __float2bfloat16(0.0f);
                }
            }
        }
        return;
    }

    // Accumulate full BLOCK_N (gate+up interleaved)
    constexpr int ELEMS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;
    float acc[ELEMS_PER_THREAD];
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    constexpr int PACKED_K = BLOCK_K / 2;
    __shared__ uint8_t A_sh[BLOCK_M * PACKED_K];
    __shared__ uint8_t B_sh[PACKED_K * BLOCK_N];
    constexpr int MX_SCALES_PER_K = BLOCK_K / MX_GROUP_SIZE;
    __shared__ float A_mx_sh[BLOCK_M * MX_SCALES_PER_K];
    __shared__ float B_mx_sh[BLOCK_N * MX_SCALES_PER_K];

    // Compute interleaved B column offsets for this tile:
    // pid_n selects which group of BLOCK_N/2 output pairs
    // Interleaving: global B col for local ln is:
    //   i_floor = ln / 2  (which pair)
    //   half_offset = pid_n * (BLOCK_N/2) + i_floor
    //   global_col = (half_offset % (N/2)) + (ln % 2) * (N/2)
    // This interleaves gate[half] and up[half] columns

    const int num_k_tiles = (g.K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BLOCK_K;

        // Load packed A tile
        for (int idx = threadIdx.x; idx < BLOCK_M * PACKED_K; idx += NUM_THREADS) {
            int lm = idx / PACKED_K, lk = idx % PACKED_K;
            int gm = pid_m * BLOCK_M + lm;
            int gk = (k_start / 2) + lk;

            int token_id = g.sorted_token_ids[gm];
            int orig_token = token_id / g.top_k;

            if (gk < g.K / 2 && token_id < g.num_valid_tokens)
                A_sh[idx] = g.A[orig_token * g.stride_am + gk * g.stride_ak];
            else
                A_sh[idx] = 0;
        }

        // Load packed B tile with interleaved column indexing
        for (int idx = threadIdx.x; idx < PACKED_K * BLOCK_N; idx += NUM_THREADS) {
            int lk = idx / BLOCK_N, ln = idx % BLOCK_N;
            int gk = (k_start / 2) + lk;

            // Interleaved column mapping
            int i_floor = ln / 2;
            int half_offset = pid_n * (BLOCK_N / 2) + i_floor;
            int gn = (half_offset % (g.N / 2)) + (ln % 2) * (g.N / 2);

            if (gk < g.K / 2 && gn < g.N)
                B_sh[idx] = g.B[expert_id * g.stride_be + gk * g.stride_bk + gn * g.stride_bn];
            else
                B_sh[idx] = 0;
        }

        // Load A MX scales
        for (int idx = threadIdx.x; idx < BLOCK_M * MX_SCALES_PER_K; idx += NUM_THREADS) {
            int lm = idx / MX_SCALES_PER_K, ls = idx % MX_SCALES_PER_K;
            int gm = pid_m * BLOCK_M + lm;
            int gs = (k_start / MX_GROUP_SIZE) + ls;

            int token_id = g.sorted_token_ids[gm];
            int orig_token = token_id / g.top_k;

            if (gs < g.K / MX_GROUP_SIZE && token_id < g.num_valid_tokens)
                A_mx_sh[idx] = decode_mx_scale(
                    g.A_mx_scale[orig_token * g.stride_amxm + gs * g.stride_amxk]);
            else
                A_mx_sh[idx] = 0.0f;
        }

        // Load B MX scales (with interleaved column mapping)
        for (int idx = threadIdx.x; idx < BLOCK_N * MX_SCALES_PER_K; idx += NUM_THREADS) {
            int ln = idx / MX_SCALES_PER_K, ls = idx % MX_SCALES_PER_K;
            int gs = (k_start / MX_GROUP_SIZE) + ls;

            int i_floor = ln / 2;
            int half_offset = pid_n * (BLOCK_N / 2) + i_floor;
            int gn = (half_offset % (g.N / 2)) + (ln % 2) * (g.N / 2);

            if (gs < g.K / MX_GROUP_SIZE && gn < g.N)
                B_mx_sh[idx] = decode_mx_scale(
                    g.B_mx_scale[expert_id * g.stride_bmxe + gs * g.stride_bmxk + gn * g.stride_bmxn]);
            else
                B_mx_sh[idx] = 0.0f;
        }
        __syncthreads();

        // MXFP4 x MXFP4 matmul
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
            if (eidx >= BLOCK_M * BLOCK_N) break;
            int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;

            float sum = 0.0f;
            for (int kk = 0; kk < BLOCK_K && (k_start + kk) < g.K; kk++) {
                int mx_group = kk / MX_GROUP_SIZE;
                float a_mx = A_mx_sh[lm * MX_SCALES_PER_K + mx_group];
                float b_mx = B_mx_sh[ln * MX_SCALES_PER_K + mx_group];

                float a_val = unpack_fp4(A_sh[lm * PACKED_K + kk / 2], kk);
                float b_val = unpack_fp4(B_sh[(kk / 2) * BLOCK_N + ln], kk);

                sum += a_val * a_mx * b_val * b_mx;
            }
            acc[i] += sum;
        }
        __syncthreads();
    }

    // Apply global scales
    float a_scale = (g.A_scale != nullptr) ? g.A_scale[0] : 1.0f;
    float b_scale = (g.B_scale != nullptr) ? g.B_scale[expert_id] : 1.0f;
    float global_scale = a_scale * b_scale;

    // Apply routing weight, then SiLU-and-mul, then store N/2 outputs
    int N_half = g.N / 2;
    constexpr int HALF_N = BLOCK_N / 2;

    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;
        acc[i] *= global_scale;

        // Apply routing weight before activation
        if (g.mul_routed_weight) {
            int gm = pid_m * BLOCK_M + lm;
            int token_id = g.sorted_token_ids[gm];
            if (token_id < g.num_valid_tokens) {
                acc[i] *= g.topk_weights[token_id];
            }
        }
    }

    // Now apply SiLU-and-mul: acc is [BLOCK_M, BLOCK_N] with interleaved gate/up
    // Even indices (ln%2==0) are gate, odd indices (ln%2==1) are up
    // output[m, n] = SiLU(gate[m, n]) * up[m, n]
    for (int i = 0; i < ELEMS_PER_THREAD; i += 2) {
        int eidx = threadIdx.x * ELEMS_PER_THREAD + i;
        if (eidx >= BLOCK_M * BLOCK_N) break;
        int lm = eidx / BLOCK_N, ln = eidx % BLOCK_N;

        // Only process pairs where ln is even (gate column)
        if (ln % 2 != 0) continue;
        if (i + 1 >= ELEMS_PER_THREAD) break;

        float gate = acc[i];      // Even = gate
        float up = acc[i + 1];    // Odd = up

        float activated = silu_exp2(gate) * up;

        // Output column
        int gm = pid_m * BLOCK_M + lm;
        int out_n = pid_n * HALF_N + ln / 2;

        if (out_n >= N_half) continue;

        int token_id = g.sorted_token_ids[gm];
        if (token_id >= g.num_valid_tokens) continue;

        g.C[token_id * g.stride_cm + out_n * g.stride_cn] = __float2bfloat16(activated);
    }
}

void dispatch_moe_mxfp4_silu(moe_mxfp4_silu_globals& g) {
    hipLaunchKernelGGL(moe_mxfp4_silu_kernel, g.grid(), g.block(),
                       g.dynamic_shared_memory(), g.stream, g);
}

// ============================================================================
// PyBind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>
#include <array>

static std::array<int,4> get_tensor_shape(pybind11::object t) {
    std::array<int,4> s = {1,1,1,1};
    auto shape = t.attr("shape").cast<pybind11::tuple>();
    int nd = shape.size();
    for (int i = 0; i < nd && i < 4; i++)
        s[4-nd+i] = pybind11::cast<int>(shape[i]);
    return s;
}
static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void moe_mxfp4_silu_py(
    pybind11::object A,
    pybind11::object B,
    pybind11::object C,
    pybind11::object A_mx_scale,
    pybind11::object B_mx_scale,
    pybind11::object A_scale,
    pybind11::object B_scale,
    pybind11::object topk_weights,
    pybind11::object sorted_token_ids,
    pybind11::object expert_ids,
    int N, int K,
    int num_valid_tokens,
    int top_k,
    bool mul_routed_weight,
    int stride_am, int stride_ak,
    int stride_be, int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    int stride_amxm, int stride_amxk,
    int stride_bmxe, int stride_bmxk, int stride_bmxn,
    int grid_m, int grid_n
) {
    moe_mxfp4_silu_globals g;
    g.A = reinterpret_cast<const uint8_t*>(get_data_ptr(A));
    g.B = reinterpret_cast<const uint8_t*>(get_data_ptr(B));
    g.C = reinterpret_cast<bf16*>(get_data_ptr(C));
    g.A_mx_scale = reinterpret_cast<const uint8_t*>(get_data_ptr(A_mx_scale));
    g.B_mx_scale = reinterpret_cast<const uint8_t*>(get_data_ptr(B_mx_scale));
    g.A_scale = A_scale.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(A_scale));
    g.B_scale = B_scale.is_none() ? nullptr : reinterpret_cast<const float*>(get_data_ptr(B_scale));
    g.topk_weights = reinterpret_cast<const float*>(get_data_ptr(topk_weights));
    g.sorted_token_ids = reinterpret_cast<const int32_t*>(get_data_ptr(sorted_token_ids));
    g.expert_ids = reinterpret_cast<const int32_t*>(get_data_ptr(expert_ids));
    g.N = N;
    g.K = K;
    g.num_valid_tokens = num_valid_tokens;
    g.top_k = top_k;
    g.mul_routed_weight = mul_routed_weight;
    g.stride_am = stride_am;
    g.stride_ak = stride_ak;
    g.stride_be = stride_be;
    g.stride_bk = stride_bk;
    g.stride_bn = stride_bn;
    g.stride_cm = stride_cm;
    g.stride_cn = stride_cn;
    g.stride_amxm = stride_amxm;
    g.stride_amxk = stride_amxk;
    g.stride_bmxe = stride_bmxe;
    g.stride_bmxk = stride_bmxk;
    g.stride_bmxn = stride_bmxn;
    g.grid_m = grid_m;
    g.grid_n = grid_n;
    g.stream = 0;

    dispatch_moe_mxfp4_silu(g);
}

PYBIND11_MODULE(moe_op_mxfp4_silu_fused_tk, m) {
    m.def("moe_mxfp4_silu_fused", &moe_mxfp4_silu_py,
          "MoE MXFP4 GEMM with fused SiLU activation kernel",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("A_mx_scale"),
          pybind11::arg("B_mx_scale"),
          pybind11::arg("A_scale"),
          pybind11::arg("B_scale"),
          pybind11::arg("topk_weights"),
          pybind11::arg("sorted_token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("N"),
          pybind11::arg("K"),
          pybind11::arg("num_valid_tokens"),
          pybind11::arg("top_k"),
          pybind11::arg("mul_routed_weight"),
          pybind11::arg("stride_am"),
          pybind11::arg("stride_ak"),
          pybind11::arg("stride_be"),
          pybind11::arg("stride_bk"),
          pybind11::arg("stride_bn"),
          pybind11::arg("stride_cm"),
          pybind11::arg("stride_cn"),
          pybind11::arg("stride_amxm"),
          pybind11::arg("stride_amxk"),
          pybind11::arg("stride_bmxe"),
          pybind11::arg("stride_bmxk"),
          pybind11::arg("stride_bmxn"),
          pybind11::arg("grid_m"),
          pybind11::arg("grid_n"));
}
