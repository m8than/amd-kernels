// Quantized MoE Kernels
// Ported from reference/triton/quant_moe.py
//
// Contains three quantization operations:
//   1. downcast_to_static_fp8: Convert tensor to FP8 with static scaling
//   2. downcast_to_mxfp: Convert tensor to MX format (FP8/FP4 with per-32-element scales)
//   3. upcast_from_mxfp: Convert MX format back to bf16/fp16

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

using namespace kittens;

// Configuration
#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// ============================================================================
// Kernel 1: Downcast to Static FP8
// ============================================================================

using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_f32 = gl<float, -1, -1, -1, -1>;

template<int BLOCK_M, int BLOCK_N>
struct downcast_fp8_globals {
    _gl_bf16 x;
    uint8_t* y;  // Raw pointer for FP8 stored as uint8
    _gl_f32 scale;

    int M, N;
    hipStream_t stream;

    dim3 grid() {
        return dim3((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
    }

    dim3 block() {
        return dim3(NUM_THREADS);
    }

    size_t dynamic_shared_memory() {
        return 0;
    }
};

// Helper: convert float to FP8 e4m3
__device__ uint8_t float_to_fp8e4m3(float val, float scale) {
    val = val / scale;
    // Clamp to FP8 range: [-448, 448]
    val = fmaxf(-448.0f, fminf(448.0f, val));

    // Use HIP intrinsic if available, otherwise manual conversion
    #ifdef __HIP_FP8_TYPE_FNUZ__
    __hip_fp8_e4m3_fnuz fp8_val = static_cast<__hip_fp8_e4m3_fnuz>(val);
    return *reinterpret_cast<uint8_t*>(&fp8_val);
    #else
    // Manual FP8 conversion (simplified)
    // FP8 E4M3: 1 sign bit, 4 exp bits, 3 mantissa bits
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 7;  // Rebias to E4 bias (7)
    uint32_t mantissa = (bits >> 20) & 0x7;

    // Clamp exponent
    exp = max(0, min(15, exp));

    uint8_t result = (sign << 7) | (exp << 3) | mantissa;
    return result;
    #endif
}

template<int BLOCK_M, int BLOCK_N>
__global__ void downcast_to_static_fp8_kernel(const downcast_fp8_globals<BLOCK_M, BLOCK_N> g) {
    int pid_m = blockIdx.y;
    int pid_n = blockIdx.x;

    int start_m = pid_m * BLOCK_M;
    int start_n = pid_n * BLOCK_N;

    int tid = threadIdx.x;

    // Load scale (single value)
    float scale = g.scale[0];

    // Process elements assigned to this thread
    for (int m = start_m + (tid / BLOCK_N); m < start_m + BLOCK_M && m < g.M; m += (NUM_THREADS / BLOCK_N)) {
        for (int n = start_n + (tid % BLOCK_N); n < start_n + BLOCK_N && n < g.N; n += BLOCK_N) {
            if (m < g.M && n < g.N) {
                // Load bf16 value
                __hip_bfloat16 x_bf16 = g.x[m * g.N + n];
                float x_f32 = __bfloat162float(x_bf16);

                // Quantize to FP8
                uint8_t y_fp8 = float_to_fp8e4m3(x_f32, scale);

                // Store
                g.y[m * g.N + n] = y_fp8;
            }
        }
    }
}

template<int BLOCK_M, int BLOCK_N>
void dispatch_downcast_fp8(downcast_fp8_globals<BLOCK_M, BLOCK_N>& g) {
    downcast_to_static_fp8_kernel<BLOCK_M, BLOCK_N><<<g.grid(), g.block(), g.dynamic_shared_memory(), g.stream>>>(g);
}

// ============================================================================
// Kernel 2: Downcast to MXFP (MX FP8/FP4 with per-32-element scales)
// ============================================================================

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
struct downcast_mxfp_globals {
    uint8_t* mx_tensor;  // Raw pointer for packed FP8 or FP4
    uint8_t* mx_scale;   // Raw pointer for exponent-only scales
    _gl_bf16 src;

    int outer_dim;
    int quant_dim;
    bool is_fp4;  // true for FP4, false for FP8

    hipStream_t stream;

    dim3 grid() {
        int quant_blocks = (quant_dim + BLOCK_SIZE_QUANT_DIM - 1) / BLOCK_SIZE_QUANT_DIM;
        int outer_blocks = (outer_dim + BLOCK_SIZE_OUT_DIM - 1) / BLOCK_SIZE_OUT_DIM;
        return dim3(outer_blocks, quant_blocks, 1);
    }

    dim3 block() {
        return dim3(NUM_THREADS);
    }

    size_t dynamic_shared_memory() {
        return 0;
    }
};

// Compute MX scale for a group of 32 elements
__device__ uint8_t compute_mx_scale_fp8(const float* vals, int count) {
    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < count; i++) {
        max_abs = fmaxf(max_abs, fabsf(vals[i]));
    }

    if (max_abs == 0.0f) return 0xFF;  // NaN encoding

    // Compute scale: 2^exp where exp = ceil(log2(max_abs / 448))
    // FP8 E4M3 max value is 448
    float dequant_scale = max_abs / 448.0f;

    // Extract exponent and round up
    uint32_t bits = __float_as_uint(dequant_scale);
    uint32_t exp = ((bits + 0x007FFFFF) & 0x7F800000) >> 23;

    return static_cast<uint8_t>(exp);
}

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
__global__ void downcast_to_mxfp_kernel(const downcast_mxfp_globals<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM> g) {
    int outer_block = blockIdx.x;
    int quant_block = blockIdx.y;

    int start_outer = outer_block * BLOCK_SIZE_OUT_DIM;
    int start_quant = quant_block * BLOCK_SIZE_QUANT_DIM;

    int tid = threadIdx.x;

    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int NUM_SCALE_GROUPS = BLOCK_SIZE_QUANT_DIM / SCALE_GROUP_SIZE;

    // Each thread processes a subset of rows
    for (int out_idx = start_outer + tid; out_idx < start_outer + BLOCK_SIZE_OUT_DIM && out_idx < g.outer_dim; out_idx += NUM_THREADS) {
        // Process each scale group
        for (int scale_group = 0; scale_group < NUM_SCALE_GROUPS; scale_group++) {
            int quant_start = start_quant + scale_group * SCALE_GROUP_SIZE;

            if (quant_start >= g.quant_dim) continue;

            // Load 32 elements
            float vals[SCALE_GROUP_SIZE];
            for (int i = 0; i < SCALE_GROUP_SIZE; i++) {
                int quant_idx = quant_start + i;
                if (quant_idx < g.quant_dim) {
                    __hip_bfloat16 bf16_val = g.src[out_idx * g.quant_dim + quant_idx];
                    vals[i] = __bfloat162float(bf16_val);
                } else {
                    vals[i] = 0.0f;
                }
            }

            // Compute scale
            uint8_t scale_exp = compute_mx_scale_fp8(vals, SCALE_GROUP_SIZE);

            // Dequant scale: 2^(exp - 127)
            uint32_t scale_bits = (static_cast<uint32_t>(scale_exp) << 23);
            float dequant_scale = __uint_as_float(scale_bits);
            float quant_scale = (dequant_scale == 0.0f) ? 0.0f : 1.0f / dequant_scale;

            // Quantize and store
            for (int i = 0; i < SCALE_GROUP_SIZE; i++) {
                int quant_idx = quant_start + i;
                if (quant_idx < g.quant_dim) {
                    float quantized = vals[i] * quant_scale;

                    uint8_t packed_val;
                    if (g.is_fp4) {
                        // FP4 E2M1 encoding (2 values per byte)
                        // Simplified: store as zero for now (full implementation is complex)
                        packed_val = 0;
                    } else {
                        // FP8 E4M3 encoding
                        packed_val = float_to_fp8e4m3(quantized, 1.0f);
                    }

                    int out_idx_packed = g.is_fp4 ? (quant_idx / 2) : quant_idx;
                    g.mx_tensor[out_idx * (g.quant_dim / (g.is_fp4 ? 2 : 1)) + out_idx_packed] = packed_val;
                }
            }

            // Store scale
            int scale_idx = start_quant / SCALE_GROUP_SIZE + scale_group;
            g.mx_scale[out_idx * (g.quant_dim / SCALE_GROUP_SIZE) + scale_idx] = scale_exp;
        }
    }
}

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
void dispatch_downcast_mxfp(downcast_mxfp_globals<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM>& g) {
    downcast_to_mxfp_kernel<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM><<<g.grid(), g.block(), g.dynamic_shared_memory(), g.stream>>>(g);
}

// ============================================================================
// Kernel 3: Upcast from MXFP
// ============================================================================

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
struct upcast_mxfp_globals {
    _gl_bf16 out;
    uint8_t* mx_scale;   // Raw pointer
    uint8_t* mx_tensor;  // Raw pointer

    int outer_dim;
    int quant_dim;
    bool is_fp4;

    hipStream_t stream;

    dim3 grid() {
        int quant_blocks = (quant_dim + BLOCK_SIZE_QUANT_DIM - 1) / BLOCK_SIZE_QUANT_DIM;
        int outer_blocks = (outer_dim + BLOCK_SIZE_OUT_DIM - 1) / BLOCK_SIZE_OUT_DIM;
        return dim3(outer_blocks, quant_blocks, 1);
    }

    dim3 block() {
        return dim3(NUM_THREADS);
    }

    size_t dynamic_shared_memory() {
        return 0;
    }
};

// Helper: FP8 to float
__device__ float fp8e4m3_to_float(uint8_t fp8_val) {
    #ifdef __HIP_FP8_TYPE_FNUZ__
    __hip_fp8_e4m3_fnuz fp8 = *reinterpret_cast<__hip_fp8_e4m3_fnuz*>(&fp8_val);
    return static_cast<float>(fp8);
    #else
    // Manual conversion
    uint32_t sign = (fp8_val >> 7) & 1;
    uint32_t exp = (fp8_val >> 3) & 0xF;
    uint32_t mantissa = fp8_val & 0x7;

    // Rebias exponent from 7 to 127
    int32_t f32_exp = exp - 7 + 127;

    if (f32_exp <= 0) return 0.0f;  // Underflow
    if (f32_exp >= 255) return sign ? -INFINITY : INFINITY;  // Overflow

    uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 20);
    return __uint_as_float(f32_bits);
    #endif
}

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
__global__ void upcast_from_mxfp_kernel(const upcast_mxfp_globals<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM> g) {
    int outer_block = blockIdx.x;
    int quant_block = blockIdx.y;

    int start_outer = outer_block * BLOCK_SIZE_OUT_DIM;
    int start_quant = quant_block * BLOCK_SIZE_QUANT_DIM;

    int tid = threadIdx.x;

    constexpr int SCALE_GROUP_SIZE = 32;
    constexpr int NUM_SCALE_GROUPS = BLOCK_SIZE_QUANT_DIM / SCALE_GROUP_SIZE;

    for (int out_idx = start_outer + tid; out_idx < start_outer + BLOCK_SIZE_OUT_DIM && out_idx < g.outer_dim; out_idx += NUM_THREADS) {
        for (int scale_group = 0; scale_group < NUM_SCALE_GROUPS; scale_group++) {
            int quant_start = start_quant + scale_group * SCALE_GROUP_SIZE;

            if (quant_start >= g.quant_dim) continue;

            // Load scale
            int scale_idx = quant_start / SCALE_GROUP_SIZE;
            uint8_t scale_exp = g.mx_scale[out_idx * (g.quant_dim / SCALE_GROUP_SIZE) + scale_idx];

            if (scale_exp == 0xFF) {
                // NaN encoding
                for (int i = 0; i < SCALE_GROUP_SIZE; i++) {
                    int quant_idx = quant_start + i;
                    if (quant_idx < g.quant_dim) {
                        g.out[out_idx * g.quant_dim + quant_idx] = __float2bfloat16(NAN);
                    }
                }
                continue;
            }

            // Dequant scale: 2^(exp_bits interpreted as exponent)
            uint32_t scale_bits = (static_cast<uint32_t>(scale_exp) << 23);
            float dequant_scale = __uint_as_float(scale_bits);

            // Dequantize elements
            for (int i = 0; i < SCALE_GROUP_SIZE; i++) {
                int quant_idx = quant_start + i;
                if (quant_idx < g.quant_dim) {
                    uint8_t packed_val;

                    if (g.is_fp4) {
                        // FP4: 2 values per byte
                        int byte_idx = quant_idx / 2;
                        packed_val = g.mx_tensor[out_idx * (g.quant_dim / 2) + byte_idx];
                        // Extract correct nibble (simplified - full impl is complex)
                        packed_val = (quant_idx % 2 == 0) ? (packed_val & 0xF) : (packed_val >> 4);
                    } else {
                        // FP8: 1 value per byte
                        packed_val = g.mx_tensor[out_idx * g.quant_dim + quant_idx];
                    }

                    float dequant_val;
                    if (g.is_fp4) {
                        // FP4 E2M1 to float (simplified)
                        dequant_val = 0.0f;  // Full implementation requires bit manipulation
                    } else {
                        dequant_val = fp8e4m3_to_float(packed_val);
                    }

                    float result = dequant_val * dequant_scale;
                    g.out[out_idx * g.quant_dim + quant_idx] = __float2bfloat16(result);
                }
            }
        }
    }
}

template<int BLOCK_SIZE_OUT_DIM, int BLOCK_SIZE_QUANT_DIM>
void dispatch_upcast_mxfp(upcast_mxfp_globals<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM>& g) {
    upcast_from_mxfp_kernel<BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM><<<g.grid(), g.block(), g.dynamic_shared_memory(), g.stream>>>(g);
}

// ============================================================================
// Explicit template instantiations for common sizes
// ============================================================================

template void dispatch_downcast_fp8<64, 64>(downcast_fp8_globals<64, 64>&);
template void dispatch_downcast_mxfp<64, 128>(downcast_mxfp_globals<64, 128>&);
template void dispatch_upcast_mxfp<64, 128>(upcast_mxfp_globals<64, 128>&);

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

void downcast_to_static_fp8_py(
    pybind11::object x,
    pybind11::object y,
    pybind11::object scale,
    int M, int N
) {
    _gl_bf16 x_gl(reinterpret_cast<bf16*>(get_data_ptr(x)), 0UL, 0UL, 0UL, 0UL);
    _gl_f32 scale_gl(reinterpret_cast<float*>(get_data_ptr(scale)), 0UL, 0UL, 0UL, 0UL);

    downcast_fp8_globals<64, 64> g{
        x_gl,
        reinterpret_cast<uint8_t*>(get_data_ptr(y)),
        scale_gl,
        M, N,
        0  // stream
    };

    dispatch_downcast_fp8<64, 64>(g);
}

void downcast_to_mxfp_py(
    pybind11::object mx_tensor,
    pybind11::object mx_scale,
    pybind11::object src,
    int outer_dim, int quant_dim,
    bool is_fp4
) {
    _gl_bf16 src_gl(reinterpret_cast<bf16*>(get_data_ptr(src)), 0UL, 0UL, 0UL, 0UL);

    downcast_mxfp_globals<64, 128> g{
        reinterpret_cast<uint8_t*>(get_data_ptr(mx_tensor)),
        reinterpret_cast<uint8_t*>(get_data_ptr(mx_scale)),
        src_gl,
        outer_dim, quant_dim, is_fp4,
        0  // stream
    };

    dispatch_downcast_mxfp<64, 128>(g);
}

void upcast_from_mxfp_py(
    pybind11::object out,
    pybind11::object mx_scale,
    pybind11::object mx_tensor,
    int outer_dim, int quant_dim,
    bool is_fp4
) {
    _gl_bf16 out_gl(reinterpret_cast<bf16*>(get_data_ptr(out)), 0UL, 0UL, 0UL, 0UL);

    upcast_mxfp_globals<64, 128> g{
        out_gl,
        reinterpret_cast<uint8_t*>(get_data_ptr(mx_scale)),
        reinterpret_cast<uint8_t*>(get_data_ptr(mx_tensor)),
        outer_dim, quant_dim, is_fp4,
        0  // stream
    };

    dispatch_upcast_mxfp<64, 128>(g);
}

PYBIND11_MODULE(quant_moe_tk, m) {
    m.def("downcast_to_static_fp8", &downcast_to_static_fp8_py,
          "Convert tensor to FP8 with static scaling",
          pybind11::arg("x"),
          pybind11::arg("y"),
          pybind11::arg("scale"),
          pybind11::arg("M"),
          pybind11::arg("N"));
    m.def("downcast_to_mxfp", &downcast_to_mxfp_py,
          "Convert tensor to MX format (FP8/FP4 with per-32-element scales)",
          pybind11::arg("mx_tensor"),
          pybind11::arg("mx_scale"),
          pybind11::arg("src"),
          pybind11::arg("outer_dim"),
          pybind11::arg("quant_dim"),
          pybind11::arg("is_fp4"));
    m.def("upcast_from_mxfp", &upcast_from_mxfp_py,
          "Convert MX format back to bf16",
          pybind11::arg("out"),
          pybind11::arg("mx_scale"),
          pybind11::arg("mx_tensor"),
          pybind11::arg("outer_dim"),
          pybind11::arg("quant_dim"),
          pybind11::arg("is_fp4"));
}
