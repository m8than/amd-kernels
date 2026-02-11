#include "kittens.cuh"
using namespace kittens;

// Fused RMSNorm + MXFP4 Quantization Kernel
// Combines RMSNorm and MXFP4 (Microscaling FP4) quantization
// MXFP4: 4-bit floating point with shared exponent per block (E2M1 format)
//   - 1 sign bit, 2 exponent bits, 1 mantissa bit
//   - Block-wise shared exponent (typically 32 elements per block)
//   - Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define MXFP4_BLOCK_SIZE 32  // Standard block size for MXFP4
#define EXP_BIAS_FP32 127
#define EXP_BIAS_FP4 1
#define MBITS_FP32 23
#define MBITS_FP4 1

// Global tensor descriptors
template<int D>
struct mxfp4_quant_globals {
    using input_gl = gl<bf16, -1, -1, -1, D>;
    using weight_gl = gl<bf16, -1, -1, -1, D>;

    input_gl input;
    weight_gl weight;
    uint8_t* output;  // Raw pointer for 2 FP4 values per byte
    uint8_t* scales;  // Raw pointer for E8M0 exponent
    int rows;
    int cols;
    float eps;
    hipStream_t stream;

    dim3 grid() { return dim3(rows); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return sizeof(float) * D * 2; }
};

// Device helper: RMSNorm operation
__device__ __forceinline__
void rmsnorm_op_mxfp4(float* out, const float* in, const float* weight, int n_cols, float eps, int tid) {
    // Compute sum(x^2)
    float sum_sq = 0.0f;
    for (int col = tid; col < n_cols; col += NUM_THREADS) {
        float val = in[col];
        sum_sq += val * val;
    }

    // Reduce across threads
    __shared__ float shared_sum[NUM_THREADS];
    shared_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float rms_norm_factor = rsqrtf(shared_sum[0] / n_cols + eps);

    // Normalize and apply weight
    for (int col = tid; col < n_cols; col += NUM_THREADS) {
        out[col] = in[col] * rms_norm_factor * weight[col];
    }
    __syncthreads();
}

// Device helper: MXFP4 quantization per block
__device__ __forceinline__
void mxfp4_quant_block(
    uint8_t* out_fp4,
    uint8_t* out_scale,
    const float* in,
    int block_idx,
    int tid
) {
    const int block_start = block_idx * MXFP4_BLOCK_SIZE;
    __shared__ float block_max;
    __shared__ int shared_exp;

    // Find max absolute value in block
    float local_max = 0.0f;
    if (tid < MXFP4_BLOCK_SIZE) {
        local_max = fabsf(in[block_start + tid]);
    }

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down(local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    if (tid == 0) {
        // Compute shared exponent (E8M0 format)
        // amax rounded to nearest power of 2
        uint32_t amax_bits = __float_as_uint(local_max);
        amax_bits = (amax_bits + 0x200000) & 0xFF800000;  // Round mantissa
        float amax_rounded = __uint_as_float(amax_bits);

        // Extract exponent and compute scale
        int exp_unbiased = (int)floorf(log2f(amax_rounded)) - 2;  // -2 for FP4 range
        exp_unbiased = max(-127, min(127, exp_unbiased));

        shared_exp = exp_unbiased;
        block_max = exp2f((float)exp_unbiased);
    }
    __syncthreads();

    float quant_scale = exp2f(-(float)shared_exp);
    uint8_t scale_e8m0 = (uint8_t)(shared_exp + 127);

    // Quantize each element to MXFP4 (E2M1)
    if (tid < MXFP4_BLOCK_SIZE) {
        float val = in[block_start + tid];
        float qx = val * quant_scale;

        // Convert to E2M1 format (4 bits: 1 sign, 2 exp, 1 mantissa)
        uint32_t qx_bits = __float_as_uint(qx);
        uint32_t sign = qx_bits & 0x80000000;

        // Remove sign for processing
        qx_bits = qx_bits & 0x7FFFFFFF;
        float qx_abs = __uint_as_float(qx_bits);

        uint8_t e2m1_value = 0;

        // Check ranges for MXFP4 values
        if (qx_abs >= 6.0f) {
            e2m1_value = 0x7;  // Max value: ±6.0 (111)
        } else if (qx_abs >= 4.0f) {
            e2m1_value = 0x6;  // ±4.0 (110)
        } else if (qx_abs >= 3.0f) {
            e2m1_value = 0x5;  // ±3.0 (101)
        } else if (qx_abs >= 2.0f) {
            e2m1_value = 0x4;  // ±2.0 (100)
        } else if (qx_abs >= 1.5f) {
            e2m1_value = 0x3;  // ±1.5 (011)
        } else if (qx_abs >= 1.0f) {
            e2m1_value = 0x2;  // ±1.0 (010)
        } else if (qx_abs >= 0.5f) {
            e2m1_value = 0x1;  // ±0.5 (001)
        } else {
            e2m1_value = 0x0;  // ±0.0 (000)
        }

        // Add sign bit
        if (sign) {
            e2m1_value |= 0x8;  // Set bit 3 for negative
        }

        // Pack two 4-bit values per byte
        const int pair_idx = tid / 2;
        if (tid % 2 == 0) {
            // Store in lower 4 bits
            out_fp4[pair_idx] = (out_fp4[pair_idx] & 0xF0) | e2m1_value;
        } else {
            // Store in upper 4 bits
            out_fp4[pair_idx] = (out_fp4[pair_idx] & 0x0F) | (e2m1_value << 4);
        }
    }

    // Store shared exponent (one per block)
    if (tid == 0) {
        *out_scale = scale_e8m0;
    }
}

// Main kernel: Fused RMSNorm + MXFP4 Quantization
template<int D>
__global__ void fused_rmsnorm_mxfp4_quant_kernel(
    const mxfp4_quant_globals<D> g
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= g.rows) return;

    extern __shared__ float shared_mem[];
    float* input_buf = shared_mem;
    float* norm_buf = &shared_mem[g.cols];

    // Load input row
    for (int col = tid; col < g.cols; col += NUM_THREADS) {
        bf16 val = g.input[{0, 0, row, col}];
        input_buf[col] = __bfloat162float(val);
    }
    __syncthreads();

    // Load weight (shared across rows)
    for (int col = tid; col < g.cols; col += NUM_THREADS) {
        bf16 w = g.weight[{0, 0, 0, col}];
        norm_buf[col] = __bfloat162float(w);  // Temporarily store weight
    }
    __syncthreads();

    // Perform RMSNorm
    rmsnorm_op_mxfp4(norm_buf, input_buf, norm_buf, g.cols, g.eps, tid);

    // Quantize in MXFP4 blocks
    const int num_blocks = (g.cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int block_start = block_idx * MXFP4_BLOCK_SIZE;
        const int block_end = min(block_start + MXFP4_BLOCK_SIZE, g.cols);
        const int block_size = block_end - block_start;

        if (block_size != MXFP4_BLOCK_SIZE && tid >= block_size) {
            continue;  // Skip threads beyond last partial block
        }

        // Shared memory for packed output
        __shared__ uint8_t packed_fp4[MXFP4_BLOCK_SIZE / 2];
        if (tid < MXFP4_BLOCK_SIZE / 2) {
            packed_fp4[tid] = 0;
        }
        __syncthreads();

        // Quantize block
        mxfp4_quant_block(packed_fp4, nullptr, norm_buf, block_idx, tid);
        __syncthreads();

        // Store packed output
        const int out_offset = row * ((g.cols + 1) / 2) + block_idx * (MXFP4_BLOCK_SIZE / 2);
        if (tid < MXFP4_BLOCK_SIZE / 2) {
            g.output[out_offset + tid] = packed_fp4[tid];
        }

        // Store scale (shared exponent)
        if (tid == 0) {
            // Recompute scale for storage
            float local_max = 0.0f;
            for (int i = 0; i < block_size; i++) {
                local_max = fmaxf(local_max, fabsf(norm_buf[block_start + i]));
            }
            uint32_t amax_bits = __float_as_uint(local_max);
            amax_bits = (amax_bits + 0x200000) & 0xFF800000;
            float amax_rounded = __uint_as_float(amax_bits);
            int exp_unbiased = (int)floorf(log2f(amax_rounded)) - 2;
            exp_unbiased = max(-127, min(127, exp_unbiased));
            uint8_t scale_e8m0 = (uint8_t)(exp_unbiased + 127);
            const int scale_blocks = (g.cols + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
            g.scales[row * scale_blocks + block_idx] = scale_e8m0;
        }
        __syncthreads();
    }
}

// Dispatch function
template<int D>
void dispatch_fused_rmsnorm_mxfp4_quant(mxfp4_quant_globals<D>& g) {
    hipLaunchKernelGGL(
        (fused_rmsnorm_mxfp4_quant_kernel<D>),
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        g.stream,
        g
    );
}

// Explicit template instantiations
template void dispatch_fused_rmsnorm_mxfp4_quant<4096>(mxfp4_quant_globals<4096>&);
template void dispatch_fused_rmsnorm_mxfp4_quant<8192>(mxfp4_quant_globals<8192>&);

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

// input: (M, D) bf16, weight: (D,) bf16
// output: (M, D/2) uint8 (packed fp4), scales: (M, D/32) uint8
template<int D>
void fused_mxfp4_quant_impl(pybind11::object input, pybind11::object weight,
                              pybind11::object output, pybind11::object scales, float eps) {
    int rows = _get_dim(input, 0);
    int cols = D;

    using globals = mxfp4_quant_globals<D>;
    globals g{
        make_gl<typename globals::input_gl>(_get_ptr(input), 1, 1, rows, D),
        make_gl<typename globals::weight_gl>(_get_ptr(weight), 1, 1, 1, D),
        (uint8_t*)_get_ptr(output),
        (uint8_t*)_get_ptr(scales),
        rows,
        cols,
        eps,
        0  // default HIP stream
    };

    dispatch_fused_rmsnorm_mxfp4_quant<D>(g);
}

void fused_rmsnorm_mxfp4_quant_fwd(pybind11::object input, pybind11::object weight,
                                     pybind11::object output, pybind11::object scales,
                                     float eps) {
    int D = _get_dim(input, 1);
    if (D == 4096)      fused_mxfp4_quant_impl<4096>(input, weight, output, scales, eps);
    else if (D == 8192) fused_mxfp4_quant_impl<8192>(input, weight, output, scales, eps);
    else throw std::runtime_error("Unsupported D=" + std::to_string(D) + ". Supported: 4096, 8192.");
}

PYBIND11_MODULE(fused_mxfp4_quant_tk, m) {
    m.doc() = "HipKittens fused RMSNorm + MXFP4 quantization kernel";
    m.def("fused_rmsnorm_mxfp4_quant_fwd", &fused_rmsnorm_mxfp4_quant_fwd,
          "Fused RMSNorm + MXFP4 quantization forward pass");
}
