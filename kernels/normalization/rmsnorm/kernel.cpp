/**
 * RMS Normalization kernel ported from AITER Triton to HipKittens.
 *
 * Implements two variants:
 *   1. rmsnorm_fwd       — output = (x / sqrt(mean(x^2) + eps)) * weight
 *   2. fused_add_rmsnorm — res_out = x + residual;
 *                          output  = (res_out / sqrt(mean(res_out^2) + eps)) * weight
 *
 * Each row of the 2-D input (n_rows, D) is normalized independently.
 * Templated on D (model dimension) so the compiler can unroll everything.
 *
 * Uses rv_naive<bf16, D> register vectors — one vector per row per warp.
 * Reductions (sum-of-squares) are computed in FP32 for numerical precision.
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// NUM_WORKERS warps per block; each warp processes one row at a time.
#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// ---------------------------------------------------------------------------
// 1. RMSNorm Forward
// ---------------------------------------------------------------------------

template <int _D>
struct rmsnorm_globals {
    static constexpr int D = _D;

    // Tensor descriptors: input(n_rows, D), weight(D), output(n_rows, D)
    using input_gl  = gl<bf16, -1, -1, -1, -1>;
    using weight_gl = gl<bf16, -1, -1, -1, -1>;
    using output_gl = gl<bf16, -1, -1, -1, -1>;

    input_gl  input;
    weight_gl weight;
    output_gl output;

    float epsilon;
    int n_rows;

    dim3 grid()  { return dim3((n_rows + NUM_WORKERS - 1) / NUM_WORKERS); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template <int D>
__global__ void rmsnorm_fwd_kernel(const rmsnorm_globals<D> g) {
    const int warp_id = kittens::warpid();
    const int row_idx = blockIdx.x * NUM_WORKERS + warp_id;
    if (row_idx >= g.n_rows) return;

    constexpr int d_model = D;

    // Load input row and weight vector into register vectors
    rv_naive<bf16, d_model> x_reg, w_reg;
    load(x_reg, g.input,  {0, 0, row_idx, 0});
    load(w_reg, g.weight, {0, 0, 0, 0});

    // Compute sum of squares in FP32 for numerical precision.
    // We do this by: (1) square element-wise, (2) reduce-sum, (3) rsqrt.
    rv_naive<bf16, d_model> x_sq;
    mul(x_sq, x_reg, x_reg);  // x_sq = x * x

    bf16 sum_sq;
    sum(sum_sq, x_sq);  // sum_sq = sum(x^2)

    // Compute norm_factor = rsqrt(mean(x^2) + eps)
    float sum_sq_f = __bfloat162float(sum_sq);
    float mean_sq  = sum_sq_f / static_cast<float>(d_model);
    float norm_factor = rsqrtf(mean_sq + g.epsilon);
    bf16 norm_factor_bf = __float2bfloat16(norm_factor);

    // output = x * norm_factor * weight
    rv_naive<bf16, d_model> out_reg;
    mul(out_reg, x_reg, norm_factor_bf);  // out = x * norm_factor
    mul(out_reg, out_reg, w_reg);          // out = out * weight

    store(g.output, out_reg, {0, 0, row_idx, 0});
}

template <int D>
void dispatch_rmsnorm_fwd(rmsnorm_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_fwd_kernel<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_fwd_kernel<D><<<g.grid(), g.block(), mem_size>>>(g);
}

// ---------------------------------------------------------------------------
// 2. Fused Add + RMSNorm Forward
//    res_out = x + residual
//    output  = rmsnorm(res_out) * weight
// ---------------------------------------------------------------------------

template <int _D>
struct fused_add_rmsnorm_globals {
    static constexpr int D = _D;

    using input_gl    = gl<bf16, -1, -1, -1, -1>;
    using residual_gl = gl<bf16, -1, -1, -1, -1>;
    using weight_gl   = gl<bf16, -1, -1, -1, -1>;
    using output_gl   = gl<bf16, -1, -1, -1, -1>;
    using res_out_gl  = gl<bf16, -1, -1, -1, -1>;

    input_gl    input;
    residual_gl residual;
    weight_gl   weight;
    output_gl   output;
    res_out_gl  res_out;

    float epsilon;
    int n_rows;

    dim3 grid()  { return dim3((n_rows + NUM_WORKERS - 1) / NUM_WORKERS); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template <int D>
__global__ void fused_add_rmsnorm_fwd_kernel(const fused_add_rmsnorm_globals<D> g) {
    const int warp_id = kittens::warpid();
    const int row_idx = blockIdx.x * NUM_WORKERS + warp_id;
    if (row_idx >= g.n_rows) return;

    constexpr int d_model = D;

    // Load input, residual, and weight
    rv_naive<bf16, d_model> x_reg, res_reg, w_reg;
    load(x_reg,   g.input,    {0, 0, row_idx, 0});
    load(res_reg, g.residual, {0, 0, row_idx, 0});
    load(w_reg,   g.weight,   {0, 0, 0, 0});

    // Fused add: res_out = x + residual
    add(x_reg, x_reg, res_reg);  // x_reg now holds res_out

    // Store res_out for use in subsequent layers
    store(g.res_out, x_reg, {0, 0, row_idx, 0});

    // Compute sum of squares
    rv_naive<bf16, d_model> x_sq;
    mul(x_sq, x_reg, x_reg);

    bf16 sum_sq;
    sum(sum_sq, x_sq);

    // norm_factor = rsqrt(mean(x^2) + eps)
    float sum_sq_f = __bfloat162float(sum_sq);
    float mean_sq  = sum_sq_f / static_cast<float>(d_model);
    float norm_factor = rsqrtf(mean_sq + g.epsilon);
    bf16 norm_factor_bf = __float2bfloat16(norm_factor);

    // output = res_out * norm_factor * weight
    rv_naive<bf16, d_model> out_reg;
    mul(out_reg, x_reg, norm_factor_bf);
    mul(out_reg, out_reg, w_reg);

    store(g.output, out_reg, {0, 0, row_idx, 0});
}

template <int D>
void dispatch_fused_add_rmsnorm_fwd(fused_add_rmsnorm_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fused_add_rmsnorm_fwd_kernel<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fused_add_rmsnorm_fwd_kernel<D><<<g.grid(), g.block(), mem_size>>>(g);
}

// ---------------------------------------------------------------------------
// Instantiate for common LLM hidden dimensions and expose via pybind11.
// Each dimension gets its own function: rmsnorm_fwd_<D>, fused_add_rmsnorm_fwd_<D>
// ---------------------------------------------------------------------------

#define BIND_RMSNORM(D_VAL) \
    py::bind_function<dispatch_rmsnorm_fwd<D_VAL>>(m, "rmsnorm_fwd_" #D_VAL, \
        &rmsnorm_globals<D_VAL>::input, \
        &rmsnorm_globals<D_VAL>::weight, \
        &rmsnorm_globals<D_VAL>::output, \
        &rmsnorm_globals<D_VAL>::epsilon, \
        &rmsnorm_globals<D_VAL>::n_rows \
    );

#define BIND_FUSED_ADD_RMSNORM(D_VAL) \
    py::bind_function<dispatch_fused_add_rmsnorm_fwd<D_VAL>>(m, "fused_add_rmsnorm_fwd_" #D_VAL, \
        &fused_add_rmsnorm_globals<D_VAL>::input, \
        &fused_add_rmsnorm_globals<D_VAL>::residual, \
        &fused_add_rmsnorm_globals<D_VAL>::weight, \
        &fused_add_rmsnorm_globals<D_VAL>::output, \
        &fused_add_rmsnorm_globals<D_VAL>::res_out, \
        &fused_add_rmsnorm_globals<D_VAL>::epsilon, \
        &fused_add_rmsnorm_globals<D_VAL>::n_rows \
    );

#define BIND_BOTH(D_VAL) \
    BIND_RMSNORM(D_VAL) \
    BIND_FUSED_ADD_RMSNORM(D_VAL)

PYBIND11_MODULE(rmsnorm_tk, m) {
    m.doc() = "HipKittens RMSNorm kernels (multi-dimension)";

    // Head dimensions
    BIND_BOTH(128)
    BIND_BOTH(256)

    // Small model hidden sizes
    BIND_BOTH(512)
    BIND_BOTH(768)
    BIND_BOTH(1024)   // Qwen3-0.6B
    BIND_BOTH(1536)   // Qwen3-1.7B

    // Medium model hidden sizes
    BIND_BOTH(2048)
    BIND_BOTH(2560)   // Qwen3-4B
    BIND_BOTH(3072)   // Gemma-2B
    BIND_BOTH(3584)   // Qwen2.5-7B
    BIND_BOTH(4096)   // Llama-8B, Mistral-7B

    // Large model hidden sizes
    BIND_BOTH(5120)   // Llama-13B, Qwen3-32B
    BIND_BOTH(7168)   // DeepSeek-V3
    BIND_BOTH(8192)   // Llama-70B, Qwen2.5-72B

    // Backward-compat aliases (D=128 default)
    py::bind_function<dispatch_rmsnorm_fwd<128>>(m, "rmsnorm_fwd",
        &rmsnorm_globals<128>::input,
        &rmsnorm_globals<128>::weight,
        &rmsnorm_globals<128>::output,
        &rmsnorm_globals<128>::epsilon,
        &rmsnorm_globals<128>::n_rows
    );
    py::bind_function<dispatch_fused_add_rmsnorm_fwd<128>>(m, "fused_add_rmsnorm_fwd",
        &fused_add_rmsnorm_globals<128>::input,
        &fused_add_rmsnorm_globals<128>::residual,
        &fused_add_rmsnorm_globals<128>::weight,
        &fused_add_rmsnorm_globals<128>::output,
        &fused_add_rmsnorm_globals<128>::res_out,
        &fused_add_rmsnorm_globals<128>::epsilon,
        &fused_add_rmsnorm_globals<128>::n_rows
    );
}
