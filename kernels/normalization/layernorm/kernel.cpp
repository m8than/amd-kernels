/**
 * Layer Normalization kernel ported from AITER Triton to HipKittens.
 *
 * Implements two variants:
 *   1. layernorm_fwd       — output = ((x - mean) / sqrt(var + eps)) * weight + bias
 *   2. fused_add_layernorm — res_out = x + residual;
 *                            output  = layernorm(res_out) * weight + bias
 *
 * Each row of the 2-D input (n_rows, D) is normalized independently.
 * Templated on D (model dimension) for compile-time unrolling.
 *
 * Uses rv_naive<bf16, D> register vectors with FP32 scalar accumulation.
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define NUM_WORKERS 4
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

// ---------------------------------------------------------------------------
// 1. LayerNorm Forward
// ---------------------------------------------------------------------------

template <int _D>
struct layernorm_globals {
    static constexpr int D = _D;

    using input_gl  = gl<bf16, -1, -1, -1, -1>;
    using weight_gl = gl<bf16, -1, -1, -1, -1>;
    using bias_gl   = gl<bf16, -1, -1, -1, -1>;
    using output_gl = gl<bf16, -1, -1, -1, -1>;

    input_gl  input;
    weight_gl weight;
    bias_gl   bias;
    output_gl output;

    float epsilon;
    int n_rows;

    dim3 grid()  { return dim3((n_rows + NUM_WORKERS - 1) / NUM_WORKERS); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template <int D>
__global__ void layernorm_fwd_kernel(const layernorm_globals<D> g) {
    const int warp_id = kittens::warpid();
    const int row_idx = blockIdx.x * NUM_WORKERS + warp_id;
    if (row_idx >= g.n_rows) return;

    constexpr int d_model = D;

    // Load input row and parameters
    rv_naive<bf16, d_model> x_reg, w_reg, b_reg;
    load(x_reg, g.input,  {0, 0, row_idx, 0});
    load(w_reg, g.weight, {0, 0, 0, 0});
    load(b_reg, g.bias,   {0, 0, 0, 0});

    // Step 1: Compute mean in FP32
    bf16 mean_bf;
    sum(mean_bf, x_reg);
    float mean_f = __bfloat162float(mean_bf) / static_cast<float>(d_model);
    bf16 mean_val = __float2bfloat16(mean_f);

    // Step 2: Subtract mean: x_centered = x - mean
    rv_naive<bf16, d_model> x_centered;
    sub(x_centered, x_reg, mean_val);

    // Step 3: Compute variance = mean((x - mean)^2)
    rv_naive<bf16, d_model> x_sq;
    mul(x_sq, x_centered, x_centered);

    bf16 var_sum_bf;
    sum(var_sum_bf, x_sq);
    float var_f = __bfloat162float(var_sum_bf) / static_cast<float>(d_model);

    // Step 4: rstd = rsqrt(var + eps)
    float rstd_f = rsqrtf(var_f + g.epsilon);
    bf16 rstd_val = __float2bfloat16(rstd_f);

    // Step 5: output = (x - mean) * rstd * weight + bias
    rv_naive<bf16, d_model> out_reg;
    mul(out_reg, x_centered, rstd_val);   // normalized = (x - mean) * rstd
    mul(out_reg, out_reg, w_reg);          // scaled = normalized * weight
    add(out_reg, out_reg, b_reg);          // output = scaled + bias

    store(g.output, out_reg, {0, 0, row_idx, 0});
}

template <int D>
void dispatch_layernorm_fwd(layernorm_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)layernorm_fwd_kernel<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    layernorm_fwd_kernel<D><<<g.grid(), g.block(), mem_size>>>(g);
}

// ---------------------------------------------------------------------------
// 2. Fused Add + LayerNorm Forward
//    res_out = x + residual
//    output  = layernorm(res_out) * weight + bias
// ---------------------------------------------------------------------------

template <int _D>
struct fused_add_layernorm_globals {
    static constexpr int D = _D;

    using input_gl    = gl<bf16, -1, -1, -1, -1>;
    using residual_gl = gl<bf16, -1, -1, -1, -1>;
    using weight_gl   = gl<bf16, -1, -1, -1, -1>;
    using bias_gl     = gl<bf16, -1, -1, -1, -1>;
    using output_gl   = gl<bf16, -1, -1, -1, -1>;
    using res_out_gl  = gl<bf16, -1, -1, -1, -1>;

    input_gl    input;
    residual_gl residual;
    weight_gl   weight;
    bias_gl     bias;
    output_gl   output;
    res_out_gl  res_out;

    float epsilon;
    int n_rows;

    dim3 grid()  { return dim3((n_rows + NUM_WORKERS - 1) / NUM_WORKERS); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template <int D>
__global__ void fused_add_layernorm_fwd_kernel(const fused_add_layernorm_globals<D> g) {
    const int warp_id = kittens::warpid();
    const int row_idx = blockIdx.x * NUM_WORKERS + warp_id;
    if (row_idx >= g.n_rows) return;

    constexpr int d_model = D;

    // Load input, residual, weight, bias
    rv_naive<bf16, d_model> x_reg, res_reg, w_reg, b_reg;
    load(x_reg,   g.input,    {0, 0, row_idx, 0});
    load(res_reg, g.residual, {0, 0, row_idx, 0});
    load(w_reg,   g.weight,   {0, 0, 0, 0});
    load(b_reg,   g.bias,     {0, 0, 0, 0});

    // Fused add: res_out = x + residual
    add(x_reg, x_reg, res_reg);  // x_reg now holds res_out

    // Store res_out
    store(g.res_out, x_reg, {0, 0, row_idx, 0});

    // Compute mean
    bf16 mean_bf;
    sum(mean_bf, x_reg);
    float mean_f = __bfloat162float(mean_bf) / static_cast<float>(d_model);
    bf16 mean_val = __float2bfloat16(mean_f);

    // Subtract mean
    rv_naive<bf16, d_model> x_centered;
    sub(x_centered, x_reg, mean_val);

    // Compute variance
    rv_naive<bf16, d_model> x_sq;
    mul(x_sq, x_centered, x_centered);

    bf16 var_sum_bf;
    sum(var_sum_bf, x_sq);
    float var_f = __bfloat162float(var_sum_bf) / static_cast<float>(d_model);

    // rstd = rsqrt(var + eps)
    float rstd_f = rsqrtf(var_f + g.epsilon);
    bf16 rstd_val = __float2bfloat16(rstd_f);

    // output = (x - mean) * rstd * weight + bias
    rv_naive<bf16, d_model> out_reg;
    mul(out_reg, x_centered, rstd_val);
    mul(out_reg, out_reg, w_reg);
    add(out_reg, out_reg, b_reg);

    store(g.output, out_reg, {0, 0, row_idx, 0});
}

template <int D>
void dispatch_fused_add_layernorm_fwd(fused_add_layernorm_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fused_add_layernorm_fwd_kernel<D>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fused_add_layernorm_fwd_kernel<D><<<g.grid(), g.block(), mem_size>>>(g);
}

// ---------------------------------------------------------------------------
// Template instantiation and pybind11 bindings
// ---------------------------------------------------------------------------

constexpr int D = 128;

PYBIND11_MODULE(layernorm_tk, m) {
    m.doc() = "HipKittens LayerNorm kernels";

    // Basic LayerNorm forward
    py::bind_function<dispatch_layernorm_fwd<D>>(m, "layernorm_fwd",
        &layernorm_globals<D>::input,
        &layernorm_globals<D>::weight,
        &layernorm_globals<D>::bias,
        &layernorm_globals<D>::output,
        &layernorm_globals<D>::epsilon,
        &layernorm_globals<D>::n_rows
    );

    // Fused Add + LayerNorm forward
    py::bind_function<dispatch_fused_add_layernorm_fwd<D>>(m, "fused_add_layernorm_fwd",
        &fused_add_layernorm_globals<D>::input,
        &fused_add_layernorm_globals<D>::residual,
        &fused_add_layernorm_globals<D>::weight,
        &fused_add_layernorm_globals<D>::bias,
        &fused_add_layernorm_globals<D>::output,
        &fused_add_layernorm_globals<D>::res_out,
        &fused_add_layernorm_globals<D>::epsilon,
        &fused_add_layernorm_globals<D>::n_rows
    );
}
