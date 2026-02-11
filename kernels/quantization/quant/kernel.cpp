#include "kittens.cuh"
using namespace kittens;

// Per-token INT8 quantization kernel
// Quantizes input tensor to INT8 per row (token)
// Formula: scale = max(abs(x)) / 127.0; q = round(x / scale).clamp(-127, 127)

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define QUANT_MAX 127.0f

// Global tensor descriptors
template<int D>
struct quant_globals {
    using input_gl = gl<bf16, -1, -1, -1, D>;
    using scale_gl = gl<float, -1, -1, -1, 1>;

    input_gl input;
    int8_t* output;  // Raw pointer for int8 output
    scale_gl scales;
    int rows;
    int cols;
    hipStream_t stream;

    dim3 grid() { return dim3(rows); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

// Device kernel: per-token dynamic quantization
template<int D, int BLOCK_SIZE>
__global__ void per_token_quant_kernel(const quant_globals<D> g) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / kittens::WARP_THREADS;
    const int lane_id = tid % kittens::WARP_THREADS;

    if (row >= g.rows) return;

    // Each warp processes a chunk of the row
    const int warps = NUM_WARPS;
    const int cols_per_warp = (g.cols + warps - 1) / warps;
    const int warp_start = warp_id * cols_per_warp;
    const int warp_end = min(warp_start + cols_per_warp, g.cols);

    // Phase 1: Find local max(abs(x)) per warp
    float local_max = 0.0f;
    for (int col = warp_start + lane_id; col < warp_end; col += kittens::WARP_THREADS) {
        bf16 val = g.input[{0, 0, row, col}];
        float val_f = __bfloat162float(val);
        local_max = fmaxf(local_max, fabsf(val_f));
    }

    // Warp-level reduction to find max within warp
    #pragma unroll
    for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset /= 2) {
        float other = __shfl_down(local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    // Share warp maxes in shared memory
    __shared__ float warp_maxes[NUM_WARPS];
    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();

    // First warp reduces across all warps
    float row_max = 0.0f;
    if (warp_id == 0) {
        if (lane_id < NUM_WARPS) {
            row_max = warp_maxes[lane_id];
        }
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
            float other = __shfl_down(row_max, offset);
            row_max = fmaxf(row_max, other);
        }
        // Broadcast final max to all lanes in first warp
        row_max = __shfl(row_max, 0);
    }
    __syncthreads();

    // Broadcast row_max to all threads
    if (warp_id == 0 && lane_id == 0) {
        warp_maxes[0] = row_max;
    }
    __syncthreads();
    row_max = warp_maxes[0];

    // Compute scale
    float scale = row_max / QUANT_MAX;
    scale = fmaxf(scale, 1e-10f); // Avoid division by zero
    float scale_recip = 1.0f / scale;

    // Store scale (only one thread per row)
    if (tid == 0) {
        g.scales[{0, 0, row, 0}] = scale;
    }

    // Phase 2: Quantize and store
    for (int col = warp_start + lane_id; col < warp_end; col += kittens::WARP_THREADS) {
        bf16 val = g.input[{0, 0, row, col}];
        float val_f = __bfloat162float(val);
        float quantized = rintf(val_f * scale_recip);
        quantized = fmaxf(fminf(quantized, QUANT_MAX), -QUANT_MAX);
        g.output[row * g.cols + col] = (int8_t)quantized;
    }
}

// Dispatch function
template<int D>
void dispatch_per_token_quant(quant_globals<D>& g) {
    hipLaunchKernelGGL(
        (per_token_quant_kernel<D, 1024>),
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        g.stream,
        g
    );
}

// Explicit template instantiations for common sizes
template void dispatch_per_token_quant<1024>(quant_globals<1024>&);
template void dispatch_per_token_quant<2048>(quant_globals<2048>&);
template void dispatch_per_token_quant<4096>(quant_globals<4096>&);
template void dispatch_per_token_quant<8192>(quant_globals<8192>&);

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

// input: (M, D) bf16, output: (M, D) int8, scales: (M, 1) f32
template<int D>
void per_token_quant_impl(pybind11::object input, pybind11::object output, pybind11::object scales) {
    int rows = _get_dim(input, 0);
    int cols = D;

    using globals = quant_globals<D>;
    globals g{
        make_gl<typename globals::input_gl>(_get_ptr(input), 1, 1, rows, D),
        (int8_t*)_get_ptr(output),
        make_gl<typename globals::scale_gl>(_get_ptr(scales), 1, 1, rows, 1),
        rows,
        cols,
        0  // default HIP stream
    };

    dispatch_per_token_quant<D>(g);
}

void per_token_quant_fwd(pybind11::object input, pybind11::object output, pybind11::object scales) {
    int D = _get_dim(input, 1);
    if (D == 1024)      per_token_quant_impl<1024>(input, output, scales);
    else if (D == 2048) per_token_quant_impl<2048>(input, output, scales);
    else if (D == 4096) per_token_quant_impl<4096>(input, output, scales);
    else if (D == 8192) per_token_quant_impl<8192>(input, output, scales);
    else throw std::runtime_error("Unsupported D=" + std::to_string(D) + ". Supported: 1024, 2048, 4096, 8192.");
}

PYBIND11_MODULE(quant_tk, m) {
    m.doc() = "HipKittens per-token INT8 quantization kernel";
    m.def("per_token_quant_fwd", &per_token_quant_fwd, "Per-token INT8 quantization forward pass");
}
