// fused_mul_add: Element-wise fused multiply-add
// Implements: out = a * x + b
//
// Supports scalar and tensor variants for a and b:
// - a, b can be scalars (single value)
// - a, b can be tensors (element-wise)
//
// This is a simple vectorized element-wise kernel.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ELEMS_PER_THREAD = 4;  // Each thread processes 4 elements
constexpr int NUM_THREADS_MUL_ADD = 256;

template<typename T, int N>
struct fused_mul_add_globals {
    using _gl_x = gl<T, -1, -1, -1, N>;
    using _gl_a = gl<T, -1, -1, -1, N>;
    using _gl_b = gl<T, -1, -1, -1, N>;
    using _gl_out = gl<T, -1, -1, -1, N>;

    _gl_x x;
    _gl_a a;
    _gl_b b;
    _gl_out out;
    int total_elements;
    bool a_is_scalar;
    bool b_is_scalar;
    hipStream_t stream;

    dim3 grid() {
        return dim3(ceil_div(total_elements, ELEMS_PER_THREAD * NUM_THREADS_MUL_ADD));
    }
    dim3 block() { return dim3(NUM_THREADS_MUL_ADD); }
};

// Simple element-wise kernel using raw pointer access
template<typename T, int N>
__global__ void fused_mul_add_kernel(
    const fused_mul_add_globals<T, N> g
) {
    const int total = g.total_elements;
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMS_PER_THREAD;

    if (base >= total) return;

    const T* x_base = (T*)&g.x[{0, 0, 0, 0}];
    const T* a_base = (T*)&g.a[{0, 0, 0, 0}];
    const T* b_base = (T*)&g.b[{0, 0, 0, 0}];
    T* out_base = (T*)&g.out[{0, 0, 0, 0}];

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx >= total) break;

        float x_f = float(x_base[idx]);
        float a_f = g.a_is_scalar ? float(a_base[0]) : float(a_base[idx]);
        float b_f = g.b_is_scalar ? float(b_base[0]) : float(b_base[idx]);
        out_base[idx] = T(a_f * x_f + b_f);
    }
}

template<typename T, int N>
void dispatch_fused_mul_add(fused_mul_add_globals<T, N>& g) {
    fused_mul_add_kernel<T, N><<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// Explicit instantiations
template void dispatch_fused_mul_add<bf16, 4096>(fused_mul_add_globals<bf16, 4096>&);
template void dispatch_fused_mul_add<bf16, 8192>(fused_mul_add_globals<bf16, 8192>&);
template void dispatch_fused_mul_add<half, 4096>(fused_mul_add_globals<half, 4096>&);
template void dispatch_fused_mul_add<half, 8192>(fused_mul_add_globals<half, 8192>&);

PYBIND11_MODULE(fused_mul_add_tk, m) {
    m.doc() = "HipKittens Fused Multiply-Add kernel: out = a * x + b";

    kittens::py::bind_function<dispatch_fused_mul_add<bf16, 4096>>(m, "dispatch_bf16_4096",
        &fused_mul_add_globals<bf16, 4096>::x,
        &fused_mul_add_globals<bf16, 4096>::a,
        &fused_mul_add_globals<bf16, 4096>::b,
        &fused_mul_add_globals<bf16, 4096>::out,
        &fused_mul_add_globals<bf16, 4096>::total_elements,
        &fused_mul_add_globals<bf16, 4096>::a_is_scalar,
        &fused_mul_add_globals<bf16, 4096>::b_is_scalar);

    kittens::py::bind_function<dispatch_fused_mul_add<bf16, 8192>>(m, "dispatch_bf16_8192",
        &fused_mul_add_globals<bf16, 8192>::x,
        &fused_mul_add_globals<bf16, 8192>::a,
        &fused_mul_add_globals<bf16, 8192>::b,
        &fused_mul_add_globals<bf16, 8192>::out,
        &fused_mul_add_globals<bf16, 8192>::total_elements,
        &fused_mul_add_globals<bf16, 8192>::a_is_scalar,
        &fused_mul_add_globals<bf16, 8192>::b_is_scalar);

    kittens::py::bind_function<dispatch_fused_mul_add<half, 4096>>(m, "dispatch_fp16_4096",
        &fused_mul_add_globals<half, 4096>::x,
        &fused_mul_add_globals<half, 4096>::a,
        &fused_mul_add_globals<half, 4096>::b,
        &fused_mul_add_globals<half, 4096>::out,
        &fused_mul_add_globals<half, 4096>::total_elements,
        &fused_mul_add_globals<half, 4096>::a_is_scalar,
        &fused_mul_add_globals<half, 4096>::b_is_scalar);

    kittens::py::bind_function<dispatch_fused_mul_add<half, 8192>>(m, "dispatch_fp16_8192",
        &fused_mul_add_globals<half, 8192>::x,
        &fused_mul_add_globals<half, 8192>::a,
        &fused_mul_add_globals<half, 8192>::b,
        &fused_mul_add_globals<half, 8192>::out,
        &fused_mul_add_globals<half, 8192>::total_elements,
        &fused_mul_add_globals<half, 8192>::a_is_scalar,
        &fused_mul_add_globals<half, 8192>::b_is_scalar);
}
