// Activation Functions HipKittens Kernel
// Ported from reference/triton/activation.py
//
// Supports: SiLU, GeLU, GeLU-tanh, ReLU, Tanh
// Also supports fused gated variant: act(a) * b (for SwiGLU/GeGLU patterns)
//
// Input layout: (M, N) bf16 — 2D tensor
// For gated variant: input is (M, 2*N), first half is 'a', second half is 'b'
// Output layout: (M, N) bf16
//
// Each thread block handles a tile of the input. We use a simple element-wise
// approach with HIP intrinsics for bf16 operations.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

// Activation type enum
enum class ActivationType : int {
    SILU = 0,
    GELU = 1,
    GELU_TANH = 2,
    RELU = 3,
    TANH = 4
};

// Device functions for activation computations (operate in fp32 for accuracy)
__device__ __forceinline__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float device_silu(float x) {
    return x * device_sigmoid(x);
}

__device__ __forceinline__ float device_gelu(float x) {
    constexpr float M_SQRT1_2_F = 0.7071067811865476f;
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2_F));
}

__device__ __forceinline__ float device_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float device_gelu_tanh(float x) {
    constexpr float BETA = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float device_relu(float x) {
    return fmaxf(0.0f, x);
}

template<ActivationType ACT>
__device__ __forceinline__ float apply_activation(float x) {
    if constexpr (ACT == ActivationType::SILU) {
        return device_silu(x);
    } else if constexpr (ACT == ActivationType::GELU) {
        return device_gelu(x);
    } else if constexpr (ACT == ActivationType::GELU_TANH) {
        return device_gelu_tanh(x);
    } else if constexpr (ACT == ActivationType::RELU) {
        return device_relu(x);
    } else if constexpr (ACT == ActivationType::TANH) {
        return device_tanh(x);
    }
    return x;
}

// Simple element-wise activation kernel
// input: (M, N), output: (M, N)
template<ActivationType ACT>
__global__ void activation_kernel(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    // Each thread processes one element
    if (idx < total) {
        float val = __bfloat162float(input[idx]);
        float result = apply_activation<ACT>(val);
        output[idx] = __float2bfloat16(result);
    }
}

// Fused gated activation kernel: output = act(a) * b
// input: (M, 2*N) where a = input[:, :N], b = input[:, N:]
// output: (M, N)
template<ActivationType ACT>
__global__ void gated_activation_kernel(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;

        // a is in first N columns, b is in second N columns
        float a = __bfloat162float(input[row * (2 * N) + col]);
        float b = __bfloat162float(input[row * (2 * N) + N + col]);

        float result = apply_activation<ACT>(a) * b;
        output[idx] = __float2bfloat16(result);
    }
}

// Dispatch functions
extern "C" {

void launch_silu(const __hip_bfloat16* input, __hip_bfloat16* output,
                 int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    activation_kernel<ActivationType::SILU><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

void launch_gelu(const __hip_bfloat16* input, __hip_bfloat16* output,
                 int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    activation_kernel<ActivationType::GELU><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

void launch_gelu_tanh(const __hip_bfloat16* input, __hip_bfloat16* output,
                      int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    activation_kernel<ActivationType::GELU_TANH><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

void launch_relu(const __hip_bfloat16* input, __hip_bfloat16* output,
                 int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    activation_kernel<ActivationType::RELU><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

void launch_tanh_act(const __hip_bfloat16* input, __hip_bfloat16* output,
                     int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    activation_kernel<ActivationType::TANH><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

// Gated variants (SwiGLU, GeGLU patterns)
void launch_silu_and_mul(const __hip_bfloat16* input, __hip_bfloat16* output,
                         int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gated_activation_kernel<ActivationType::SILU><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

void launch_gelu_and_mul(const __hip_bfloat16* input, __hip_bfloat16* output,
                         int M, int N, hipStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gated_activation_kernel<ActivationType::GELU><<<blocks, threads, 0, stream>>>(input, output, M, N);
}

} // extern "C"

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

static uint64_t _get_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}
static int _get_dim(pybind11::object t, int d) {
    return pybind11::cast<int>(t.attr("shape").cast<pybind11::tuple>()[d]);
}

void silu_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    launch_silu((const __hip_bfloat16*)_get_ptr(input),
                (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void gelu_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    launch_gelu((const __hip_bfloat16*)_get_ptr(input),
                (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void gelu_tanh_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    launch_gelu_tanh((const __hip_bfloat16*)_get_ptr(input),
                     (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void relu_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    launch_relu((const __hip_bfloat16*)_get_ptr(input),
                (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void tanh_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0), N = _get_dim(input, 1);
    launch_tanh_act((const __hip_bfloat16*)_get_ptr(input),
                    (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void silu_and_mul_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0);
    int N = _get_dim(output, 1);  // output is (M, N), input is (M, 2*N)
    launch_silu_and_mul((const __hip_bfloat16*)_get_ptr(input),
                        (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}
void gelu_and_mul_fwd(pybind11::object input, pybind11::object output) {
    int M = _get_dim(input, 0);
    int N = _get_dim(output, 1);
    launch_gelu_and_mul((const __hip_bfloat16*)_get_ptr(input),
                        (__hip_bfloat16*)_get_ptr(output), M, N, 0);
}

PYBIND11_MODULE(activation_tk, m) {
    m.doc() = "HipKittens activation kernels";
    m.def("silu_fwd", &silu_fwd, "SiLU activation");
    m.def("gelu_fwd", &gelu_fwd, "GeLU activation");
    m.def("gelu_tanh_fwd", &gelu_tanh_fwd, "GeLU-tanh activation");
    m.def("relu_fwd", &relu_fwd, "ReLU activation");
    m.def("tanh_fwd", &tanh_fwd, "Tanh activation");
    m.def("silu_and_mul_fwd", &silu_and_mul_fwd, "Gated SiLU (SwiGLU)");
    m.def("gelu_and_mul_fwd", &gelu_and_mul_fwd, "Gated GeLU (GeGLU)");
}
