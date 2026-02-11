// RoPE (Rotary Position Embeddings) HipKittens Kernel
// Ported from reference/triton/rope/rope.py
// Closely follows reference/hipkittens/rotary/kernel.cpp pattern
//
// Applies rotary position encoding using NeoX-style rotation:
//   x_out[..., :D/2] = x[..., :D/2] * cos - x[..., D/2:] * sin
//   x_out[..., D/2:] = x[..., D/2:] * cos + x[..., :D/2] * sin
//
// Layout: (B, H, N, D) — batch, heads, sequence length, head dimension
// cos/sin: (N, D/2) — precomputed rotary frequencies

#include "kittens.cuh"

#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

using namespace kittens;

// Block size for sequence tiling — each warp processes BLOCK_SIZE positions
constexpr int BLOCK_SIZE = 32;

// Register tile shape matching the rotary reference
using shape = kittens::rt_32x32_8_s;

template<int HEAD_DIM>
struct rope_globals {
    static constexpr int d_model = HEAD_DIM;
    static constexpr int HALF_DIM = HEAD_DIM / 2;

    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    o_gl o;
    sin_gl sin_freq;
    cos_gl cos_freq;

    int B;   // batch size
    int H;   // num heads
    int N;   // sequence length

    hipStream_t stream;

    dim3 grid() {
        return dim3(H, (B + NUM_WORKERS - 1) / NUM_WORKERS, N / BLOCK_SIZE);
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int HEAD_DIM>
__global__ void rope_kernel(const rope_globals<HEAD_DIM> g) {
    constexpr int HALF_DIM = HEAD_DIM / 2;

    const int b = blockIdx.y * NUM_WORKERS + kittens::warpid();
    const int h = blockIdx.x;
    const int n = blockIdx.z;

    if (b >= g.B) return;

    // Register tiles: full x (BLOCK_SIZE x HEAD_DIM), cos/sin (BLOCK_SIZE x HALF_DIM)
    using tile_t = rt<bf16, BLOCK_SIZE, BLOCK_SIZE, row_l, shape>;
    constexpr int half_dim_tiles = HALF_DIM / BLOCK_SIZE;
    constexpr int PT = tile_t::packed_per_thread;

    rt<bf16, BLOCK_SIZE, HEAD_DIM, row_l, shape> x_reg;
    rt<bf16, BLOCK_SIZE, HALF_DIM, row_l, shape> cos_reg, sin_reg;

    // Load cos/sin for this sequence position block
    // cos/sin shape: (1, 1, N/BLOCK_SIZE blocks, HALF_DIM)
    load<2>(cos_reg, g.cos_freq, {0, 0, n, 0});
    load<2>(sin_reg, g.sin_freq, {0, 0, n, 0});

    // Load input x for this (batch, head, seq_block)
    load<2>(x_reg, g.x, {b, h, n, 0});

    asm volatile("s_waitcnt lgkmcnt(0)");

    // Apply NeoX-style rotary embedding:
    // out[..., :D/2] = x[..., :D/2] * cos - x[..., D/2:] * sin
    // out[..., D/2:] = x[..., D/2:] * cos + x[..., :D/2] * sin
    #pragma unroll
    for (int i = 0; i < half_dim_tiles; ++i) {
        #pragma unroll
        for (int j = 0; j < PT; ++j) {
            const auto x1 = x_reg.tiles[0][i].data[j];               // x[..., :D/2]
            const auto x2 = x_reg.tiles[0][i + half_dim_tiles].data[j]; // x[..., D/2:]
            // out_first_half = x1 * cos - x2 * sin
            x_reg.tiles[0][i].data[j] = __hsub2(
                __hmul2(x1, cos_reg.tiles[0][i].data[j]),
                __hmul2(x2, sin_reg.tiles[0][i].data[j])
            );
            // out_second_half = x2 * cos + x1 * sin
            x_reg.tiles[0][i + half_dim_tiles].data[j] = __hadd2(
                __hmul2(x2, cos_reg.tiles[0][i].data[j]),
                __hmul2(x1, sin_reg.tiles[0][i].data[j])
            );
        }
    }

    // Store output
    store(g.o, x_reg, {b, h, n, 0});
}

template<int HEAD_DIM>
void dispatch_rope(rope_globals<HEAD_DIM>& g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rope_kernel<HEAD_DIM>,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rope_kernel<HEAD_DIM><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Explicit instantiations for common head dimensions
template void dispatch_rope<64>(rope_globals<64>&);
template void dispatch_rope<128>(rope_globals<128>&);
template void dispatch_rope<256>(rope_globals<256>&);

// ---- pybind11 bindings ----
#include <pybind11/pybind11.h>

// Helper: extract shape from a torch.Tensor pybind11 object
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

template<int HEAD_DIM>
void rope_fwd_impl(pybind11::object x, pybind11::object out, pybind11::object cos_freq, pybind11::object sin_freq) {
    auto xs = get_tensor_shape(x);
    int B = xs[0], H = xs[1], N = xs[2];

    using globals = rope_globals<HEAD_DIM>;
    globals g{
        make_gl<typename globals::x_gl>(get_data_ptr(x), B, H, N, HEAD_DIM),
        make_gl<typename globals::o_gl>(get_data_ptr(out), B, H, N, HEAD_DIM),
        make_gl<typename globals::sin_gl>(get_data_ptr(sin_freq), 1, 1, N, HEAD_DIM / 2),
        make_gl<typename globals::cos_gl>(get_data_ptr(cos_freq), 1, 1, N, HEAD_DIM / 2),
        B, H, N,
        0  // default HIP stream
    };

    dispatch_rope<HEAD_DIM>(g);
}

void rope_fwd(pybind11::object x, pybind11::object out, pybind11::object cos_freq, pybind11::object sin_freq) {
    auto xs = get_tensor_shape(x);
    int D = xs[3];
    if (D == 64)       rope_fwd_impl<64>(x, out, cos_freq, sin_freq);
    else if (D == 128) rope_fwd_impl<128>(x, out, cos_freq, sin_freq);
    else if (D == 256) rope_fwd_impl<256>(x, out, cos_freq, sin_freq);
    else throw std::runtime_error("Unsupported head dimension: " + std::to_string(D) + ". Supported: 64, 128, 256.");
}

PYBIND11_MODULE(rope_tk, m) {
    m.doc() = "HipKittens RoPE kernel";
    m.def("rope_fwd", &rope_fwd, "RoPE forward pass");
}
