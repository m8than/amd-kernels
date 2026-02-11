// MoE Expert Data Preparation Kernel
// Ported from reference/triton/moe_routing/moe_routing/expt_data.py
//
// Prepares metadata for expert computation:
//   - TokenStart[expert]: cumulative token offset for each expert
//   - TileStart[expert]: cumulative tile offset (for tiled GEMM)
//   - MDTileInfo[tile]: metadata (tile_id, expert_id) for each tile
//
// 2-stage algorithm:
//   Stage 1: Compute cumulative token/tile starts from histogram
//   Stage 2: Fill tile metadata array with expert IDs

#include <hip/hip_runtime.h>
#include <cstdint>

// Helper: ceiling division for power-of-2
__device__ __forceinline__ int32_t cdiv_pow2(int32_t n, int32_t log2_k) {
    return (n + ((1 << log2_k) - 1)) >> log2_k;
}

// Stage 1: Compute cumulative starts
// Hist[expert]: number of tokens assigned to each expert
// TokenStart[expert]: cumulative sum - where this expert's tokens start
// TileStart[expert]: cumulative sum of tiles
// Also initializes MDTileInfo with 0xFFFFFFFF for unused tiles
__global__ void expt_data_stage1_kernel(
    const int32_t* __restrict__ Hist,
    int32_t* __restrict__ TokenStart,
    int32_t* __restrict__ TileStart,
    uint32_t* __restrict__ MDTileInfo,
    int32_t n_expts_tot,
    int32_t max_num_tiles,
    int32_t n_gates,
    int32_t tile_dim_log2
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only one thread does the main work
    if (pid > 0) return;

    // Compute cumulative sums
    int32_t token_acc = 0;
    int32_t tile_acc = 0;

    for (int expert_id = 0; expert_id < n_expts_tot; expert_id++) {
        int32_t hist_token = Hist[expert_id];
        int32_t hist_tile = cdiv_pow2(hist_token, tile_dim_log2);

        TokenStart[expert_id] = token_acc;
        TileStart[expert_id] = tile_acc;

        token_acc += hist_token;
        tile_acc += hist_tile;
    }

    // Sentinel values at the end
    TokenStart[n_expts_tot] = n_gates;
    TileStart[n_expts_tot] = tile_acc;

    // Initialize unused tiles with 0xFFFFFFFF
    for (int tile_idx = tile_acc; tile_idx < max_num_tiles; tile_idx++) {
        MDTileInfo[tile_idx] = 0xFFFFFFFF;
    }
}

// Stage 2: Fill tile metadata
// For each expert, fill its tile range with (tile_local_id << 16 | expert_id)
// This is launched with one block per expert
__global__ void expt_data_stage2_kernel(
    const int32_t* __restrict__ Hist,
    const int32_t* __restrict__ TileStart,
    uint32_t* __restrict__ TileInfo,
    int32_t tile_dim_log2
) {
    int expert_id = blockIdx.x;

    int32_t n_tokens = Hist[expert_id];
    if (n_tokens == 0) return;

    int32_t n_tiles = cdiv_pow2(n_tokens, tile_dim_log2);
    int32_t tile_offset = TileStart[expert_id];

    // Each thread handles multiple tiles
    constexpr int BLOCK = 8;
    for (int i = threadIdx.x; i < n_tiles; i += blockDim.x) {
        uint32_t data = ((uint32_t)i << 16) | (uint32_t)expert_id;
        TileInfo[tile_offset + i] = data;
    }
}

// Fused Stage 2: simplified for single-tile-per-expert case
// Just writes expert_id to the tile position
__global__ void expt_data_stage2_fused_kernel(
    const int32_t* __restrict__ Hist,
    const int32_t* __restrict__ TileStart,
    uint32_t* __restrict__ TileInfo
) {
    int expert_id = blockIdx.x;

    int32_t n_tokens = Hist[expert_id];
    if (n_tokens == 0) return;

    int32_t tile_offset = TileStart[expert_id];
    if (threadIdx.x == 0) {
        TileInfo[tile_offset] = (uint32_t)expert_id;
    }
}

extern "C" {

// Launch Stage 1
void launch_expt_data_stage1(
    const int32_t* Hist,
    int32_t* TokenStart,
    int32_t* TileStart,
    uint32_t* MDTileInfo,
    int32_t n_expts_tot,
    int32_t max_num_tiles,
    int32_t n_gates,
    int32_t tile_dim_log2,
    hipStream_t stream
) {
    // Single thread kernel
    hipLaunchKernelGGL(
        expt_data_stage1_kernel,
        dim3(1), dim3(1), 0, stream,
        Hist, TokenStart, TileStart, MDTileInfo,
        n_expts_tot, max_num_tiles, n_gates, tile_dim_log2
    );
}

// Launch Stage 2
void launch_expt_data_stage2(
    const int32_t* Hist,
    const int32_t* TileStart,
    uint32_t* TileInfo,
    int32_t n_expts_tot,
    int32_t tile_dim_log2,
    hipStream_t stream
) {
    // One block per expert
    constexpr int BLOCK_SIZE = 64;
    hipLaunchKernelGGL(
        expt_data_stage2_kernel,
        dim3(n_expts_tot), dim3(BLOCK_SIZE), 0, stream,
        Hist, TileStart, TileInfo, tile_dim_log2
    );
}

// Launch Stage 2 Fused
void launch_expt_data_stage2_fused(
    const int32_t* Hist,
    const int32_t* TileStart,
    uint32_t* TileInfo,
    int32_t n_expts_tot,
    hipStream_t stream
) {
    // One block per expert, single thread
    hipLaunchKernelGGL(
        expt_data_stage2_fused_kernel,
        dim3(n_expts_tot), dim3(1), 0, stream,
        Hist, TileStart, TileInfo
    );
}

// Launch both stages together
void launch_expt_data_compute(
    const int32_t* Hist,
    int32_t* TokenStart,
    int32_t* TileStart,
    uint32_t* MDTileInfo,
    int32_t n_expts_tot,
    int32_t max_num_tiles,
    int32_t n_gates,
    int32_t tile_dim_log2,
    bool use_fused_stage2,
    hipStream_t stream
) {
    // Stage 1
    launch_expt_data_stage1(
        Hist, TokenStart, TileStart, MDTileInfo,
        n_expts_tot, max_num_tiles, n_gates, tile_dim_log2, stream
    );

    // Stage 2
    if (use_fused_stage2) {
        launch_expt_data_stage2_fused(
            Hist, TileStart, MDTileInfo, n_expts_tot, stream
        );
    } else {
        launch_expt_data_stage2(
            Hist, TileStart, MDTileInfo, n_expts_tot, tile_dim_log2, stream
        );
    }
}

} // extern "C"

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

void expt_data_stage1_py(
    pybind11::object Hist,
    pybind11::object TokenStart,
    pybind11::object TileStart,
    pybind11::object MDTileInfo,
    int32_t n_expts_tot,
    int32_t max_num_tiles,
    int32_t n_gates,
    int32_t tile_dim_log2
) {
    launch_expt_data_stage1(
        reinterpret_cast<const int32_t*>(get_data_ptr(Hist)),
        reinterpret_cast<int32_t*>(get_data_ptr(TokenStart)),
        reinterpret_cast<int32_t*>(get_data_ptr(TileStart)),
        reinterpret_cast<uint32_t*>(get_data_ptr(MDTileInfo)),
        n_expts_tot, max_num_tiles, n_gates, tile_dim_log2,
        0  // default stream
    );
}

void expt_data_stage2_py(
    pybind11::object Hist,
    pybind11::object TileStart,
    pybind11::object TileInfo,
    int32_t n_expts_tot,
    int32_t tile_dim_log2
) {
    launch_expt_data_stage2(
        reinterpret_cast<const int32_t*>(get_data_ptr(Hist)),
        reinterpret_cast<const int32_t*>(get_data_ptr(TileStart)),
        reinterpret_cast<uint32_t*>(get_data_ptr(TileInfo)),
        n_expts_tot, tile_dim_log2,
        0  // default stream
    );
}

void expt_data_stage2_fused_py(
    pybind11::object Hist,
    pybind11::object TileStart,
    pybind11::object TileInfo,
    int32_t n_expts_tot
) {
    launch_expt_data_stage2_fused(
        reinterpret_cast<const int32_t*>(get_data_ptr(Hist)),
        reinterpret_cast<const int32_t*>(get_data_ptr(TileStart)),
        reinterpret_cast<uint32_t*>(get_data_ptr(TileInfo)),
        n_expts_tot,
        0  // default stream
    );
}

void expt_data_compute_py(
    pybind11::object Hist,
    pybind11::object TokenStart,
    pybind11::object TileStart,
    pybind11::object MDTileInfo,
    int32_t n_expts_tot,
    int32_t max_num_tiles,
    int32_t n_gates,
    int32_t tile_dim_log2,
    bool use_fused_stage2
) {
    launch_expt_data_compute(
        reinterpret_cast<const int32_t*>(get_data_ptr(Hist)),
        reinterpret_cast<int32_t*>(get_data_ptr(TokenStart)),
        reinterpret_cast<int32_t*>(get_data_ptr(TileStart)),
        reinterpret_cast<uint32_t*>(get_data_ptr(MDTileInfo)),
        n_expts_tot, max_num_tiles, n_gates, tile_dim_log2,
        use_fused_stage2,
        0  // default stream
    );
}

PYBIND11_MODULE(moe_expt_data_tk, m) {
    m.def("expt_data_stage1", &expt_data_stage1_py,
          "Expert data preparation - stage 1 (cumulative starts)",
          pybind11::arg("Hist"),
          pybind11::arg("TokenStart"),
          pybind11::arg("TileStart"),
          pybind11::arg("MDTileInfo"),
          pybind11::arg("n_expts_tot"),
          pybind11::arg("max_num_tiles"),
          pybind11::arg("n_gates"),
          pybind11::arg("tile_dim_log2"));
    m.def("expt_data_stage2", &expt_data_stage2_py,
          "Expert data preparation - stage 2 (fill tile metadata)",
          pybind11::arg("Hist"),
          pybind11::arg("TileStart"),
          pybind11::arg("TileInfo"),
          pybind11::arg("n_expts_tot"),
          pybind11::arg("tile_dim_log2"));
    m.def("expt_data_stage2_fused", &expt_data_stage2_fused_py,
          "Expert data preparation - stage 2 fused (single tile per expert)",
          pybind11::arg("Hist"),
          pybind11::arg("TileStart"),
          pybind11::arg("TileInfo"),
          pybind11::arg("n_expts_tot"));
    m.def("expt_data_compute", &expt_data_compute_py,
          "Expert data preparation - both stages combined",
          pybind11::arg("Hist"),
          pybind11::arg("TokenStart"),
          pybind11::arg("TileStart"),
          pybind11::arg("MDTileInfo"),
          pybind11::arg("n_expts_tot"),
          pybind11::arg("max_num_tiles"),
          pybind11::arg("n_gates"),
          pybind11::arg("tile_dim_log2"),
          pybind11::arg("use_fused_stage2"));
}
