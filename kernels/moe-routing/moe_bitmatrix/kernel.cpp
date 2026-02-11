// MoE Bitmatrix Operations Kernel
// Ported from reference/triton/moe_routing/moe_routing/bitmatrix.py
//
// Compact representation of token->expert assignments as bit vectors.
// Each row is a token, each column group of 32 represents expert IDs 0-31, 32-63, etc.
// Bit matrix is used for efficient parallel computation of token counts per expert.
//
// Key operation: vpopc (vertical popcount)
//   Input: uint32[M, N] where each uint32 holds 32 bits
//   Output: uint32[M, 32*N] counts of set bits in each bit position
//   Used to aggregate token counts across many tokens in parallel

#include <hip/hip_runtime.h>
#include <cstdint>

// Device function: vertical popcount
// Input: bits[BLOCK_M] - array of uint32, each with 32 bits
// Output: counts[32] - count of set bits at each bit position across BLOCK_M elements
// Credits: @apgoucher (from Triton vpopc implementation)
template<int BLOCK_M>
__device__ void vpopc_block(const uint32_t* bits, uint32_t* counts) {
    static_assert(BLOCK_M >= 8, "BLOCK_M must be >= 8");
    static_assert((BLOCK_M & (BLOCK_M - 1)) == 0, "BLOCK_M must be power of 2");

    // Algorithm: hierarchical summation with bit packing
    // Step 1: 8-way sums in 4-bit fields
    constexpr int SA1 = (BLOCK_M >= 8) ? 8 : BLOCK_M;
    uint32_t y[BLOCK_M / SA1][4];

    for (int i = 0; i < BLOCK_M / SA1; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t acc = 0;
            for (int k = 0; k < SA1; k++) {
                uint32_t val = bits[i * SA1 + k];
                acc += (val >> j) & 0x11111111u;
            }
            y[i][j] = acc;
        }
    }

    // Step 2: 128-way sums in 8-bit fields
    constexpr int SA2 = (BLOCK_M >= 128) ? 16 : (BLOCK_M / SA1);
    uint32_t z[BLOCK_M / (SA1 * SA2)][2][4];

    for (int i = 0; i < BLOCK_M / (SA1 * SA2); i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                uint32_t acc = 0;
                for (int l = 0; l < SA2; l++) {
                    uint32_t val = y[i * SA2 + l][k];
                    acc += (val >> (j * 4)) & 0x0F0F0F0Fu;
                }
                z[i][j][k] = acc;
            }
        }
    }

    // Step 3: Full sum in 32-bit fields
    constexpr int SA3 = BLOCK_M / (SA1 * SA2);
    uint32_t w[4][8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            uint32_t acc = 0;
            for (int k = 0; k < SA3; k++) {
                uint32_t val = z[k][j / 4][j % 4];
                acc += (val >> (i * 8)) & 0xFFu;
            }
            w[i][j] = acc;
        }
    }

    // Flatten to output: [32]
    for (int i = 0; i < 32; i++) {
        counts[i] = w[i / 8][i % 8];
    }
}

// Sum bitmatrix rows kernel
// Computes histogram of expert assignments from bitmatrix representation
// B: [shape_bm, shape_bn] uint32 - input bitmatrix
// Ret: [shape_bn * 32] uint32 - output counts per expert
// Partials: [num_tiles, shape_bn * 32] uint32 - partial sums per tile (for cumsum)
template<int BLOCK_MM, int BLOCK_M>
__global__ void sum_bitmatrix_rows_kernel(
    const uint32_t* __restrict__ B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    uint32_t* __restrict__ Ret,
    uint32_t* __restrict__ Partials,
    int32_t stride_pm,
    int32_t stride_pn,
    int32_t num_pids_m
) {
    static_assert(BLOCK_MM % BLOCK_M == 0, "BLOCK_MM must be multiple of BLOCK_M");
    constexpr int TILE_SIZE = BLOCK_MM / BLOCK_M;

    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;

    int offs_m_start = pid_m * BLOCK_MM;
    int offs_n_start = pid_n * 32;

    // Load bitmatrix tile: [BLOCK_MM]
    uint32_t bits[BLOCK_MM];
    for (int i = 0; i < BLOCK_MM; i++) {
        int row = offs_m_start + i;
        if (row < shape_bm) {
            bits[i] = B[pid_n * stride_bn + row * stride_bm];
        } else {
            bits[i] = 0;
        }
    }

    // Compute vertical popcount in sub-tiles
    uint32_t tile_counts[TILE_SIZE][32];
    for (int tile_idx = 0; tile_idx < TILE_SIZE; tile_idx++) {
        vpopc_block<BLOCK_M>(&bits[tile_idx * BLOCK_M], tile_counts[tile_idx]);
    }

    // Atomic add total to Ret
    uint32_t total_counts[32] = {0};
    for (int tile_idx = 0; tile_idx < TILE_SIZE; tile_idx++) {
        for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
            total_counts[bit_pos] += tile_counts[tile_idx][bit_pos];
        }
    }

    for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
        atomicAdd(&Ret[offs_n_start + bit_pos], total_counts[bit_pos]);
    }

    // Write partial cumulative sums to Partials for later stages
    // Partials[tile_id, expert_bit] = cumsum up to (not including) this tile
    int tile_id_base = pid_m * TILE_SIZE;
    for (int tile_idx = 0; tile_idx < TILE_SIZE; tile_idx++) {
        int tile_id = tile_id_base + tile_idx;
        uint32_t cumsum_counts[32] = {0};

        // Cumsum up to this tile (exclusive)
        for (int prev_tile = 0; prev_tile < tile_idx; prev_tile++) {
            for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
                cumsum_counts[bit_pos] += tile_counts[prev_tile][bit_pos];
            }
        }

        for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
            atomicAdd(&Partials[tile_id * stride_pm + (offs_n_start + bit_pos) * stride_pn],
                      cumsum_counts[bit_pos]);
        }
    }

    // Add cumulative contribution to all subsequent tiles in other thread blocks
    for (int future_pid_m = pid_m + 1; future_pid_m < num_pids_m; future_pid_m++) {
        int future_tile_base = future_pid_m * TILE_SIZE;
        for (int future_tile_idx = 0; future_tile_idx < TILE_SIZE; future_tile_idx++) {
            int future_tile_id = future_tile_base + future_tile_idx;
            for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
                atomicAdd(&Partials[future_tile_id * stride_pm +
                                    (offs_n_start + bit_pos) * stride_pn],
                          total_counts[bit_pos]);
            }
        }
    }
}

// Fused version: single-block, computes full histogram in one pass
// Used when bitmatrix fits in a single block (common for small MoE problems)
template<int BLOCK_M>
__global__ void sum_bitmatrix_rows_fused_kernel(
    const uint32_t* __restrict__ B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    uint32_t* __restrict__ Ret,
    int32_t N_BLKS_BITMATRIX
) {
    // Single block handles entire matrix
    if (blockIdx.x > 0 || blockIdx.y > 0) return;

    for (int blk_n = 0; blk_n < N_BLKS_BITMATRIX; blk_n++) {
        int offs_n_start = blk_n * 32;

        // Load bits for this column block
        uint32_t bits[BLOCK_M];
        for (int i = 0; i < BLOCK_M; i++) {
            if (i < shape_bm) {
                bits[i] = B[blk_n * stride_bn + i * stride_bm];
            } else {
                bits[i] = 0;
            }
        }

        // Compute vertical popcount
        uint32_t counts[32];
        vpopc_block<BLOCK_M>(bits, counts);

        // Write to output
        for (int bit_pos = 0; bit_pos < 32; bit_pos++) {
            Ret[offs_n_start + bit_pos] = counts[bit_pos];
        }
    }
}

extern "C" {

// Launch sum_bitmatrix_rows (general case with partials)
void launch_sum_bitmatrix_rows(
    const uint32_t* B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    uint32_t* Ret,
    uint32_t* Partials,
    int32_t stride_pm,
    int32_t stride_pn,
    int32_t shape_pn,
    int32_t shape_bn,
    hipStream_t stream
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_MM = 256;

    int num_pids_m = (shape_bm + BLOCK_MM - 1) / BLOCK_MM;
    int num_pids_n = shape_bn;

    dim3 grid(num_pids_m, num_pids_n);
    dim3 block(1);  // Single thread per block (vpopc is sequential)

    // Zero outputs
    int num_experts = shape_bn * 32;
    hipMemsetAsync(Ret, 0, num_experts * sizeof(uint32_t), stream);
    hipMemsetAsync(Partials, 0, stride_pm * shape_pn * sizeof(uint32_t), stream);

    hipLaunchKernelGGL(
        (sum_bitmatrix_rows_kernel<BLOCK_MM, BLOCK_M>),
        grid, block, 0, stream,
        B, shape_bm, stride_bm, stride_bn, Ret, Partials,
        stride_pm, stride_pn, num_pids_m
    );
}

// Launch sum_bitmatrix_rows_fused (single-block case)
void launch_sum_bitmatrix_rows_fused(
    const uint32_t* B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    uint32_t* Ret,
    int32_t N_BLKS_BITMATRIX,
    hipStream_t stream
) {
    // Choose BLOCK_M based on shape_bm
    // Must be power of 2 and >= 8
    int block_m = 128;
    if (shape_bm <= 64) block_m = 64;
    if (shape_bm <= 32) block_m = 32;
    if (shape_bm <= 16) block_m = 16;
    if (shape_bm <= 8) block_m = 8;

    // Zero output
    hipMemsetAsync(Ret, 0, N_BLKS_BITMATRIX * 32 * sizeof(uint32_t), stream);

    if (block_m == 128) {
        hipLaunchKernelGGL(
            (sum_bitmatrix_rows_fused_kernel<128>),
            dim3(1), dim3(1), 0, stream,
            B, shape_bm, stride_bm, stride_bn, Ret, N_BLKS_BITMATRIX
        );
    } else if (block_m == 64) {
        hipLaunchKernelGGL(
            (sum_bitmatrix_rows_fused_kernel<64>),
            dim3(1), dim3(1), 0, stream,
            B, shape_bm, stride_bm, stride_bn, Ret, N_BLKS_BITMATRIX
        );
    } else if (block_m == 32) {
        hipLaunchKernelGGL(
            (sum_bitmatrix_rows_fused_kernel<32>),
            dim3(1), dim3(1), 0, stream,
            B, shape_bm, stride_bm, stride_bn, Ret, N_BLKS_BITMATRIX
        );
    } else if (block_m == 16) {
        hipLaunchKernelGGL(
            (sum_bitmatrix_rows_fused_kernel<16>),
            dim3(1), dim3(1), 0, stream,
            B, shape_bm, stride_bm, stride_bn, Ret, N_BLKS_BITMATRIX
        );
    } else {  // block_m == 8
        hipLaunchKernelGGL(
            (sum_bitmatrix_rows_fused_kernel<8>),
            dim3(1), dim3(1), 0, stream,
            B, shape_bm, stride_bm, stride_bn, Ret, N_BLKS_BITMATRIX
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

void sum_bitmatrix_rows_py(
    pybind11::object B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    pybind11::object Ret,
    pybind11::object Partials,
    int32_t stride_pm,
    int32_t stride_pn,
    int32_t shape_pn,
    int32_t shape_bn
) {
    launch_sum_bitmatrix_rows(
        reinterpret_cast<const uint32_t*>(get_data_ptr(B)),
        shape_bm, stride_bm, stride_bn,
        reinterpret_cast<uint32_t*>(get_data_ptr(Ret)),
        reinterpret_cast<uint32_t*>(get_data_ptr(Partials)),
        stride_pm, stride_pn, shape_pn, shape_bn,
        0  // default stream
    );
}

void sum_bitmatrix_rows_fused_py(
    pybind11::object B,
    int32_t shape_bm,
    int32_t stride_bm,
    int32_t stride_bn,
    pybind11::object Ret,
    int32_t N_BLKS_BITMATRIX
) {
    launch_sum_bitmatrix_rows_fused(
        reinterpret_cast<const uint32_t*>(get_data_ptr(B)),
        shape_bm, stride_bm, stride_bn,
        reinterpret_cast<uint32_t*>(get_data_ptr(Ret)),
        N_BLKS_BITMATRIX,
        0  // default stream
    );
}

PYBIND11_MODULE(moe_bitmatrix_tk, m) {
    m.def("sum_bitmatrix_rows", &sum_bitmatrix_rows_py,
          "Sum bitmatrix rows (general case with partials)",
          pybind11::arg("B"),
          pybind11::arg("shape_bm"),
          pybind11::arg("stride_bm"),
          pybind11::arg("stride_bn"),
          pybind11::arg("Ret"),
          pybind11::arg("Partials"),
          pybind11::arg("stride_pm"),
          pybind11::arg("stride_pn"),
          pybind11::arg("shape_pn"),
          pybind11::arg("shape_bn"));
    m.def("sum_bitmatrix_rows_fused", &sum_bitmatrix_rows_fused_py,
          "Sum bitmatrix rows (single-block fused case)",
          pybind11::arg("B"),
          pybind11::arg("shape_bm"),
          pybind11::arg("stride_bm"),
          pybind11::arg("stride_bn"),
          pybind11::arg("Ret"),
          pybind11::arg("N_BLKS_BITMATRIX"));
}
