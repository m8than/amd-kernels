// MoE Routing Kernel
// Ported from reference/triton/moe_routing/moe_routing/routing.py
//
// Computes dispatch and gather indices for MoE routing:
//   - GatherIndx: maps output position -> input token ID
//   - ScatterIndx: maps input token -> output position
//   - GateScal: gating weights for each routed token
//
// Algorithm:
//   1. Load expert assignments (ExptIndx) and weights (ExptScal)
//   2. Sort by expert ID (stable sort using key-value pairs)
//   3. Compute run lengths per expert using keyed associative scan
//   4. Compute output positions using TokenStart offsets + run lengths
//   5. Write gather/scatter indices and gating weights

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// Device function: keyed add for associative scan
// Key is in upper 16 bits, value in lower 16 bits
// Only adds if keys match, otherwise returns y
__device__ __forceinline__ uint32_t keyed_add(uint32_t x, uint32_t y) {
    constexpr uint32_t KEY_MASK = 0xFFFF0000u;
    uint32_t kx = x & KEY_MASK;
    uint32_t ky = y & KEY_MASK;
    return (kx == ky) ? (x + y - kx) : y;
}

// Warp-level inclusive scan with custom operator
__device__ uint32_t warp_scan_inclusive(uint32_t val, uint32_t (*op)(uint32_t, uint32_t)) {
    constexpr int WARP_SIZE = 64;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint32_t other = __shfl_up(val, offset, WARP_SIZE);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val = op(val, other);
        }
    }
    return val;
}

// Block-level inclusive scan with keyed_add operator
template<int BLOCK_SIZE>
__device__ uint32_t block_scan_inclusive_keyed(uint32_t val, uint32_t* smem) {
    constexpr int WARP_SIZE = 64;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Warp-level scan
    uint32_t warp_result = warp_scan_inclusive(val, keyed_add);

    // Last thread in each warp writes to shared memory
    if (lane_id == WARP_SIZE - 1) {
        smem[warp_id] = warp_result;
    }
    __syncthreads();

    // First warp scans the warp results
    uint32_t warp_sum = 0;
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        warp_sum = smem[lane_id];
        warp_sum = warp_scan_inclusive(warp_sum, keyed_add);
        smem[lane_id] = warp_sum;
    }
    __syncthreads();

    // Add prefix from previous warps
    uint32_t prefix = (warp_id > 0) ? smem[warp_id - 1] : 0;
    uint32_t result = keyed_add(warp_result, prefix);

    __syncthreads();
    return result;
}

// Routing indices computation kernel
// Sorts tokens by expert, computes run lengths, and generates scatter/gather indices
template<int BLOCK_M, int N_EXPTS_ACT>
__global__ void routing_compute_indx_kernel(
    int32_t* __restrict__ GatherIndx,
    int32_t* __restrict__ ScatterIndx,
    __hip_bfloat16* __restrict__ GateScal,
    const __hip_bfloat16* __restrict__ ExptScal,
    const int32_t* __restrict__ ExptIndx,
    const int32_t* __restrict__ PartialOffs,
    int32_t stride_pm,
    int32_t stride_pn,
    const int32_t* __restrict__ TokensStart,
    int32_t n_gates,
    bool even_m
) {
    static_assert(N_EXPTS_ACT * BLOCK_M <= 32768, "Block size too large");

    int pid_m = blockIdx.x;
    int tid = threadIdx.x;

    constexpr int TOTAL_ELEMENTS = N_EXPTS_ACT * BLOCK_M;

    __shared__ uint32_t kv_pairs[TOTAL_ELEMENTS];
    __shared__ uint32_t scan_buffer[TOTAL_ELEMENTS / 64];  // For warp intermediates

    // Load expert indices and create key-value pairs
    int local_offs = tid;
    while (local_offs < TOTAL_ELEMENTS) {
        int global_offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs;
        uint32_t expert;

        if (even_m || global_offs < n_gates) {
            expert = (uint32_t)ExptIndx[global_offs];
        } else {
            expert = 0xFFFFu;  // Sentinel for out-of-bounds
        }

        // Create key-value pair: (expert_id << 16) | local_offset
        kv_pairs[local_offs] = (expert << 16) | (uint32_t)local_offs;

        local_offs += blockDim.x;
    }
    __syncthreads();

    // Sort by expert ID (simple bitonic sort for small arrays)
    // For larger arrays, would need a more sophisticated sort
    // Here we use a simplified approach: bubble sort within shared memory
    // This is suboptimal but matches Triton's tl.sort() semantics

    // Simplified sort: each thread handles a portion
    // In production, would use radix sort or bitonic sort
    // For now, use a basic parallel sort
    for (int stride = TOTAL_ELEMENTS / 2; stride > 0; stride >>= 1) {
        int idx = tid;
        while (idx < TOTAL_ELEMENTS) {
            int pair_idx = idx ^ stride;
            if (pair_idx > idx) {
                uint32_t a = kv_pairs[idx];
                uint32_t b = kv_pairs[pair_idx];

                // Sort by key (upper 16 bits)
                bool should_swap = (a > b);

                if (should_swap) {
                    kv_pairs[idx] = b;
                    kv_pairs[pair_idx] = a;
                }
            }
            idx += blockDim.x;
        }
        __syncthreads();
    }

    // Extract sorted expert IDs and offsets
    local_offs = tid;
    while (local_offs < TOTAL_ELEMENTS) {
        uint32_t kv = kv_pairs[local_offs];
        uint32_t expert = kv >> 16;
        uint32_t orig_offs = kv & 0xFFFFu;
        int global_offs = pid_m * BLOCK_M * N_EXPTS_ACT + orig_offs;

        // Skip invalid entries
        if (expert == 0xFFFFu) {
            local_offs += blockDim.x;
            continue;
        }

        // Load gate scaling
        __hip_bfloat16 gate_scal;
        if (even_m || global_offs < n_gates) {
            gate_scal = ExptScal[global_offs];
        } else {
            gate_scal = __float2bfloat16(0.0f);
        }

        // Compute run length using keyed scan
        // Create scan input: (expert << 16) | 1
        uint32_t scan_input = (expert << 16) | 0x00000001u;

        // Perform inclusive scan
        __shared__ uint32_t scan_smem[TOTAL_ELEMENTS / 64];
        uint32_t scan_result = block_scan_inclusive_keyed<blockDim.x>(scan_input, scan_smem);

        // Extract exclusive run length
        uint32_t exclusive_run_length = (scan_result - 1) & 0xFFFFu;

        // Compute output gate position
        int32_t gate_pos;
        if (PartialOffs != nullptr) {
            gate_pos = PartialOffs[pid_m * stride_pm + expert * stride_pn];
        } else {
            gate_pos = 0;
        }
        gate_pos += TokensStart[expert];
        gate_pos += exclusive_run_length;

        // Write outputs
        if (even_m || global_offs < n_gates) {
            ScatterIndx[global_offs] = gate_pos;
            GatherIndx[gate_pos] = global_offs;
            GateScal[gate_pos] = gate_scal;
        }

        local_offs += blockDim.x;
    }
}

// Fused variant without PartialOffs
template<int BLOCK_M, int N_EXPTS_ACT>
__global__ void routing_compute_indx_fused_kernel(
    int32_t* __restrict__ GatherIndx,
    int32_t* __restrict__ ScatterIndx,
    __hip_bfloat16* __restrict__ GateScal,
    const __hip_bfloat16* __restrict__ ExptScal,
    const int32_t* __restrict__ ExptIndx,
    const int32_t* __restrict__ TokensStart,
    int32_t n_gates,
    bool even_m
) {
    // Same as routing_compute_indx_kernel but without PartialOffs
    // Simplified: assume single block (pid_m = 0, no partial offsets)
    routing_compute_indx_kernel<BLOCK_M, N_EXPTS_ACT>(
        GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
        nullptr, 0, 0, TokensStart, n_gates, even_m
    );
}

extern "C" {

// Launch routing indices computation
void launch_routing_compute_indx(
    int32_t* GatherIndx,
    int32_t* ScatterIndx,
    __hip_bfloat16* GateScal,
    const __hip_bfloat16* ExptScal,
    const int32_t* ExptIndx,
    const int32_t* PartialOffs,
    int32_t stride_pm,
    int32_t stride_pn,
    const int32_t* TokensStart,
    int32_t n_gates,
    int32_t N_EXPTS_ACT,
    hipStream_t stream
) {
    constexpr int BLOCK_M = 128;
    int num_blocks = (n_gates + (BLOCK_M * N_EXPTS_ACT) - 1) / (BLOCK_M * N_EXPTS_ACT);
    bool even_m = (n_gates % (BLOCK_M * N_EXPTS_ACT)) == 0;

    // Choose kernel based on N_EXPTS_ACT
    if (N_EXPTS_ACT == 2) {
        hipLaunchKernelGGL(
            (routing_compute_indx_kernel<BLOCK_M, 2>),
            dim3(num_blocks), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            PartialOffs, stride_pm, stride_pn, TokensStart, n_gates, even_m
        );
    } else if (N_EXPTS_ACT == 4) {
        hipLaunchKernelGGL(
            (routing_compute_indx_kernel<BLOCK_M, 4>),
            dim3(num_blocks), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            PartialOffs, stride_pm, stride_pn, TokensStart, n_gates, even_m
        );
    } else if (N_EXPTS_ACT == 8) {
        hipLaunchKernelGGL(
            (routing_compute_indx_kernel<BLOCK_M, 8>),
            dim3(num_blocks), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            PartialOffs, stride_pm, stride_pn, TokensStart, n_gates, even_m
        );
    }
}

// Launch fused variant
void launch_routing_compute_indx_fused(
    int32_t* GatherIndx,
    int32_t* ScatterIndx,
    __hip_bfloat16* GateScal,
    const __hip_bfloat16* ExptScal,
    const int32_t* ExptIndx,
    const int32_t* TokensStart,
    int32_t n_gates,
    int32_t N_EXPTS_ACT,
    hipStream_t stream
) {
    constexpr int BLOCK_M = 128;
    bool even_m = (n_gates % (BLOCK_M * N_EXPTS_ACT)) == 0;

    if (N_EXPTS_ACT == 2) {
        hipLaunchKernelGGL(
            (routing_compute_indx_fused_kernel<BLOCK_M, 2>),
            dim3(1), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            TokensStart, n_gates, even_m
        );
    } else if (N_EXPTS_ACT == 4) {
        hipLaunchKernelGGL(
            (routing_compute_indx_fused_kernel<BLOCK_M, 4>),
            dim3(1), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            TokensStart, n_gates, even_m
        );
    } else if (N_EXPTS_ACT == 8) {
        hipLaunchKernelGGL(
            (routing_compute_indx_fused_kernel<BLOCK_M, 8>),
            dim3(1), dim3(256), 0, stream,
            GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx,
            TokensStart, n_gates, even_m
        );
    }
}

} // extern "C"
