// MoE Align Block Size Kernel
// Ported from reference/triton/moe_align_block_size.py
//
// Aligns token counts per expert to block size multiples for efficient GEMM batching.
// This is essential for batched expert computation where each expert processes tokens
// in blocks of a fixed size.
//
// 4-stage algorithm:
//   Stage 1: Count tokens per expert (histogram)
//   Stage 2: Parallel prefix sum to compute cumulative counts
//   Stage 3: Compute aligned cumulative sums (rounded up to block_size multiples)
//   Stage 4: Scatter tokens to aligned positions and mark expert boundaries

#include <hip/hip_runtime.h>
#include <cstdint>

// Stage 1: Histogram - count tokens assigned to each expert
// Each thread processes tokens_per_thread tokens
// topk_ids: [numel] int32 - expert IDs for each token
// tokens_cnts: [(num_experts+1) * num_experts] int32 - output histogram
//   Layout: tokens_cnts[thread_id * num_experts + expert_id]
__global__ void moe_align_stage1_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ tokens_cnts,
    int32_t num_experts,
    int32_t numel,
    int32_t tokens_per_thread
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = pid * tokens_per_thread;
    int off_c = (pid + 1) * num_experts;

    for (int i = 0; i < tokens_per_thread; i++) {
        int idx_pos = start_idx + i;
        if (idx_pos < numel) {
            int expert_id = topk_ids[idx_pos];
            int count_pos = off_c + expert_id;
            // Atomic since multiple threads may increment the same expert
            atomicAdd(&tokens_cnts[count_pos], 1);
        }
    }
}

// Stage 2: Parallel prefix sum across threads for each expert
// Computes cumulative sum: tokens_cnts[i][e] = sum of counts from threads 0..i for expert e
__global__ void moe_align_stage2_kernel(
    int32_t* __restrict__ tokens_cnts,
    int32_t num_experts
) {
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id >= num_experts) return;

    int last_cnt = 0;
    // Iterate through thread partitions
    for (int i = 1; i <= num_experts; i++) {
        int count_pos = i * num_experts + expert_id;
        int token_cnt = tokens_cnts[count_pos];
        last_cnt = last_cnt + token_cnt;
        tokens_cnts[count_pos] = last_cnt;
    }
}

// Stage 3: Compute aligned cumulative sums with block size padding
// cumsum[i] = cumulative sum of aligned token counts for experts 0..i-1
// cumsum[0] = 0, cumsum[num_experts] = total_tokens_post_pad
__global__ void moe_align_stage3_kernel(
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t* __restrict__ tokens_cnts,
    int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t block_size
) {
    // Single thread does this (small num_experts, typically < 64)
    if (blockIdx.x > 0 || threadIdx.x > 0) return;

    int last_cumsum = 0;
    int off_cnt = num_experts * num_experts;

    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; i++) {
        int token_cnt = tokens_cnts[off_cnt + i - 1];
        // Align to block_size: cdiv(token_cnt, block_size) * block_size
        int aligned_cnt = ((token_cnt + block_size - 1) / block_size) * block_size;
        last_cumsum = last_cumsum + aligned_cnt;
        cumsum[i] = last_cumsum;
    }
    *total_tokens_post_pad = last_cumsum;
}

// Stage 4: Scatter tokens to aligned positions and write expert IDs
// sorted_token_ids: output array mapping aligned position -> original token index
// expert_ids: output array mapping block -> expert ID
__global__ void moe_align_stage4_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ tokens_cnts,  // Used as local counter, modified in-place
    const int32_t* __restrict__ cumsum,
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    int32_t tokens_per_thread
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    // Part 1: Write expert_ids for blocks belonging to this expert
    if (pid < num_experts) {
        int start_idx = cumsum[pid];
        int end_idx = cumsum[pid + 1];
        for (int i = start_idx; i < end_idx; i += block_size) {
            expert_ids[i / block_size] = pid;
        }
    }
    __syncthreads();

    // Part 2: Scatter tokens
    int start_token_idx = pid * tokens_per_thread;
    int off_t = pid * num_experts;

    for (int i = start_token_idx; i < min(start_token_idx + tokens_per_thread, numel); i++) {
        int expert_id = topk_ids[i];
        // Atomic increment to get position
        int local_count = atomicAdd(&tokens_cnts[off_t + expert_id], 1);
        int rank_post_pad = local_count + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
    }
}

extern "C" {

// Launch all 4 stages
void launch_moe_align_block_size(
    const int32_t* topk_ids,           // [numel] expert assignments
    int32_t* sorted_token_ids,         // [total_tokens_post_pad] output
    int32_t* expert_ids,               // [num_blocks] output
    int32_t* tokens_cnts_buffer,       // [(num_experts+1)*num_experts] workspace
    int32_t* cumsum_buffer,            // [num_experts+1] workspace
    int32_t* total_tokens_post_pad,    // [1] output
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    hipStream_t stream
) {
    // Tokens per thread - balance between parallelism and overhead
    int tokens_per_thread = 32;
    int num_threads = (numel + tokens_per_thread - 1) / tokens_per_thread;

    // Zero out tokens_cnts
    hipMemsetAsync(tokens_cnts_buffer, 0,
                   (num_experts + 1) * num_experts * sizeof(int32_t), stream);

    // Stage 1: Histogram
    int block_size_s1 = 256;
    int grid_size_s1 = (num_threads + block_size_s1 - 1) / block_size_s1;
    hipLaunchKernelGGL(moe_align_stage1_kernel, dim3(grid_size_s1), dim3(block_size_s1),
                       0, stream,
                       topk_ids, tokens_cnts_buffer, num_experts, numel, tokens_per_thread);

    // Stage 2: Prefix sum per expert
    int block_size_s2 = 256;
    int grid_size_s2 = (num_experts + block_size_s2 - 1) / block_size_s2;
    hipLaunchKernelGGL(moe_align_stage2_kernel, dim3(grid_size_s2), dim3(block_size_s2),
                       0, stream,
                       tokens_cnts_buffer, num_experts);

    // Stage 3: Aligned cumulative sum (single thread)
    hipLaunchKernelGGL(moe_align_stage3_kernel, dim3(1), dim3(1),
                       0, stream,
                       total_tokens_post_pad, tokens_cnts_buffer, cumsum_buffer,
                       num_experts, block_size);

    // Stage 4: Scatter tokens
    // Reset first row of tokens_cnts for use as local counters
    hipMemsetAsync(tokens_cnts_buffer, 0, num_experts * sizeof(int32_t), stream);

    int block_size_s4 = 256;
    int grid_size_s4 = (max(num_threads, num_experts) + block_size_s4 - 1) / block_size_s4;
    hipLaunchKernelGGL(moe_align_stage4_kernel, dim3(grid_size_s4), dim3(block_size_s4),
                       0, stream,
                       topk_ids, sorted_token_ids, expert_ids, tokens_cnts_buffer,
                       cumsum_buffer, num_experts, block_size, numel, tokens_per_thread);
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

void moe_align_block_size_py(
    pybind11::object topk_ids,
    pybind11::object sorted_token_ids,
    pybind11::object expert_ids,
    pybind11::object tokens_cnts_buffer,
    pybind11::object cumsum_buffer,
    pybind11::object total_tokens_post_pad,
    int num_experts,
    int block_size,
    int numel
) {
    launch_moe_align_block_size(
        reinterpret_cast<const int32_t*>(get_data_ptr(topk_ids)),
        reinterpret_cast<int32_t*>(get_data_ptr(sorted_token_ids)),
        reinterpret_cast<int32_t*>(get_data_ptr(expert_ids)),
        reinterpret_cast<int32_t*>(get_data_ptr(tokens_cnts_buffer)),
        reinterpret_cast<int32_t*>(get_data_ptr(cumsum_buffer)),
        reinterpret_cast<int32_t*>(get_data_ptr(total_tokens_post_pad)),
        num_experts,
        block_size,
        numel,
        0  // default stream
    );
}

PYBIND11_MODULE(moe_align_block_size_tk, m) {
    m.def("moe_align_block_size", &moe_align_block_size_py,
          "MoE align block size kernel",
          pybind11::arg("topk_ids"),
          pybind11::arg("sorted_token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("tokens_cnts_buffer"),
          pybind11::arg("cumsum_buffer"),
          pybind11::arg("total_tokens_post_pad"),
          pybind11::arg("num_experts"),
          pybind11::arg("block_size"),
          pybind11::arg("numel"));
}
