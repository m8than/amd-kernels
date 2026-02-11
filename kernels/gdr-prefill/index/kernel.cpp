// SPDX-License-Identifier: MIT
// Index preparation utilities ported from Triton/Python to HIP C++
// These are host-side utilities (no GPU kernels) for preparing indices
// used by chunk-based variable-length sequence operations.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Host utility functions (ported from Python)
// ============================================================================

// Compute sequence lengths from cumulative sequence lengths
// cu_seqlens: [N+1], returns lens: [N]
std::vector<int> prepare_lens(const std::vector<int>& cu_seqlens) {
    std::vector<int> lens(cu_seqlens.size() - 1);
    for (size_t i = 0; i < lens.size(); i++) {
        lens[i] = cu_seqlens[i + 1] - cu_seqlens[i];
    }
    return lens;
}

// Convert sequence lengths to cumulative sequence lengths
// lens: [N], returns cu_seqlens: [N+1]
std::vector<int> prepare_cu_seqlens_from_lens(const std::vector<int>& lens) {
    std::vector<int> cu_seqlens(lens.size() + 1);
    cu_seqlens[0] = 0;
    for (size_t i = 0; i < lens.size(); i++) {
        cu_seqlens[i + 1] = cu_seqlens[i] + lens[i];
    }
    return cu_seqlens;
}

// Generate position IDs for each sequence
// cu_seqlens: [N+1], returns position_ids: [total_tokens]
std::vector<int> prepare_position_ids(const std::vector<int>& cu_seqlens) {
    auto lens = prepare_lens(cu_seqlens);
    std::vector<int> pos_ids;
    for (int len : lens) {
        for (int i = 0; i < len; i++) {
            pos_ids.push_back(i);
        }
    }
    return pos_ids;
}

// Generate sequence IDs indicating which sequence each token belongs to
// cu_seqlens: [N+1], returns seq_ids: [total_tokens]
std::vector<int> prepare_sequence_ids(const std::vector<int>& cu_seqlens) {
    auto pos_ids = prepare_position_ids(cu_seqlens);
    std::vector<int> seq_ids(pos_ids.size());
    int seq = -1;
    for (size_t i = 0; i < pos_ids.size(); i++) {
        if (pos_ids[i] == 0) seq++;
        seq_ids[i] = seq;
    }
    return seq_ids;
}

// Ceiling division
int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// Prepare chunk indices for variable-length sequences
// Returns vector of pairs (sequence_id, chunk_idx_in_seq)
std::vector<std::pair<int, int>> prepare_chunk_indices(
    const std::vector<int>& cu_seqlens,
    int chunk_size
) {
    auto lens = prepare_lens(cu_seqlens);
    std::vector<std::pair<int, int>> indices;
    int seq_id = 0;
    for (int len : lens) {
        int num_chunks = cdiv(len, chunk_size);
        for (int c = 0; c < num_chunks; c++) {
            indices.push_back({seq_id, c});
        }
        seq_id++;
    }
    return indices;
}

// Prepare cumulative chunk offsets for variable-length sequences
// Returns cu_chunk_offsets: [N+1]
std::vector<int> prepare_chunk_offsets(
    const std::vector<int>& cu_seqlens,
    int chunk_size
) {
    auto lens = prepare_lens(cu_seqlens);
    std::vector<int> offsets(lens.size() + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < lens.size(); i++) {
        offsets[i + 1] = offsets[i] + cdiv(lens[i], chunk_size);
    }
    return offsets;
}

// Get maximum number of chunks across all sequences
int get_max_num_splits(const std::vector<int>& cu_seqlens, int chunk_size) {
    auto lens = prepare_lens(cu_seqlens);
    int max_chunks = 0;
    for (int len : lens) {
        max_chunks = std::max(max_chunks, cdiv(len, chunk_size));
    }
    return max_chunks;
}

// ============================================================================
// GPU kernel for preparing chunk indices on device
// This is useful when cu_seqlens is already on GPU
// ============================================================================
__global__ void prepare_chunk_indices_kernel(
    const int* __restrict__ cu_seqlens,
    int* __restrict__ chunk_indices,  // [num_chunks, 2]
    const int* __restrict__ chunk_offsets,  // [N+1]
    int N,
    int chunk_size
) {
    int seq_id = blockIdx.x;
    if (seq_id >= N) return;

    int bos = cu_seqlens[seq_id];
    int eos = cu_seqlens[seq_id + 1];
    int seq_len = eos - bos;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    int base_offset = chunk_offsets[seq_id];

    for (int c = threadIdx.x; c < num_chunks; c += blockDim.x) {
        chunk_indices[(base_offset + c) * 2] = seq_id;
        chunk_indices[(base_offset + c) * 2 + 1] = c;
    }
}

// ============================================================================
// Test harness
// ============================================================================
// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void gpu_prepare_chunk_indices_wrapper(pybind11::object cu_seqlens, pybind11::object chunk_indices,
                                       pybind11::object chunk_offsets, int N, int chunk_size) {
    dim3 grid(N);
    dim3 block(64);
    hipLaunchKernelGGL(prepare_chunk_indices_kernel, grid, block, 0, 0,
        (const int*)get_data_ptr(cu_seqlens),
        (int*)get_data_ptr(chunk_indices),
        (const int*)get_data_ptr(chunk_offsets),
        N, chunk_size);
}

PYBIND11_MODULE(index_tk, m) {
    m.doc() = "Index preparation utilities for chunk-based variable-length sequence operations";
    m.def("prepare_lens", &prepare_lens, "Compute sequence lengths from cumulative seqlens");
    m.def("prepare_cu_seqlens_from_lens", &prepare_cu_seqlens_from_lens, "Convert lens to cu_seqlens");
    m.def("prepare_position_ids", &prepare_position_ids, "Generate position IDs per sequence");
    m.def("prepare_sequence_ids", &prepare_sequence_ids, "Generate sequence IDs per token");
    m.def("prepare_chunk_offsets", &prepare_chunk_offsets, "Prepare cumulative chunk offsets",
          pybind11::arg("cu_seqlens"), pybind11::arg("chunk_size"));
    m.def("get_max_num_splits", &get_max_num_splits, "Get max chunks across sequences",
          pybind11::arg("cu_seqlens"), pybind11::arg("chunk_size"));
    m.def("gpu_prepare_chunk_indices", &gpu_prepare_chunk_indices_wrapper,
          "GPU kernel to prepare chunk indices",
          pybind11::arg("cu_seqlens"), pybind11::arg("chunk_indices"),
          pybind11::arg("chunk_offsets"), pybind11::arg("N"), pybind11::arg("chunk_size"));
}
