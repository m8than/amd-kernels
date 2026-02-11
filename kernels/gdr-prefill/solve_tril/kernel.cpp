// SPDX-License-Identifier: MIT
// Solve lower triangular system: compute (I + A)^{-1} where A is strictly lower triangular
// Ported from Triton solve_tril.py
// Supports BT=16, BT=32, and BT=64 chunk sizes
// Input: A [B, T, H, BT] (strictly lower triangular per chunk)
// Output: Ai = (I + A)^{-1} [B, T, H, BT]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// 16x16 triangular inverse kernel
// Each block handles one 16x16 chunk for one (batch, head) pair
// ============================================================================
__global__ void solve_tril_16x16_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    int T, int H, int BT
) {
    int i_t = blockIdx.x;   // chunk index within the BT-sized blocks
    int i_bh = blockIdx.y;
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;

    int bos = i_b * T;

    // A is laid out as [B, T, H, BT] -> element at (b, t, h, j) = A[((b*T + t)*H + h)*BT + j]
    // But the Triton code addresses it as A + (bos*H + i_h)*BT with strides (H*BT, 1)
    // So A[row][col] = A_base[row * H * BT + col] where row is time offset, col is position in chunk

    const float* A_base = A + ((long long)(bos) * H + i_h) * BT;
    float* Ai_base = Ai + ((long long)(bos) * H + i_h) * 16;  // Ai uses 16 columns for 16x16

    // We need to invert a 16x16 lower triangular matrix per chunk
    // Load the 16x16 block into shared memory
    __shared__ float s_A[16][16];  // The matrix to invert
    __shared__ float s_Ai[16][16]; // The inverse

    // Load A block (only lower triangular part)
    int offset = i_t * 16;
    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        float val = 0.0f;
        if (row > col && offset + row < T) {
            // A is stored with offset within the BT block
            val = A_base[(offset + row) * H * BT + (offset % BT) + col];
        }
        s_A[row][col] = val;
        // Initialize Ai as identity
        s_Ai[row][col] = (row == col) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Forward substitution to compute (I + A)^{-1}
    // The Triton code does:
    // b_A = -A (negate the strictly lower triangular part)
    // Then for i in range(2, 16):
    //   b_a = -A[i, :] (load row i)
    //   b_a = b_a + sum(b_a[:, None] * b_Ai, 0)  // update row using accumulated inverse
    //   b_Ai[i, :] = b_a
    // Finally b_Ai += I

    // This is equivalent to computing the Neumann series / back-substitution
    // We'll do this sequentially since it's inherently serial across rows
    if (tid == 0) {
        // Initialize: s_Ai = -A (strictly lower) + 0 elsewhere
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                if (i > j) {
                    s_Ai[i][j] = -s_A[i][j];
                } else if (i == j) {
                    s_Ai[i][j] = 1.0f;
                } else {
                    s_Ai[i][j] = 0.0f;
                }
            }
        }

        // Forward substitution: for row i, update using rows 0..i-1
        for (int i = 2; i < 16 && offset + i < T; i++) {
            // b_a[j] = -A[i][j] for j < i
            float b_a[16];
            for (int j = 0; j < 16; j++) {
                b_a[j] = (j < i) ? -s_A[i][j] : 0.0f;
            }
            // b_a += sum over k of b_a[k] * s_Ai[k][:]
            float new_a[16] = {};
            for (int j = 0; j < 16; j++) {
                for (int k = 0; k < i; k++) {
                    new_a[j] += b_a[k] * s_Ai[k][j];
                }
            }
            for (int j = 0; j < 16; j++) {
                new_a[j] += b_a[j];
            }
            // Update row i
            for (int j = 0; j < 16; j++) {
                s_Ai[i][j] = new_a[j];
            }
        }

        // Add identity
        for (int i = 0; i < 16; i++) {
            s_Ai[i][i] += 0.0f;  // Already included above
        }
    }
    __syncthreads();

    // Store result
    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        if (offset + row < T) {
            Ai_base[(offset + row) * H * 16 + col] = s_Ai[row][col];
        }
    }
}

// ============================================================================
// 32x32 inverse kernel using block structure
// Partition 32x32 into four 16x16 blocks and use Schur complement
// ============================================================================
__global__ void solve_tril_32x32_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    int T, int H, int BT
) {
    int i_t = blockIdx.x;
    int i_bh = blockIdx.y;
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;
    int bos = i_b * T;

    const float* A_base = A + ((long long)(bos) * H + i_h) * BT;
    float* Ai_base = Ai + ((long long)(bos) * H + i_h) * BT;

    __shared__ float s_Ai_11[16][16];
    __shared__ float s_Ai_22[16][16];
    __shared__ float s_Ai_21[16][16];
    __shared__ float s_A_21[16][16];

    int chunk_offset = i_t * BT;

    if (tid == 0) {
        // Compute 16x16 inverse for block (0,0)
        float A11[16][16] = {};
        for (int i = 0; i < 16 && chunk_offset + i < T; i++) {
            for (int j = 0; j < i; j++) {
                A11[i][j] = A_base[(chunk_offset + i) * H * BT + j];
            }
        }

        // Initialize s_Ai_11 with forward substitution
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                s_Ai_11[i][j] = (i == j) ? 1.0f : ((i > j) ? -A11[i][j] : 0.0f);

        for (int i = 2; i < 16 && chunk_offset + i < T; i++) {
            float row[16] = {};
            for (int j = 0; j < i; j++) row[j] = -A11[i][j];
            float new_row[16] = {};
            for (int j = 0; j < 16; j++)
                for (int k = 0; k < i; k++)
                    new_row[j] += row[k] * s_Ai_11[k][j];
            for (int j = 0; j < 16; j++)
                new_row[j] += row[j];
            for (int j = 0; j < 16; j++) s_Ai_11[i][j] = new_row[j];
        }

        // Compute 16x16 inverse for block (1,1)
        float A22[16][16] = {};
        for (int i = 0; i < 16 && chunk_offset + 16 + i < T; i++) {
            for (int j = 0; j < i; j++) {
                A22[i][j] = A_base[(chunk_offset + 16 + i) * H * BT + 16 + j];
            }
        }

        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                s_Ai_22[i][j] = (i == j) ? 1.0f : ((i > j) ? -A22[i][j] : 0.0f);

        for (int i = 2; i < 16 && chunk_offset + 16 + i < T; i++) {
            float row[16] = {};
            for (int j = 0; j < i; j++) row[j] = -A22[i][j];
            float new_row[16] = {};
            for (int j = 0; j < 16; j++)
                for (int k = 0; k < i; k++)
                    new_row[j] += row[k] * s_Ai_22[k][j];
            for (int j = 0; j < 16; j++) new_row[j] += row[j];
            for (int j = 0; j < 16; j++) s_Ai_22[i][j] = new_row[j];
        }

        // Load A_21 block
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) {
                s_A_21[i][j] = 0.0f;
                if (chunk_offset + 16 + i < T) {
                    s_A_21[i][j] = A_base[(chunk_offset + 16 + i) * H * BT + j];
                }
            }

        // Compute Ai_21 = -Ai_22 * A_21 * Ai_11
        float tmp[16][16] = {};
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                for (int k = 0; k < 16; k++)
                    tmp[i][j] += s_Ai_22[i][k] * s_A_21[k][j];

        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) {
                float val = 0.0f;
                for (int k = 0; k < 16; k++)
                    val += tmp[i][k] * s_Ai_11[k][j];
                s_Ai_21[i][j] = -val;
            }
    }
    __syncthreads();

    // Store all blocks
    for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        if (chunk_offset + row < T) {
            Ai_base[(chunk_offset + row) * H * BT + col] = s_Ai_11[row][col];
        }
        if (chunk_offset + 16 + row < T) {
            Ai_base[(chunk_offset + 16 + row) * H * BT + 16 + col] = s_Ai_22[row][col];
            Ai_base[(chunk_offset + 16 + row) * H * BT + col] = s_Ai_21[row][col];
        }
    }
}

// ============================================================================
// 64x64 inverse kernel
// Partition into 4x4 blocks of 16x16 and use block triangular inverse
// ============================================================================
__global__ void solve_tril_64x64_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    int T, int H, int BT
) {
    int i_t = blockIdx.x;
    int i_bh = blockIdx.y;
    int i_b = i_bh / H;
    int i_h = i_bh % H;
    int tid = threadIdx.x;
    int bos = i_b * T;

    const float* A_base = A + ((long long)(bos) * H + i_h) * BT;
    float* Ai_base = Ai + ((long long)(bos) * H + i_h) * BT;

    int chunk_offset = i_t * BT;

    // We need 10 16x16 blocks for the lower triangular part
    // Diagonal: Ai_11, Ai_22, Ai_33, Ai_44
    // Sub-diagonal: Ai_21, Ai_31, Ai_32, Ai_41, Ai_42, Ai_43
    __shared__ float s_blocks[10][16][16];

    // Block indices: 0=Ai_11, 1=Ai_22, 2=Ai_33, 3=Ai_44
    //               4=Ai_21, 5=Ai_31, 6=Ai_32, 7=Ai_41, 8=Ai_42, 9=Ai_43

    if (tid == 0) {
        // Helper lambda for 16x16 triangular inverse
        auto invert_diag = [&](int block_row, float out[16][16]) {
            int off = chunk_offset + block_row * 16;
            float A_blk[16][16] = {};
            for (int i = 0; i < 16 && off + i < T; i++)
                for (int j = 0; j < i; j++)
                    A_blk[i][j] = A_base[(off + i) * H * BT + block_row * 16 + j];

            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    out[i][j] = (i == j) ? 1.0f : ((i > j) ? -A_blk[i][j] : 0.0f);

            for (int i = 2; i < 16 && off + i < T; i++) {
                float row[16] = {};
                for (int j = 0; j < i; j++) row[j] = -A_blk[i][j];
                float new_row[16] = {};
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < i; k++)
                        new_row[j] += row[k] * out[k][j];
                for (int j = 0; j < 16; j++) new_row[j] += row[j];
                for (int j = 0; j < 16; j++) out[i][j] = new_row[j];
            }
        };

        // Compute diagonal inverses
        invert_diag(0, s_blocks[0]);
        invert_diag(1, s_blocks[1]);
        invert_diag(2, s_blocks[2]);
        invert_diag(3, s_blocks[3]);

        // Load off-diagonal A blocks
        auto load_A_block = [&](int br, int bc, float out[16][16]) {
            int off_r = chunk_offset + br * 16;
            int off_c = bc * 16;
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++) {
                    out[i][j] = 0.0f;
                    if (off_r + i < T)
                        out[i][j] = A_base[(off_r + i) * H * BT + off_c + j];
                }
        };

        // Matrix multiply helper: C = A * B
        auto matmul = [](const float A_m[16][16], const float B_m[16][16], float C_m[16][16]) {
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++) {
                    C_m[i][j] = 0.0f;
                    for (int k = 0; k < 16; k++)
                        C_m[i][j] += A_m[i][k] * B_m[k][j];
                }
        };

        // Matrix negate
        auto negate = [](float M[16][16]) {
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    M[i][j] = -M[i][j];
        };

        // Matrix add: A += B
        auto matadd = [](float A_m[16][16], const float B_m[16][16]) {
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    A_m[i][j] += B_m[i][j];
        };

        float A_21[16][16], A_31[16][16], A_32[16][16];
        float A_41[16][16], A_42[16][16], A_43[16][16];

        load_A_block(1, 0, A_21);
        load_A_block(2, 0, A_31);
        load_A_block(2, 1, A_32);
        load_A_block(3, 0, A_41);
        load_A_block(3, 1, A_42);
        load_A_block(3, 2, A_43);

        // Ai_21 = -Ai_22 * A_21 * Ai_11
        float tmp1[16][16], tmp2[16][16];
        matmul(s_blocks[1], A_21, tmp1);
        matmul(tmp1, s_blocks[0], s_blocks[4]);
        negate(s_blocks[4]);

        // Ai_32 = -Ai_33 * A_32 * Ai_22
        matmul(s_blocks[2], A_32, tmp1);
        matmul(tmp1, s_blocks[1], s_blocks[6]);
        negate(s_blocks[6]);

        // Ai_43 = -Ai_44 * A_43 * Ai_33
        matmul(s_blocks[3], A_43, tmp1);
        matmul(tmp1, s_blocks[2], s_blocks[9]);
        negate(s_blocks[9]);

        // Ai_31 = -Ai_33 * (A_31 * Ai_11 + A_32 * Ai_21)
        matmul(A_31, s_blocks[0], tmp1);
        matmul(A_32, s_blocks[4], tmp2);
        matadd(tmp1, tmp2);
        matmul(s_blocks[2], tmp1, s_blocks[5]);
        negate(s_blocks[5]);

        // Ai_42 = -Ai_44 * (A_42 * Ai_22 + A_43 * Ai_32)
        matmul(A_42, s_blocks[1], tmp1);
        matmul(A_43, s_blocks[6], tmp2);
        matadd(tmp1, tmp2);
        matmul(s_blocks[3], tmp1, s_blocks[8]);
        negate(s_blocks[8]);

        // Ai_41 = -Ai_44 * (A_41 * Ai_11 + A_42 * Ai_21 + A_43 * Ai_31)
        matmul(A_41, s_blocks[0], tmp1);
        matmul(A_42, s_blocks[4], tmp2);
        matadd(tmp1, tmp2);
        matmul(A_43, s_blocks[5], tmp2);
        matadd(tmp1, tmp2);
        matmul(s_blocks[3], tmp1, s_blocks[7]);
        negate(s_blocks[7]);
    }
    __syncthreads();

    // Store all blocks
    // Layout: block (br, bc) at rows [br*16, br*16+16), cols [bc*16, bc*16+16)
    struct BlockInfo { int blk_idx, row_off, col_off; };
    BlockInfo infos[] = {
        {0, 0, 0}, {1, 16, 16}, {2, 32, 32}, {3, 48, 48},
        {4, 16, 0}, {5, 32, 0}, {6, 32, 16},
        {7, 48, 0}, {8, 48, 16}, {9, 48, 32}
    };

    for (auto& info : infos) {
        for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
            int row = idx / 16;
            int col = idx % 16;
            int t_row = chunk_offset + info.row_off + row;
            int t_col = info.col_off + col;
            if (t_row < T && t_col < BT) {
                Ai_base[t_row * H * BT + t_col] = s_blocks[info.blk_idx][row][col];
            }
        }
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void dispatch_solve_tril(
    const float* A, float* Ai,
    int B, int T, int H, int BT,
    hipStream_t stream
) {
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * H);
    dim3 block(64);

    if (BT == 16) {
        hipLaunchKernelGGL(solve_tril_16x16_kernel, grid, block, 0, stream, A, Ai, T, H, BT);
    } else if (BT == 32) {
        hipLaunchKernelGGL(solve_tril_32x32_kernel, grid, block, 0, stream, A, Ai, T, H, BT);
    } else if (BT == 64) {
        hipLaunchKernelGGL(solve_tril_64x64_kernel, grid, block, 0, stream, A, Ai, T, H, BT);
    }
}

// ============================================================================
// Reference: compute (I + A)^{-1} using direct forward substitution
// ============================================================================
void reference_solve_tril(
    const float* A, float* Ai,
    int B, int T, int H, int BT
) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int NT = (T + BT - 1) / BT;
            for (int nt = 0; nt < NT; nt++) {
                int off = nt * BT;
                int sz = std::min(BT, T - off);

                // Build (I + A) for this chunk
                std::vector<float> M(BT * BT, 0.0f);
                for (int i = 0; i < sz; i++) {
                    M[i * BT + i] = 1.0f;  // identity
                    for (int j = 0; j < i; j++) {
                        M[i * BT + j] = A[((long long)(b * T + off + i) * H + h) * BT + j];
                    }
                }

                // Invert by forward substitution: Ai = (I+A)^{-1}
                // Since A is strictly lower triangular, (I+A) is unit lower triangular
                // Its inverse is also unit lower triangular
                std::vector<float> Inv(BT * BT, 0.0f);
                for (int i = 0; i < sz; i++) {
                    Inv[i * BT + i] = 1.0f;
                    for (int j = 0; j < i; j++) {
                        float sum = 0.0f;
                        for (int k = j; k < i; k++) {
                            sum += M[i * BT + k] * Inv[k * BT + j];
                        }
                        Inv[i * BT + j] = -sum;
                    }
                }

                // Store
                for (int i = 0; i < sz; i++) {
                    for (int j = 0; j < BT; j++) {
                        Ai[((long long)(b * T + off + i) * H + h) * BT + j] = Inv[i * BT + j];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Pybind11 bindings
// ============================================================================
#include <pybind11/pybind11.h>

static uint64_t get_data_ptr(pybind11::object t) {
    return t.attr("data_ptr")().cast<uint64_t>();
}

void solve_tril_wrapper(pybind11::object A, pybind11::object Ai,
                        int B, int T, int H, int BT) {
    dispatch_solve_tril(
        (const float*)get_data_ptr(A),
        (float*)get_data_ptr(Ai),
        B, T, H, BT, 0);
}

PYBIND11_MODULE(solve_tril_tk, m) {
    m.doc() = "Solve lower triangular: compute (I + A)^{-1}";
    m.def("solve_tril", &solve_tril_wrapper,
          "Compute (I + A)^{-1} for strictly lower triangular A",
          pybind11::arg("A"), pybind11::arg("Ai"),
          pybind11::arg("B"), pybind11::arg("T"), pybind11::arg("H"), pybind11::arg("BT"));
}
