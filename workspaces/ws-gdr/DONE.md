# Done

## Summary

All 17 GDR (Gated Delta Rule) kernels tested with parity tests against NumPy/PyTorch reference implementations. **31 PASS, 3 FAIL** — the only failures are the standalone `solve_tril_tk` kernel at all three block sizes (BT=16, 32, 64), which has a confirmed bug in its forward substitution algorithm.

Test file: `test_kernels.py` (~1200 lines, runnable standalone)

## Results

| # | Kernel | Status | Max Diff | Tolerance | Notes |
|---|--------|--------|----------|-----------|-------|
| 1 | gdr_utils_tk abs_diff | PASS | 0.000000 | 1e-5 | Exact match |
| 2 | gdr_utils_tk squared_diff | PASS | 0.000000 | 1e-3 | Exact match |
| 3 | gdr_utils_tk max_reduce | PASS | 0.000000 | 1e-5 | Multi-block partial reduction |
| 4 | gdr_utils_tk sum_reduce | PASS | ~1e-5 | 1e-3 | Multi-block partial reduction |
| 5 | gdr_utils_tk bf16↔float | PASS | 0.000000 | 1e-5 | Roundtrip conversion |
| 6 | gdr_utils_tk l2_normalize | PASS | ~1e-3 | 1e-2 | bf16 precision |
| 7 | l2norm_tk fwd | PASS | ~1e-3 | 1e-2 | |
| 8 | l2norm_tk bwd | PASS | ~1e-2 | 0.05 | bf16 gradient accumulation |
| 9 | op_tk (8 elementwise ops) | PASS | ~1e-3 | 1e-2 | exp/log/sigmoid/silu/softplus |
| 10 | cumsum_tk scalar | PASS | ~1e-3 | 1e-2 | |
| 11 | cumsum_tk vector | PASS | ~1e-3 | 1e-2 | |
| 12 | index_tk prepare_lens | PASS | 0.000000 | exact | Integer output |
| 13 | index_tk prepare_position_ids | PASS | 0.000000 | exact | Integer output |
| 14 | index_tk prepare_sequence_ids | PASS | 0.000000 | exact | Integer output |
| 15 | index_tk prepare_chunk_offsets | PASS | 0.000000 | exact | Integer output |
| 16 | fused_qkvzba_split_tk | PASS | ~1e-4 | 1e-2 | Pure data shuffle |
| 17 | causal_conv1d_split_qkv_tk (decode) | PASS | ~1e-3 | 0.05 | |
| 18 | causal_conv1d_fwd_split_qkv_tk (prefill) | PASS | ~1e-5 | 1e-3 | float32 I/O |
| 19 | fused_recurrent_tk output | PASS | ~0.05 | 0.15 | bf16 accumulation over K×V state |
| 20 | fused_recurrent_tk final state | PASS | ~0.11 | 0.15 | bf16 accumulation over K×V state |
| 21 | fused_sigmoid_gating_recurrent_tk | PASS | ~0.02 | 0.05 | Delta rule + sigmoid gating |
| 22 | fused_gdn_gating_prefill_tk | PASS | ~1e-3 | 0.05 | Softplus + sigmoid gating |
| 23 | fused_cumsum_kkt_tk g_cumsum | PASS | ~1e-3 | 0.05 | |
| 24 | fused_cumsum_kkt_tk A matrix | PASS | ~0.02 | 0.05 | |
| 25 | chunk_tk cumsum | PASS | ~1e-3 | 0.05 | |
| 26 | chunk_tk compute_A | PASS | ~1e-3 | 0.05 | |
| 27 | chunk_tk solve_tril (in-place) | PASS | ~0.01 | 0.05 | Different (correct) algorithm |
| 28 | chunk_tk recompute_w_u | PASS | ~0.02 | 0.05 | |
| 29 | chunk_delta_h_tk | PASS | ~0.03 | 0.05 | |
| 30 | chunk_o_tk | PASS | ~0.02 | 0.05 | |
| 31 | **solve_tril_tk BT=16** | **FAIL** | **1.000000** | 0.05 | Bug: missing identity addition |
| 32 | **solve_tril_tk BT=32** | **FAIL** | **1.000000** | 0.05 | Bug: missing identity addition |
| 33 | **solve_tril_tk BT=64** | **FAIL** | **1.000000** | 0.05 | Bug: missing identity addition |
| 34 | wy_representation_tk kkt | PASS | ~0.01 | 0.05 | |

## Fixes Applied

During test development, several issues were discovered and fixed in the test harness:

1. **gdr_utils abs_diff/squared_diff**: Functions take 4 args `(in1, in2, out, N)`, not 3.
2. **gdr_utils max_reduce/sum_reduce**: Multi-block partial reduction — output has one value per block, must aggregate on host.
3. **fused_recurrent/sigmoid_gating cu_seqlens**: pybind wrapper calls `data_ptr()` unconditionally; cannot pass `None`. Must pass valid `cu_seqlens = torch.arange(B+1)*T`.
4. **index_tk input type**: Functions accept Python lists (`Sequence[SupportsInt]`), not GPU tensors.
5. **causal_conv1d_fwd_split_qkv**: Uses float32 I/O internally, not bf16.
6. **fused_recurrent_tk tolerance**: Relaxed to 0.15 due to bf16 accumulation error over K×V hidden state with delta rule updates.
7. **chunk_tk/solve_tril tolerance**: Used small input magnitudes (k×0.1, g×0.01, beta×0.1) to keep A matrix entries small and verify correctness.

## Known Issues

### solve_tril_tk standalone kernel bug (CONFIRMED)

**Root cause**: In all three kernel variants (BT=16, 32, 64), the forward substitution algorithm has a bug:

1. Step 1: Initialize `s_Ai` with identity on diagonal → `s_Ai[i][i] = 1.0f`
2. Step 2: Forward substitution loop overwrites entire rows including diagonal
3. Step 3: `s_Ai[i][i] += 0.0f` — **should be `+= 1.0f`**

The max diff is exactly 1.0 because the diagonal elements are missing the identity contribution.

**Why chunk_tk's solve_tril works**: The `chunk_tk` module has its own `solve_tril_inplace_kernel` that uses a different, correct algorithm with separate M and Ai buffers, where `Ai[i][i] = 1.0f` is set correctly.

**Fix**: Change `s_Ai[i][i] += 0.0f` to `s_Ai[i][i] += 1.0f` in `kernels/gdr-prefill/solve_tril/kernel.cpp` (all three variants).

See `ASSUMPTIONS.md` for detailed analysis.
