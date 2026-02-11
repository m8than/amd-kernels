# Progress

## Status: DONE

## Completed
- Read all kernel.cpp pybind11 bindings and globals structs for all 18 kernels
- Verified correct dispatch argument orders for all kernels
- Wrote test_kernels.py with subprocess isolation (18 tests)
- Ran all tests — all 18 FAIL due to shared memory hardware limit
- Diagnosed root cause: MAX_SHARED_MEMORY=160KB exceeds MI325X 64KB LDS
- Wrote DONE.md with full results

## Root Cause
All kernels request 160KB dynamic shared memory via `MAX_SHARED_MEMORY = 160000` (in `HipKittens/include/common/util.cuh:104`). The MI325X (gfx942/CDNA3) only supports 64KB per workgroup. This causes `hipErrorInvalidConfiguration` on every kernel launch.
