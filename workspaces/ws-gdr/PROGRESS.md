# Progress

## Completed Steps

1. **Explored all 17 GDR kernel directories** — read kernel.cpp pybind11 signatures and test_parity.py reference implementations
2. **Wrote test_kernels.py** — ~1200 lines covering all 17 kernels (34 test cases)
3. **Ran tests, fixed issues, re-ran** — iterative debugging of argument formats, types, tolerances
4. **Final results: 31 PASS, 3 FAIL** — only solve_tril_tk standalone fails
5. **Investigated solve_tril_tk bug** — confirmed root cause: `+= 0.0f` should be `+= 1.0f` on diagonal
6. **Wrote DONE.md** with full results table, fixes applied, and known issues
