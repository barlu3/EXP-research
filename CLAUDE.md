# Project Instructions

## Commands

```bash
# Build (run from implementations/)
g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench

# Run benchmarks (run from implementations/)
./bench

# Capture output (run from implementations/)
./bench 2>&1 | tee output/bench_results.txt
```

## Key Decisions

- glibc-2.43 is the reference implementation. `exp.hpp` is a faithful C++ port
  of `e_expf.c` (float32) and `e_exp.c` (float64) from `sysdeps/ieee754/`.
- AVX2 symbols in `exp.hpp` are scalar-loop stubs so the benchmark compiles
  unchanged with or without AVX2 hardware.
