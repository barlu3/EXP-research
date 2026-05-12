# EXP-research

- Extract folders with `tar xf glibc-2.43.tar.xz`
- Verify `find glibc-2.43/ -name "e_exp.c"`
- Compile with `g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench 2>&1`
- Run benchmarks with `./bench 2>&1`