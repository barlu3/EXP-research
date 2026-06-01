# EXP-research

## Setup:
- Extract folders with `tar xf glibc-2.43.tar.xz`
- Verify `find glibc-2.43/ -name "e_exp.c"`
- `cd` into `implementations/`
- Compile benchmarks for homemade with `g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark-home.cpp -o output/bench 2>&1`
- Compile benchmarks for Inria with `g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark-inria.cpp -o output/bench_inria 2>&1`
- Run benchmarks with `./output/bench 2>&1` and `./output/bench_inria 2>&1`

## For MPFR cross check:
- Ensure you have the MPFR downloaded on your system by navigating to root and compiling with `cc -o version version.c -lmpfr -lgmp`
- If the above compilation results in a `version` file being output, all is well
- Otherwise, install MPFR on your system with `sudo apt install libmpfr-dev`

## Cross-evaluation (exp / sin / log):
The `cross-eval/` directory holds two MPFR-driven tools. Both select the target
function at **compile time** (default = exp; `-DVERIFY_*` / `-DROUND_*` for the
others). Run every command from inside `cross-eval/`.

### `verify_mpfr.c` — exhaustive CORE-MATH vs MPFR check
Compares all 65536 bfloat16 patterns against correctly-rounded MPFR and logs each
discrepancy with the CORE-MATH lookup-table indices (T1/T2 for exp, S1/C1/S2/C2 or
S3/C3 for sin, T1/T2 or T3 for log).
- exp → `MPFR-result16bitp.txt`:
  `gcc -O2 -std=c11 verify_mpfr.c -lmpfr -lgmp -lm -o verify_mpfr && ./verify_mpfr`
- sin → `sin/MPFR-result16bitp-sin.txt`:
  `gcc -O2 -std=c11 -DVERIFY_SIN verify_mpfr.c -lmpfr -lgmp -lm -o verify_sin && ./verify_sin`
- log → `log/MPFR-result16bitp-log.txt`:
  `gcc -O2 -std=c11 -DVERIFY_LOG verify_mpfr.c -lmpfr -lgmp -lm -o verify_log && ./verify_log`
- Each source `#include`s the 16-bit-rounded tables by default (`inria-exp16bitp.c`,
  `sin/inria-sin16bitp.c`, `log/inria-log16bitp.c`). To check the unmodified 24-bit
  Inria tables, swap the `#include` (and matching `OUT_FILE`) at the top of the file.

### `round-16bitp.c` — round lookup tables to 16-bit precision
Loads each binary32 table value exactly into MPFR, re-rounds to 16-bit (RNDN), and
emits a per-entry diff plus paste-ready C arrays.
- exp T1/T2 → `round-16bitp.txt`:
  `gcc -O2 -std=c11 round-16bitp.c -lmpfr -lgmp -lm -o round-16bitp && ./round-16bitp`
- sin S1/C1/S2/C2/S3/C3 → `sin/round-16bitp-sin.txt`:
  `gcc -O2 -std=c11 -DROUND_SIN round-16bitp.c -lmpfr -lgmp -lm -o round-sin && ./round-sin`
- log T1/T2/T3 → `log/round-16bitp-log.txt`:
  `gcc -O2 -std=c11 -DROUND_LOG round-16bitp.c -lmpfr -lgmp -lm -o round-log && ./round-log`

## Why the homemade version is faster (most inputs)

The benchmark labels glibc as "called via PLT (shared-library ABI)". PLT is the
mechanism, but it is not the primary cost.

**PLT** (Procedure Linkage Table) is the dynamic linker's indirection layer for
calls into shared libraries. `exp()` lives in `libm.so`; the binary doesn't know
its load address at compile time, so the compiler emits a call through a PLT
trampoline that resolves the address on first call and caches it in the GOT.
glibc also cannot avoid this because POSIX requires `exp()` to set `errno` on
overflow/underflow — that contract forces a stable ABI boundary, making inlining
impossible at the library level.

The indirect jump through PLT itself costs ~1–2 cycles. The real cost is what PLT
implies: **the compiler can never inline `libm.so`'s `exp()`**. Every call pays:

- Full function-call overhead (push/pop caller-saved registers, `call`/`ret`)
- A hard optimization barrier — the compiler cannot fold constants, eliminate
  unreachable special-case branches, or keep intermediate values in registers
  across the call

The homemade version (`exp.hpp`) is declared `inline` in a header. The compiler
pastes the entire body at the call site, eliminates the call overhead, and can
apply cross-boundary optimizations. It also omits `errno` handling, removing
branches that glibc must execute on every call.

**The tiny-x exception** (`x ~ 2e-10`): glibc wins there despite the PLT cost
because its libm has a very short fast-exit path for near-zero inputs that
completes in fewer instructions than the homemade branch structure, even after
accounting for call overhead.