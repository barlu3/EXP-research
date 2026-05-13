# EXP-research

## Setup:
- Extract folders with `tar xf glibc-2.43.tar.xz`
- Verify `find glibc-2.43/ -name "e_exp.c"`
- Compile with `g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench 2>&1`
- Run benchmarks with `./bench 2>&1`

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