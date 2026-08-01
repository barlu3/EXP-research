# EXP-research

Correctly-rounded `exp`, `sin`, and `log` for 16-bit floats, cross-checked
exhaustively against MPFR.

## Build

Requires CMake ≥ 3.20, a C11/C++20 compiler, and MPFR + GMP
(`sudo apt install libmpfr-dev libgmp-dev`, or `brew install mpfr gmp`).

```
make            # build everything
make help       # list every target
```

See [BUILD.md](BUILD.md) for the full target reference.

| Target | Purpose |
| --- | --- |
| `make verify` | Exhaustive 65536-pattern check vs MPFR (exp, sin, log) |
| `make bench` | Build + run the throughput benchmarks |
| `make round` | Regenerate the 16-bit-rounded table reports |
| `make glibc` | Fetch + extract the glibc reference source |
| `make check-mpfr` | Confirm MPFR is installed and print its version |
| `make test` | Run the verification suite through CTest |

## Cross-evaluation (exp / sin / log)

`cross-eval/` holds two MPFR-driven tools. Both select the target function at
**compile time**, so the build produces one executable per function.

`verify_mpfr.c` compares all 65536 bfloat16 patterns against correctly-rounded
MPFR and logs each discrepancy with the CORE-MATH lookup-table indices (T1/T2
for exp, S1/C1/S2/C2 or S3/C3 for sin, T1/T2 or T3 for log). Reports land in
`MPFR-result16bitp.txt`, `sin/MPFR-result16bitp-sin.txt`, and
`log/MPFR-result16bitp-log.txt`.

Each source `#include`s the 16-bit-rounded tables by default
(`inria-exp16bitp.c`, `sin/inria-sin16bitp.c`, `log/inria-log16bitp.c`). To
check the unmodified 24-bit Inria tables, swap the `#include` (and matching
`OUT_FILE`) at the top of the file.

`round-16bitp.c` loads each binary32 table value exactly into MPFR, re-rounds to
16-bit (RNDN), and emits a per-entry diff plus paste-ready C arrays into
`round-16bitp.txt`, `sin/round-16bitp-sin.txt`, and `log/round-16bitp-log.txt`.

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