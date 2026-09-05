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
| `make verify` | Exhaustive 65536-pattern check vs MPFR (exp, sin, log; float32 and bf16 limb tables) |
| `make bench` | Build + run the throughput benchmarks |
| `make round` | Regenerate the 16-bit-rounded table reports |
| `make limb-tables` | Regenerate the bf16 limb tables (ln, exp, sin) |
| `make limb-sweep` | Measure the bf16 limb-configuration frontier |
| `make glibc` | Fetch + extract the glibc reference source |
| `make check-mpfr` | Confirm MPFR is installed and print its version |
| `make test` | Run the verification suite through CTest |

## bf16-only limb tables

Alongside the CORE-MATH implementations, all three functions have variants
whose lookup tables contain **no float32 at all**: each entry is stored as a
sum of non-overlapping bf16 limbs, summed back in float32 at use. The target is
hardware with bf16 storage or bf16 MACs and no float32 table path — the tables
are larger, not smaller — except `exp`'s minimal variant, which is exactly the
same size.

| function | reconstruction | minimal config | tuned entries | verified |
|---|---|---|---|---|
| `cr_log_bf16_limb` | `T1 + T2` | 3×2 | 0 | 0 discrepancies / 65536 |
| `cr_exp_bf16_limb` | `T1 * T2` | 2×2 | 3 | 0 discrepancies / 65536 |
| `cr_sin_bf16_limb` | `fma(S1, C2, C1*S2)` | 2×3 (mid path) | 0 | 0 discrepancies / 65536 |

A sum splits into limbs trivially; a product looks like it should not, since
expanding `(a1+a2)(b1+b2)` gives `n·m` cross terms. It does not have to be
expanded — rebuild each *factor* from its limbs first, then do the single
multiply, for `n+m` adds at any limb count. Three bf16 limbs carry float32's
24 significand bits exactly, so that variant is bit-identical to CORE-MATH by
construction. The minimal variants give that up. `ln` and `sin` drop one limb
from a single factor, keeping the other exact so the error stays on one side of
the product; `sin` needs no tuning at all. `exp` goes further and drops **both**
factors to two limbs — neither is exact, so correctness is established by the
exhaustive check rather than by construction, and three entries carry a one- or
two-ULP adjustment. That configuration is the same size as the float32 table it
replaces, so on `exp` the scheme costs no storage at all.

See [log-research/EXP-SIN-LIMB-RESULTS.md](log-research/EXP-SIN-LIMB-RESULTS.md).

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