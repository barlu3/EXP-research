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

See [BUILD.md](BUILD.md) for the full target reference. Every target below is a
thin wrapper over CMake; the build tree is configured on first use. Override
`BUILD_DIR` (default `build`), `JOBS`, or `CMAKE` on the command line.

**Build**

| Target | Purpose |
| --- | --- |
| `make` / `make all` | Configure if needed, then build every target |
| `make configure` | Create the CMake build tree without building |
| `make clean` | Remove build artifacts; generated reports and the glibc tree stay |
| `make distclean` | Delete the `build/` tree outright |
| `make help` | List every available target |

**Verify**

| Target | Purpose |
| --- | --- |
| `make verify` | Exhaustive 65536-pattern check vs MPFR (exp, sin, log; float32 and bf16 limb tables) |
| `make verify-limb` | The same check restricted to the bf16-only limb tables |
| `make test` | Run the verification suite through CTest |

**Generate**

| Target | Purpose |
| --- | --- |
| `make limb-tables` | Regenerate the bf16 limb tables (ln, exp, sin) |
| `make round` | Regenerate the 16-bit-rounded table reports |
| `make tables` | Regenerate the T1/T2 lookup tables via SageMath |

**Measure**

| Target | Purpose |
| --- | --- |
| `make bench` | Build + run the throughput benchmarks |
| `make limb-sweep` | Measure the bf16 limb-configuration frontier |

**Environment**

| Target | Purpose |
| --- | --- |
| `make check-mpfr` | Confirm MPFR is installed and print its version |
| `make glibc` | Fetch + extract the glibc reference source |

Two targets exit nonzero by design, not from build breakage:

- `make test` — 6 of 9 tests pass. `verify-exp`, `verify-sin` and `verify-log`
  report 7, 158 and 33 discrepancies against MPFR: open research findings in the
  16-bit-rounded tables. The six bf16 limb tests pass with zero discrepancies.
- `make tables` — requires SageMath on `PATH`; it reports that and stops if
  absent. Lookup tables are never edited by hand.

## Layout

Every top-level directory answers one question, and the three functions —
`exp`, `sin`, `log` — sit level with each other inside the ones that are
per-function.

```
implementations/  the math kernels             exp/ sin/ log/
cross-eval/       is it correct?               exp/ sin/ log/ + shared MPFR tools
table-gen/        build the bf16 limb tables   exp/ sin/ log/ sweep/
benchmarks/       is it fast?
log-research/     does a correctly-rounded bf16 ln() exist?  (LP/MILP)
docs/             research writeups
cmake/            build support
dependencies/     glibc reference fetch
```

`log-research/` is scoped to `ln` on purpose: it is the feasibility question for
log's table layout, not shared tooling. The limb *generators* for all three
functions live in `table-gen/`.

## bf16-only limb tables

Alongside the CORE-MATH implementations, all three functions have variants
whose lookup tables contain **no float32 at all**: each entry is stored as a
sum of bf16 limbs, summed back in float32 at use. The target is hardware with
bf16 storage or bf16 MACs and no float32 table path. At the minimal
configuration `ln` is *smaller* than the float32 table it replaces and `exp` is
exactly the same size; only `sin` still pays for the scheme.

| function | reconstruction | minimal config | tuned entries | storage vs float32 | verified |
|---|---|---|---|---|---|
| `cr_log_bf16_limb` | `T1 + T2` | 2×2 | 14 | −12.5% | 0 discrepancies / 65536 |
| `cr_exp_bf16_limb` | `T1 * T2` | 2×2 | 3 | ±0% | 0 discrepancies / 65536 |
| `cr_sin_bf16_limb` | `fma(S1, C2, C1*S2)` | 2×3 (mid path) | 0 | +25% | 0 discrepancies / 65536 |

A sum splits into limbs trivially; a product looks like it should not, since
expanding `(a1+a2)(b1+b2)` gives `n·m` cross terms. It does not have to be
expanded — rebuild each *factor* from its limbs first, then do the single
multiply, for `n+m` adds at any limb count. Three bf16 limbs carry float32's
24 significand bits exactly, so a 3-limb variant is bit-identical to CORE-MATH
by construction. The minimal variants give that up: `ln` and `exp` both drop to
two limbs on both sides, where neither side is exact, so correctness rests on
the exhaustive check rather than on a property of the split. Each needs a small
number of entries whose last limb sits a few ULP off the canonical
round-and-subtract value — 14 for `ln`, 3 for `exp`. `sin` keeps one factor
exact and needs no tuning at all.

Finding those entries is its own problem, and for `ln` a naive search reports
the minimal configuration as infeasible when it is not. See
[docs/LIMB-TUNING.md](docs/LIMB-TUNING.md) for the search, and
[docs/EXP-SIN-LIMB-RESULTS.md](docs/EXP-SIN-LIMB-RESULTS.md) for the frontier.

## Cross-evaluation (exp / sin / log)

`cross-eval/` holds two MPFR-driven tools. Both select the target function at
**compile time**, so the build produces one executable per function.

`verify_mpfr.c` compares all 65536 bfloat16 patterns against correctly-rounded
MPFR and logs each discrepancy with the CORE-MATH lookup-table indices (T1/T2
for exp, S1/C1/S2/C2 or S3/C3 for sin, T1/T2 or T3 for log). Reports land in
`exp/MPFR-result16bitp.txt`, `sin/MPFR-result16bitp-sin.txt`, and
`log/MPFR-result16bitp-log.txt`.

Each source `#include`s the 16-bit-rounded tables by default
(`exp/inria-exp16bitp.c`, `sin/inria-sin16bitp.c`, `log/inria-log16bitp.c`). To
check the unmodified 24-bit Inria tables, swap the `#include` (and matching
`OUT_FILE`) at the top of the file.

`round-16bitp.c` loads each binary32 table value exactly into MPFR, re-rounds to
16-bit (RNDN), and emits a per-entry diff plus paste-ready C arrays into
`exp/round-16bitp.txt`, `sin/round-16bitp-sin.txt`, and `log/round-16bitp-log.txt`.

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