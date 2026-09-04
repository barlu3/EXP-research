# Build Reference

The build is CMake-driven. The top-level `Makefile` is a thin wrapper that
configures on first use, so `make <target>` works without touching CMake
directly.

## Requirements

| Dependency | Purpose | Install |
| --- | --- | --- |
| CMake ≥ 3.20 | Build system | `sudo apt install cmake` |
| C11 + C++20 compiler | Tools + benchmarks | `sudo apt install build-essential` |
| MPFR, GMP | Ground truth for verification | `sudo apt install libmpfr-dev libgmp-dev` |
| SageMath *(optional)* | Regenerating lookup tables | [sagemath.org](https://www.sagemath.org) |
| GLPK or HiGHS *(optional)* | Solving the emitted MILP | `sudo apt install glpk-utils` |

On macOS use `brew install mpfr gmp` or `sudo port install mpfr gmp`. If MPFR
sits in a non-standard prefix, point the configure step at it with
`MPFR_ROOT=/opt/local cmake -S . -B build`.

Configuration fails with an explicit install hint when MPFR or GMP is missing.
`make check-mpfr` prints the detected library and header versions — it replaces
the old manual `cc -o version version.c -lmpfr -lgmp` probe.

## Targets

### Verification

`make verify` builds and runs all three checkers; `make test` runs the same
binaries through CTest with a pass/fail gate.

| Target | Output |
| --- | --- |
| `make verify` | All three functions |
| `cmake --build build --target verify-exp` | `cross-eval/MPFR-result16bitp.txt` |
| `cmake --build build --target verify-sin` | `cross-eval/sin/MPFR-result16bitp-sin.txt` |
| `cmake --build build --target verify-log` | `cross-eval/log/MPFR-result16bitp-log.txt` |

The pass condition is zero discrepancies across all 65536 bfloat16 patterns.
The checkers always exit 0, so CTest derives pass/fail from the reported
discrepancy count instead of the exit status.

**These currently fail** — exp 7, sin 158, log 33 discrepancies against the
16-bit-rounded tables. Those are open research findings, not build breakage.

The bf16 limb tables have their own verifiers, and these are expected to
**pass**: they exit nonzero on any discrepancy, so CTest uses the exit status
directly.

| Target | Output |
| --- | --- |
| `cmake --build build --target verify-exp-limb-run` | `cross-eval/MPFR-result-limb-exp.txt` |
| `cmake --build build --target verify-sin-limb-run` | `cross-eval/sin/MPFR-result-limb-sin.txt` |
| `cmake --build build --target verify-log-limb-run` | `cross-eval/log/MPFR-result-limb-log.txt` |
| `cmake --build build --target verify-limb` | all three |

`verify-limb.c` selects its function at compile time (`-DVERIFY_SIN_LIMB`) and
checks both variants — exact and minimal — of that function in one run. The ln
verifier is a separate, older file (`cross-eval/log/verify-limb.c`).

### bf16 limb tables

| Target | Output |
| --- | --- |
| `make limb-tables` | all three headers below |
| `cmake --build build --target exp-limb-tables` | `implementations/expbf16-limb.h` |
| `cmake --build build --target sin-limb-tables` | `implementations/sin/sinbf16-limb.h` |
| `cmake --build build --target ln-limb-tables` | `implementations/log/logbf16-limb.h` |
| `make limb-sweep` | the limb-configuration frontier, to stdout |

The exp and sin generators split the **shipped** CORE-MATH tables rather than
recomputing ideals in MPFR, so they need no MPFR. That is deliberate: those
tables carry manual ULP adjustments and an overflow cap, and splitting the
shipped values is what makes the generated limb tables reproduce
`cr_exp_bf16` / `cr_sin_bf16` exactly. `limb-gen.c` (ln) does use MPFR, because
CORE-MATH's ln tables are plain correctly-rounded values.

Like `ln`, these write into the source tree and so are manual targets — nothing
regenerates them as a build dependency.

### Benchmarks

`make bench` builds and runs both benchmarks. They measure throughput, not
correctness — run `make verify` first.

Benchmarks compile with `-O3` plus whichever of `-march=native`, `-mavx2` and
`-mfma` the compiler accepts — each is probed rather than assumed, since the
AVX flags hard-error on ARM targets. `-march=native` makes binaries
non-portable across machines; configure with
`cmake -S . -B build -DEXP_NATIVE_ARCH=OFF` to drop the architecture flags.

Executables land in `implementations/output/`.

| Benchmark | Compares |
| --- | --- |
| `bench-home`, `bench-inria` | float64 `exp`: homemade vs CORE-MATH vs libm |
| `bench-log-limb` | bf16 `ln`: limb tables vs CORE-MATH's float32 tables |
| `bench-exp-limb` | bf16 `exp`: 3×3 and 3×2 limb tables vs float32 |
| `bench-sin-limb` | bf16 `sin`: exact and minimal limb tables vs float32 |

The three limb benchmarks end with an exhaustive agreement check against the
float32 tables and exit nonzero if it fails, so CTest runs them as regression
tests too. Each reports the best of three timed runs per measurement: a single
run is noisy enough to swing a ratio by ~0.4×, which the no-table control
cluster in each benchmark makes visible.

**`bench-home` and `bench-inria` are x86-only.** `exp.hpp`, `inria-exp.hpp` and
`inria-exp-seg.hpp` include `<immintrin.h>` / `<x86intrin.h>`, so those two
targets do not build on ARM. The limb benchmarks are unaffected; build them by
name, or use `make bench` on x86.

### Lookup tables

`make round` regenerates the 16-bit-rounded table reports for all three
functions (`round-exp`, `round-sin`, `round-log` individually).

`make tables` regenerates the Inria T1/T2 tables from the `.sage` sources and
requires SageMath. Never edit the generated tables by hand.

### glibc reference source

`make glibc` runs `dependencies/fethc-glibc.sh` to fetch the glibc 2.43 tarball,
extracts it, and confirms `sysdeps/ieee754/dbl-64/e_exp.c` is present. Both
steps are incremental — an existing tarball or extracted tree is left alone.

### log-research (ln table LP/MILP workflow)

Three sequential steps, each depending on the previous step's output:

| Step | Target | Output |
| --- | --- | --- |
| 1 | `ln-precompute` | `log-research/ln-precompute.txt` |
| 2 | `ln-bounds` | `log-research/ln-bounds.txt`, `lp-constraints.txt` |
| 3 | `ln-milp` | `log-research/milp-constraints.lp` |

Run them with `cmake --build build --target <name>`. Step 3 emits a CPLEX-format
MILP; solve it for feasibility with:

```
glpsol --lp log-research/milp-constraints.lp    # "OPTIMAL" => table exists
highs log-research/milp-constraints.lp          # "Optimal" => table exists
```

Regenerating these overwrites large committed research artifacts. The output
depends on `CAND_ULP` in `log-research/milp-gen.cc`, so a changed value yields a
different (still deterministic) file.

## Notes

`log-research/milp-gen.cc` is C despite its extension and is compiled with
`LANGUAGE C` to match; letting CMake infer C++ breaks the build. That file and
`bound-calc.c` hardcode `log-research/`-prefixed output paths, so their targets
run from the repo root while every other tool runs from its own directory.

`make clean` removes build artifacts but leaves generated reports and the
extracted glibc tree in place. `make distclean` deletes the `build/` tree.
