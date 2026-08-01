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

### Benchmarks

`make bench` builds and runs both benchmarks. They measure throughput, not
correctness — run `make verify` first.

Benchmarks compile with `-O3 -march=native -mavx2 -mfma`. `-march=native` makes
binaries non-portable across machines; configure with
`cmake -S . -B build -DEXP_NATIVE_ARCH=OFF` to drop the architecture flags.

Executables land in `implementations/output/`.

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
