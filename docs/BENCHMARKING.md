# Benchmarking

How the throughput benchmarks in `benchmarks/` work, what their numbers mean,
and — as importantly — what they cannot tell you.

The short version: five benchmark programs share one measurement core
(`benchmarks/bench-harness.hpp`). Each reports, per input cluster and per
variant, a **median ns/call with an interquartile range**, taken across
**separate processes**. Ratios are medians of per-epoch *paired* ratios, and
each cluster carries a **control kernel** so the loop's own cost can be
subtracted rather than blamed on the function under test.

---

## 1. What is being measured

| Benchmark | Question | Output |
|---|---|---|
| `benchmark-limb.cpp` | Do bf16 limb tables cost throughput vs CORE-MATH's float32 tables, for `ln`? | `output/bench_results_limb.txt` |
| `benchmark-exp-limb.cpp` | Same, for `exp` — 3×3 and 2×2 limb configurations | `output/bench_results_exp_limb.txt` |
| `benchmark-sin-limb.cpp` | Same, for `sin` — exact and minimal configurations | `output/bench_results_sin_limb.txt` |
| `benchmark-inria.cpp` | float64 `exp`: CORE-MATH `cr_exp` vs the homemade port vs stdlib, plus per-stage timings | `output/bench_results_inria.txt` |
| `benchmark-home.cpp` | float64/float32 homemade `exp`/`expf` vs stdlib, plus the polynomial step alone | `output/bench_results.txt` |

### What they cannot measure

The bf16 limb scheme exists to shrink table *storage* — for `ln`, 2036 B down
to 1782 B. **No benchmark here can show that benefit.** Both tables fit in the
L1 data cache of any machine this runs on many times over (128 KB on the Apple
M5 these were developed against), so the smaller table never wins a cache it was
not already going to win.

What these benchmarks measure is the other side of the trade: the **arithmetic
the limb scheme costs** to buy that storage. A result of "1.15×" means the limb
path costs 15% more time, and the storage saving is real but invisible here.
Demonstrating the saving needs a different experiment — cache pressure from a
co-running workload, or the constrained target hardware itself.

---

## 2. Layout

```
benchmarks/
  bench-harness.hpp     the measurement core — statistics, timing, epochs, reporting
  bench-clusters.hpp    the input ranges, shared with the tests
  harness-test.cpp      unit tests for the core and the cluster invariants
  benchmark-*.cpp       the five programs above
  output/               binaries and the generated reports
```

The kernels being measured live in `implementations/`; `benchmarks/CMakeLists.txt`
is the single place that coupling is spelled out. The CORE-MATH sources must be
compiled as C — under a C++ compiler they would get C++ linkage and fail to
match the benchmarks' `extern "C"` declarations.

## 3. Running them

```sh
make bench                 # build + run all five, regenerating every report
cd build && ctest -R bench # the same programs as pass/fail tests
```

Individual targets are `run-bench-home`, `run-bench-inria`, `run-bench-log-limb`,
`run-bench-exp-limb`, `run-bench-sin-limb`. Each program writes its report to
`benchmarks/output/` *and* to stdout, so `make bench` output and the committed
file are the same text.

Compile flags come from `BENCH_FLAGS` in `benchmarks/CMakeLists.txt`: `-O3` plus
`-march=native -mavx2 -mfma`, each **probed** rather than assumed, because the
x86 flags hard-error on Apple Silicon. `-DEXP_NATIVE_ARCH=OFF` drops all of
them. The banner prints the flags actually used, read from `__AVX2__` /
`__FMA__`, not a hardcoded string.

**Run on an idle machine.** These are timing measurements; a parallel build or a
second benchmark run will distort them badly. A benchmark that takes 54 s idle
was observed taking 858 s under contention, and its ratios moved with it.

---

## 4. Anatomy of a measurement

A single reading is `time_once_ns(fn, data, mask, iters)`: a prewarm loop, then a
timed loop of `iters` iterations of `acc += fn(data[i & mask])`, divided by
`iters`. Buffers are power-of-two sized so indexing is a mask.

Everything else in the core exists because a naive version of that loop gives
answers that are confidently wrong. Each layer below was added against a
measured failure, listed with the evidence.

### 4.1 The control kernel

Every cluster times a variant that only loads and converts the input — no table,
no arithmetic. It costs about **0.57–0.60 ns/call**, against about 0.98 ns for
`cr_log_bf16`: roughly **61%** of what a naive harness would attribute to the
function is the loop itself.

That offset is additive and identical across variants, so it drags every ratio
toward 1.0. Reports therefore print both:

- **ratio** — gross, what a caller in this loop shape would observe.
- **net** — the ratio after subtracting the control from both sides, which
  estimates the kernels' relative cost.

For `ln`'s narrow clusters the gross ratio understated the net one by around
0.2–0.4×.

### 4.2 Interleaving, with rotation

Variants are timed A, B, C, A, B, C — not all of A then all of B — so frequency
and thermal drift land on all variants within a rep instead of accumulating
across a run. Ratios are then computed **per rep and then medianed**, not as a
ratio of two medians, so drift shared inside a rep divides out.

Interleaving alone still pins each variant to a fixed slot, and the slot matters:
whichever variant runs first after the gap between reps absorbs the frequency
ramp. That showed up as a **10× difference in IQR** between two variants whose
medians were stable. The running order therefore rotates by one position each
rep, so every variant occupies every slot.

### 4.3 Per-run prewarm

Because variants alternate, the one that just ran left its own table in cache and
evicted the next one's. Without a prewarm, the first slice of every timed run
pays another variant's eviction — one variant swung 0.92–1.41 ns between reps
while the inlined control held to within 2%. Each timed run is therefore preceded
by `PREWARM_ITERS` untimed iterations of the same kernel.

This makes the measurement "steady-state throughput of this function called
repeatedly", which is the standard microbenchmark definition and the reproducible
one. It is a real choice, and it moves results: it is not the cost of calling the
function once, cold.

### 4.4 Discarded reps

The first reps are contaminated for every variant at once — 1.14 ns against a
0.57 ns steady state for the control — which is the process still settling rather
than anything about the code. `DISCARD_REPS` reps run and are thrown away before
recording starts.

### 4.5 Layout re-basing

The input buffer is allocated with `LAYOUT_SLACK` elements of slack past the
timed window, and each rep reads from a different offset inside that slack. This
moves the buffer's base address, and so its cache-set alignment against the
tables.

Without it, a process is internally consistent and consistently wrong: IQRs as
low as 0.005 while the *answer* moved between processes.

### 4.6 Epochs — the outer unit of replication

**This is the one that matters most, and the one a normal harness gets wrong.**

Repetition inside one process cannot average out code layout. The kernels are
reached by a call into another translation unit, and where that code and its
`.rodata` tables land relative to each other is fixed for the life of the process
and re-drawn by ASLR on the next one.

Eight runs of the same binary on an idle machine, same cluster:

```
ratio  0.848  1.235  0.914  1.100  1.061  1.210  0.944  1.006
```

— a 45% spread, while the inlined control held to **0.4%** across all eight. The
machine was quiet. Each process was simply laying out memory differently, and
each was reporting its own layout with high confidence.

So the parent process re-runs its own binary with `--epoch`; each child measures
every cluster and prints its per-variant medians; the parent takes the median and
IQR **across those processes**. Everything in 4.1–4.5 still applies inside each
child. Epochs add the one axis a single process cannot sample.

An epoch is committed only if the child exits cleanly *and* emits exactly the row
set the first good epoch did. Partial output from a crashed child must be
discarded whole, because readings are paired positionally — a child dying between
emitting variant 0 and variant 2 of a cluster would otherwise leave the series
different lengths and silently divide one variant's epoch 3 by another's epoch 4.

---

## 5. Inputs

### 5.1 Why bf16 needs codepoint enumeration

bf16 has 7 mantissa bits, so a narrow real interval collapses onto very few
distinct values. The ranges the benchmarks originally sampled resolved to:

| range | distinct bf16 values |
|---|---|
| `[0.9, 1.1]` | 38 |
| `[79.5, 80.5]` | **3** |
| `[1e-10, 3e-10]` | 201 |
| `[1e-40, 1e-38]` | 107 |

`[79.5, 80.5]` drew 500,000 samples from **three** values. The table stayed in
one or two cache lines permanently, and the benchmark could not observe any
property of the table as a whole.

Clusters are therefore built by **enumerating the distinct bf16 codepoints a
range contains** (`enumerate_bf16_in_range`) and cycling a reshuffled permutation
of them. Each pass is shuffled afresh so the access order has no short period for
the branch predictor and prefetcher to learn.

`harness-test.cpp` asserts every shipped cluster reaches `MIN_DISTINCT_INPUTS`
(100), so a starved range cannot be added without a test failing.

### 5.2 Sweeps must stay on the table path

Each function has a sweep cluster covering **every codepoint that actually
reaches the table**. Those are the rows to quote — the narrow clusters keep a
couple of cache lines hot and measure a corner of the table.

Sweeps deliberately do *not* span every finite bf16. An early version of the
`ln` sweep did, and reported **0.69×** — an apparently large win for the limb
tables. The breakdown:

| subset | n | inria | limb | ratio |
|---|---|---|---|---|
| all finite | 65280 | 3.881 | 2.763 | 0.71 |
| negative only (→ NaN) | 32639 | 1.136 | 0.683 | 0.60 |
| positive only | 32641 | 1.382 | 1.385 | **1.00** |

Half of "all finite" is `x < 0`, where `log` returns NaN without consulting a
table at all and the limb build's early exit happens to be laid out faster.
Shuffling two paths that differ that much together also mispredicts constantly,
which is why the mixed row costs 3.9 ns against ~1.2 ns for either half alone.
The headline was measuring error handling and branch misprediction.

Sweeping only the valid domain gives 1.00×. The path boundaries, read off the
CORE-MATH kernels and encoded in `bench-clusters.hpp`:

| function | no-table region | table path |
|---|---|---|
| `exp` | `abs(x) <= 0x1p-9` (→1), `abs(x) >= 0x1.74p+6` (→const) | between them |
| `sin` | `abs(x) <= 0x1.dp-4` (→x) | mid `< 4096`, large `>= 4096` |
| `log` | `x <= 0` (→NaN/−Inf) | all `x > 0`, normals and subnormals |

`test_sweeps_stay_on_table_path` enforces this.

### 5.3 float64 and float32

The float64 benchmarks keep plain interval sampling. Their inputs do not collapse
— 500k draws from `[0.9, 1.1]` land on ~500k distinct doubles — and there is no
table whose footprint a wider range would exercise.

---

## 6. Reading a report

```
  -- x near 1            257 distinct inputs   control 0.5663 ns/call
     variant           ns med   ns IQR    ratio      IQR      net
     -------------- --------- -------- -------- -------- --------
     inria (f32)       1.3541   0.0379     1.00       --     1.00
     limb (bf16)       1.3655   0.0053     1.01    0.024     1.48
```

- **distinct inputs** — how many different values this cluster actually
  exercises. A small number here means the row describes a corner of the table.
  Printed so the starvation of §5.1 cannot silently return.
- **control** — the loop's own cost for this cluster (§4.1).
- **ns med / ns IQR** — median and interquartile range of the per-epoch medians.
- **ratio / IQR** — median of the per-epoch paired ratios against the baseline.
- **net** — the same with the control removed from both sides.

**Read the IQR before the median.** It is the honest signal of whether the
difference is resolvable. `ln`'s kernels run near 1 ns against a 0.57 ns control,
so only ~0.5 ns is signal and small perturbations move the ratio a lot; `sin`'s
large path runs near 10 ns and its ratios are correspondingly tight (IQR ~0.04).
A median quoted without its IQR is the mistake the old harness made.

The baseline row shows `1.00` and `--` by construction — it is being divided by
itself. Any column whose underlying series is missing prints `n/a`, never a
number: a reader cannot tell a fabricated value from a measured one. A variant
that lost epochs the baseline kept is labelled `(only N of M epochs paired)`
rather than silently compared on fewer samples.

---

## 7. Benchmarks as regression tests

The three limb benchmarks end with an exhaustive agreement check over all 65536
bf16 inputs, comparing each limb variant against the float32 implementation, and
**exit non-zero on any mismatch**. A benchmark that had silently drifted from the
verified build would otherwise report meaningless timings. That is why they are
registered in CTest alongside `bench-harness-test`.

`bench-harness-test` covers the core itself: quantile/median/IQR against known
vectors, codepoint enumeration, buffer coverage and slack bounds, the cluster
invariants of §5, epoch failure handling, the "never print a number you did not
measure" rule of §6, and a noise-floor check that runs two *identical* kernels
through the real timing path and asserts they come out equal.

That last test is load-aware: if the control's own spread exceeds 15%, the
machine is too loaded for the check to conclude anything and it reports SKIP
rather than failing. Under 16 competing CPU hogs it measured two identical
kernels 7% apart — a fact about the machine, not the harness.

---

## 8. Known limits

- **The storage saving is not measurable here** (§1). This is the big one.
- **Residual cross-process variation.** Epochs reduce it but do not remove it;
  successive `ln` runs were observed at 1.12, 1.12, 1.11, 1.08, 1.01, 0.95 as the
  machine warmed under sustained load. Treat differences inside the IQR as
  not resolved.
- **`ln`'s ratios are the least reproducible** of the three, because its kernels
  are the fastest relative to loop overhead.
- **Steady-state only.** The prewarm (§4.3) means these are throughput numbers
  for repeated calls, not first-call or cold-cache costs.
- **`i % size` was never the problem.** An earlier hypothesis blamed the runtime
  modulo in the old indexing; measured, it cost 0.031 ns/iter against a mask.
  The power-of-two buffers are tidier, not faster.

---

## 9. Extending it

**A new cluster** — add a `Cluster` to the right table in `bench-clusters.hpp`.
Keep it inside one code path (§5.2) and above `MIN_DISTINCT_INPUTS`; the tests
check both. Mark it `sweep = true` only if it covers a whole table path.

**A new variant** — add a lambda to the `run_interleaved` call in that
benchmark's `run_epoch`, and a row to its `report_cluster` call. The core is
variadic; nothing else changes. Keep the control first, and guard every variant's
series for emptiness before reporting.

**A new benchmark** — follow `benchmark-limb.cpp`: a `run_epoch()` that measures
and calls `emit_epoch_row`, a `main` that dispatches on `is_epoch_child` and
otherwise calls `gather_epochs` and reports. Register it in
`benchmarks/CMakeLists.txt` via `add_limb_bench` and add it to the `bench` target
in the root `CMakeLists.txt`.

**Tuning cost vs precision** — `DEFAULT_EPOCHS` (7), `DEFAULT_REPS` (5),
`DISCARD_REPS` (3), and each benchmark's `BENCH_ITERS`. Total timed runs per
cluster are `epochs × (reps + discard) × variants`. Prefer more epochs over more
reps: epochs sample the axis that actually dominates (§4.6).

---

## 10. Reference

| Constant | Where | Value | Meaning |
|---|---|---|---|
| `DEFAULT_EPOCHS` | harness | 7 | child processes per report |
| `DEFAULT_REPS` | harness | 5 | recorded reps per epoch |
| `DISCARD_REPS` | harness | 3 | settling reps thrown away |
| `PREWARM_ITERS` | harness | 200,000 | untimed iterations before each timed run |
| `LAYOUT_SLACK` | harness | 4096 | elements of slack for per-rep re-basing |
| `MIN_DISTINCT_INPUTS` | harness | 100 | floor enforced on every cluster |
| `BENCH_ITERS` | per benchmark | 10M (bf16), 2M (float64) | iterations per timed run |
| `MIN_BUFFER` / `BUFFER_N` | per benchmark | 4096 / 16384 | input window, rounded to a power of two |

Related: [EXP-SIN-LIMB-RESULTS.md](EXP-SIN-LIMB-RESULTS.md) for what the limb
tables are and what they cost, [LIMB-TUNING.md](LIMB-TUNING.md) for how they are
generated and tuned.
