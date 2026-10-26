# FRECPE as a Range Reduction: Method and Findings

How an ARM "reciprocal estimate" instruction can stand in for a lookup table
inside logarithm-based functions, and how much faster that makes them on an
Apple M5.

---

## The short version

- **FRECPE** is an ARM instruction that returns a rough (8-bit) approximation
  of 1/x in a single step.
- On its own, a rough 1/x is not much use: a real division (`FDIV`) on the M5
  is fast enough that FRECPE never wins as a division replacement.
- But a **logarithm** needs exactly a rough 1/x as its first step, and it
  normally gets one from a lookup table. FRECPE gives that same value without
  the table lookup.
- Across seven functions — `logf`, `log2f`, `log10f`, `log1pf`, `powf`,
  `asinhf`, `acoshf` — the FRECPE version is **6.5 to 8 cycles faster per call** on a
  dependent chain and **7–10% faster** in a loop of independent calls, with
  **identical results** to the table version.

---

## 1. Background

### How a logarithm is computed

A computer can't evaluate `log(x)` directly. Library code splits the problem
into easy pieces. Any positive float can be written as

```
x = 2^k × m        where k is an integer and m is in [1, 2)
```

so `log(x) = k·log(2) + log(m)`. The first term is easy. For the second, a
polynomial only works well on inputs very close to 1, and `m` can be anywhere
in [1, 2). So the library shrinks it with a trick called **range reduction**:

1. Look up a value `c ≈ 1/m` in a small table, indexed by the first few bits
   of `m`.
2. Compute `r = m·c − 1`. Because `c` is close to `1/m`, `r` is tiny
   (around ±0.003).
3. Use `log(m) = log(1 + r) − log(c)`. The `log(c)` values come from a
   second table, and `log(1 + r)` is a short polynomial because `r` is tiny.

The table lookup in step 1 is the part this work replaces.

### What FRECPE is

`FRECPE` ("floating-point reciprocal estimate") returns an approximation of
`1/x` accurate to about 8 bits. Normally it is the starting point for a
couple of Newton–Raphson steps that refine it into a full-precision `1/x`.

Measured exhaustively on the M5, FRECPE has three properties that make it a
drop-in replacement for step 1 above:

| Property | Why it matters |
|---|---|
| Its output is always one of **256 fixed values**, chosen by the top 8 bits of `x`'s mantissa, and scaled by `2^-k` | This is exactly what a 256-entry reciprocal table holds |
| `r = x · FRECPE(x) − 1` is computed with **no rounding error** | The reduction adds no error of its own |
| `|r| ≤ 2^-8.45` (about 0.0029) | Small enough for a short polynomial |

It also applies the `2^-k` scaling itself, which a software table has to undo
by hand.

---

## 2. The method

Both versions compute the same thing. The only difference is where the
reciprocal `c` comes from.

**Table version**

```
x (float register)
  └─► copy bits to an integer register     ◄── slow on the M5
        └─► extract the 8-bit index
              └─► load c from the table (memory)
                    └─► r = m·c − 1
                          └─► polynomial ─► result
```

**FRECPE version**

```
x (float register)
  └─► c = FRECPE(x)                        ◄── one instruction
        └─► r = x·c − 1
              └─► polynomial ─► result

  (in parallel: copy bits → index → load log(c) → joins at the very end)
```

### Why this is faster

The key is the **critical path**: the longest chain of steps where each step
waits for the one before it.

- In the table version, computing `r` has to wait for a float-to-integer
  register copy, then an index calculation, then a memory load. On the M5
  that chain costs around 11 cycles before `r` can even start.
- In the FRECPE version, `r` is ready after two floating-point instructions.
  The table of `log(c)` values is still needed, but its lookup runs **in
  parallel** with the polynomial and is only needed at the final addition.
  So it is hidden.

FRECPE also replaces three instructions (a scale, a multiply and a load) with
one. That is where the throughput gain comes from: fewer instructions per
result.

### Using it in other functions

The same trick works anywhere a logarithm sits inside a function:

| Function | Where the log is |
|---|---|
| `log2f`, `log10f` | Same as `logf`; only the table values and one constant change |
| `log1pf(x)` | `log(1 + x)`: the log's input is one addition away from `x` |
| `powf(x, y)` | Computed as `2^(y · log2(x))`, so the log is the first half |
| `asinhf(x)` | `log(|x| + √(x² + 1))` |
| `acoshf(x)` | `log(x + √(x² − 1))` |

`powf` is the hardest case. Multiplying by `y` also multiplies any error in
`log2(x)`, by up to 150×. So its logarithm has to be hundreds of times more
accurate than `logf`'s. FRECPE's exact `r` makes that cheap: the only
remaining errors are the polynomial's and the table's.

In `asinhf` and `acoshf`, the log's input is calculated mid-function, behind a
square root. These two test whether the win survives when the log is not the
first thing the function does.

---

## 3. How it was measured

### What was compared

Each function was timed in three versions, and the four logs in a fourth:

| Version | What it is |
|---|---|
| **glibc** | glibc 2.44's `logf`, `log2f`, `log10f` or `log1pf`, built here from glibc's own source so the baseline does not change with the Mac it runs on. The baseline for those four. |
| **libm** | The system (libSystem) math library's function. The baseline for `powf`, `asinhf` and `acoshf`; a comparison row for the logs. |
| **table** | Hand-written assembly; `c` comes from a 256-entry table |
| **FRECPE** | The same assembly, with `c` from the FRECPE instruction |

The table holds **FRECPE's own 256 values**. So the table and FRECPE versions
produce bit-for-bit identical results, and any timing difference comes from
how `c` is fetched and nothing else.

### Two kinds of speed

- **Latency**: each call's input depends on the previous call's result, so
  calls cannot overlap. This is the cost of one call on a critical path. The
  cost of the test loop itself is subtracted, so the number is the function
  alone.
- **Throughput**: calls are independent, so the CPU can overlap many of them.
  This is the average cost per result in a loop.

A change can help one and not the other, so both are reported. All numbers are
in **CPU cycles**, and lower is better.

### Making the numbers trustworthy

- **Repeated runs.** Each figure is the median over 7 separate processes, with
  the versions interleaved so none of them always runs first.
- **Control input ranges.** Each function has an input range where neither
  hand-written version uses the reciprocal (for example, inputs very close to
  1 take a separate path). There the table and FRECPE versions must time the
  same. They did, in every function, which shows the measured gap is real and
  not noise.
- **Cycle counts.** Cycles come from hardware counters when run with `sudo`,
  and otherwise from a calibrated timer. The tables below use the timer. For
  `logf` a hardware-counter run gave the same answer within 0.3 cycles
  (FRECPE 22.2 vs table 30.4 latency; 2.55 vs 2.83 throughput).

### Checking accuracy

Faster is useless if the answer is wrong. Every version is held to **1 ulp**
("unit in the last place": the gap between one float and the next) of the
**correctly rounded** answer, computed with the MPFR high-precision library.

- **One-argument functions:** checked on all 4,278,190,080 finite float32
  inputs. Every single one.
- **`powf(x, y)`:** checking every `(x, y)` pair is impossible (about 10^19
  pairs). Instead, every `x` was checked against each of four `y` values
  (2.2, −1/3, 3 and −20000), about 17 billion pairs. Special `y` values
  (±0, ±∞, NaN, huge, tiny, integer edges) were checked separately.

All versions of all seven functions passed. The special-value check also caught
a real bug during development: `pow(2^-149, y)` returned 1 for large `y`.

---

## 4. Results

Cycles per call. Latency is per call with the loop's own cost removed;
throughput is per result. Lower is better.

### Main input range of each function

| Function | Latency: glibc / libm / table / **FRECPE** | Throughput: glibc / libm / table / **FRECPE** | FRECPE vs table |
|---|---|---|---|
| `logf` | 32.7 / 29.2 / 30.2 / **22.0** | 5.05 / 6.05 / 2.86 / **2.58** | −8.1 cycles, −10% |
| `log2f` | 32.7 / 37.0 / 31.1 / **23.0** | 5.02 / 5.13 / 2.82 / **2.56** | −8.0 cycles, −9% |
| `log10f` | 28.6 / 35.1 / 31.1 / **23.0** | 7.37 / 5.12 / 2.82 / **2.54** | −8.1 cycles, −10% |
| `log1pf` | 45.5 / 35.8 / 33.1 / **26.7** | 10.47 / 6.88 / 3.39 / **3.00** | −6.4 cycles, −12% |
| `powf` (y = 2.2) | — / 63.4 / 56.6 / **48.7** | — / 8.55 / 5.97 / **5.56** | −7.9 cycles, −7% |
| `asinhf` | — / 34.3 / 53.0 / **46.5** | — / 6.58 / 4.98 / **4.55** | −6.5 cycles, −9% |
| `acoshf` | — / 34.7 / 49.0 / **42.5** | — / 7.80 / 4.45 / **4.06** | −6.5 cycles, −9% |

The four log rows were rerun on 2026-10-25 with the glibc baseline; the other
three are from the earlier libm-only runs.

### Control ranges (no reciprocal used, so table and FRECPE should match)

| Function | Range | Latency: table / FRECPE | Throughput: table / FRECPE |
|---|---|---|---|
| `logf` | x in [1, 1 + 2^-8) | 18.0 / 18.0 | 2.14 / 2.25 |
| `powf` | y = −20000, x just above 1 | 44.0 / 44.2 | 4.84 / 4.83 |
| `log1pf` | \|x\| < 2^-8 | 15.1 / 15.1 | 2.12 / 2.12 |
| `asinhf` | \|x\| < 2^-8 | 15.0 / 15.0 | 1.79 / 1.74 |
| `acoshf` | x − 1 < 2^-18 | 34.0 / 34.0 | 3.42 / 3.45 |

### What the results say

1. **The win is consistent.** FRECPE beats the table in every function: about
   8 cycles of latency where the log reads the input directly, about 6.5
   where the log's input is computed first (even by a single addition, as in
   `log1pf`).
2. **It carries through `powf`.** The ~8-cycle gain survives the extra
   exponential step. In relative terms it is smaller (14% of latency instead
   of 26%) only because `powf` does more work overall.
3. **The throughput gain is smaller than the latency gain.** In a loop the
   CPU already overlaps the slow table lookup with other calls' work. What
   remains is the saving from running fewer instructions.
4. **The base of the logarithm doesn't matter.** `logf`, `log2f` and `log10f`
   behave identically, as expected: they share one reduction.
5. **`asinhf` and `acoshf` are a mixed picture.** FRECPE still beats the
   table, but both hand-written versions are slower than libm on latency.
   The square root in front of the log is the cause, not the reduction. They
   are still faster than libm on throughput.
6. **glibc is a different trade.** glibc's `logf` and `log2f` (a 16-entry
   table, within 1 ulp) run about 32.7 cycles of latency and 5.0 of
   throughput, behind FRECPE in both. Its `log10f` and `log1pf` are CORE-MATH's
   and correctly rounded, which the hand kernels (1 ulp) are not, so their gap
   to FRECPE is not the reduction alone. The table-against-FRECPE column is the
   comparison that isolates the fetch.

---

## 5. Where FRECPE does *not* help

This work started from a broader question: where on the M5 do ARM's estimate
instructions pay off? Most places they don't.

| Use | Result | Why |
|---|---|---|
| FRECPE + Newton steps instead of division | **Loses** | `FDIV` on the M5 takes 7 cycles and can start one per cycle, no slower than the estimate itself |
| `tan`, `tanh`, `sinh`, `cosh`, … (they divide) | **Loses** | Same reason: there the estimate only replaces a division |
| `exp`, `sin`, `cos` | **Not applicable** | Their range reductions use no reciprocal |
| bfloat16 log | **No benefit** | Inputs are few enough to use a direct table with no reduction step |
| FRSQRTE + Newton steps for `1/√x` | **Wins** (1.84 vs 3.0 cycles) | It replaces a square root *and* a division |
| FRSQRTE for the `√` inside `acosf` | **Loses** (+12 cycles latency) | `acosf` needs `√x`, not `1/√x`, so the estimate replaces only one fast instruction |

The pattern: an estimate instruction helps when it removes something genuinely
slow, either several slow instructions at once or a memory lookup on the
critical path. A logarithm's range reduction is the clearest case.

---

## 6. Limits of these results

- **One CPU.** All numbers are from an Apple M5. The win depends on how slow
  the float-to-integer register copy is, which differs between chips.
- **Timer-based cycle counts.** The table above uses the calibrated timer, not
  hardware counters. The `logf` hardware run agreed closely, but the others
  have not been confirmed with `sudo`.
- **Random inputs.** The latency gap assumes inputs vary. When consecutive
  inputs hit the same table entry, the CPU can predict the table load, and
  the gap roughly halves.
- **Warm caches.** Every table here stays in L1. `make bench-cache` reruns the
  four logs with the caches swept between bursts of calls, which is where a
  saved table fetch should matter most; see
  [BENCHMARKING.md §10.10](BENCHMARKING.md).
- **`powf` is checked by sample in `y`.** Every `x` is checked, but only for
  the chosen `y` values and the special cases.

---

## 7. Reproducing

The benchmarks live in `ARM-approx-research/benchmarks/scalar/`:

| Directory | Programs |
|---|---|
| `logf/` | `bench-logf`, `bench-log2f`, `bench-log10f` |
| `powf/` | `bench-powf` |
| `asinhf-acoshf/` | `bench-asinhf`, `bench-acoshf`, `bench-log1pf` |

```bash
make bench
```

That runs everything without `sudo` (timer-estimated cycles). For hardware
cycle counts, run a single program as root, for example:

```bash
sudo ARM-approx-research/benchmarks/scalar/logf/bench-logf
```

Each program writes its full report, `bench_results_<function>.txt`, next to
its binary. For how the measurement works in detail, see §10 of
[BENCHMARKING.md](BENCHMARKING.md).
