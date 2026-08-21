# log-research

Does a correctly-rounded `ln()` for bfloat16 exist using a two-table
(`T1`, `T2`) reconstruction, where both tables are themselves bf16 values?

The question is posed as a MILP: if the model is **feasible**, such a table
pair exists. If it is **infeasible**, no bf16-grid table can be correctly
rounded at that candidate width — a property of the format, not of any
particular implementation.

## Pipeline

```bash
./build-lp.sh          # generate milp-constraints.lp + fragments
./solve-fragments.sh   # solve each fragment, print status table
```

`build-lp.sh` drives three CMake targets from the repo root (`bound-calc` and
`milp-gen` hardcode `log-research/`-prefixed output paths):

| Step | Target | Output |
|---|---|---|
| 1 | `ln-precompute` | `ln-precompute.txt` — `ln(x)` for every bf16 input |
| 2 | `ln-bounds` | `ln-bounds.txt`, `lp-constraints.txt` |
| 3 | `ln-milp` | `milp-constraints.lp` — the MILP |

It then splits the model and checks that no coupling rows were lost.

## Model structure

254 `T1` entries, 128 `T2` entries. Each is pinned to one of `2*CAND_ULP+1`
bf16 candidates by a `selT*` (choose exactly one) and `linkT*` (bind the
continuous value to the chosen candidate) row pair.

Every coupling row has the exact form:

```
c<N>_lo: T1_a + T2_b >= <bound>
c<N>_hi: T1_a + T2_b <= <bound>
```

254 × 128 × 2 = **65024 coupling rows**. There is no T1–T1 or T2–T2 coupling,
which is what makes the model decomposable.

```mermaid
graph LR
  subgraph f1["fragment 1"]
    A1["T1_1..T1_16"]
  end
  subgraph f2["fragment 2"]
    A2["T1_17..T1_32"]
  end
  SH["T2_0..T2_127<br/>(duplicated into<br/>every fragment)"]
  A1 -->|2048 rows| SH
  A2 -->|2048 rows| SH
```

## Why fragment at all

The full model makes `glpsol` hang. Each fragment solves in well under a
second. A fragment keeps a subset of `T1` plus **all** of `T2`, so it is a
valid standalone LP.

### Reading the results — the logic is asymmetric

| Result | Means |
|---|---|
| Any fragment **INFEASIBLE** | Full model **INFEASIBLE**, proven. Fragment rows are a subset, so infeasibility lifts. |
| All fragments **OPTIMAL** | **Nothing.** Each fragment picks its own `T2`; they need not agree. Feasibility does not compose. |

## Fragment modes

```bash
./build-lp.sh                      # 16 T1 chunks (default)
./build-lp.sh --blocks 32
./build-lp.sh --t2-blocks 16       # slice T2 instead, to localize a conflict
./build-lp.sh --t1-range 1 8       # one partial fragment
python3 split-milp.py --pin-t2 fragments/solutions/frag-t1-001-016.sol
```

`--pin-t2` freezes every `zT2_k_j` to a solved fragment's assignment, which
makes the `T1` blocks genuinely independent. Infeasible blocks then name
exactly which `T1` entries that `T2` cannot serve.

## Do not trust a bare `Optimal`

The coupling bounds sit **~1e-9 apart** (`-87.749999998999996` vs
`-87.250000001000004`), tighter than GLPK's default primal and integer
tolerances. `glpsol` will report `OPTIMAL` on assignments that violate the
source rows by up to ~1e-3.

```bash
python3 check-solution.py fragments/solutions/fragpin-t1-*.sol
```

`check-solution.py` recomputes every coupling row in exact decimal arithmetic.
It reads values from the `zT*_k_j` binaries and the LP's literal `link`
coefficients — never the `.sol` Activity column, which prints only ~6
significant digits, an error ~65x the tolerance being tested.

This is not hypothetical. Pinning a `T2` taken from one solved block at
`CAND_ULP=6` produced `OPTIMAL` on all 16 `T1` blocks, yet 97 genuine
violations (worst slack -7.7e-4), with one landing at exactly -1.0e-9.

## T2 precision sweep

`t2-precision-sweep.c` asks the natural follow-up: if `T2` were *stored wider
than bf16* — `T1` still bf16, the correctness target still bf16 rounding — how
many significand bits does `T2` need?

```bash
cmake --build build --target t2-precision-sweep
./log-research/t2-precision-sweep               # widths 8..24
./log-research/t2-precision-sweep --min-bits 8 --max-bits 32 --keep
```

It drives the existing pipeline once per width (`milp-gen p FILE` →
`split-milp.py` → `glpsol` → `check-solution.py`) and accepts a width only when
**every fragment is OPTIMAL *and* the exact re-check is clean** — never on a
bare `Optimal`, for the reason in the section above.

The search runs linearly upward rather than bisecting: feasibility is not known
to be monotone in `p`. A finer `T2` grid moves every candidate, so a width that
fails says nothing rigorous about a narrower one.

`milp-gen` takes `[T2_PREC [OUT_FILE [T1_PREC]]]`. At the defaults it emits the
original model, identical apart from the header comment; only `T2`'s candidate
grid changes above that. `T1` and the coupling bounds are identical at every
width.

### Result: T2 width is not the binding constraint

Every width from 8 to 24 bits is **INFEASIBLE**, all at the LP relaxation:

| T2 bits | Result |
|---|---|
| 8 … 24 | INFEASIBLE (16/16 T1 fragments, LP relaxation) |

This is not a limit of the sweep range. With `T2` given *unlimited* precision
and `T1` free to take any of its 7 candidates, 95 of the 128 `T2` indices still
have an **empty** feasible interval — no real number, at any storage width,
satisfies their rows.

The constraint is `CAND_ULP`, which confines `T1` to ±3 bf16 ULPs of its ideal.
Widening `T2` cannot rescue a row that `T1`'s own window already excludes.

## T1 precision sweep

The mirror experiment: pin `T2` and widen `T1` instead.

```bash
cmake --build build --target t1-precision-sweep
./log-research/t1-precision-sweep                          # T2 pinned at 10
./log-research/t1-precision-sweep --t2-bits 16 --tmlim 120
```

`milp-gen` takes `[T2_PREC [OUT_FILE [T1_PREC]]]`; `T1_PREC` is last so existing
two-argument calls keep working. Both default to 8, which reproduces the
original model exactly — the sole difference is the `\`-prefixed header comment,
which now also records `T1_PREC` (solvers ignore comment lines).

### With T2 pinned at 10 bits: blocked by one index

| T1 bits | Result |
|---|---|
| 8 … 12 | INFEASIBLE (1/17 fragments optimal) |
| 13 … 15 | INFEASIBLE (2–15/17) |
| 16 … 24 | INFEASIBLE (**16/17** — one fragment holds out) |

Unlike the T2 sweep, this one makes real progress: fragments flip to OPTIMAL as
`T1` widens, and from 16 bits up exactly **one** fragment remains infeasible.
At `T1 >= 16` the LP relaxation is feasible and only the *integer* problem
fails — a qualitative change from the T2 sweep, where every width died at the
relaxation.

Bisecting that fragment isolates the obstruction to a single entry, **`T1_126`**
— exponent −1, the binade `x` in `[0.5, 1)`. It is infeasible *on its own*, with
all 128 `T2` entries free to help. Testing all 254 indices individually at
`T1=20, T2=10`, it is the **only** singleton-infeasible one.

That obstruction is `T2`-driven, not `T1`-driven. Holding `T1` at a generous 24
bits and varying `T2`:

| T2 bits | `T1_126` alone |
|---|---|
| 10, 12 | INFEASIBLE |
| 16, 20, 24 | OPTIMAL |

So `T2 = 10` is what caps this sweep. `T2` needs ~16 bits before the near-1
binade can be served at all.

### With T2 pinned at 16 bits: OPTIMAL, but not correct

| T1 bits | Result |
|---|---|
| 8 … 14 | INFEASIBLE (1/17) |
| 15 | INFEASIBLE (7/17) |
| 16 | INFEASIBLE (16/17) |
| 17 … 24 | **VIOLATED** — 17/17 optimal, exact re-check fails |

This is the trap the section above warns about, caught automatically. From 17
bits every fragment reports `INTEGER OPTIMAL`; a sweep trusting bare status
would report **17 bits** as the answer. `check-solution.py` rejects it — the
composed assignment violates real coupling rows by 1e-5 to 9e-5, four orders of
magnitude past the 1e-9 tolerance:

```
VIOLATION c13797_hi: T1_107 + T2_101 <= -13.281250001
                     -> -13.281158447265625  (slack -9.155e-5)
```

`glpsol` accepts these because they fall inside its default integer tolerance.
They are genuine violations of the source model, not composition artifacts.

**No `T1` width in 8..24 yields a correct table at either pinned `T2`.**

## Current status

At the current `CAND_ULP=3` (see `milp-gen.cc:42`), **every fragment is
INFEASIBLE**, detected at the LP relaxation before branching — so this verdict
is not tolerance-sensitive. Infeasibility composes upward, so the full model is
infeasible: no bf16-grid `T1`/`T2` pair reproduces a correctly-rounded `ln()`
within ±3 ULP candidate windows.

Note the continuous relaxation (`lp-constraints.txt`, both tables free reals) is
**OPTIMAL**. A real-valued solution exists; it is the grid plus the ±3 ULP
window that destroys it. Since the sweep above rules out `T2` width as the
lever, widening `CAND_ULP` is what remains to test.

## Files

| File | Role |
|---|---|
| `precompute.c`, `bound-calc.c`, `milp-gen.cc` | pipeline stages 1–3 |
| `t2-precision-sweep.c` | sweeps T2 significand width over the whole pipeline |
| `build-lp.sh` | build + split, with a row-conservation check |
| `split-milp.py` | decomposition (`--blocks`, `--t2-blocks`, `--t1-range`, `--pin-t2`) |
| `solve-fragments.sh` | run `glpsol` per fragment; nonzero exit if any is not OPTIMAL |
| `check-solution.py` | exact re-validation of a composed assignment |

`fragments/` and all `.lp` files are gitignored — regenerate with `build-lp.sh`.

`core-near1.lp`, `core-i1_126.lp`, `core-i1_127.lp` and
`milp-constraints-min.lp` are kept artifacts with **no generator in this repo**;
nothing here rebuilds or overwrites them.
