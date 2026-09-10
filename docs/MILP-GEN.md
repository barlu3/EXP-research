# Turning "is this table possible?" into a solvable program

How `log-research/` decides whether a correctly-rounded bf16 `ln` can be built
from tables that are themselves bf16 — by rewriting an awkward question about
rounding into a **mixed-integer linear program** and handing it to a solver.

Assumes you've seen linear programming once (constraints, feasible region,
"maximise subject to"). Everything else is built up here.

## The 30-second version

`ln(x)` for bf16 is computed as `T1[i1] + T2[i2]`, two table lookups added
together. CORE-MATH stores those tables as 32-bit floats. **Can they be 16-bit
instead?**

You can't answer that by trying tables at random — there are far too many. So we
turn the question into a system of linear inequalities with some integer
variables (a MILP), and ask a solver whether *any* assignment satisfies all of
them. The answer for one bf16 per entry is **no**. For a *sum* of bf16 pieces per
entry, it's **yes**.

Getting there requires three rewrites, and knowing exactly what each verdict
does and doesn't prove. That second part is most of this document.

## The question, stated precisely

For every one of the 32512 valid inputs `x`, let `y = correctly_rounded_ln(x)` —
the single closest bf16 value to the true `ln(x)`, computed once and for all in
high precision. We want to know whether tables exist such that:

```
round(T1[i1] + T2[i2]) == y        for all 32512 normal inputs
round(T3[i2])          == y        for all 127 subnormal inputs
```

with every table entry being a bf16 value. (`T3` is a separate direct-lookup
table for very small inputs, where the exponent/mantissa split doesn't work.)

Note what kind of question this is. It's not "compute the best table" — it's
**does one exist at all?** That's a *feasibility* problem: there's no objective
function to optimise, we just want to know if the feasible region is empty.

## Why you can't just write this down

Three things about that specification are hostile to a linear solver.

### Obstruction 1: `round()` is a staircase

`round()` is piecewise constant. It's not linear, not continuous, not even
convex or concave. Writing `round(S) == y` isn't an algebraic constraint at all —
you can't hand a solver a step function.

### Obstruction 2: the bf16 grid isn't evenly spaced

"`T1[k]` must be a bf16 value" is a membership test in a finite set — and the
spacing of that set **doubles at every power of two**. Near `T2`'s values the
gap is about `2^-9`; near `T1`'s extremes (magnitude ~88) it's `2^-1`. So you
*can't* say "`T1[k]` is an integer multiple of δ" for any single δ. Ordinary
integer variables don't capture this.

### Obstruction 3: how the pieces combine

For `ln` this one is free, because `T1 + T2` is a **sum** — already linear in the
unknowns. It is *not* free for `exp` (`T1 × T2`) or `sin`, where you get a
product of two unknowns. That's a separate story, and it's in the "where this
stops working" section below.

### Plus one benign annoyance

If you add a constant `c` to every `T1` entry and subtract it from every `T2`
entry, every constraint is unchanged — the sum is the same. So the solution set
is an infinite family (a "gauge freedom"). Not a nonlinearity, but if you don't
pin it down the feasible region is unbounded and the solver wanders.

## Rewrite 1: invert the rounding operator

This is the pivot of the whole construction, and it's a genuinely nice trick.

Don't ask *"what does `round` do to `S`?"* — that's the staircase. Ask
**"which values of `S` round to `y`?"** That's a *level set* of `round`, and
level sets of round-to-nearest are **intervals**:

```
        |<------- these values round to y ------->|
   -----+--------------------+--------------------+-----
       y⁻       midpoint      y      midpoint     y⁺
                  = lb              = ub
```

So `round(S) == y` becomes:

```
lb  ≤  T1[i1] + T2[i2]  ≤  ub
```

Two linear inequalities. And critically, **`lb` and `ub` are constants** — we
computed `y` in advance from `x`, so we know exactly which interval to use. The
transcendental content (`ln 2`, `ln(1+k/128)`, the midpoints) is all evaluated
offline in MPFR at 200 bits. The solver never sees a logarithm; it sees a sparse
matrix of rational numbers.

That single move kills Obstruction 1 for all 32512 inputs at once.

### Two details that are easy to get fatally wrong

**Ties-to-even decides whether the interval is open or closed.** When `S` lands
exactly on a midpoint, IEEE rounds to whichever neighbour has an even mantissa.
So `y` *owns* both its midpoints if its own mantissa is even, and neither if
it's odd. The generator checks the parity bit and picks accordingly. Getting
this wrong is not cosmetic:

- treat every interval as **open** → you exclude tie points that genuinely
  round to `y` → the model reports **infeasible on exactly the hard cases the
  study is about**
- treat every interval as **closed** → you admit values that round to the
  neighbour → a "feasible" answer is worthless

**Strict inequalities don't exist in LP file formats.** LP and MILP formats can
only express `≤` and `≥`, never `<`. So the open sides get tightened by
`ε = 1e-9`: `(lb, ub)` becomes `[lb + ε, ub - ε]`. This is an **inner**
approximation — we're asking for slightly *less* than the truth. Consequence:
"feasible" is fully sound, but "infeasible" is only sound up to `ε`. Given that
neighbouring bf16 values here are ~`0.1` apart, shaving `1e-9` is immaterial —
but it isn't *nothing*, and the same `ε` comes back later.

One special case: at `x = 1`, `ln(x) = 0` exactly and `lb == ub`. Nudging both
sides there would flip the inequalities and manufacture a fake infeasibility on
the one row whose answer is exactly representable. The code never nudges a
closed interval.

## Rewrite 2: pick from a menu

Now the model is linear — but over *real-valued* tables. That's the continuous
LP, and it comes back **feasible**: real-valued tables exist. Unsurprising and
not what we asked. We need entries that are actually on the bf16 grid
(Obstruction 2), and rounding a real solution afterwards is not good enough —
it might round out of its interval.

The fix is **one-hot encoding**. For each table entry, enumerate a small set of
candidate bf16 values offline, then let binary variables choose one:

```
candidates for T1[k]:  all bf16 values within 3 ULPs of the ideal   (7 of them)

sel  row:   z₀ + z₁ + ... + z₆ = 1        exactly one is chosen
link row:   T1[k] = c₀z₀ + c₁z₁ + ... + c₆z₆    and T1[k] takes its value
```

Since `sel` forces the `z`s to a corner of the simplex, no fractional blend
survives to an integer solution — the encoding is **exact**, not an
approximation. And every `cⱼ` is a *constant* computed offline.

Here's a real emitted pair, for `T1[1]` whose ideal is `ln(2^-126) ≈ -87.3365`:

```
selT1_1:  + zT1_1_0 + zT1_1_1 + zT1_1_2 + zT1_1_3 + zT1_1_4 + zT1_1_5 + zT1_1_6 = 1
linkT1_1: T1_1 + 87.5 zT1_1_0 + 88 zT1_1_1 + 88.5 zT1_1_2 + 89 zT1_1_3
               + 87 zT1_1_4 + 86.5 zT1_1_5 + 86 zT1_1_6 = 0
```

Seven candidates, spaced `0.5` apart — the local ULP at magnitude 88.

### Two entries are pinned to a single candidate

**`T1[127] = 0`** — this is the gauge fix. The ideal is `ln(2⁰) = 0`; giving it
one candidate removes the additive-shift freedom, so the solution is a genuine
split rather than a translation family.

**`T2[0] = 0`** — this one is a real numerical hazard, not a convenience.
`ln(1) = 0`, and the bf16 neighbours of zero are the tiniest subnormals,
`±2⁻¹³³ ≈ 9.2e-41`. Admitting those as candidates would put coefficients of
magnitude `1e-41` in the same matrix as coefficients of magnitude `88` — a
**dynamic range of ~10⁴²**, far beyond what double-precision simplex arithmetic
can hold together. The result wouldn't be a slow solve, it would be a *wrong
answer*: false infeasibility from numerical noise. Pinning it costs nothing (a
zero entry is exact; there's nothing to search).

## What the model actually looks like

At the default settings (one bf16 per entry, candidate window ±3 ULP):

| Block | Rows | Where from |
|---|---|---|
| `sel` | 509 | one per entry (254 + 128 + 127) |
| `link` | 509 | one per entry |
| `couple` | 65,024 | 2 per normal input (`≤` and `≥`) |
| `subnormal` | 254 | 2 per subnormal input |
| **total** | **66,296** | |

| Columns | Count |
|---|---|
| continuous (the table entries) | 509 |
| binary (the selectors) | 3,551 |
| **total** | **4,060** |

About 138,000 nonzeros. The objective is identically zero — it's pure
feasibility. (Amusing detail: GLPK's parser rejects a bare `obj: 0`, so the
generator emits `Minimize obj: 0 T1_1`.)

## Reading the verdicts — this is the important part

A solver says "OPTIMAL" or "INFEASIBLE". Neither of those is automatically an
answer to *your* question, and getting this backwards is the classic way to
publish a wrong result.

The rule is about which direction you changed the feasible region:

| If your model is a... | ...then | licensed verdict |
|---|---|---|
| **restriction** (smaller feasible set than the truth) | feasible ⟹ truth feasible | **"yes" is sound** |
| **relaxation** (larger feasible set than the truth) | infeasible ⟹ truth infeasible | **"no" is sound** |

Our model is a **restriction**, in two ways: the `ε` tightening, and — much more
significantly — the candidate window only holds values within **±3 ULP of the
ideal**. So:

| MILP verdict | What it actually licenses |
|---|---|
| **feasible** | A bf16-grid table exists. Sound. |
| **infeasible** | No bf16 table exists ***within ±3 ULP of the ideals.*** |

That second row is genuinely weaker than "no table exists". The project's
headline negative result — single-bf16 `T1` is infeasible — is a statement about
a ±3 ULP neighbourhood, and widening the window is the remaining lever. Stating
it that way isn't hedging; it's the only honest reading.

## Structure you get for free

Look at the shape of a coupling row: `T1_a + T2_b`, unit coefficients, one
variable from each family. There is **no** `T1`–`T1` or `T2`–`T2` row anywhere.

So the constraint graph is **complete bipartite** — every `T1` entry talks to
every `T2` entry and to nothing else. Two useful things follow.

**You can split the problem.** A fragment that keeps *some* `T1` entries plus
*all* of `T2` is a self-contained sub-LP, not an approximation. `split-milp.py`
does this. But the inference is asymmetric, and again only one direction works:

| Observation | What you may conclude |
|---|---|
| **any** fragment infeasible | full model infeasible — **proven** |
| **all** fragments feasible | **nothing.** Different fragments may want different `T2`. |

That second line is the trap. Each fragment is a *relaxation* (fewer rows), so
feasibility doesn't compose. To fix it, `--pin-t2` freezes `T2` to the values
one solved fragment chose; now the `T1` blocks really are independent, and an
infeasible block names exactly which entries that particular `T2` can't serve.
(If you've seen Benders decomposition, this is a hand-rolled version of fixing
the complicating variables.)

**`T3` falls out completely.** The subnormal rows are `lo ≤ T3[i2] ≤ hi` —
*single-variable*. They share no variable with anything, so the 127 entries are
127 independent one-variable questions. There's no composition step, so nothing
for composition to break, and "feasible" means exactly what it says.

Result: **`T3` works at plain bf16 width**, all 254 rows clean under exact
re-check. Which incidentally means CORE-MATH's 32-bit `T3` is wider than
correctness actually requires.

## The limb model

Single-bf16 entries don't work. So ask a weaker question: can each entry be a
**sum** of several bf16 values?

This stays linear, which is the whole point. Add one row per entry:

```
T1[k] - (limb0 + limb1 + limb2) = 0
```

A sum of unknowns is affine, so nothing about the earlier reasoning changes —
and the coupling rows are *byte-for-byte identical*, because what "correctly
rounded" means doesn't depend on how you store the table.

| | single-value | per-limb (3×`T1`, 2×`T2`) |
|---|---|---|
| total rows | 66,296 | 67,950 |
| continuous columns | 509 | 1,527 |
| **binary columns** | **3,551** | **7,985** |

Only 2.5% more rows — but **2.3× the binaries**, and binaries are where
branch-and-bound cost actually lives. It shows: `glpsol` ran 611 s on the full
model without finishing. Solving entry-by-entry instead: 253 of 254 immediately
optimal, and one (`T1[134]`) timed out after 3.1 million nodes, then solved in
0.0 s once its own binaries were pinned.

Verdict: **`3×2` limb tables are feasible**, 0 violations across 65,278 rows
under exact re-check.

One caveat worth stating plainly. The MILP is a model over the **real numbers**;
the actual implementation sums limbs in **float32**. Those agree only if the
float32 accumulation is error-free. So the project carries *two* separate proofs
and neither subsumes the other:

- the MILP proves the **real-arithmetic model** feasible, in exact decimal
  arithmetic
- `cross-eval/` proves the **floating-point implementation** correctly rounded,
  exhaustively over all 65,536 inputs against MPFR

## Where linearization stops: multiplication

`exp` reconstructs as `T1[i1] × T2[i2]`. Apply the interval trick and you get:

```
lo  ≤  T1[i1] · T2[i2]  ≤  hi
```

That's **bilinear** — a product of two unknowns. The feasible region is
non-convex, and no amount of clever encoding of the *values* removes it. Two
textbook fixes exist; both were rejected, for instructive reasons.

**(a) McCormick envelopes.** The standard convexification: bound both factors
and replace the product with an auxiliary variable plus four inequalities. But
it's a **relaxation** — so infeasible proves infeasibility, and feasible proves
nothing. Since the expected (and actual) answer here was "feasible", a
relaxation is simply the wrong instrument. It'd also be weak: the envelope's
slack scales with box width, and the boxes are the same size as the intervals
being tested.

**(b) Products of binaries.** Exact, and the obvious route: keep one-hot on both
factors and linearize each product of selectors `w = z⁽¹⁾z⁽²⁾` with the classic
triple `w ≤ z⁽¹⁾`, `w ≤ z⁽²⁾`, `w ≥ z⁽¹⁾ + z⁽²⁾ − 1`. No relaxation gap. The
cost kills it: with `K` candidates per side you need `K²` product variables and
`3K²` rows *per coupling row*. At `K = 25` over `exp`'s 3954 inputs that's about
**2.5 million auxiliary binaries and 7.4 million rows** — for a problem whose
real answer turns out to be obtainable by inspection.

### What was done instead: make one factor a constant

Neither. Remove the bilinearity **at its source**.

> A bf16 value carries 8 significand bits; a float32 carries 24. So **three
> non-overlapping bf16 limbs represent any float32 exactly.**

That's a fact about bit widths, not a test result. If `T1` is stored at three
limbs, the reconstructed `T1[i1]` *is* the shipped float32 value — a known
constant. And then:

```
lo  ≤  τ · T2[i2]  ≤  hi          where τ is a constant
```

which is an ordinary linear row in one unknown. Two things follow, and the
second matters more:

1. **Linearity is restored.**
2. **The model separates completely.** Every row involving `T2[i2]` involves
   *only* `T2[i2]`. It isn't one MILP with 256 coupled entries — it's **256
   independent single-entry problems.**

And once it's separated, **you don't need a solver at all.** Each entry's
feasible set can be searched by *complete enumeration* over a small window of
limb displacements, running the real float32 arithmetic and comparing
bit-for-bit against the shipped implementation. That's strictly stronger
evidence than an LP verdict, because it sidesteps the real-vs-float32 fidelity
gap entirely — and it finishes in seconds.

> **Note on scope.** The `exp` figures recorded in this line of work describe
> the `3×2` configuration (254 of 256 entries fine at the canonical split, two
> needing a 1-ULP nudge). `exp` has since shipped at `2×2`, where *both* factors
> are inexact so this decoupling argument no longer applies — see
> [LIMB-TUNING.md](LIMB-TUNING.md) for how that case is handled.

`sin`'s mid path is the same argument on a two-variable row, giving a joint 2-D
search per index. Its long path (`|x| ≥ 4096`) is where everything fails: it
chains up to eight lookup pairs, each step's output feeding the next, so **no
factor can be held exact**. The constant-coefficient trick has no foothold,
per-entry independence is gone, and a greedy search bottoms out at 8
misroundings. Three limbs are kept there, where the exactness fact applies and
the question doesn't arise.

## Don't trust the solver

This model has an unusually nasty numerical profile, and it's the most
transferable lesson here.

Look at an actual pair of emitted rows:

```
c128_lo: T1_1 + T2_0 >= -87.749999998999996
c128_hi: T1_1 + T2_0 <= -87.250000001000004
```

Those bounds are `1e-9` from the true midpoints — which is **tighter than
GLPK's default feasibility tolerances**. The consequence, stated bluntly:
`glpsol` will happily report `INTEGER OPTIMAL` on an assignment that violates
the source constraints by up to about `1e-3`.

**This is not hypothetical.** On record: pinning a `T2` from one solved block
produced `OPTIMAL` on all 16 `T1` blocks — and 97 genuine violations, worst
slack `-7.7e-4`. Separately, a precision sweep found that from 17 bits upward
*every* fragment reports `INTEGER OPTIMAL`; a sweep that trusted status would
have published "17 bits" as the answer. The exact re-check rejects it, with
violations of `1e-5` to `9e-5`.

Three defences:

1. **Never trust a bare status.** `check-solution.py` recomputes every row in
   exact decimal arithmetic (Python's `Decimal`).
2. **Read values from the model, not the solution report.** The checkers resolve
   each entry through the binaries and the LP's *literal* `link` coefficients —
   never the `.sol` file's Activity column, which prints ~6 significant digits,
   an error roughly **65× the tolerance under test**. Subtle, easy to miss, and
   it would silently invalidate every check.
3. **Conserve rows across decomposition.** `build-lp.sh` counts rows in the
   source and in the fragments and aborts on mismatch — a fragment set that
   silently dropped rows would solve fast and report optimal while proving
   nothing.

Two of the project's verdicts are immune to all this, and it's worth knowing
why: the headline `T1` infeasibility is detected at the **LP relaxation, before
any branching**, so integer tolerances never enter. And the `T3` result is immune
*structurally* — single-variable rows have no composition step for tolerance
drift to corrupt.

## Results

| Question | Model | Verdict |
|---|---|---|
| Real-valued `T1 + T2` tables exist? | continuous LP | **yes** |
| Single-bf16 `T1`/`T2` tables exist? | grid MILP | **no**, at ±3 ULP (caught at the LP relaxation) |
| Does widening `T2` rescue it? | MILP, `T2` at 8–24 bits | **no** at every width |
| Does widening `T1` rescue it? | MILP, `T1` at 8–24 bits | **no** — "optimal" at ≥17 bits is a tolerance artefact, rejected by exact re-check |
| Single-bf16 `T3` (subnormals)? | 127 one-variable rows | **yes** — 254/254 rows clean, zero slack |
| `3×2` bf16 limb tables? | per-limb MILP | **yes** — 0 violations over 65,278 rows |
| Same trick for `exp`? | 256 separated enumerations | **yes** at `3×2` |
| Same trick for `sin` (mid)? | 128 separated 2-D enumerations | **yes** at `2×3`, nothing tuned |
| Same trick for `sin` (large)? | greedy descent | **no** — floor of 8 misroundings |

## The one idea worth taking away

The same pattern recurs at every stage: **move the nonlinearity out of the model
and into precomputation.**

- the rounding operator → inverted offline into interval endpoints
- the non-uniform grid → enumerated offline into candidate constants
- the transcendental functions → evaluated offline in MPFR
- one whole factor of a product → made exact offline, so it enters as a
  *coefficient* instead of a variable

What reaches the solver is, every time, a sparse matrix of rationals.

The last one is the most transferable. Forcing exactness on one side of a
bilinear constraint isn't just a linearization — it's a **separation**. It
collapsed a model that would have needed ~2.5 million auxiliary binaries into
256 independent searches small enough to enumerate exhaustively, which is a
*stronger* form of evidence than any solver verdict on the original model could
have been.

## Running it

Three sequential stages, each consuming the previous stage's output:

```bash
cmake --build build --target ln-precompute   # MPFR ground truth  -> ln-precompute.txt
cmake --build build --target ln-bounds       # rounding intervals -> ln-bounds.txt,
                                             #                      lp-constraints.txt
cmake --build build --target ln-milp         # the MILP           -> milp-constraints.lp
```

Then solve for feasibility with either solver:

```bash
glpsol --lp log-research/milp-constraints.lp    # "OPTIMAL"  => a table exists
highs log-research/milp-constraints.lp          # "Optimal"  => a table exists
```

And — per the section above — **always re-check the result**:

```bash
python3 log-research/check-solution.py          # exact decimal re-validation
python3 log-research/check-limb-solution.py     # same, against the emitted header
```

Useful knobs and notes:

- `CAND_ULP` in `log-research/milp-gen.cc` sets the candidate window (default 3).
  Changing it changes every verdict's meaning — see "Reading the verdicts".
- `log-research/split-milp.py` splits the model into fragments; `--pin-t2`
  freezes `T2` to make the `T1` blocks genuinely independent.
- Regenerating overwrites large committed research artifacts.
- `milp-gen.cc` is C despite the extension, and CMake is told so explicitly.
- `log-research/README.md` is the detailed run log, with the full record of
  verdicts, timings and sweeps.
