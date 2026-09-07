# Finding the entries that need tuning

How `table-gen/exp/exp-limb-gen.c` and `table-gen/log/limb-gen.c` decide which
bf16 limbs to move off their canonical value, and why the two generators need
different amounts of machinery to do it.

This is the search that turns a *nearly* correct limb table into a correct one.
Both `exp` and `ln` ship minimal configurations that the plain round-and-subtract
split does not reach on its own.

## The problem

Splitting a table entry into bf16 limbs by iterated round-and-subtract — limb
`k` is the bf16 nearest whatever limbs `0..k-1` left over — is the obvious
construction and usually the right one. At enough limbs it is exact and nothing
else is needed.

At the *minimal* limb count it is not exact, and a handful of inputs land on the
wrong side of a bf16 rounding boundary:

| function | minimal config | canonical misrounds | inputs on that path |
|---|---|---|---|
| `exp` | 2×2 | 3 | 3954 |
| `ln`  | 2×2 | 14 | 32512 |

`ln`'s full canonical frontier, printed by `limb-gen` on every run so the
numbers below cannot drift from the tool that produces them:

```
    n1\n2      1      2      3
       1     8091   8078   8078
       2      239     14     15
       3      231      0      0
```

3×2 is the smallest configuration that needs **no** tuning, which is what `ln`
shipped until tuning was tried. Note 2×3 (15) is marginally *worse* than 2×2
(14) — at this scale the residual failures are boundary coincidences, not a
smooth function of precision.

The canonical split is not the only legal one. Any limb values that sum into the
correct rounding interval for every input that reads them will do. So the
question becomes a search: *which entries should carry a limb a few ULP off
canonical, and by how much?*

The alternative — spend another limb — is what both generators used to do. It
costs 512 B and one dependent add per function, so it is worth a search first.

## The shared machinery

Both generators use the same three pieces.

**Displacement, not replacement.** An entry is stored as its canonical limbs
with an integer ULP displacement applied to the *last* limb only. The search
space per entry is therefore a small integer window (±12 for `exp`, ±16 for
`ln`), not the whole bf16 grid. This is a neighbourhood search around the
canonical split, not a complete search over all limb tuples — see
[Completeness](#completeness-what-this-search-does-not-prove).

**Use-lists.** Every table entry is indexed by the inputs that read it. Moving
`T1[i]` can only change results for inputs whose `i1` is `i`, so scoring that
entry against its own use-list is *exactly* equivalent to rescoring the whole
path. That equivalence is what makes the descent cheap enough to run inside the
generator on every build.

**Bounded coordinate descent.** Repeatedly: for each entry that currently
misrounds something, scan its window and keep the displacement that minimises
its own error count, preferring the smallest displacement on ties. Stop at a
fixpoint.

```
for each pass until nothing moves:
    for each entry e with score(e) > 0:
        d[e] <- argmin over window of score(e)      # ties -> smallest |d|
```

## `exp`: two regimes, and why one of them is easy

`exp` reconstructs by a product, `T1[i1] * T2[i2]`, and its two shipped
configurations sit on opposite sides of an important line.

### 3×2 — the search decouples completely

Three bf16 limbs carry 24 significand bits, exactly float32's. Because
`exp-limb-gen` splits the *shipped float32 table values* rather than recomputed
reals, three limbs reproduce `T1` **exactly**. All residual error then lives on
the `T2` side of the product.

That collapses the problem. The correctness condition for an input is a bound on
`T1[i1] * T2[i2]`; with `T1` exact, each `T2` entry can be chosen without
reference to any other. There is no coupling to resolve — it is 256 independent
single-entry searches, and enumerating a ±12 ULP window per entry is a
*complete* search of that neighbourhood rather than a heuristic.

### 2×2 — bilinear, but barely broken

At 2×2 both factors are approximations. The correctness rows become genuinely
bilinear (`T1_a * T2_b`), and the per-entry decoupling argument is gone —
`EXP-LIMB-PLAN.md` read this as requiring a MILP.

It does not, for a reason specific to how little is actually wrong. The
canonical 2×2 split misrounds **three** of 3954 table-path inputs, and those
three read three *disjoint* entries. Repairs that touch disjoint entries cannot
interact, so coordinate descent finds all three in a single pass:

```
2x2 descent: 3 misrounded inputs at the canonical split, 1 pass, 3 tuned entries
  T1[16]  limb1 +1 ULP
  T2[105] limb1 +2 ULP
  T2[182] limb1 +1 ULP
2x2 verified: 0 misroundings over all 3954 table-path inputs
```

The bilinear coupling is real; it just never gets exercised, because no two
repairs share an entry.

## `ln`: additive, and the one case descent cannot see

`ln` reconstructs by a sum, `T1[i1] + T2[i2]`, which decouples even more
cleanly than a product: the error at `(i1, i2)` is exactly `e1[i1] + e2[i2]`,
one term per axis, with no interaction at all. That should make it the easy
case. It is not, and the reason is worth recording.

Plain descent clears 13 of the 14 canonical misrounds in one pass and then
**stalls at 1**:

```
T1[14] T1[29] T1[36] T1[71] T1[73] T1[74] T1[107]
T1[186] T1[209] T1[217] T1[235] T1[248]   -> fixed
x = 0.9921875                             -> stuck
```

### Why it stalls

The survivor is `x = 0.9921875` (`i1=126`, `i2=126`), in the cancellation band
just below 1. There `T1 = -ln2 ≈ -0.693` and `T2 = ln(1.984…) ≈ 0.685` nearly
annihilate, leaving `ln(x) ≈ -0.00784` — about 88× smaller than either operand,
so roughly 6.5 bits of cancellation.

Two things go wrong at once:

1. **`ln(x)` sits 1.599e-7 below a bf16 midpoint** — an unusually hard rounding
   case.
2. **The 2-limb reconstruction lands on that midpoint exactly.** The residuals
   of the two entries cancel to precisely the gap. Round-half-to-even then picks
   `0xbc00`; the correct answer is `0xbc01`.

So the failure is a tie broken the wrong way, not an accuracy shortfall. Note
that at 2×2 this entry's error (1.599e-7) is about **8× smaller** than at 3×2
(1.27e-6) — 3×2 is correct here because its larger error lands on the harmless
side of the tie.

Descent cannot escape because the finest available move — one ULP on a second
limb — is 7.63e-6, about 24× wider than the ±1.599e-7 window that would fix it.
Every displacement of `T1[126]` (and of `T2[126]`, and of either entry's *first*
limb) moves the sum along the same 2^-17 lattice, so no single-coordinate move
lands inside the window. Worse, none of them improves that row's own score
either, so descent has no gradient to follow and simply rejects them all.

### The escape: a seeded uphill move

The fix is a move descent will never take on its own — one that does not improve
the row it touches, and is only redeemed afterwards by a different axis:

```
for each row i1 that still misrounds, for |d| = 1, 2, 3, … :
    force  T1[i1] last limb += d ULP        # uphill or neutral; not an improvement
    re-run full descent over the T2 columns  # let the other axis repair the damage
    keep the first seed that reaches zero
```

Seeding `T1[126]` with **-1 ULP** relocates its failure from column 126 to
column 77, and a `T2[77] +4 ULP` column move then clears it — a two-step repair
whose first step looks worthless in isolation. Total: 14 tuned entries, zero
misrounds.

Seeds are tried smallest-magnitude first so the emitted table stays as close to
the canonical split as possible. Any successful seed is equally correct; the
smallest is just the least surprising.

**This is the whole reason `ln` shipped at 3×2 for as long as it did.** A single
level of descent reports "2×2 is infeasible, 1 input cannot be repaired", which
looks like a structural result and is not one.

## Completeness: what this search does not prove

Both generators search a *neighbourhood* of the canonical split, so a failure
proves nothing:

- The window is bounded (±12 / ±16 ULP), so a repair needing a larger
  displacement would be missed.
- Only the last limb moves. Earlier limbs stay canonical.
- An entry whose last limb is **zero** cannot be moved at all — stepping from
  bf16 zero lands on the smallest subnormal, ~9e-41, which changes nothing.
  `exp`'s `T2[8]` holds exactly 1.0 and is the real instance of this; the repair
  for the input reading it has to come from the `T1` side.

Every one of these limitations **fails closed**: the emitted table is scored
with the same restricted accessors the search uses, so a repair the restriction
blocks shows up as a nonzero unsolved count and *aborts the write*. Neither
generator can emit a table it has not scored clean.

Success, on the other hand, is not a heuristic claim. A found solution is
checked exhaustively before it ships:

- `exp`: all 3954 table-path inputs inside the generator, then all 65536 bf16
  inputs against MPFR in `cross-eval/exp/verify-limb.c`.
- `ln`: all 32512 normal inputs against MPFR inside the generator, then all
  65536 against MPFR in `cross-eval/log/verify-limb.c`.

The tuned tables are correct by verification, not by construction — which is a
weaker guarantee than the exact configurations offer, and the reason both
generators keep their exact variants around.

## Where it does not work: `sin`'s large path

`sin`'s `|x| >= 4096` path chains up to eight lookup pairs and reuses each entry
across many inputs, so errors compound and no per-entry choice fixes them.
Bounded descent reaches a floor of 8 misrounds and stops. `S3`/`C3` therefore
keep three limbs.

This is the honest negative: descent stalling is not proof of infeasibility
there either — it is the same non-exhaustive search — but unlike `ln`'s stall,
a seeded retry does not rescue it, because the failures are not confined to one
axis that another axis can repair.

## Results

| function | config | canonical misrounds | passes | tuned entries | final |
|---|---|---|---|---|---|
| `exp` | 2×2 | 3 | 1 | 3 | 0 |
| `ln`  | 2×2 | 14 | 1 + seeded retry | 14 | 0 |
| `sin` | 2×3 mid | 0 | — | 0 | 0 |
| `sin` | large path, 2 limbs | 70 | floor at 8 | — | **not solved** |

## Reproducing

```bash
cmake --build build --target limb-gen exp-limb-gen sin-limb-gen
./table-gen/log/limb-gen        # ln frontier, the seed it needed, every tuned entry
./table-gen/exp/exp-limb-gen    # prints its 3 tuned entries
./table-gen/sweep/limb-config-sweep   # the exp/sin configuration frontier

cmake --build build --target verify-log-limb verify-exp-limb verify-sin-limb
./cross-eval/log/verify-log-limb      # 0 discrepancies / 65536
```
