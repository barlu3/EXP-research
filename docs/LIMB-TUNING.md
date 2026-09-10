# Finding the entries that need tuning

How `table-gen/log/limb-gen.c` and `table-gen/exp/exp-limb-gen.c` decide which
bf16 table entries to nudge off their "correct" values, and why nudging them is
the right thing to do.

No floating-point background assumed beyond knowing that a `float` has limited
precision.

## The 30-second version

These functions compute `exp` and `ln` by looking up two numbers in tables and
combining them. We want to store those tables using only 16-bit floats, which
aren't precise enough on their own — so each table entry is stored as a *sum* of
two 16-bit floats instead of one.

At two pieces per entry, a handful of inputs come out wrong: 3 of 3954 for
`exp`, 14 of 32512 for `ln`. Rather than spend a third piece (which costs
memory), we **deliberately make a few entries slightly less accurate**, chosen
so that the *final answers* come out right. A search finds which entries to
nudge and by how much.

That sounds like cheating. It isn't, and the rest of this document explains why.

## Background: what the tables are for

### bf16

`bf16` (bfloat16) is a 16-bit float: 1 sign bit, 8 exponent bits, 7 stored
mantissa bits. That's **8 bits of precision** — about 2 decimal digits. It has
the same range as a 32-bit `float` but far less precision, which is why it's
popular in machine learning.

### Correct rounding

A function is **correctly rounded** if it always returns the closest bf16 value
to the true mathematical answer. This is a strong guarantee: not "accurate to
within a bit or two", but *always the single best answer*. Because bf16 has only
65536 possible bit patterns, we can verify it by brute force — just test every
one.

### The table trick

`ln(x)` is computed by splitting `x` into its exponent and mantissa:

```
x = 2^e  ×  m          (where 1 ≤ m < 2)
ln(x) = ln(2^e) + ln(m)
        \______/  \____/
         T1[i1]   T2[i2]
```

Both pieces are looked up in tables, then added. `exp` does the analogous thing
with a product instead of a sum. CORE-MATH's original tables store each entry as
a 32-bit `float`.

**Our goal:** get identical results while storing nothing wider than 16 bits.
The kernel's arithmetic, its indexing and its rounding all stay exactly as they
are. The *only* thing we're allowed to change is the bits in the tables.

## Step 1: splitting an entry into limbs

One bf16 per entry is hopeless — 8 bits of precision means `ln` gets the wrong
answer on **8091 of 32512** inputs.

So store each entry as a sum of several bf16 values, called **limbs**. The idea
is the same as writing a number in pieces:

```
π ≈ 3.14159...
  ≈ 3.1     +  0.04    +  0.0016
    piece 1    piece 2    piece 3
```

Each piece captures what the previous ones missed. The construction is
**iterated round-and-subtract**:

```
remainder = ideal_value
for k in 0 .. n-1:
    limb[k] = nearest_bf16(remainder)
    remainder = remainder - limb[k]
```

The kernel adds the limbs back up in 32-bit arithmetic and rounds once. Two
limbs give roughly 16 bits of precision, three give 24 — and 24 happens to be
exactly a 32-bit `float`'s precision, so **three limbs reproduce the original
table exactly**. That's the easy, boring option. It also costs 50% more memory
than the tables we're replacing, which is the whole thing we were trying to
avoid.

## Step 2: what goes wrong

Here's how many inputs come out wrong at each configuration, using the plain
round-and-subtract split. `limb-gen` prints this on every run so the numbers
can't drift:

```
ln  (of 32512 inputs)        exp  (of 3954 inputs)
  n1\n2   1     2     3        n1\n2   1    2    3
     1  8091  8078  8078          1   886  722  722
     2   239    14    15          2   664    3    3
     3   231     0     0          3   656    2    0
```

Read `n1` as "limbs per T1 entry" and `n2` as "limbs per T2 entry".

Two things to notice:

**It's a step function.** Going from 1 limb to 2 is transformative; going from 2
to 3 barely matters. One limb gives about 0.4% relative error, which is *worse*
than the 0.2% margin bf16 rounding allows — so anything with a 1 in it fails
badly. Two limbs give 0.01% or better, comfortably inside the margin, so only
freak cases fail.

**More precision isn't always better.** For `ln`, 2×3 (15 failures) is slightly
*worse* than 2×2 (14). That looks impossible until you see why the failures
happen — Step 6 explains it.

So `2×2` is the configuration we want: it's the smallest, and for `exp` it uses
*exactly* the same memory as the 32-bit tables it replaces. It just has 3 and 14
wrong answers we need to get rid of.

## Step 3: the key idea

Here's the insight that makes tuning work.

**The canonical split is not the only legal table.** We don't need each entry to
be the most accurate bf16 sum available. We need the *final answer*, after the
kernel combines two entries and rounds, to be correct.

Think of it as buckets. The true answer `ln(x)` is some real number. The bf16
values around it divide the number line into buckets, and correct rounding means
landing in the right bucket:

```
        |<------ this bucket rounds to A ------>|<--- rounds to B --->|
   -----+---------------------+-----------------+----------+----------+-----
     midpoint                 A             midpoint       B      midpoint
                              ^                     ^
                        ideal answer          where we land
```

We don't have to hit the ideal answer. We have to land anywhere in its bucket.
That's a *much* weaker requirement, and it gives us room to move entries around.

So: take an entry and shift its **last limb** by a small whole number of steps.
That deliberately makes the entry slightly less accurate, but if it pushes a
misrounding input back into the right bucket without knocking anything else out,
we've won.

Three restrictions keep this manageable:

1. **Only the last limb moves.** Earlier limbs stay canonical.
2. **Only whole steps.** A "step" is one increment of the last limb's least
   significant bit, so the reachable values form a grid, not a continuum.
3. **Only a few steps** (±12 for `exp`, ±16 for `ln`). It's a search of the
   neighbourhood around the canonical split, not of all possible tables.

## Step 4: the search

Two ideas make the search cheap.

**Use-lists.** Moving `T1[i]` can only change the answer for inputs whose index
happens to be `i`. So we score that entry against just *its own* inputs, which
is exactly equivalent to rescoring everything — but far faster. For `exp` an
entry is read by only 8–16 inputs.

**Bounded coordinate descent.** Repeatedly: for each entry that currently gets
something wrong, try every shift in its window and keep whichever gives the
fewest errors. Prefer no shift on ties, so the table stays close to canonical.

```
repeat until nothing moves:
    for each entry e with score(e) > 0:
        d[e] <- the shift minimising score(e)      # ties -> smallest shift
```

That's it. It's hill-climbing on "number of wrong answers".

## Step 5: `exp` — easy, and we can prove it

The canonical 2×2 split gets 3 of 3954 inputs wrong. Descent fixes all three in
a single pass:

```
2x2 descent: 3 misrounded inputs at the canonical split, 1 pass, 3 tuned entries
  T1[16]  limb1 +1 ULP
  T2[105] limb1 +2 ULP
  T2[182] limb1 +1 ULP
2x2 verified: 0 misroundings over all 3954 table-path inputs
```

Why so easy? Because **the three repairs can't interfere with each other.** We
checked directly: the sets of inputs reading those three entries are pairwise
disjoint — no input reads two of them. When repairs are independent, one pass of
descent is guaranteed to find them if they exist.

One wrinkle worth knowing. The first failing input reads `T2[8]`, which holds
exactly `1.0` — and an entry that's exactly representable has a *zero* last
limb, so there's nothing to shift. (Stepping from zero lands on the smallest
subnormal, ~9e-41, which changes nothing.) The repair had to come from the other
side of the product, and it did: `T1[16]`. This isn't hypothetical; it's a real
constraint the search worked around.

## Step 6: `ln` — harder, and the reason is interesting

`ln` adds its two entries instead of multiplying, which makes the error
*perfectly* separable: the error at input `(i,j)` is exactly `e1[i] + e2[j]`, one
term per axis, no interaction. That should make it the easy case.

It doesn't. Descent fixes 13 of the 14 failures in one sweep and then **stalls at
1**:

```
T1[14] T1[29] T1[36] T1[71] T1[73] T1[74] T1[107]
T1[186] T1[209] T1[217] T1[235] T1[248]   -> fixed
x = 0.9921875                             -> stuck
```

### The stuck input

`x = 127/128 = 0.9921875`, just below 1. Here `T1 ≈ -0.693` and `T2 ≈ +0.685`
nearly cancel, leaving `ln(x) ≈ -0.00784` — about 88× smaller than either
operand. Two things go wrong at once:

1. `ln(x)` sits just `1.599e-7` from a bucket boundary — an unusually hard
   rounding case to begin with.
2. **The computed sum lands exactly on that boundary.** The two entries' errors
   happen to cancel to precisely the gap. Ties round to even, which picks the
   wrong neighbour.

So this is a **tie broken the wrong way**, not an accuracy problem. And that
explains the odd non-monotonicity from Step 2: at 2×3 this entry's error is *8×
bigger* and lands harmlessly off the boundary. Being less accurate saved it.

### Why descent can't fix it

It's tempting to say "one step is 7.63e-6, the gap is 1.6e-7, the steps are too
coarse". **That's wrong.** We don't need to hit the gap — we need to land in the
bucket, which is `6.1e-5` wide. Eight grid points fit inside it. A single step of
`T1[126]` *does* land in the right bucket and *does* give the correct answer.

Descent refuses to take it anyway. Here's the score of row 126 at each shift:

```
shift:  ... -3   -2   -1    0   +1   +2   +3  ...
score:  ...  4    3    1    1    4    6    9  ...
                       ^    ^
                  fixes 126  the starting point
                  breaks 77
```

The shift that repairs column 126 **breaks column 77 instead**. So the score
stays at 1 either way. Descent only accepts *strict* improvements and prefers no
shift on ties — so it keeps the starting point and stops.

This is a **plateau**: a flat spot with no downhill direction. Anyone who has
written a hill-climbing algorithm has seen one. The window width is irrelevant;
widening it doesn't help.

The lesson: the *error* separates cleanly per axis, but the *score* is a count
over shared inputs and doesn't separate at all. `ln` is the easy case for error
and the hard case for search.

## Step 7: escaping the plateau

The fix is a move descent will never make on its own — one that doesn't improve
the row it touches, and only pays off later via the other axis:

```
for each row that still fails, for shift = 1, 2, 3, ... :
    force that row's last limb to shift          # not an improvement!
    pin the row so descent can't undo it
    re-run descent over the columns              # let the other axis clean up
    keep the first seed that reaches zero errors
```

The **pin** is essential. Without it, the very next sweep would revisit the row,
notice the forced move didn't help, and revert it before any column ever saw the
consequences.

What actually happens: forcing `T1[126]` by one step moves the failure from
column 126 to column 77, exactly as the score table predicts. Then a `T2[77]`
shift of +4 clears it. Two steps, the first of which looks worthless alone.
Total: 14 tuned entries, zero wrong answers.

**This is why `ln` shipped at 3×2 for as long as it did.** Plain descent reports
"2×2 is infeasible, 1 input can't be repaired", which looks like a structural
result and is not one.

Is the solution fragile? No — we enumerated every possible seed, and **7 of them
reach zero**, across two different rows. Three are just as compact as the one we
ship. The rule "smallest shift, lowest row first" just picks a canonical
representative.

## Step 8: how we know it's right

A tuned entry is no longer "the nearest bf16 to the remainder", so nothing about
the table is self-evident any more. Correctness comes from checking:

1. **The generator refuses to ship a bad table.** Before writing the header it
   re-scores every input against a high-precision MPFR reference. Nonzero
   failures → print "refusing to write" and exit. Every limitation of the search
   *fails closed* this way.
2. **An independent exhaustive check.** `cross-eval/` tests all **65536** bf16
   bit patterns against MPFR and reports 0 discrepancies. It calls the shipped
   function through its public entry point and doesn't share the
   index-computing code with the generator, so a bug in one wouldn't hide a bug
   in the other.

Worth knowing how thin the margin is: the closest-passing input clears its
bucket boundary by only about **3×** the worst-case rounding error of the
32-bit accumulation. Comfortable, not vast — which is why the table is valid
only for the exact arithmetic it was scored against (see Maintenance).

## Step 9: knowing in advance whether a solution exists

Steps 1–8 tell us the table we *have* is correct. They say nothing about whether
a table *exists* when the search fails — and that ambiguity is what kept `ln` at
3×2. So the generator now settles it before searching at all.

Here's the trick. Since only the last limb moves, and `ln` *adds* its entries,
the computed answer is:

```
result(i,j) = base(i,j) + a[i] + b[j]
```

where `a[i]` and `b[j]` are the two shifts. Correct rounding means landing in a
bucket, i.e. a two-sided constraint:

```
L(i,j)  ≤  a[i] + b[j]  ≤  U(i,j)
```

Substitute `b[j] → -b'[j]` and every one of those 32512 constraints becomes

```
a[i] - b'[j]  ≤  U(i,j)          and          b'[j] - a[i]  ≤  -L(i,j)
```

which is a system of **difference constraints** — the classic reduction to
shortest paths. Build a graph with an edge of weight `w` for each constraint
`x - y ≤ w`; the system is satisfiable **if and only if the graph has no negative
cycle**, which **Bellman–Ford** decides in `O(V·E)`. Here `V = 383` and
`E = 65024`: milliseconds.

This is qualitatively better than searching, because a negative cycle is a
*proof*, not a search that gave up. The verdict has three values:

| Verdict | Meaning |
|---|---|
| **SAT** | A correct table provably exists — and here it is |
| **UNSAT** | No correct table exists, even allowing fractional shifts |
| **UNKNOWN** | Neither proved; fall back to searching |

Two details make it honest:

- **SAT is earned, not inferred.** The constraints get shrunk slightly to
  account for 32-bit rounding and for the fact that shifts are whole numbers,
  and any constraint too narrow to survive shrinking gets *dropped*. At 2×2 that
  happens once; at 1×1 it happens ~32000 times out of 32512. So we don't trust
  "feasible" — we build the actual table from the solution and test all 32512
  inputs. Only zero failures counts as SAT.
- **UNSAT can never happen for `ln`.** Because the error is exactly separable,
  setting `a[i] = -e1[i]` and `b[j] = -e2[j]` cancels it perfectly, so a
  *fractional* solution always exists at any limb count. Every real obstruction
  comes from shifts having to be whole numbers, and shows up as UNKNOWN.

Running it on `ln`:

```
  cfg   verdict   minW  moved  verified
  1x1   UNKNOWN    -1      0   -
  2x1   UNKNOWN    -1      0   -
  2x2   SAT         2    325   yes
  3x2   SAT         1    356   yes
  3x3   SAT         1    195   yes
```

**2×2 is the smallest configuration that certifies SAT** — so when descent later
stalls on `x = 0.9921875`, we already know a correct table exists and the stall
is a search failure, not a dead end.

### So why keep the search?

Because Bellman–Ford answers the wrong question well. It returns a *corner* of
the solution space, which shifts **325 of 382 entries**. Descent shifts **14**.
For a generated header full of constants that humans audit, compact matters a
lot. So the two are used together: the certificate *decides*, the search
*minimises*.

## Results

| function | config | canonical wrong | search | tuned entries | final |
|---|---|---|---|---|---|
| `exp` | 3×3 | 0 | none (exact by construction) | 0 | 0 |
| `exp` | 2×2 | 3 | descent, 1 pass | 3 | 0 |
| `ln` | 3×2 | 0 | none | 0 | 0 |
| `ln` | 2×2 | 14 | descent + seeded retry | 14 | 0 |
| `sin` | 2×3 mid | 0 | none | 0 | 0 |
| `sin` | large path, 2 limbs | 70 | descent, floor at 8 | — | **not solved** |

Memory, against the 32-bit tables being replaced:

- `ln` 2×2: **1792 B vs 2048 B** — 12.5% smaller
- `exp` 2×2: **3072 B vs 3072 B** — exact parity, so bf16-only storage costs
  nothing at all here
- The exact variants (3×3, 3×2) are 50% and 33% *larger*, which is what tuning
  buys us out of

Regenerating everything takes 0.12 s for `ln` and 0.008 s for `exp`.

## What this does and doesn't prove

The search looks at a *neighbourhood* of the canonical split, so **a failure
proves nothing**. We measured each limitation rather than just listing them:

| Limitation | Status |
|---|---|
| Window is bounded (±12 / ±16) | **Real** — but Step 9 removes it for `ln` |
| Sweeps are capped, not run to a proved fixpoint | **Real** |
| Only `T1` rows are used as seeds | **Cheap to fix** — seeding `T2` columns also works (20 seeds reach zero) and would find the answer in one step instead of two |
| Only the last limb moves | **Vacuous** — we scanned all ±16×±16 combinations of both limbs and nothing beats leaving limb 0 alone. The first limb's step is 512× larger than the second's and 64× wider than the whole bucket, so any move throws the sum far out and nothing pulls it back |
| Entries with a zero last limb can't move | **Real**, and it binds once (`exp`'s `T2[8]`, Step 5) |

Every one of these **fails closed**: the emitted table is scored with the same
restrictions the search used, so a blocked repair shows up as a nonzero failure
count and aborts the write. Neither generator can emit a table it hasn't scored
clean.

### Where it doesn't work: `sin`'s large path

`sin`'s `|x| ≥ 4096` path chains up to *eight* lookup pairs and reuses entries
across many inputs, so errors compound and no per-entry choice fixes them.
Descent bottoms out at 8 failures. `S3`/`C3` therefore keep three limbs.

We can now say *why* this differs from `ln`'s plateau, not just that it does.
The whole difference-constraint reduction in Step 9 works because each input
reads exactly **two** tuned entries, one per axis. Chain eight and the
constraint becomes a bound on a sum of eight variables — a genuine integer
program with no shortest-path structure. So `sin` isn't evidence the method is
weak; it's evidence the method needs the two-axis shape.

## Maintenance

A tuned table is correct only for the exact arithmetic it was scored against,
and the margin it relies on is smaller than a 32-bit rounding error. So:

- **Accumulation order is part of the table.** `ln` adds smallest limb first;
  `exp` rebuilds each factor then multiplies once. Reordering needs re-scoring.
  (Measured: the orders tried are currently equivalent — but 38.7% of `ln`'s
  accumulations do round, so that's a coincidence of these tables, not a rule.)
- **`-ffp-contract` is *not* the hazard here.** Neither kernel contains a
  multiply-add pair to fuse, and every contract setting emits byte-identical
  output. The flag that matters is **`FLT_EVAL_METHOD`** — computing the 32-bit
  partial sums at wider precision (x87) would move exactly the values the
  tuning depends on.
- **Regenerate on any change** to the kernel, compiler or flags. It takes a
  tenth of a second, so it belongs in CI: regenerate, diff against the committed
  header, fail on any difference. `ctest -R limb-gen-golden` does exactly that.

The exact configurations (3×3 for `exp`, 3×2 for `ln`) carry none of these
obligations, which is why both generators keep them around.

## Reproducing

```bash
cmake --build build --target limb-gen exp-limb-gen sin-limb-gen
./table-gen/log/limb-gen        # certificate, frontier, seed, every tuned entry
./table-gen/exp/exp-limb-gen    # its 3 tuned entries
./table-gen/sweep/limb-config-sweep   # the exp/sin configuration frontier

cmake --build build --target limb-analysis
./table-gen/analysis/limb-lp           # every configuration, decided
./table-gen/analysis/limb-diagnostics  # the score table, seed enumeration
./table-gen/analysis/limb-margins      # how thin the margins actually are

ctest -R 'limb-cert-test|limb-gen-golden|verify-.*-limb'
```
