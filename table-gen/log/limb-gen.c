/* Generate the 2xT1 + 2xT2 bf16 limb tables for cr_log_bf16_limb.

   Why limbs. A single bf16 table entry carries 8 significand bits, which is
   not enough: T1[i1]+T2[i2] then misrounds 8091 of the 32512 normal bf16
   inputs. Storing each ideal as a sum of bf16 limbs recovers precision using
   only bf16 storage -- the target here is hardware that has bf16 storage or
   bf16 MACs and no float32 table path.

   Split. Iterated round-and-subtract: limb k is the bf16 nearest to what the
   previous limbs left over. That canonical split leaves 14 misrounds at 2x2,
   so a tuning pass then moves individual second limbs off their canonical
   value by a few ULP until every input rounds correctly. See
   docs/LIMB-TUNING.md for the search and why the obvious version of it fails.

   Widths. 2 limbs for T1, 2 for T2, chosen as the minimum that a tuned table
   reaches. The canonical frontier is:

       n1\n2      1      2      3
          1   8091   8078   8078
          2    239     14     15
          3    231      0      0

   so 3x2 is the smallest configuration that needs NO tuning -- which is what
   this generator used to ship. Tuning 14 entries brings 2x2 to zero as well,
   saving 512 B and one dependent add, and matching how exp-limb-gen.c already
   treats its own minimal configuration.

   Correctness here is established by exhaustive search, not by construction:
   a tuned limb is no longer "the bf16 nearest the previous remainder", so the
   non-overlapping property that makes the canonical split self-evidently
   accurate does not hold. This generator therefore re-checks all 32512 normal
   inputs against MPFR before writing, and refuses to emit a failing table.

   T3 is not emitted with limbs: the subnormal band already solves at plain
   bf16 width.

   Usage: limb-gen [OUT_FILE]     (default implementations/log/logbf16-limb.h)
   Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>

#include "table-gen/common/limb-certificate.h"

#define WORK_PREC 300
#define BF16_PREC 8
#define T1_LIMBS 2
#define T2_LIMBS 2
#define T1_N 256
#define T2_N 128
#define T3_N 128

/* Tuning window, in ULP of the limb being moved. The shipped solution needs
   at most 12, so 16 leaves headroom without making the seeded retry slow.

   Do NOT derive this from the certificate's minimal window. That number (2)
   bounds the LINEAR PROGRAM's own solution, which is a different point in the
   feasible set; the descent reaches zero by a different route and needs +4 ULP
   on T2[77]. Narrowing TUNE_WIN to the certificate's W would change the emitted
   table, which is the one thing this generator may not do. */
#define TUNE_WIN 16
_Static_assert (TUNE_WIN >= 4,
  "the shipped solution displaces T2[77] by 4 ULP; a narrower window "
  "cannot reach it and would emit a different table");

typedef union { __bf16 f; uint16_t u; } b16u16_u;

/* Nearest bf16 to `v`, returned as a double (exact -- bf16 is a float subset). */
static double round_bf16 (mpfr_t v) {
  b16u16_u t;
  t.f = (__bf16) mpfr_get_d (v, MPFR_RNDN);
  return (double) t.f;
}

/* `v` stepped by `n` ULP along the bf16 grid, sign-magnitude. `v` must already
   be an exact bf16 value; the result is one too.

   An entry whose last limb is exactly zero cannot be usefully moved: stepping
   from bf16 zero lands on the smallest subnormal, ~9e-41, which cannot change
   any sum. Here that affects only T1[127] and T2[0], whose ideals are exactly
   ln(1) = 0 and so are already exact and never need tuning -- but it is the
   same completeness gap exp-limb-gen.c documents around its T2[8], and it
   fails closed the same way: the final total_bad() gate below rejects any
   table the restriction left unsolved. */
static double bf16_step (double v, int n) {
  b16u16_u t;
  t.f = (__bf16) v;
  int32_t k = (t.u & 0x8000) ? -(int32_t) (t.u & 0x7fff) : (int32_t) (t.u & 0x7fff);
  k += n;
  b16u16_u r;
  r.u = (k < 0) ? (uint16_t) (0x8000 | (uint32_t) (-k)) : (uint16_t) k;
  return (double) r.f;
}

/* Split `ideal` into `n` bf16 limbs by iterated round-and-subtract.
   Residual arithmetic is exact: each subtraction is of a bf16 from a
   WORK_PREC value, and WORK_PREC leaves ample room. */
static void split_limbs (mpfr_t ideal, double *limb, int n, mpfr_t residual) {
  mpfr_set (residual, ideal, MPFR_RNDN);
  for (int k = 0; k < n; k++) {
    limb[k] = round_bf16 (residual);
    mpfr_sub_d (residual, residual, limb[k], MPFR_RNDN);
  }
}

static double t1[T1_N][4], t2[T2_N][4], t3[T3_N][4];
static uint16_t ref[T1_N][T2_N];   /* correctly-rounded bf16 ln, by (i1,i2) */

/* Reconstruct and round exactly as inria-logbf16-limb.c does. Any change to
   the accumulation order there must be mirrored here. */
static uint16_t eval (int i1, int i2) {
  float s = 0.0f;
  for (int k = T1_LIMBS - 1; k >= 0; k--) s += (float) (__bf16) t1[i1][k];
  for (int k = T2_LIMBS - 1; k >= 0; k--) s += (float) (__bf16) t2[i2][k];
  b16u16_u o;
  o.f = (__bf16) s;
  return o.u;
}

static int row_bad (int i1) {
  int bad = 0;
  for (int i2 = 0; i2 < T2_N; i2++) if (eval (i1, i2) != ref[i1][i2]) bad++;
  return bad;
}

static int col_bad (int i2) {
  int bad = 0;
  for (int i1 = 1; i1 < T1_N - 1; i1++) if (eval (i1, i2) != ref[i1][i2]) bad++;
  return bad;
}

static int total_bad (void) {
  int bad = 0;
  for (int i1 = 1; i1 < T1_N - 1; i1++) bad += row_bad (i1);
  return bad;
}

/* One sweep of greedy coordinate descent: every failing T1 row, then every
   failing T2 column, each moved to the ULP offset that minimises its own
   errors. `pin` is a T1 row the caller has seeded and does not want disturbed,
   or -1. Returns nonzero if anything moved. */
static int descent_sweep (int pin) {
  int changed = 0;
  for (int i1 = 1; i1 < T1_N - 1; i1++) {
    if (i1 == pin) continue;
    int base = row_bad (i1);
    if (!base) continue;
    double keep = t1[i1][T1_LIMBS - 1];
    int best_d = 0, best = base;
    for (int d = -TUNE_WIN; d <= TUNE_WIN; d++) {
      if (!d) continue;
      t1[i1][T1_LIMBS - 1] = bf16_step (keep, d);
      int b = row_bad (i1);
      if (b < best || (b == best && abs (d) < abs (best_d)))
        { best = b; best_d = d; }
    }
    t1[i1][T1_LIMBS - 1] = best_d ? bf16_step (keep, best_d) : keep;
    if (best_d) changed = 1;
  }
  for (int i2 = 0; i2 < T2_N; i2++) {
    int base = col_bad (i2);
    if (!base) continue;
    double keep = t2[i2][T2_LIMBS - 1];
    int best_d = 0, best = base;
    for (int d = -TUNE_WIN; d <= TUNE_WIN; d++) {
      if (!d) continue;
      t2[i2][T2_LIMBS - 1] = bf16_step (keep, d);
      int b = col_bad (i2);
      if (b < best || (b == best && abs (d) < abs (best_d)))
        { best = b; best_d = d; }
    }
    t2[i2][T2_LIMBS - 1] = best_d ? bf16_step (keep, best_d) : keep;
    if (best_d) changed = 1;
  }
  return changed;
}

static void descend (int pin) {
  for (int pass = 0; pass < 12 && descent_sweep (pin); pass++) { }
}

int main (int argc, char **argv) {
  const char *out_file = (argc > 1) ? argv[1]
                                    : "implementations/log/logbf16-limb.h";

  mpfr_t ln2, ideal, residual, x, lg;
  mpfr_init2 (ln2, WORK_PREC);
  mpfr_init2 (ideal, WORK_PREC);
  mpfr_init2 (residual, WORK_PREC);
  mpfr_init2 (x, WORK_PREC);
  mpfr_init2 (lg, WORK_PREC);
  mpfr_const_log2 (ln2, MPFR_RNDN);

  /* T1[k] approximates ln(2^(k-127)) = (k-127)*ln2. k=0 and k=255 are the
     subnormal and NaN/Inf slots; cr_log_bf16 never sums them, so leave them
     zero to match the shipping table's shape. */
  for (int k = 1; k < T1_N - 1; k++) {
    mpfr_mul_si (ideal, ln2, k - 127, MPFR_RNDN);
    split_limbs (ideal, t1[k], T1_LIMBS, residual);
  }

  /* T2[k] approximates ln(1 + k/128). */
  for (int k = 0; k < T2_N; k++) {
    mpfr_set_si (ideal, 128 + k, MPFR_RNDN);
    mpfr_div_si (ideal, ideal, 128, MPFR_RNDN);
    mpfr_log (ideal, ideal, MPFR_RNDN);
    split_limbs (ideal, t2[k], T2_LIMBS, residual);
  }

  /* T3[k] serves the subnormal input x = k * 2^-133 directly -- a lookup, not
     a summand. One bf16 limb suffices; the MILP fragment frag-t3-only proves
     the whole 127-entry band feasible on the plain bf16 grid. T3[0] is
     ln(+0) = -Inf, hardcoded by the caller and left zero here. */
  for (int k = 1; k < T3_N; k++) {
    mpfr_set_si (ideal, k, MPFR_RNDN);
    mpfr_mul_2si (ideal, ideal, -133, MPFR_RNDN);
    mpfr_log (ideal, ideal, MPFR_RNDN);
    split_limbs (ideal, t3[k], 1, residual);
  }

  /* Correctly-rounded reference for every normal input, from MPFR. Rounding
     300 -> 8 bits in one step avoids the double-rounding a float32 detour
     would introduce; ln() of a normal bf16 never lands outside bf16's normal
     range, so no subnormal handling is needed here. */
  for (int i1 = 1; i1 < T1_N - 1; i1++)
    for (int i2 = 0; i2 < T2_N; i2++) {
      b16u16_u xb;
      xb.u = (uint16_t) (i1 * 128 + i2);
      mpfr_set_d (x, (double) xb.f, MPFR_RNDN);
      mpfr_log (lg, x, MPFR_RNDN);
      mpfr_t r;
      mpfr_init2 (r, BF16_PREC);
      mpfr_set (r, lg, MPFR_RNDN);
      b16u16_u o;
      o.f = (__bf16) mpfr_get_d (r, MPFR_RNDN);
      ref[i1][i2] = o.u;
      mpfr_clear (r);
    }

  /* Configuration frontier, measured rather than asserted: how many inputs the
     canonical split misrounds at each (n1, n2). The header comment above
     quotes this table, so it is printed on every run to keep the two honest.
     Uses its own scratch arrays and leaves the working tables untouched. */
  {
    static double f1[T1_N][3], f2[T2_N][3];
    for (int k = 1; k < T1_N - 1; k++) {
      mpfr_mul_si (ideal, ln2, k - 127, MPFR_RNDN);
      split_limbs (ideal, f1[k], 3, residual);
    }
    for (int k = 0; k < T2_N; k++) {
      mpfr_set_si (ideal, 128 + k, MPFR_RNDN);
      mpfr_div_si (ideal, ideal, 128, MPFR_RNDN);
      mpfr_log (ideal, ideal, MPFR_RNDN);
      split_limbs (ideal, f2[k], 3, residual);
    }
    printf ("limb-gen: canonical frontier (misrounds of 32512 normal inputs)\n"
            "    n1\\n2      1      2      3\n");
    for (int n1 = 1; n1 <= 3; n1++) {
      printf ("       %d  ", n1);
      for (int n2 = 1; n2 <= 3; n2++) {
        int bad = 0;
        for (int i1 = 1; i1 < T1_N - 1; i1++)
          for (int i2 = 0; i2 < T2_N; i2++) {
            float acc = 0.0f;
            for (int k = n1 - 1; k >= 0; k--) acc += (float) (__bf16) f1[i1][k];
            for (int k = n2 - 1; k >= 0; k--) acc += (float) (__bf16) f2[i2][k];
            b16u16_u o;
            o.f = (__bf16) acc;
            if (o.u != ref[i1][i2]) bad++;
          }
        printf ("%7d", bad);
      }
      printf ("\n");
    }
  }

  int canonical = total_bad ();

  /* Feasibility certificate, before any searching.

     The descent below is a bounded neighbourhood search: when it stalls, that
     is a fact about the search, not about the problem. ln shipped at 3x2 for
     exactly that reason -- a single level of descent reports "2x2 is
     infeasible, 1 input cannot be repaired", which looks like a structural
     result and is not one. The certificate settles the question first, by
     Bellman-Ford over the difference-constraint form of the same problem (see
     table-gen/common/limb-certificate.h).

     It runs on the CANONICAL tables, so it must come before descent mutates
     them. It never writes to t1/t2: the emitted table is still the descent's,
     because the LP returns a polytope vertex that moves 325 of 382 entries
     against the descent's 14, and a header of hand-audited constants is worth
     keeping sparse. */
  {
    limb_problem prob = {
      .n1 = T1_N, .n2 = T2_N, .row_lo = 1, .row_hi = T1_N - 1,
      .l1 = T1_LIMBS, .l2 = T2_LIMBS,
      .c1 = (const double (*)[4]) t1, .c2 = (const double (*)[4]) t2,
      .ref = &ref[0][0]
    };
    limb_cert cert;
    limb_cert_verdict v = limb_certify (&prob, LIMB_CERT_WINDOW_NONE, &cert,
                                        NULL, NULL);
    switch (v) {
    case LIMB_CERT_SAT:
      printf ("limb-gen: certificate %dx%d -> SAT "
              "(constructed and verified, 0 misrounds over %d inputs)\n"
              "limb-gen:   smallest lattice-safe window %d ULP; a correct table "
              "provably exists,\n"
              "limb-gen:   so a stall below is a search failure, not infeasibility\n",
              T1_LIMBS, T2_LIMBS, (T1_N - 2) * T2_N, cert.min_window);
      break;
    case LIMB_CERT_UNSAT:
      fprintf (stderr,
        "limb-gen: certificate %dx%d -> UNSAT\n"
        "limb-gen:   the difference-constraint system has a negative cycle of "
        "%d constraints,\n"
        "limb-gen:   so no %dx%d table is correct even with real-valued "
        "displacements.\n"
        "limb-gen:   not searching; spend a limb instead.\n",
        T1_LIMBS, T2_LIMBS, cert.cycle_len, T1_LIMBS, T2_LIMBS);
      mpfr_clear (ln2); mpfr_clear (ideal); mpfr_clear (residual);
      mpfr_clear (x); mpfr_clear (lg);
      return 1;
    case LIMB_CERT_UNKNOWN:
    default:
      printf ("limb-gen: certificate %dx%d -> UNKNOWN "
              "(real system feasible; no lattice-safe solution found)\n"
              "limb-gen:   %d interval(s) closed under the quantisation shrink"
              "%s\n"
              "limb-gen:   searching anyway -- neither outcome below is a proof\n",
              T1_LIMBS, T2_LIMBS, cert.closed_intervals,
              cert.safe_cycle_len ? ", witness cycle recorded" : "");
      break;
    }
  }

  /* Pass 1 -- plain greedy descent. This clears 13 of the 14 canonical
     misrounds and stalls on x = 0.9921875, whose float32 reconstruction lands
     exactly on a bf16 midpoint. No single row or column move improves that
     row's own score, so descent will not take the step that fixes it. */
  descend (-1);

  /* Pass 2 -- seeded retry. For each still-failing row, force its last limb to
     each offset in turn (an uphill move descent would reject), then let the
     columns repair the damage. Keeps the first seed that reaches zero. */
  if (total_bad () > 0) {
    static double save1[T1_N][4], save2[T2_N][4];
    int seed_row = -1, seed_d = 0;
    for (int i1 = 1; i1 < T1_N - 1 && seed_row < 0; i1++) {
      if (!row_bad (i1)) continue;
      memcpy (save1, t1, sizeof t1);
      memcpy (save2, t2, sizeof t2);
      /* Smallest perturbation first: |d| ascending, negative before positive.
         Any seed that reaches zero is correct, but the smallest one keeps the
         emitted table closest to the canonical split. */
      for (int m = 1; m <= TUNE_WIN && seed_row < 0; m++)
        for (int sgn = -1; sgn <= 1 && seed_row < 0; sgn += 2) {
          int d = sgn * m;
          memcpy (t1, save1, sizeof t1);
          memcpy (t2, save2, sizeof t2);
          t1[i1][T1_LIMBS - 1] = bf16_step (t1[i1][T1_LIMBS - 1], d);
          descend (i1);
          if (total_bad () == 0) { seed_row = i1; seed_d = d; }
        }
      if (seed_row < 0) { memcpy (t1, save1, sizeof t1); memcpy (t2, save2, sizeof t2); }
    }
    if (seed_row >= 0)
      printf ("limb-gen: seeded retry fixed T1[%d] with %+d ULP\n", seed_row, seed_d);
  }

  int remaining = total_bad ();
  if (remaining) {
    fprintf (stderr,
      "limb-gen: %d input(s) still misround at %dx%d after tuning -- refusing "
      "to write %s\n", remaining, T1_LIMBS, T2_LIMBS, out_file);
    return 1;
  }

  /* Report which entries ended up off their canonical split, so the tuned
     table is auditable rather than a pile of unexplained constants. */
  int tuned = 0;
  for (int k = 1; k < T1_N - 1; k++) {
    mpfr_mul_si (ideal, ln2, k - 127, MPFR_RNDN);
    double canon[4];
    split_limbs (ideal, canon, T1_LIMBS, residual);
    for (int j = 0; j < T1_LIMBS; j++)
      if (canon[j] != t1[k][j]) { printf ("  T1[%d] limb%d %a -> %a\n", k, j, canon[j], t1[k][j]); tuned++; }
  }
  for (int k = 0; k < T2_N; k++) {
    mpfr_set_si (ideal, 128 + k, MPFR_RNDN);
    mpfr_div_si (ideal, ideal, 128, MPFR_RNDN);
    mpfr_log (ideal, ideal, MPFR_RNDN);
    double canon[4];
    split_limbs (ideal, canon, T2_LIMBS, residual);
    for (int j = 0; j < T2_LIMBS; j++)
      if (canon[j] != t2[k][j]) { printf ("  T2[%d] limb%d %a -> %a\n", k, j, canon[j], t2[k][j]); tuned++; }
  }

  FILE *out = fopen (out_file, "w");
  if (!out) {
    fprintf (stderr, "limb-gen: cannot write %s\n", out_file);
    return 1;
  }

  fprintf (out,
    "/* Generated by table-gen/log/limb-gen.c -- do not edit by hand.\n"
    "\n"
    "   bf16 limb tables for cr_log_bf16_limb: each ln() table entry is stored\n"
    "   as a sum of %d bf16 limbs for T1 and %d for T2, so the table needs no\n"
    "   float32 storage. Accumulating the limbs in float32 and rounding once\n"
    "   reproduces correctly-rounded bf16 ln() on every input.\n"
    "\n"
    "   %d entries below are NOT the canonical round-and-subtract split: their\n"
    "   last limb is moved a few ULP so that the 14 inputs the canonical %dx%d\n"
    "   split misrounds come out correct. The generator re-verifies all 32512\n"
    "   normal inputs against MPFR before writing this file; see\n"
    "   docs/LIMB-TUNING.md for how those entries are found.\n"
    "*/\n\n"
    "#pragma once\n\n"
    "/* Every literal below is an exact bf16 value, so the initialiser loses\n"
    "   nothing. C++ still warns on the double->__bf16 conversion rank, which\n"
    "   is noise here -- silence it for the tables only. */\n"
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic push\n"
    "#pragma GCC diagnostic ignored \"-Wconversion\"\n"
    "#endif\n\n"
    "#define LOGBF16_T1_LIMBS %d\n"
    "#define LOGBF16_T2_LIMBS %d\n\n",
    T1_LIMBS, T2_LIMBS, tuned, T1_LIMBS, T2_LIMBS, T1_LIMBS, T2_LIMBS);

  /* emit_table is inlined here so the tuned arrays are written verbatim. */
  struct { const char *name, *comment; double (*data)[4]; int rows, nlimb; } tabs[] = {
    { "T1L", "/* T1L[i1] sums to ln(x1), x1 the bf16 with encoding i1*2^7. */",
      t1, T1_N, T1_LIMBS },
    { "T2L", "/* T2L[i2] sums to ln(x2), 1 <= x2 < 2, i2 the low 7 encoding bits. */",
      t2, T2_N, T2_LIMBS },
    { "T3L", "/* T3L[i2] is ln(i2 * 2^-133) outright -- the subnormal band is a direct\n"
             "   lookup, so one bf16 limb is enough. T3L[0] is unused (ln(+0) = -Inf). */",
      t3, T3_N, 1 },
  };
  for (size_t t = 0; t < sizeof tabs / sizeof *tabs; t++) {
    fprintf (out, "%s\n", tabs[t].comment);
    fprintf (out, "static const __bf16 %s[%d][%d] = {\n",
             tabs[t].name, tabs[t].rows, tabs[t].nlimb);
    for (int i = 0; i < tabs[t].rows; i++) {
      fprintf (out, "  {");
      for (int k = 0; k < tabs[t].nlimb; k++)
        fprintf (out, "%s%af", k ? ", " : "", tabs[t].data[i][k]);
      fprintf (out, "},%s", (i % 2) ? "\n" : "");
    }
    if (tabs[t].rows % 2) fprintf (out, "\n");
    fprintf (out, "};\n\n");
  }

  fprintf (out,
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic pop\n"
    "#endif\n");
  fclose (out);

  mpfr_clear (ln2); mpfr_clear (ideal); mpfr_clear (residual);
  mpfr_clear (x); mpfr_clear (lg);
  printf ("limb-gen: %d T1 x %d limbs, %d T2 x %d limbs, %d T3 x 1 limb -> %s\n"
          "limb-gen: canonical split misrounded %d input(s); %d entr%s tuned; "
          "0 misrounds after tuning\n",
          T1_N, T1_LIMBS, T2_N, T2_LIMBS, T3_N, out_file,
          canonical, tuned, tuned == 1 ? "y" : "ies");
  return 0;
}
