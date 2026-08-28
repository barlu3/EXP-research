/* Per-limb MILP for the 3xT1 + 2xT2 bf16 limb tables.

   Question. milp-gen.cc asks whether ONE bf16 grid value per entry can serve
   every input; the answer is no (every T1 fragment is infeasible). This asks
   the weaker, and affirmative, question: can each entry be a SUM of bf16
   limbs -- 3 for T1, 2 for T2 -- that lands in the bf16 rounding interval of
   every input it serves?

   Model. Each limb L is chosen independently from the bf16 grid values within
   +/-CAND_ULP ULPs of that limb's ideal residual, exactly the selT/linkT
   pattern milp-gen.cc uses. The entry value is then the free sum of its limbs:

     T1_k = T1_k_0 + T1_k_1 + T1_k_2
     T2_k = T2_k_0 + T2_k_1

   and the coupling rows are unchanged from milp-gen.cc -- the correctness
   bounds are bf16 rounding intervals either way, since what "correctly
   rounded" means does not depend on how the table is stored.

   Limb ideals. Limb 0's ideal is the entry's ideal; limb j's ideal is what
   limbs 0..j-1 leave over when each is rounded to bf16. This is the same
   round-and-subtract split limb-gen.c emits, so the generated tables are
   always inside the candidate windows and a feasible solution is guaranteed
   to exist -- the model's job is to prove it, and to show how much slack the
   window leaves.

   T3 keeps one limb: the subnormal band already solves at plain bf16 width.

   Usage: milp-limb-gen [OUT_FILE]
   Run from the repo root; solve with:
     glpsol --lp log-research/milp-limb-constraints.lp
*/

#include <stdio.h>
#include <stdint.h>
#include <mpfr.h>

#define WORK_PREC 300
#define BF16_PREC 8
#define BF16_EMIN (-133)
#define BF16_EMAX 128
#define CAND_ULP 3
#define MAX_CAND (2 * CAND_ULP + 1)
#define T1_LIMBS 3
#define T2_LIMBS 2

typedef union { float f; uint32_t u; } f32u32_l;
typedef union { __bf16 f; uint16_t u; } b16u16_l;

static const char *out_file = "log-research/milp-limb-constraints.lp";

/* Round `value` to bf16 (RNDN) into `y`. */
static void round_bf16 (mpfr_t value, mpfr_t y) {
  b16u16_l t;
  t.f = (__bf16) mpfr_get_d (value, MPFR_RNDN);
  mpfr_set_d (y, (double) t.f, MPFR_RNDN);
}

/* Fill cand[0..*n-1] with bf16 grid values within +/-CAND_ULP ULP of `ideal`.
   cand[0] is the nearest. Duplicates at an exponent boundary are dropped. */
static void candidates (mpfr_t ideal, double *cand, int *n) {
  mpfr_t t; mpfr_init2 (t, BF16_PREC);
  round_bf16 (ideal, t);
  double center = mpfr_get_d (t, MPFR_RNDN);
  *n = 0;
  cand[(*n)++] = center;

  /* A zero center has no meaningful neighbourhood here: stepping off it lands
     on bf16 subnormals ~4.6e-41, whose coefficients give the constraint matrix
     a ~1e41 dynamic range and make branch-and-bound fail on a model whose LP
     relaxation is fine. A zero limb is exact, so one candidate is right --
     there is nothing to search. */
  if (mpfr_zero_p (t)) { mpfr_clear (t); return; }

  mpfr_t lo; mpfr_init2 (lo, BF16_PREC); mpfr_set (lo, t, MPFR_RNDN);
  for (int i = 0; i < CAND_ULP; i++) {
    mpfr_nextbelow (lo);
    double v = mpfr_get_d (lo, MPFR_RNDN);
    int dup = 0;
    for (int j = 0; j < *n; j++) if (cand[j] == v) dup = 1;
    if (!dup) cand[(*n)++] = v;
  }
  mpfr_t hi; mpfr_init2 (hi, BF16_PREC); mpfr_set (hi, t, MPFR_RNDN);
  for (int i = 0; i < CAND_ULP; i++) {
    mpfr_nextabove (hi);
    double v = mpfr_get_d (hi, MPFR_RNDN);
    int dup = 0;
    for (int j = 0; j < *n; j++) if (cand[j] == v) dup = 1;
    if (!dup) cand[(*n)++] = v;
  }

  mpfr_clear (t); mpfr_clear (lo); mpfr_clear (hi);
}

/* Candidate sets for every limb of every entry. */
static double t1c[255][T1_LIMBS][MAX_CAND];
static int    t1n[255][T1_LIMBS];
static double t2c[128][T2_LIMBS][MAX_CAND];
static int    t2n[128][T2_LIMBS];
static double t3c[128][MAX_CAND];
static int    t3n[128];

/* Build candidate sets for one entry's limb chain by round-and-subtract. */
static void limb_candidates (mpfr_t ideal, double cand[][MAX_CAND], int *n,
                             int nlimb, mpfr_t residual, mpfr_t rounded) {
  mpfr_set (residual, ideal, MPFR_RNDN);
  for (int j = 0; j < nlimb; j++) {
    candidates (residual, cand[j], &n[j]);
    round_bf16 (residual, rounded);
    mpfr_sub (residual, residual, rounded, MPFR_RNDN);
  }
}

static void emit_entry (FILE *o, const char *tab, int k, int nlimb,
                        double cand[][MAX_CAND], const int *n) {
  /* One selection + link row per limb, then the entry as the limb sum. */
  for (int j = 0; j < nlimb; j++) {
    fprintf (o, " sel%s_%d_%d:", tab, k, j);
    for (int c = 0; c < n[j]; c++)
      fprintf (o, " + z%s_%d_%d_%d", tab, k, j, c);
    fprintf (o, " = 1\n");

    fprintf (o, " link%s_%d_%d: %s_%d_%d", tab, k, j, tab, k, j);
    for (int c = 0; c < n[j]; c++) {
      double v = cand[j][c];
      if (v >= 0) fprintf (o, " - %.17g z%s_%d_%d_%d", v, tab, k, j, c);
      else        fprintf (o, " + %.17g z%s_%d_%d_%d", -v, tab, k, j, c);
    }
    fprintf (o, " = 0\n");
  }
  fprintf (o, " sum%s_%d: %s_%d", tab, k, tab, k);
  for (int j = 0; j < nlimb; j++) fprintf (o, " - %s_%d_%d", tab, k, j);
  fprintf (o, " = 0\n");
}

int main (int argc, char **argv) {
  if (argc > 1) out_file = argv[1];

  mpfr_t ln2, ideal, residual, rounded, x, lnx, y, n_prev, n_next, lb, ub;
  mpfr_init2 (ln2, WORK_PREC);     mpfr_init2 (ideal, WORK_PREC);
  mpfr_init2 (residual, WORK_PREC); mpfr_init2 (rounded, WORK_PREC);
  mpfr_init2 (x, WORK_PREC);       mpfr_init2 (lnx, WORK_PREC);
  mpfr_init2 (y, BF16_PREC);
  mpfr_init2 (n_prev, BF16_PREC);  mpfr_init2 (n_next, BF16_PREC);
  mpfr_init2 (lb, WORK_PREC);      mpfr_init2 (ub, WORK_PREC);
  mpfr_const_log2 (ln2, MPFR_RNDN);

  for (int k = 1; k <= 254; k++) {
    mpfr_mul_si (ideal, ln2, k - 127, MPFR_RNDN);
    limb_candidates (ideal, t1c[k], t1n[k], T1_LIMBS, residual, rounded);
  }
  for (int k = 0; k <= 127; k++) {
    mpfr_set_si (ideal, 128 + k, MPFR_RNDN);
    mpfr_div_si (ideal, ideal, 128, MPFR_RNDN);
    mpfr_log (ideal, ideal, MPFR_RNDN);
    limb_candidates (ideal, t2c[k], t2n[k], T2_LIMBS, residual, rounded);
  }
  for (int k = 1; k <= 127; k++) {
    mpfr_set_si (ideal, k, MPFR_RNDN);
    mpfr_mul_2si (ideal, ideal, -133, MPFR_RNDN);
    mpfr_log (ideal, ideal, MPFR_RNDN);
    candidates (ideal, t3c[k], &t3n[k]);
  }
  t3n[0] = 0;

  FILE *o = fopen (out_file, "w");
  if (!o) { fprintf (stderr, "milp-limb-gen: cannot write %s\n", out_file); return 1; }

  fprintf (o, "\\ Per-limb MILP for correctly-rounded ln() bf16 limb tables\n");
  fprintf (o, "\\ Generated by log-research/milp-limb-gen.cc (MPFR %s)\n",
           mpfr_get_version ());
  fprintf (o, "\\ CAND_ULP=%d, T1_LIMBS=%d, T2_LIMBS=%d, T3 limbs=1\n",
           CAND_ULP, T1_LIMBS, T2_LIMBS);
  fprintf (o, "\\ Each limb is pinned to one bf16 candidate; the entry is the limb sum.\n");
  fprintf (o, "\\ Feasible ('Optimal') => a correctly-rounded bf16 limb table exists.\n");
  fprintf (o, "Minimize\n obj: 0 T1_1\n");
  fprintf (o, "Subject To\n");

  for (int k = 1; k <= 254; k++) emit_entry (o, "T1", k, T1_LIMBS, t1c[k], t1n[k]);
  for (int k = 0; k <= 127; k++) emit_entry (o, "T2", k, T2_LIMBS, t2c[k], t2n[k]);
  for (int k = 1; k <= 127; k++) {
    fprintf (o, " selT3_%d:", k);
    for (int c = 0; c < t3n[k]; c++) fprintf (o, " + zT3_%d_%d", k, c);
    fprintf (o, " = 1\n");
    fprintf (o, " linkT3_%d: T3_%d", k, k);
    for (int c = 0; c < t3n[k]; c++) {
      double v = t3c[k][c];
      if (v >= 0) fprintf (o, " - %.17g zT3_%d_%d", v, k, c);
      else        fprintf (o, " + %.17g zT3_%d_%d", -v, k, c);
    }
    fprintf (o, " = 0\n");
  }

  /* Coupling rows: identical to milp-gen.cc. The bounds are bf16 rounding
     intervals, which do not depend on how the tables are stored. */
  const double EPS = 1e-9;
  long rows = 0, t3_rows = 0;
  for (uint32_t uu = 0; uu <= 0xFFFF; uu++) {
    uint16_t u = (uint16_t) uu;
    int i1 = u >> 7, i2 = u & 0x7f;
    if (u >> 15) continue;                 /* negative: no table path */
    if (i1 == 0xff) continue;              /* NaN / Inf */
    if (i1 == 0 && i2 == 0) continue;      /* ln(+0) = -Inf, hardcoded */

    f32u32_l c; c.u = (uint32_t) u << 16;
    mpfr_set_d (x, (double) c.f, MPFR_RNDN);
    mpfr_log (lnx, x, MPFR_RNDN);
    mpfr_set (y, lnx, MPFR_RNDN);          /* correctly-rounded bf16 result */

    mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
    mpfr_set_emin (BF16_EMIN); mpfr_set_emax (BF16_EMAX);
    mpfr_set (n_prev, y, MPFR_RNDN); mpfr_nextbelow (n_prev);
    mpfr_set (n_next, y, MPFR_RNDN); mpfr_nextabove (n_next);
    mpfr_set_emin (emin); mpfr_set_emax (emax);

    mpfr_add (lb, n_prev, y, MPFR_RNDN); mpfr_div_2ui (lb, lb, 1, MPFR_RNDN);
    mpfr_add (ub, y, n_next, MPFR_RNDN); mpfr_div_2ui (ub, ub, 1, MPFR_RNDN);

    /* An even result owns both midpoints under ties-to-even, so its interval
       is closed and must not be nudged. Nudging it would also invert the row
       pair where lb == ub (ln(1) == 0 exactly), making the model infeasible
       on a row whose correct value is representable. */
    b16u16_l yv; yv.f = (__bf16) mpfr_get_d (y, MPFR_RNDN);
    int y_even = ((yv.u & 1u) == 0);
    double lbd = mpfr_get_d (lb, MPFR_RNDN), ubd = mpfr_get_d (ub, MPFR_RNDN);
    double lo_b = y_even ? lbd : lbd + EPS;
    double hi_b = y_even ? ubd : ubd - EPS;

    if (i1 == 0) {
      fprintf (o, " s%u_lo: T3_%d >= %.17g\n", u, i2, lo_b);
      fprintf (o, " s%u_hi: T3_%d <= %.17g\n", u, i2, hi_b);
      t3_rows += 2;
    } else {
      fprintf (o, " c%ld_lo: T1_%d + T2_%d >= %.17g\n", rows, i1, i2, lo_b);
      fprintf (o, " c%ld_hi: T1_%d + T2_%d <= %.17g\n", rows, i1, i2, hi_b);
      rows += 2;
    }
  }

  /* Binaries. Limb and entry variables stay free (default bounds would clamp
     the negative ln values to >= 0). */
  fprintf (o, "Bounds\n");
  for (int k = 1; k <= 254; k++) {
    fprintf (o, " T1_%d free\n", k);
    for (int j = 0; j < T1_LIMBS; j++) fprintf (o, " T1_%d_%d free\n", k, j);
  }
  for (int k = 0; k <= 127; k++) {
    fprintf (o, " T2_%d free\n", k);
    for (int j = 0; j < T2_LIMBS; j++) fprintf (o, " T2_%d_%d free\n", k, j);
  }
  for (int k = 1; k <= 127; k++) fprintf (o, " T3_%d free\n", k);

  long nbin = 0;
  fprintf (o, "Binary\n");
  for (int k = 1; k <= 254; k++)
    for (int j = 0; j < T1_LIMBS; j++)
      for (int c = 0; c < t1n[k][j]; c++) { fprintf (o, " zT1_%d_%d_%d\n", k, j, c); nbin++; }
  for (int k = 0; k <= 127; k++)
    for (int j = 0; j < T2_LIMBS; j++)
      for (int c = 0; c < t2n[k][j]; c++) { fprintf (o, " zT2_%d_%d_%d\n", k, j, c); nbin++; }
  for (int k = 1; k <= 127; k++)
    for (int c = 0; c < t3n[k]; c++) { fprintf (o, " zT3_%d_%d\n", k, c); nbin++; }
  fprintf (o, "End\n");
  fclose (o);

  printf ("milp-limb-gen: %ld T1+T2 rows, %ld T3 rows, %ld binaries, "
          "CAND_ULP=%d, T1_LIMBS=%d, T2_LIMBS=%d -> %s\n",
          rows, t3_rows, nbin, CAND_ULP, T1_LIMBS, T2_LIMBS, out_file);

  mpfr_clear (ln2); mpfr_clear (ideal); mpfr_clear (residual);
  mpfr_clear (rounded); mpfr_clear (x); mpfr_clear (lnx); mpfr_clear (y);
  mpfr_clear (n_prev); mpfr_clear (n_next); mpfr_clear (lb); mpfr_clear (ub);
  return 0;
}
