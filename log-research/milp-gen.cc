/* Emit a bf16-GRID MILP for the correctly-rounded ln() tables T1, T2.

   The continuous LP (lp-constraints.txt) lets T1[k]/T2[k] be any real. A feasible
   real solution may fail once rounded to bf16, so this file emits the exact
   integer model: every table entry is forced onto a bf16-representable value, and
   a feasible solution is directly usable as the final table (no post-rounding).

   Model. For each table entry V in {T1[k] (k=1..254), T2[k] (k=0..127)}:
     - Candidates C(V) = the bf16 values within +/-CAND_ULP ULPs of V's ideal
       ln value (T1[k] ideal = ln(2^(k-127)) = (k-127)*ln2; T2[k] ideal =
       ln(1+k/128)). The correctly-rounded ideal is candidate 0.
     - Binaries zV_j pick exactly one candidate:  sum_j zV_j = 1.
     - V = sum_j c_j * zV_j   (V continuous, pinned to the chosen bf16 value).
   Each normal input u then contributes the same interval it did in the LP:
     lb_u <= T1[i1] + T2[i2] <= ub_u   (closed/open per ties-to-even, EPS on open).

   T1[0], T1[255] are unused (T1[0]=0 for x=2^-127 exponent field 0 is subnormal;
   index 255 is the Inf/NaN slot). T1[127]=0 exactly (ln 2^0) is pinned as a
   single-candidate entry, which also fixes the additive gauge.

   Output: milp-constraints.lp (CPLEX LP format with a Binary section) — read by
   HiGHS, GLPK (glpsol --lp), Gurobi, or CPLEX. Solve for feasibility:
     highs milp-constraints.lp        # "Optimal" => a correctly-rounded table exists
     glpsol --lp milp-constraints.lp   # "OPTIMAL" => a correctly-rounded table exists
   then read chosen candidates from the zV_j set to 1 in the solution.

   Build (compile as C; matches the rest of log-research):
     gcc -O2 -xc -std=c11 log-research/milp-gen.cc \
         -I/usr/include/x86_64-linux-gnu -lmpfr -lgmp -lm -o log-research/milp-gen
     ./log-research/milp-gen */

#include <stdint.h>
#include <stdio.h>
#include <mpfr.h>

typedef union { __bf16 f; uint16_t u; } b16u16;

#define BF16_PREC 8
#define BF16_EMIN (-133)
#define BF16_EMAX 128
#define WORK_PREC 200
#define CAND_ULP 3          /* candidates within +/- this many bf16 ULPs of ideal */
#define MAX_CAND (2 * CAND_ULP + 1)

#define OUT_FILE "log-research/milp-constraints.lp"

/* Round value to bf16 (RNDN, bf16 range) into y at BF16_PREC. */
static void bf16_round (mpfr_t value, mpfr_t y) {
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (BF16_EMIN); mpfr_set_emax (BF16_EMAX);
  mpfr_set_prec (y, BF16_PREC);
  int inex = mpfr_set (y, value, MPFR_RNDN);
  mpfr_subnormalize (y, inex, MPFR_RNDN);
  mpfr_set_emin (emin); mpfr_set_emax (emax);
}

/* Fill cand[0..*n-1] with the bf16 values within +/-CAND_ULP ULP of ideal,
   cand[0] = round_bf16(ideal). Values are returned as doubles (bf16 fits double
   exactly). Duplicates at the exponent-boundary ULP change are dropped. */
static void candidates (mpfr_t ideal, mpfr_t tmp, double *cand, int *n) {
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (BF16_EMIN); mpfr_set_emax (BF16_EMAX);

  bf16_round (ideal, tmp);          /* tmp = nearest bf16 = candidate 0 */
  double center = mpfr_get_d (tmp, MPFR_RNDN);

  /* Walk down CAND_ULP steps, then up CAND_ULP steps, collecting uniques. */
  double list[MAX_CAND];
  int c = 0;
  list[c++] = center;

  mpfr_t lo; mpfr_init2 (lo, BF16_PREC); mpfr_set (lo, tmp, MPFR_RNDN);
  for (int i = 0; i < CAND_ULP; i++) {
    mpfr_nextbelow (lo);
    double v = mpfr_get_d (lo, MPFR_RNDN);
    list[c++] = v;
  }
  mpfr_t hi; mpfr_init2 (hi, BF16_PREC); mpfr_set (hi, tmp, MPFR_RNDN);
  for (int i = 0; i < CAND_ULP; i++) {
    mpfr_nextabove (hi);
    double v = mpfr_get_d (hi, MPFR_RNDN);
    list[c++] = v;
  }
  mpfr_clear (lo); mpfr_clear (hi);
  mpfr_set_emin (emin); mpfr_set_emax (emax);

  /* dedupe (keep first occurrence, preserving candidate-0-first order) */
  int m = 0;
  for (int i = 0; i < c; i++) {
    int dup = 0;
    for (int j = 0; j < m; j++) if (cand[j] == list[i]) { dup = 1; break; }
    if (!dup) cand[m++] = list[i];
  }
  *n = m;
}

int main (void) {
  mpfr_set_emin (BF16_EMIN); mpfr_set_emax (BF16_EMAX);
  FILE *o = fopen (OUT_FILE, "w");
  if (!o) { perror ("fopen " OUT_FILE); return 1; }

  mpfr_t ideal, tmp, ln2, x, lnx, y, n_prev, n_next, lb, ub;
  mpfr_init2 (ideal, WORK_PREC); mpfr_init2 (tmp, BF16_PREC);
  mpfr_init2 (ln2, WORK_PREC);   mpfr_init2 (x, WORK_PREC);
  mpfr_init2 (lnx, WORK_PREC);   mpfr_init2 (y, BF16_PREC);
  mpfr_init2 (n_prev, BF16_PREC); mpfr_init2 (n_next, BF16_PREC);
  mpfr_init2 (lb, WORK_PREC);    mpfr_init2 (ub, WORK_PREC);
  mpfr_const_log2 (ln2, MPFR_RNDN);

  /* --- Precompute candidate sets per entry. --- */
  static double t1c[256][MAX_CAND]; static int t1n[256];
  static double t2c[128][MAX_CAND]; static int t2n[128];

  for (int k = 1; k <= 254; k++) {
    mpfr_mul_si (ideal, ln2, k - 127, MPFR_RNDN); /* ln(2^(k-127)) */
    candidates (ideal, tmp, t1c[k], &t1n[k]);
  }
  /* T1[127] = ln(2^0) = 0 exactly: pin to a single candidate. This keeps the
     additive gauge fixed and stops the near-zero subnormal neighbors from
     letting T1[127] drift. */
  t1c[127][0] = 0.0; t1n[127] = 1;
  for (int k = 0; k <= 127; k++) {
    mpfr_set_si (ideal, 128 + k, MPFR_RNDN);
    mpfr_div_si (ideal, ideal, 128, MPFR_RNDN);  /* 1 + k/128 */
    mpfr_log (ideal, ideal, MPFR_RNDN);          /* ln(1+k/128) */
    candidates (ideal, tmp, t2c[k], &t2n[k]);
  }
  /* T2[0] = ln(1) = 0 exactly: pin it. Its +/-ULP neighbors are subnormal bf16
     values (~1e-41) that both are meaningless candidates and wreck LP scaling
     (coefficient ratio ~1e42 -> false infeasibility from numeric instability). */
  t2c[0][0] = 0.0; t2n[0] = 1;

  /* --- LP-format preamble. --- */
  fprintf (o, "\\ bf16-grid MILP for correctly-rounded ln() tables T1, T2\n");
  fprintf (o, "\\ Generated by log-research/milp-gen.cc (MPFR %s), CAND_ULP=%d\n",
           mpfr_get_version (), CAND_ULP);
  fprintf (o, "\\ Each T1[k]/T2[k] is pinned to one bf16 candidate via binaries zT*_j.\n");
  fprintf (o, "\\ Feasible ('Optimal') => a correctly-rounded single-rounded table exists.\n");
  /* Pure feasibility: zero objective. GLPK's LP parser rejects a bare "obj: 0"
     (needs a variable term), so reference one var with coefficient 0. */
  fprintf (o, "Minimize\n obj: 0 T1_1\n");
  fprintf (o, "Subject To\n");

  /* Selection + linking rows per entry: sum z = 1 ; V - sum c_j z_j = 0. */
  for (int k = 1; k <= 254; k++) {
    fprintf (o, " selT1_%d:", k);
    for (int j = 0; j < t1n[k]; j++) fprintf (o, " + zT1_%d_%d", k, j);
    fprintf (o, " = 1\n");
    fprintf (o, " linkT1_%d: T1_%d", k, k);
    for (int j = 0; j < t1n[k]; j++) {
      /* V - c*z: fold c's sign so no "- -" appears (some LP parsers reject it). */
      double c = t1c[k][j];
      if (c >= 0) fprintf (o, " - %.17g zT1_%d_%d", c, k, j);
      else        fprintf (o, " + %.17g zT1_%d_%d", -c, k, j);
    }
    fprintf (o, " = 0\n");
  }
  for (int k = 0; k <= 127; k++) {
    fprintf (o, " selT2_%d:", k);
    for (int j = 0; j < t2n[k]; j++) fprintf (o, " + zT2_%d_%d", k, j);
    fprintf (o, " = 1\n");
    fprintf (o, " linkT2_%d: T2_%d", k, k);
    for (int j = 0; j < t2n[k]; j++) {
      double c = t2c[k][j];
      if (c >= 0) fprintf (o, " - %.17g zT2_%d_%d", c, k, j);
      else        fprintf (o, " + %.17g zT2_%d_%d", -c, k, j);
    }
    fprintf (o, " = 0\n");
  }

  /* --- Interval rows: one pair per normal input (same as the LP). --- */
  const double EPS = 1e-9;
  long rows = 0;
  for (uint32_t uu = 0; uu <= 0xFFFF; uu++) {
    uint16_t u = (uint16_t) uu;
    b16u16 v; v.u = u;
    uint16_t exp = (u >> 7) & 0xFF, mant = u & 0x7f;
    if (u >> 15 || exp == 0xFF || exp == 0) continue; /* normal inputs only */
    uint16_t i1 = u >> 7, i2 = mant;

    mpfr_set_d (x, (double) (float) v.f, MPFR_RNDN);
    mpfr_log (lnx, x, MPFR_RNDN);
    bf16_round (lnx, y);

    mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
    mpfr_set_emin (BF16_EMIN); mpfr_set_emax (BF16_EMAX);
    mpfr_set (n_prev, y, MPFR_RNDN); mpfr_nextbelow (n_prev);
    mpfr_set (n_next, y, MPFR_RNDN); mpfr_nextabove (n_next);
    mpfr_set_emin (emin); mpfr_set_emax (emax);

    mpfr_add (lb, n_prev, y, MPFR_RNDN); mpfr_div_2ui (lb, lb, 1, MPFR_RNDN);
    mpfr_add (ub, y, n_next, MPFR_RNDN); mpfr_div_2ui (ub, ub, 1, MPFR_RNDN);

    b16u16 yv; yv.f = (__bf16) mpfr_get_d (y, MPFR_RNDN);
    int y_even = ((yv.u & 1u) == 0);
    double lbd = mpfr_get_d (lb, MPFR_RNDN), ubd = mpfr_get_d (ub, MPFR_RNDN);
    double lo = y_even ? lbd : lbd + EPS;
    double hi = y_even ? ubd : ubd - EPS;

    fprintf (o, " c%u_lo: T1_%u + T2_%u >= %.17g\n", u, i1, i2, lo);
    fprintf (o, " c%u_hi: T1_%u + T2_%u <= %.17g\n", u, i1, i2, hi);
    rows += 2;
  }

  /* Bounds: T1/T2 continuous & free (candidates set their actual range). */
  fprintf (o, "Bounds\n");
  for (int k = 1; k <= 254; k++) fprintf (o, " T1_%d free\n", k);
  for (int k = 0; k <= 127; k++) fprintf (o, " T2_%d free\n", k);

  /* Binary section: every candidate selector. */
  fprintf (o, "Binary\n");
  long nbin = 0;
  for (int k = 1; k <= 254; k++)
    for (int j = 0; j < t1n[k]; j++) { fprintf (o, " zT1_%d_%d\n", k, j); nbin++; }
  for (int k = 0; k <= 127; k++)
    for (int j = 0; j < t2n[k]; j++) { fprintf (o, " zT2_%d_%d\n", k, j); nbin++; }
  fprintf (o, "End\n");
  fclose (o);

  fprintf (stderr,
    "milp-gen: %ld interval rows, %ld binaries, CAND_ULP=%d -> %s\n",
    rows, nbin, CAND_ULP, OUT_FILE);

  mpfr_clear (ideal); mpfr_clear (tmp); mpfr_clear (ln2);
  mpfr_clear (x); mpfr_clear (lnx); mpfr_clear (y);
  mpfr_clear (n_prev); mpfr_clear (n_next); mpfr_clear (lb); mpfr_clear (ub);
  return 0;
}
