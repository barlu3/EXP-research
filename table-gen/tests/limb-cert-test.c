/* Unit tests for the limb-tuning feasibility certificate.

   The certificate is what lets limb-gen distinguish "no such table exists" from
   "the search did not find one" -- a distinction the descent alone cannot make,
   and the reason ln shipped at 3x2 longer than it needed to. These tests pin
   both halves of that: the difference-constraint solver underneath, and the ln
   constraint system built on top of it.

   No test framework, matching benchmarks/harness-test.cpp and cross-eval: a
   counter and a nonzero exit.

   Run from the repo root. */

#include "table-gen/common/diffcon.h"
#include "table-gen/common/limb-certificate.h"

#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static const char *g_case = "";

static void begin (const char *name) { g_case = name; }
static void check (int ok, const char *what) {
  if (!ok) { printf ("  FAIL  %-34s %s\n", g_case, what); g_failures++; }
}

/* -- diffcon: a feasible system -------------------------------------------
   x1 - x0 <= 5, x2 - x1 <= 3, x0 - x2 <= -2.  Cycle weight 5+3-2 = 6 >= 0, so
   it is feasible and the returned potentials must satisfy every edge. */
static void test_diffcon_feasible (void) {
  begin ("diffcon_feasible");
  diffcon g;
  check (diffcon_init (&g, 3, 8) == 0, "init failed");
  diffcon_add (&g, 0, 1,  5.0);
  diffcon_add (&g, 1, 2,  3.0);
  diffcon_add (&g, 2, 0, -2.0);

  double pot[3]; int cyc[8], clen = 0, rounds = 0;
  diffcon_status st = diffcon_solve (&g, pot, &rounds, cyc, (int) (sizeof cyc / sizeof *cyc), &clen);
  check (st == DIFFCON_FEASIBLE, "expected FEASIBLE");
  check (clen == 0, "feasible system must report no cycle");

  for (int k = 0; k < g.ne; k++)
    check (pot[g.ev[k]] <= pot[g.eu[k]] + g.ew[k] + 1e-12,
           "returned potentials violate a constraint");
  diffcon_free (&g);
}

/* -- diffcon: an infeasible system, and the witness ------------------------
   x1 - x0 <= -1 and x0 - x1 <= -1 together imply 0 <= -2. The solver must say
   INFEASIBLE *and* hand back a real cycle of negative total weight -- a
   boolean alone would be indistinguishable from a search giving up, which is
   the whole point of the certificate. */
static void test_diffcon_negative_cycle (void) {
  begin ("diffcon_negative_cycle");
  diffcon g;
  check (diffcon_init (&g, 2, 8) == 0, "init failed");
  diffcon_add (&g, 0, 1, -1.0);
  diffcon_add (&g, 1, 0, -1.0);

  double pot[2]; int cyc[8], clen = 0, rounds = 0;
  diffcon_status st = diffcon_solve (&g, pot, &rounds, cyc, (int) (sizeof cyc / sizeof *cyc), &clen);
  check (st == DIFFCON_INFEASIBLE, "expected INFEASIBLE");
  check (clen >= 2, "witness must name at least two edges");

  double w = 0.0; int chained = 1;
  for (int k = 0; k < clen; k++) {
    check (cyc[k] >= 0 && cyc[k] < g.ne, "witness edge index out of range");
    w += g.ew[cyc[k]];
    if (g.ev[cyc[k]] != g.eu[cyc[(k + 1) % clen]]) chained = 0;
  }
  check (chained, "witness edges do not form a closed walk");
  check (w < 0.0, "witness cycle is not negative");
  diffcon_free (&g);
}

/* -- diffcon: a long feasible chain does not false-positive ----------------
   Bellman-Ford must terminate before the vertex bound on a feasible system;
   a solver that always burns nv rounds would report every system infeasible. */
static void test_diffcon_terminates_early (void) {
  begin ("diffcon_terminates_early");
  enum { N = 64 };
  diffcon g;
  check (diffcon_init (&g, N, 2 * N) == 0, "init failed");
  for (int i = 0; i + 1 < N; i++) diffcon_add (&g, i, i + 1, 1.0);

  double pot[N]; int cyc[8], clen = 0, rounds = 0;
  diffcon_status st = diffcon_solve (&g, pot, &rounds, cyc, (int) (sizeof cyc / sizeof *cyc), &clen);
  check (st == DIFFCON_FEASIBLE, "expected FEASIBLE");
  check (rounds <= N, "solver exceeded the vertex bound");
  diffcon_free (&g);
}

/* -- the real ln 2x2 system ------------------------------------------------
   Built here from MPFR rather than shared with limb-gen.c, deliberately: the
   test is then an independent construction of the same problem, so a
   divergence between it and the generator is itself a finding. */

#define WP 300
#define N1 256
#define N2 128
typedef union { __bf16 f; uint16_t u; } b16;

static double c1[N1][4], c2[N2][4];
static uint16_t ref[N1 * N2];

static double round_bf16 (mpfr_t v) {
  b16 t; t.f = (__bf16) mpfr_get_d (v, MPFR_RNDN); return (double) t.f;
}
static void split_limbs (mpfr_t id, double *l, int n, mpfr_t r) {
  mpfr_set (r, id, MPFR_RNDN);
  for (int k = 0; k < n; k++) { l[k] = round_bf16 (r); mpfr_sub_d (r, r, l[k], MPFR_RNDN); }
}
static void build_ln_model (void) {
  mpfr_t ln2, id, r, x, lg;
  mpfr_init2 (ln2, WP); mpfr_init2 (id, WP); mpfr_init2 (r, WP);
  mpfr_init2 (x, WP);   mpfr_init2 (lg, WP);
  mpfr_const_log2 (ln2, MPFR_RNDN);
  for (int k = 1; k < N1 - 1; k++) {
    mpfr_mul_si (id, ln2, k - 127, MPFR_RNDN); split_limbs (id, c1[k], 2, r);
  }
  for (int k = 0; k < N2; k++) {
    mpfr_set_si (id, 128 + k, MPFR_RNDN); mpfr_div_si (id, id, 128, MPFR_RNDN);
    mpfr_log (id, id, MPFR_RNDN); split_limbs (id, c2[k], 2, r);
  }
  for (int i = 1; i < N1 - 1; i++)
    for (int j = 0; j < N2; j++) {
      b16 xb; xb.u = (uint16_t) (i * 128 + j);
      mpfr_set_d (x, (double) (float) xb.f, MPFR_RNDN);
      mpfr_log (lg, x, MPFR_RNDN);
      mpfr_t rr; mpfr_init2 (rr, 8); mpfr_set (rr, lg, MPFR_RNDN);
      b16 o; o.f = (__bf16) mpfr_get_d (rr, MPFR_RNDN);
      ref[i * N2 + j] = o.u; mpfr_clear (rr);
    }
  mpfr_clear (ln2); mpfr_clear (id); mpfr_clear (r); mpfr_clear (x); mpfr_clear (lg);
}
static limb_problem ln_problem (void) {
  limb_problem p;
  p.n1 = N1; p.n2 = N2; p.row_lo = 1; p.row_hi = N1 - 1;
  p.l1 = 2;  p.l2 = 2;  p.c1 = c1; p.c2 = c2; p.ref = ref;
  return p;
}

/* The shipped configuration must certify as satisfiable, not merely "descent
   happened to find something". */
static void test_ln_cert_2x2_sat (void) {
  begin ("ln_cert_2x2_sat");
  limb_problem p = ln_problem ();
  static int d1[N1], d2[N2];
  limb_cert c;
  limb_cert_verdict v = limb_certify (&p, LIMB_CERT_WINDOW_NONE, &c, d1, d2);
  check (v == LIMB_CERT_SAT, "2x2 must certify SAT");
  check (c.cycle_len == 0, "SAT must not report a cycle");
}

/* The window scan brackets the answer: a lattice-safe solution exists within
   +-2 ULP and provably does not within +-1. The negative side is the half that
   matters -- it is a proof, not a search outcome. */
static void test_ln_cert_window_bracket (void) {
  begin ("ln_cert_window_bracket");
  limb_problem p = ln_problem ();
  static int d1[N1], d2[N2];
  limb_cert c;

  limb_cert_verdict v2 = limb_certify (&p, 2, &c, d1, d2);
  check (v2 == LIMB_CERT_SAT, "W=2 must be SAT");
  check (c.min_window == 2, "minimal lattice-safe window must be 2");

  limb_cert_verdict v1 = limb_certify (&p, 1, &c, d1, d2);
  check (v1 == LIMB_CERT_UNKNOWN || v1 == LIMB_CERT_UNSAT,
         "W=1 must not certify SAT");
  check (c.safe_cycle_len > 0, "W=1 must produce a negative-cycle witness");
}

/* An entry whose last limb is zero cannot be displaced -- stepping from bf16
   zero lands on the smallest subnormal, which changes nothing. The constraint
   system must pin those rather than hand back a displacement that the emitter
   would silently ignore. ln's cases are T1[127] and T2[0], both exactly
   ln(1) = 0. */
static void test_ln_cert_zero_limb_pinned (void) {
  begin ("ln_cert_zero_limb_pinned");
  limb_problem p = ln_problem ();
  static int d1[N1], d2[N2];
  limb_cert c;
  check (c1[127][1] == 0.0, "precondition: T1[127] last limb is zero");
  check (c2[0][1]   == 0.0, "precondition: T2[0] last limb is zero");
  limb_certify (&p, LIMB_CERT_WINDOW_NONE, &c, d1, d2);
  check (d1[127] == 0, "T1[127] must stay pinned at zero displacement");
  check (d2[0]   == 0, "T2[0] must stay pinned at zero displacement");
}

/* ln(1) = 0 exactly at (127,0), whose bf16 rounding interval is centred on
   zero and has no ordinary neighbours. A naive midpoint computation underflows
   the encoding and yields an inverted interval, which fabricates a negative
   cycle and would make the certificate report a false UNSAT. */
static void test_ln_cert_degenerate_interval (void) {
  begin ("ln_cert_degenerate_interval");
  double L = 1.0, U = -1.0;
  limb_interval (0x0000, &L, &U);
  check (L < U, "zero-result interval must not be inverted");
  check (L < 0.0 && U > 0.0, "zero-result interval must bracket zero");
  check (ref[127 * N2 + 0] == 0x0000, "precondition: ln(1) reference is +0");
}

/* A false SAT is the one dangerous failure mode: the lattice shrink drops any
   interval it closes, and at one limb it closes nearly all 32512 of them. If
   SAT were inferred from the surviving system rather than earned by evaluating
   the table, 1x1 would certify SAT and the generator would trust a table that
   misrounds a quarter of its inputs. */
static void test_ln_cert_1x1_not_sat (void) {
  begin ("ln_cert_1x1_not_sat");
  limb_problem p = ln_problem ();
  p.l1 = 1; p.l2 = 1;
  limb_cert c;
  limb_cert_verdict v = limb_certify (&p, LIMB_CERT_WINDOW_NONE, &c, NULL, NULL);
  check (v != LIMB_CERT_SAT, "1x1 must never certify SAT");
  check (c.closed_intervals > 0, "precondition: the lattice shrink closes intervals here");
  check (c.min_window == -1, "1x1 must report no feasible window");
}

/* SAT is a claim about a table, so it must carry the evidence: the returned
   displacements, re-scored through the float32 path, misround nothing. */
static void test_ln_cert_sat_is_verified (void) {
  begin ("ln_cert_sat_is_verified");
  limb_problem p = ln_problem ();
  static int d1[N1], d2[N2];
  limb_cert c;
  limb_cert_verdict v = limb_certify (&p, LIMB_CERT_WINDOW_NONE, &c, d1, d2);
  check (v == LIMB_CERT_SAT, "precondition: 2x2 is SAT");
  check (c.verified_misrounds == 0, "SAT must report a verified solution");
  check (limb_cert_verify (&p, d1, d2) == 0,
         "re-verifying the returned solution must also give zero");
  check (c.entries_moved > 0, "a constructive solution must move something");
}

/* The witness buffer belongs to the caller and its size has nothing to do with
   the graph's vertex count, so diffcon_solve must be told the capacity rather
   than infer one. A cycle longer than the buffer must report NO witness --
   truncating would hand back a walk that does not close, which cannot be
   checked, and an uncheckable witness is worse than none. Caught by review:
   the extraction loop previously bounded itself only by nv, so a graph with
   nv > the caller's buffer overflowed it (ASan-confirmed), on exactly the
   UNSAT path the certificate exists to make trustworthy. */
static void test_diffcon_cycle_buffer_capacity (void) {
  begin ("diffcon_cycle_buffer_capacity");
  enum { N = 20 };
  diffcon g;
  check (diffcon_init (&g, N, 2 * N) == 0, "init failed");
  for (int i = 0; i < N; i++) diffcon_add (&g, i, (i + 1) % N, -1.0);

  double pot[N];
  int small[4], clen = -1, rounds = 0;      /* deliberately too small for N */
  diffcon_status st = diffcon_solve (&g, pot, &rounds, small,
                                     (int) (sizeof small / sizeof *small), &clen);
  check (st == DIFFCON_INFEASIBLE, "a negative cycle must still be detected");
  check (clen == 0, "a witness that does not fit must be reported as absent");

  /* Given room, the same graph yields a real witness. */
  int big[N + 1]; clen = -1;
  diffcon_solve (&g, pot, &rounds, big, N + 1, &clen);
  check (clen > 0 && clen <= N, "an adequate buffer must receive the witness");
  double w = 0.0;
  for (int k = 0; k < clen; k++) w += g.ew[big[k]];
  check (w < 0.0, "witness cycle is not negative");
  diffcon_free (&g);
}

int main (void) {
  printf ("limb-cert-test\n");
  test_diffcon_feasible ();
  test_diffcon_negative_cycle ();
  test_diffcon_terminates_early ();
  test_diffcon_cycle_buffer_capacity ();
  build_ln_model ();
  test_ln_cert_2x2_sat ();
  test_ln_cert_window_bracket ();
  test_ln_cert_zero_limb_pinned ();
  test_ln_cert_degenerate_interval ();
  test_ln_cert_1x1_not_sat ();
  test_ln_cert_sat_is_verified ();
  if (g_failures) { printf ("  %d failure(s)\n", g_failures); return 1; }
  printf ("  all checks passed\n");
  return 0;
}
