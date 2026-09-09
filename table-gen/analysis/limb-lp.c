/* ln 2x2 as a difference-constraint system, solved by Bellman-Ford.

   Because only the last limb moves, the reconstruction is
       yhat_ij = base_ij + a_i + b_j,
   and correct rounding is the two-sided constraint L_ij <= a_i + b_j <= U_ij.
   Substituting b'_j = -b_j makes every constraint a difference constraint
   a_i - b'_j in [lo,hi], so the system is feasible iff its constraint graph has
   no negative cycle -- a DECISION procedure, where the descent in limb-gen.c is
   only a search. A negative cycle certifies that no table exists; a feasible
   run constructs one.

   Two shrinks make the certificate sound against the shipped kernel:
     - the per-input float32 accumulation error, so the real solution survives
       being evaluated in float32 rather than exactly;
     - (ulp(limb1 of T1[i]) + ulp(limb1 of T2[j]))/2, so rounding the real
       solution onto the ULP lattice cannot leave the interval.
   With both applied the result is a certified-correct integer table.

   Run from the repo root.
*/

#include "table-gen/common/limb-certificate.h"

#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
static void build_model (int n1, int n2) {
  mpfr_t ln2, id, r, x, lg;
  mpfr_init2 (ln2, WP); mpfr_init2 (id, WP); mpfr_init2 (r, WP);
  mpfr_init2 (x, WP);   mpfr_init2 (lg, WP);
  mpfr_const_log2 (ln2, MPFR_RNDN);
  for (int k = 1; k < N1 - 1; k++) {
    mpfr_mul_si (id, ln2, k - 127, MPFR_RNDN); split_limbs (id, c1[k], n1, r);
  }
  for (int k = 0; k < N2; k++) {
    mpfr_set_si (id, 128 + k, MPFR_RNDN); mpfr_div_si (id, id, 128, MPFR_RNDN);
    mpfr_log (id, id, MPFR_RNDN); split_limbs (id, c2[k], n2, r);
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

static const char *verdict_name (limb_cert_verdict v) {
  return v == LIMB_CERT_SAT ? "SAT" : v == LIMB_CERT_UNSAT ? "UNSAT" : "UNKNOWN";
}

int main (void) {
  static int d1[N1], d2[N2];
  printf ("=== ln limb configurations, decided by difference constraints ===\n");
  printf ("  V = %d nodes (254 T1 rows + 128 T2 columns + a zero reference)\n",
          (N1 - 2) + N2 + 1);
  printf ("  E = %d constraints (two per scored input) plus window bounds\n\n",
          2 * (N1 - 2) * N2);
  printf ("  cfg   verdict   minW  moved  BF rounds  closed  verified\n");

  for (int n1 = 1; n1 <= 3; n1++)
    for (int n2 = 1; n2 <= 3; n2++) {
      build_model (n1, n2);
      limb_problem p = {
        .n1 = N1, .n2 = N2, .row_lo = 1, .row_hi = N1 - 1,
        .l1 = n1, .l2 = n2,
        .c1 = (const double (*)[4]) c1, .c2 = (const double (*)[4]) c2,
        .ref = ref
      };
      limb_cert c;
      limb_cert_verdict v = limb_certify (&p, LIMB_CERT_WINDOW_NONE, &c, d1, d2);
      printf ("  %dx%d   %-7s  %4d  %5d  %9d  %6d  %s\n",
              n1, n2, verdict_name (v), c.min_window, c.entries_moved,
              c.rounds_safe, c.closed_intervals,
              c.verified_misrounds == 0 ? "yes"
                : c.verified_misrounds > 0 ? "no" : "-");
      if (v != LIMB_CERT_SAT)
        printf ("        why: %s\n",
                c.verified_misrounds > 0
                  ? "system feasible, but the solution needs a displacement past "
                    "one binade,\n             where d*ulp stops being linear and "
                    "the reduction no longer applies"
                  : "no lattice-safe solution within the window: the movable "
                    "limb's ULP\n             is too coarse for the rounding "
                    "intervals");
    }

  printf ("\n  Reading this. UNSAT is the only genuinely negative answer and it\n"
          "  does not occur for ln: real-valued displacements are continuous, so\n"
          "  the interval system alone is satisfiable even at 1x1. Every real\n"
          "  obstruction is a LATTICE effect and surfaces as UNKNOWN -- at 1x1 a\n"
          "  single limb's ULP dwarfs the rounding interval, which is why nearly\n"
          "  every constraint closes under the quantisation shrink.\n\n"
          "  2x2 is the smallest configuration that certifies SAT, which is the\n"
          "  result the shipped generator depends on: its descent may stall, but\n"
          "  the table it is looking for is known to exist before it starts.\n\n"
          "  'moved' is the LP's own sparsity, not the shipped table's. The\n"
          "  descent in table-gen/log/limb-gen.c reaches zero by moving 14\n"
          "  entries; Bellman-Ford returns a polytope vertex and moves far more.\n"
          "  That is why the generator keeps the search and uses this only to\n"
          "  decide.\n");
  return 0;
}
