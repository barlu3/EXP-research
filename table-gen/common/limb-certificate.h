/* Is a given limb configuration achievable at all?

   The tuning search in table-gen/log/limb-gen.c is a bounded coordinate
   descent. When it stalls, that is a fact about the search, not about the
   problem -- which is why ln shipped at 3x2 long after 2x2 had become
   reachable. This header answers the question the search cannot.

   The reduction. Only the last limb of an entry moves, and ln reconstructs by
   a SUM, so the displaced result is affine and separable in the two
   displacements:

       yhat_ij = base_ij + a_i + b_j + delta_ij,
       a_i = d1[i] * ulp(last limb of T1[i]),   b_j = d2[j] * ulp(... T2[j]).

   Correct rounding is not an equation but a two-sided interval constraint --
   yhat need only land between the midpoints bracketing the correct bf16 result:

       L_ij <= a_i + b_j <= U_ij.

   Substituting b_j -> -b'_j makes every one of those a DIFFERENCE constraint,
   a_i - b'_j in [L_ij, U_ij], so the whole system is decided by Bellman-Ford
   (see diffcon.h) in milliseconds -- with a negative cycle as a proof of
   infeasibility rather than a report that a search gave up.

   Two shrinks keep the verdict sound against the shipped kernel:

     delta   the float32 accumulation is not exact (38.7% of ln inputs round at
             least one partial sum), so each interval loses the per-input
             |delta_ij|. Measured worst case is 1.53e-5 of a bf16 ULP.
     lattice a_i is an integer multiple of its ULP, not a real, so shrinking
             additionally by (ulp1(i) + ulp2(j))/2 guarantees that ROUNDING a
             real solution onto the lattice cannot leave the interval.

   That yields a three-valued verdict, and both outer values are sound:

     UNSAT    the unshrunk system has a negative cycle => no such table exists,
              even with real-valued displacements. A genuine negative result.
     SAT      the lattice-safe system is feasible AND the rounded solution was
              re-scored input by input and misrounds nothing. SAT is therefore
              constructive and unconditional: it does not rest on the shrinks
              being tight, only on an exhaustive check of an actual table.
     UNKNOWN  neither proved; fall back to search.

   Why SAT is verified rather than inferred. When the lattice shrink is wider
   than an interval, that constraint closes up and is dropped -- at 2x2 exactly
   one does (ln(1) = 0 at (127,0), whose variables are both pinned anyway), but
   at 1x1 nearly 32000 do, because a single limb's ULP dwarfs the rounding
   interval. Inferring SAT from a feasible system that quietly dropped
   constraints would be unsound, so the verdict is instead earned by evaluating
   the constructed table. Dropping constraints can then only cost a SAT that
   was really available, never manufacture one.

   In practice UNSAT is unreachable for ln: real displacements are continuous,
   so the interval system alone is satisfiable even at 1x1, and every real
   obstruction is a lattice effect showing up as UNKNOWN. The UNSAT branch is
   kept because it is sound and cheap, not because it is expected to fire.

   What this does NOT do is produce a good table. Bellman-Ford returns a
   shortest-path potential -- a vertex of the feasible polytope -- which sits
   against constraint boundaries and moves nearly every entry it may (325 of
   382, against the descent's 14). Sparsity matters for a header of constants
   that humans audit, so the descent stays. This decides; the descent
   sparsifies.

   Scope. ln only. exp reconstructs by a PRODUCT, so the additive form needs
   the constraints taken in log space (and the quantisation shrink taken there
   too); sin's large path chains up to eight lookup pairs, which destroys the
   two-variable structure entirely and admits no shortest-path formulation.
   diffcon.h is the layer either would reuse. */

#ifndef TABLE_GEN_COMMON_LIMB_CERTIFICATE_H
#define TABLE_GEN_COMMON_LIMB_CERTIFICATE_H

#include "table-gen/common/diffcon.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define LIMB_CERT_WINDOW_NONE 0    /* unbounded displacement */
/* How far the minimal-window search looks, in ULP of the movable limb.

   The cap is a modelling limit, not a budget. The reduction assumes the
   displacement is LINEAR in d -- a_i = d * ulp(canonical last limb) -- and that
   holds only while the displaced limb stays in its own binade. bf16 has 2^7
   steps per binade, so past ~128 ULP the step size changes underneath the model
   and the constraint system stops describing the table. Reporting UNKNOWN there
   is correct; widening the cap would only buy false confidence.

   The search doubles then bisects, so this costs ~2*log2(cap) solves. */
#define LIMB_CERT_SCAN_CAP 128

#define LIMB_CERT_MAX_CYCLE 512

typedef struct {
  int n1, n2;                 /* table entry counts */
  int row_lo, row_hi;         /* scored T1 rows, half-open [lo, hi) */
  int l1, l2;                 /* limbs per entry; the last one is the movable one */
  const double (*c1)[4];      /* canonical limbs, as the generator holds them */
  const double (*c2)[4];
  const uint16_t *ref;        /* correctly rounded results, ref[i * n2 + j] */
} limb_problem;

typedef enum {
  LIMB_CERT_UNSAT   = 0,
  LIMB_CERT_UNKNOWN = 1,
  LIMB_CERT_SAT     = 2
} limb_cert_verdict;

typedef struct {
  limb_cert_verdict verdict;
  int min_window;             /* smallest lattice-safe feasible |d|, or -1 */
  int rounds_real, rounds_safe;
  int cycle_len;              /* UNSAT witness, from the unshrunk system */
  int safe_cycle_len;         /* why no constructive solution at this window */
  int cycle[LIMB_CERT_MAX_CYCLE];
  int safe_cycle[LIMB_CERT_MAX_CYCLE];
  int closed_intervals;       /* intervals the lattice shrink closed entirely */
  int entries_moved;          /* nonzero displacements in the returned solution */
  int verified_misrounds;     /* direct re-score of that solution; SAT needs 0 */
} limb_cert;

/* ULP of the bf16 grid at |v|. Zero has no useful ULP here: stepping from bf16
   zero lands on the smallest subnormal (~9.2e-41), which cannot move any sum,
   so callers must pin such entries rather than scale by this. */
static inline double limb_ulp (double v) {
  if (v == 0.0) return ldexp (1.0, -133);
  int e; frexp (fabs (v), &e);
  double u = ldexp (1.0, (e - 1) - 7);
  return u < ldexp (1.0, -133) ? ldexp (1.0, -133) : u;
}

/* The reals that round to bf16 encoding `enc`, as [*L, *U].

   The zero case is special and must stay that way: ln(1) = 0 exactly, so the
   correct result at (127,0) is +0, whose neighbours are not reachable by
   decrementing the magnitude field -- doing so underflows to 0xffff and yields
   an inverted interval, which fabricates a negative cycle and reports a false
   UNSAT. Zero's interval is instead the half-subnormal band around it. */
static inline void limb_interval (uint16_t enc, double *L, double *U) {
  typedef union { __bf16 f; uint16_t u; } b16c;
  uint16_t mag = enc & 0x7fff;
  if (mag == 0) {
    double h = 0.5 * ldexp (1.0, -133);
    *L = -h; *U = h; return;
  }
  b16c t; t.u = enc; double v = (double) (float) t.f;
  int neg = (enc & 0x8000) != 0;
  b16c up, dn;
  up.u = (uint16_t) (neg ? mag - 1 : mag + 1); if (neg) up.u |= 0x8000;
  dn.u = (uint16_t) (neg ? mag + 1 : mag - 1); if (neg) dn.u |= 0x8000;
  double vu = (double) (float) up.f, vd = (double) (float) dn.f;
  *U = 0.5 * (v + vu);
  *L = 0.5 * (v + vd);
  if (*L > *U) { double s = *L; *L = *U; *U = s; }
}

/* One solve. `lattice_safe` adds the quantisation shrink; `window` bounds the
   displacement in ULP (0 = unbounded). Potentials land in `pot`. */
static inline diffcon_status limb_cert_solve (const limb_problem *p,
                                              int lattice_safe, int window,
                                              double *pot, int *rounds,
                                              int *cycle, int *cycle_len,
                                              int *closed) {
  int nrows = p->row_hi - p->row_lo;
  int nv = nrows + p->n2 + 1;
  int zn = nrows + p->n2;
  diffcon g;
  if (diffcon_init (&g, nv, 2 * nrows * p->n2 + 4 * nv + 8) != 0)
    return DIFFCON_ERROR;   /* never INFEASIBLE: that would fake a proof */
  if (closed) *closed = 0;

  for (int i = p->row_lo; i < p->row_hi; i++) {
    for (int j = 0; j < p->n2; j++) {
      double base = 0.0;
      for (int k = 0; k < p->l1; k++) base += p->c1[i][k];
      for (int k = 0; k < p->l2; k++) base += p->c2[j][k];

      /* The kernel evaluates in float32; the model is exact. Charge the
         difference to the interval so the verdict holds for the real path. */
      float s = 0.0f;
      for (int k = p->l1 - 1; k >= 0; k--) s += (float) (__bf16) p->c1[i][k];
      for (int k = p->l2 - 1; k >= 0; k--) s += (float) (__bf16) p->c2[j][k];
      double eps = fabs ((double) s - base);
      if (lattice_safe)
        eps += 0.5 * (limb_ulp (p->c1[i][p->l1 - 1]) + limb_ulp (p->c2[j][p->l2 - 1]));

      double L, U;
      limb_interval (p->ref[i * p->n2 + j], &L, &U);
      L += eps; U -= eps;
      if (U <= L) { if (closed) (*closed)++; continue; }

      int A = i - p->row_lo, B = nrows + j;
      diffcon_add (&g, B, A,  U - base);   /* a_i - b'_j <=  U - base */
      diffcon_add (&g, A, B, -(L - base)); /* b'_j - a_i <= -(L - base) */
    }
  }

  /* Window bounds, and the pins. An entry whose last limb is zero cannot be
     displaced at all, so it is pinned to the zero node regardless of window --
     otherwise the solver would hand back a displacement the emitter ignores,
     and the certificate would be describing a table nobody can build. */
  for (int i = p->row_lo; i < p->row_hi; i++) {
    int A = i - p->row_lo;
    double cap = (p->c1[i][p->l1 - 1] == 0.0) ? 0.0
               : (window > 0 ? window * limb_ulp (p->c1[i][p->l1 - 1]) : -1.0);
    if (cap >= 0.0) { diffcon_add (&g, zn, A, cap); diffcon_add (&g, A, zn, cap); }
  }
  for (int j = 0; j < p->n2; j++) {
    int B = nrows + j;
    double cap = (p->c2[j][p->l2 - 1] == 0.0) ? 0.0
               : (window > 0 ? window * limb_ulp (p->c2[j][p->l2 - 1]) : -1.0);
    if (cap >= 0.0) { diffcon_add (&g, zn, B, cap); diffcon_add (&g, B, zn, cap); }
  }

  diffcon_status st = diffcon_solve (&g, pot, rounds, cycle,
                                     LIMB_CERT_MAX_CYCLE, cycle_len);
  diffcon_free (&g);
  return st;
}

/* Round a real displacement onto the ULP lattice.

   Guarded rather than a bare (int) llround: the quotient is a displacement
   divided by a limb ULP, and nothing in the type system bounds it. A window
   edge that failed to clamp -- because the ULP was small enough for
   DIFFCON_EPS to swallow the constraint -- would hand llround a value outside
   long long, which is undefined. Anything past a binade is meaningless here
   anyway (the linear model does not survive it), so an out-of-range quotient
   becomes zero and the verification step then rejects the solution. */
static inline int limb_round_disp (double a, double u) {
  if (!(u > 0.0)) return 0;
  double q = a / u;
  if (!(q > -1.0e9 && q < 1.0e9)) return 0;
  return (int) llround (q);
}

/* Step a bf16 value by n ULP along the monotone sign-magnitude key, matching
   bf16_step in limb-gen.c: this crosses zero and binade boundaries cleanly,
   which a raw bit increment does not. */
static inline double limb_step (double v, int n) {
  typedef union { __bf16 f; uint16_t u; } b16s;
  b16s t; t.f = (__bf16) v;
  int32_t k = (t.u & 0x8000) ? -(int32_t) (t.u & 0x7fff) : (int32_t) (t.u & 0x7fff);
  k += n;
  b16s r; r.u = (k < 0) ? (uint16_t) (0x8000 | (uint32_t) (-k)) : (uint16_t) k;
  return (double) r.f;
}

/* Score a displacement vector by direct evaluation through the kernel's own
   float32 path. This is what turns a feasible LP into a claim about a table. */
static inline int limb_cert_verify (const limb_problem *p,
                                    const int *d1, const int *d2) {
  double a[4], b[4];
  int bad = 0;
  for (int i = p->row_lo; i < p->row_hi; i++) {
    for (int k = 0; k < p->l1; k++) a[k] = p->c1[i][k];
    if (d1 && d1[i]) a[p->l1 - 1] = limb_step (p->c1[i][p->l1 - 1], d1[i]);
    for (int j = 0; j < p->n2; j++) {
      for (int k = 0; k < p->l2; k++) b[k] = p->c2[j][k];
      if (d2 && d2[j]) b[p->l2 - 1] = limb_step (p->c2[j][p->l2 - 1], d2[j]);
      float s = 0.0f;
      for (int k = p->l1 - 1; k >= 0; k--) s += (float) (__bf16) a[k];
      for (int k = p->l2 - 1; k >= 0; k--) s += (float) (__bf16) b[k];
      typedef union { __bf16 f; uint16_t u; } b16v;
      b16v o; o.f = (__bf16) s;
      if (o.u != p->ref[i * p->n2 + j]) bad++;
    }
  }
  return bad;
}

/* One window attempt: solve the lattice-safe system at `window`, round the
   potentials onto the ULP lattice, and re-score the resulting table. Returns
   nonzero only if that table misrounds nothing -- the verdict is earned by
   evaluation, never inferred from feasibility, because the quantisation shrink
   may have dropped intervals it closed. */
static inline int limb_cert_try (const limb_problem *p, int window, double *pot,
                                 limb_cert *out, int *s1, int *s2) {
  int nrows = p->row_hi - p->row_lo;
  int rounds = 0, clen = 0, closed = 0;
  diffcon_status safe = limb_cert_solve (p, 1, window, pot, &rounds,
                                         out->safe_cycle, &clen, &closed);
  out->rounds_safe = rounds;
  out->closed_intervals = closed;
  out->safe_cycle_len = clen;
  if (safe != DIFFCON_FEASIBLE) return 0;

  double z = pot[nrows + p->n2];
  int moved = 0;
  for (int i = 0; i < p->n1; i++) s1[i] = 0;
  for (int j = 0; j < p->n2; j++) s2[j] = 0;
  for (int i = p->row_lo; i < p->row_hi; i++) {
    if (p->c1[i][p->l1 - 1] == 0.0) continue;      /* pinned: cannot move */
    s1[i] = limb_round_disp (pot[i - p->row_lo] - z, limb_ulp (p->c1[i][p->l1 - 1]));
    if (s1[i]) moved++;
  }
  for (int j = 0; j < p->n2; j++) {
    if (p->c2[j][p->l2 - 1] == 0.0) continue;
    s2[j] = limb_round_disp (-(pot[nrows + j] - z), limb_ulp (p->c2[j][p->l2 - 1]));
    if (s2[j]) moved++;
  }
  int bad = limb_cert_verify (p, s1, s2);
  out->verified_misrounds = bad;
  out->entries_moved = moved;
  return bad == 0;
}

/* Decide the configuration. Fills `d1`/`d2` with a constructive integer
   solution when the verdict is SAT; both may be NULL.

   `window` bounds the displacement in ULP; LIMB_CERT_WINDOW_NONE scans up to
   LIMB_CERT_SCAN_CAP. The constructive witness always comes from the SMALLEST
   feasible window, never from an unbounded solve: Bellman-Ford returns a vertex
   of the polytope, and an unbounded one sits far out (|d| in the hundreds of
   thousands), where a displacement crosses binades and the ULP the solution was
   rounded against is no longer the ULP of the displaced value. Bounding first
   keeps the lattice step meaningful, which is what makes the rounding sound. */
static inline limb_cert_verdict limb_certify (const limb_problem *p, int window,
                                              limb_cert *out, int *d1, int *d2) {
  int nrows = p->row_hi - p->row_lo;
  int nv = nrows + p->n2 + 1;
  double *pot = (double *) malloc ((size_t) nv * sizeof *pot);
  out->verdict = LIMB_CERT_UNKNOWN;
  out->min_window = -1;
  out->cycle_len = out->safe_cycle_len = 0;
  out->closed_intervals = out->entries_moved = 0;
  out->rounds_real = out->rounds_safe = 0;
  out->verified_misrounds = -1;
  if (d1) for (int i = 0; i < p->n1; i++) d1[i] = 0;
  if (d2) for (int j = 0; j < p->n2; j++) d2[j] = 0;
  if (!pot) return out->verdict;

  /* The unshrunk system. Infeasible here means no table exists at all, even
     with real-valued displacements -- the only genuinely negative answer. */
  diffcon_status real = limb_cert_solve (p, 0, window, pot, &out->rounds_real,
                                         out->cycle, &out->cycle_len, NULL);
  /* Only a genuine negative cycle may produce UNSAT; DIFFCON_ERROR means the
     system was never decided, which is UNKNOWN. */
  if (real == DIFFCON_INFEASIBLE) out->verdict = LIMB_CERT_UNSAT;
  else out->cycle_len = 0;

  /* Smallest lattice-safe window that both solves and verifies.

     Doubling then bisecting, rather than a linear scan: an infeasible solve
     costs a full O(V) Bellman-Ford (383 passes here) while a feasible one
     settles in a handful, so a linear walk to a large cap is dominated by its
     failures. */
  int *s1 = d1 ? d1 : (int *) calloc ((size_t) p->n1, sizeof (int));
  int *s2 = d2 ? d2 : (int *) calloc ((size_t) p->n2, sizeof (int));

  if (s1 && s2) {
    int scan_cap = (window > 0) ? window : LIMB_CERT_SCAN_CAP;
    int lo = 0, hi = 0;                      /* lo: known bad, hi: known good */
    for (int w = 1; w <= scan_cap; w *= 2) {
      if (limb_cert_try (p, w, pot, out, s1, s2)) { hi = w; break; }
      lo = w;
      if (w > scan_cap / 2) break;
    }
    if (hi) {
      while (lo + 1 < hi) {                  /* bisect for the exact threshold */
        int mid = lo + (hi - lo) / 2;
        if (limb_cert_try (p, mid, pot, out, s1, s2)) hi = mid; else lo = mid;
      }
      /* Re-run at the threshold so the returned solution is the one min_window
         names, not whichever probe happened to land last. */
      limb_cert_try (p, hi, pot, out, s1, s2);
      out->min_window = hi;
      out->verified_misrounds = 0;
      out->safe_cycle_len = 0;
      if (out->verdict != LIMB_CERT_UNSAT) out->verdict = LIMB_CERT_SAT;
    }
  }

  if (out->verdict != LIMB_CERT_SAT) {
    if (s1) for (int i = 0; i < p->n1; i++) s1[i] = 0;
    if (s2) for (int j = 0; j < p->n2; j++) s2[j] = 0;
    out->entries_moved = 0;
  }
  if (!d1) free (s1);
  if (!d2) free (s2);
  free (pot);
  return out->verdict;
}

#endif /* TABLE_GEN_COMMON_LIMB_CERTIFICATE_H */
