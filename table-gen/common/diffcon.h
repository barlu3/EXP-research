/* Difference-constraint systems, decided by Bellman-Ford.

   A difference constraint has the form  x_v - x_u <= w. A system of them is
   feasible iff the graph with an edge u -> v of weight w for each constraint
   has no negative cycle -- and Bellman-Ford decides that in O(V*E), returning
   either a satisfying assignment or a cycle that witnesses infeasibility.

   Why this file exists. The limb tuning problem is exactly such a system (see
   table-gen/common/limb-certificate.h), so the question "does a correct 2x2
   table exist?" is decidable rather than merely searchable. That distinction is
   the point: a coordinate descent that stalls proves nothing, whereas a
   negative cycle is a proof. The witness is returned as edge indices so the
   caller -- or a test -- can re-derive the contradiction rather than trust it.

   Deliberately free of MPFR, bf16 and anything ln-specific: the ln system is
   one client, and an exp variant (which needs the constraints taken in log
   space, since exp reconstructs by a product) would be another.

   Header-only, matching benchmarks/bench-harness.hpp. Single-threaded. */

#ifndef TABLE_GEN_COMMON_DIFFCON_H
#define TABLE_GEN_COMMON_DIFFCON_H

#include <stdlib.h>

typedef enum {
  DIFFCON_FEASIBLE   = 0,
  DIFFCON_INFEASIBLE = 1,
  /* Could not decide -- allocation failed. Distinct from INFEASIBLE on
     purpose: this header's whole claim is that INFEASIBLE is a PROOF, so an
     out-of-memory condition must never be able to impersonate one. */
  DIFFCON_ERROR      = 2
} diffcon_status;

typedef struct {
  int    *eu, *ev;   /* edge tails and heads */
  double *ew;        /* edge weights: the bound in x_ev - x_eu <= ew */
  int     ne, cap;
  int     nv;
} diffcon;

/* A relaxation must beat the incumbent by more than this to be taken. Ties are
   rejected, so a system whose constraints are exactly tight converges instead
   of churning on floating-point noise. The bound is far below any interval
   width the limb problem produces (the narrowest is ~7.6e-6). */
#define DIFFCON_EPS 1e-12

static inline void diffcon_free (diffcon *g);

static inline int diffcon_init (diffcon *g, int nv, int cap) {
  g->nv = nv; g->ne = 0; g->cap = cap;
  g->eu = NULL; g->ev = NULL; g->ew = NULL;
  g->eu = (int *)    malloc ((size_t) cap * sizeof *g->eu);
  g->ev = (int *)    malloc ((size_t) cap * sizeof *g->ev);
  g->ew = (double *) malloc ((size_t) cap * sizeof *g->ew);
  /* Release whatever did succeed: a partial allocation is not the caller's to
     clean up, and the caller has no way to tell which of the three failed. */
  if (!g->eu || !g->ev || !g->ew) { diffcon_free (g); return -1; }
  return 0;
}

static inline void diffcon_free (diffcon *g) {
  free (g->eu); free (g->ev); free (g->ew);
  g->eu = NULL; g->ev = NULL; g->ew = NULL; g->ne = g->cap = 0;
}

/* Add  x_v - x_u <= w. Silently drops past capacity; callers size `cap` from
   the constraint count they are about to add, so overflow is a programming
   error rather than a runtime condition. */
static inline void diffcon_add (diffcon *g, int u, int v, double w) {
  if (g->ne >= g->cap) return;
  g->eu[g->ne] = u; g->ev[g->ne] = v; g->ew[g->ne] = w; g->ne++;
}

/* Decide the system.

   `pot`   receives a satisfying assignment when FEASIBLE (size nv).
   `rounds` receives the number of relaxation passes performed, so a caller can
            report how far below the O(V) bound the system actually settled.
   `cycle` / `cycle_len` receive the witness when INFEASIBLE: edge indices in
            forward order, so ev[cycle[k]] == eu[cycle[k+1]] and the last closes
            back onto the first. `cycle_cap` is that buffer's capacity and is
            enforced here -- a cycle can be as long as nv, which is a property
            of the caller's problem and has no relationship to the caller's
            buffer size. If the witness does not fit, `*cycle_len` is set to 0
            rather than truncated: a partial cycle is not a closed walk, so it
            cannot be checked, and an uncheckable witness is worse than none.
            Pass cycle == NULL to skip extraction.

   Every potential starts at 0, which is equivalent to a virtual source joined
   to every vertex by a zero-weight edge -- so the system is decided as a whole
   rather than only the part reachable from one node. */
static inline diffcon_status diffcon_solve (diffcon *g, double *pot,
                                            int *rounds,
                                            int *cycle, int cycle_cap,
                                            int *cycle_len) {
  int *pred = (int *) malloc ((size_t) g->nv * sizeof *pred);
  if (!pred) { if (rounds) *rounds = 0; return DIFFCON_ERROR; }
  for (int v = 0; v < g->nv; v++) { pot[v] = 0.0; pred[v] = -1; }
  if (cycle_len) *cycle_len = 0;

  int passes = 0, changed = 1, last = -1;
  while (changed && passes < g->nv) {
    changed = 0;
    for (int k = 0; k < g->ne; k++) {
      if (pot[g->eu[k]] + g->ew[k] < pot[g->ev[k]] - DIFFCON_EPS) {
        pot[g->ev[k]] = pot[g->eu[k]] + g->ew[k];
        pred[g->ev[k]] = k;
        changed = 1; last = g->ev[k];
      }
    }
    passes++;
  }
  if (rounds) *rounds = passes;

  if (!changed) { free (pred); return DIFFCON_FEASIBLE; }

  /* A relaxation on pass nv means some walk uses nv edges, so it repeats a
     vertex: there is a negative cycle. Stepping back nv times from the last
     relaxed vertex lands strictly inside it, whatever the tail leading in. */
  if (cycle && cycle_len && last >= 0) {
    int v = last;
    for (int i = 0; i < g->nv && pred[v] >= 0; i++) v = g->eu[pred[v]];
    int len = 0, u = v;
    do {
      if (pred[u] < 0) { len = 0; break; }
      if (len >= g->nv) { len = 0; break; }   /* defensive: no closure found */
      if (len >= cycle_cap) { len = 0; break; } /* does not fit: report none */
      cycle[len++] = pred[u];
      u = g->eu[pred[u]];
    } while (u != v);
    /* Collected against the predecessor chain, i.e. backwards; reverse so the
       witness reads as a walk and can be checked edge by edge. */
    for (int a = 0, b = len - 1; a < b; a++, b--) {
      int t = cycle[a]; cycle[a] = cycle[b]; cycle[b] = t;
    }
    *cycle_len = len;
  }
  free (pred);
  return DIFFCON_INFEASIBLE;
}

#endif /* TABLE_GEN_COMMON_DIFFCON_H */
