/* Generate the bf16 limb tables for cr_exp_bf16_limb.

   Companion to limb-gen.c (which does the same job for ln), but the problem
   is shaped differently, and so is this generator.

   Why exp is not ln. cr_log_bf16 ends in a SUM, so splitting each entry into
   limbs keeps the reconstruction linear: (a1+a2+a3) + (b1+b2) is five terms
   added. cr_exp_bf16 ends in a PRODUCT, T1[i1] * T2[i2], and the obvious
   transcription -- split both factors, expand the product -- gives n*m cross
   terms instead of n+m summands. That is the reading in EXP-LIMB-PLAN.md, and
   it is what made exp look like a dead end.

   It is avoidable. Nothing requires the product to be expanded: reconstruct
   each FACTOR from its limbs first, in float32, then do the single multiply
   CORE-MATH already does. Cost is n+m adds and one multiply -- no cross terms
   at any limb count -- and the table still needs nothing but bf16 storage,
   which is the whole point of the scheme.

   Why this splits the shipped tables rather than recomputing from MPFR.
   limb-gen.c recomputes ln() ideals in MPFR because CORE-MATH's ln tables are
   plain correctly-rounded values. The exp tables are not: entries are capped
   to FLT_MAX at the overflow end, and the .sage generator hand-adjusts entries
   by an ULP to kill exceptional cases (see report.md 6f). Splitting the
   shipped float32 values keeps those adjustments, which is what makes the
   3x3 table bit-identical to cr_exp_bf16 rather than merely close to it.

   Two configurations are emitted.

     3x3  exact by construction. Three bf16 limbs carry 24 significand bits,
          exactly float32's, so the split is lossless and the reconstructed
          factors ARE the shipped floats. Identical output on every input, as
          a theorem about significand widths, not a test result.

     3x2  minimal. T2 at two limbs loses ~3e-6 relative, which misrounds two
          of the 3954 table-path inputs -- both landing within 3e-5 of a bf16
          rounding midpoint. Because T1 stays exact, the residual error is
          confined to one factor, so the repair decouples: each T2 entry can
          be chosen independently of every other. A one-ULP nudge on the
          second limb of T2[91] and T2[182] fixes both. This is the same
          manual-adjustment trick the .sage generators use, applied per limb.

   The decoupling is the reason 3x2 is reachable at all. With both factors
   inexact the correctness rows are bilinear (T1_a * T2_b) and the per-entry
   search does not apply -- see EXP-LIMB-PLAN.md task 7. Holding one factor
   exact is what turns that bilinear model back into 256 independent searches.

   Usage: exp-limb-gen [OUT_FILE]   (default implementations/expbf16-limb.h)
   Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "implementations/inria-expbf16.c"

#define T1_N       512
#define T2_N       256
#define T1_LIMBS     3
#define T2_LIMBS_X   3   /* exact configuration   */
#define T1_LIMBS_M   2   /* minimal configuration */
#define T2_LIMBS_M   2
#define TUNE_WIN    12   /* bf16 ULPs searched either side of a limb */
#define DESCENT_PASSES 8
#define MAX_USE1    16   /* inputs per T1 entry: i1 = au>>3, so at most 8 */
#define MAX_USE2    64   /* inputs per T2 entry: ~15 in practice          */

typedef union { __bf16 f; uint16_t u; } b16;

static float bf16_of (float v) { b16 t; t.f = (__bf16) v; return (float) t.f; }

/* Monotone sign-magnitude key over the bf16 grid. Stepping in this key crosses
   zero and exponent boundaries cleanly, which a raw bit increment does not. */
static int32_t key_of (float v) {
  b16 t; t.f = (__bf16) v;
  return (t.u & 0x8000) ? -(int32_t) (t.u & 0x7fff) : (int32_t) (t.u & 0x7fff);
}
static float from_key (int32_t k) {
  b16 t;
  t.u = (k < 0) ? (uint16_t) (0x8000 | (uint32_t) (-k)) : (uint16_t) k;
  return (float) t.f;
}
static float bstep (float v, int n) { return from_key (key_of (v) + n); }

/* Iterated round-and-subtract, with a guard the ln generator does not need.

   T1[247..255] hold FLT_MAX (the overflow cap). bf16 has no such value -- it
   rounds to +Inf -- so the residual FLT_MAX - Inf is -Inf and would poison
   every later limb with NaN. Stopping the chain on a non-finite or zero
   residual leaves {+Inf, 0, 0}, whose sum is +Inf and whose product with any
   T2 is +Inf: exactly what the shipped table produces for those slots.

   T1[504..505] sit below bf16's 2^-133 subnormal floor and round to zero. The
   split is genuinely lossy there; the 28 inputs that reach those entries
   underflow to bf16 zero regardless, which the exhaustive check confirms. */
static void split_limbs (float v, int n, float *limb) {
  float r = v;
  for (int j = 0; j < n; j++) {
    if (!isfinite (r) || r == 0.0f) {
      limb[j] = (j == 0) ? r : 0.0f;
      r = 0.0f;
    } else {
      limb[j] = bf16_of (r);
      r -= limb[j];              /* exact: both operands are float32 */
    }
  }
}

/* Inputs each T2 index serves, with the correctly-rounded result it must
   produce. The reference is the shipped float32 product -- cr_exp_bf16 is
   verified correctly rounded on every bfloat16 input, so matching it is
   equivalent to matching MPFR, and needs no MPFR here. */
static int      n_use[T2_N];
static uint16_t use_i1[T2_N][MAX_USE2];
static uint16_t use_ref[T2_N][MAX_USE2];

/* The mirror lists, which 3x2 never needed. With T1 exact a T1 entry could not
   be the cause of a misrounding, so only T2 was ever searched. At 2x2 both
   factors are approximations and either side can carry the repair, so each T1
   entry needs the inputs that read it too. */
static int      n_use1[T1_N];
static uint16_t use1_i2[T1_N][MAX_USE1];
static uint16_t use1_ref[T1_N][MAX_USE1];

static void collect_usage (void) {
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) continue;   /* not the table path */
    uint16_t i1 = ((u >> 15) << 8) + (au >> 3) - 0x760;
    uint16_t i2 = ((u >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0;
    b16 ref; ref.f = T1[i1] * T2[i2];
    if (n_use[i2] >= MAX_USE2 || n_use1[i1] >= MAX_USE1) {
      fprintf (stderr, "exp-limb-gen: usage list overflow (i1=%u i2=%u)\n", i1, i2);
      exit (1);
    }
    use_i1[i2][n_use[i2]]  = i1;
    use_ref[i2][n_use[i2]] = ref.u;
    n_use[i2]++;
    use1_i2[i1][n_use1[i1]]  = i2;
    use1_ref[i1][n_use1[i1]] = ref.u;
    n_use1[i1]++;
  }
}

/* ââ The 2x2 search ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

   At 3x2 the repair decoupled. T1 was exact, so every residual error sat on one
   side of the product and each T2 entry could be chosen without reference to
   any other -- 256 independent single-entry searches. At 2x2 that argument is
   gone: both factors are approximations, the correctness rows are bilinear
   (T1_a * T2_b), and EXP-LIMB-PLAN.md task 7 reads that as needing a MILP.

   It does not, for a reason specific to how little is actually broken. The
   canonical 2x2 split misrounds three of the 3954 table-path inputs, and those
   three read three disjoint entries, so the repairs cannot interact. Bounded
   coordinate descent finds all three in a single pass.

   Note the asymmetry with sin's large path, where the same descent stalls.
   Failure under descent would prove nothing, since descent is not exhaustive.
   Success is different in kind: the emitted table is checked against every
   table-path input below before it is written, and cross-eval/verify-limb.c
   re-checks all 65536 against MPFR. A found solution is verified, not assumed. */

static float c1a[T1_N], c1b[T1_N], c2a[T2_N], c2b[T2_N];  /* canonical limbs */
static int   d1[T1_N], d2[T2_N];                    /* second-limb displacement */

/* A zero second limb is not a repair site: stepping from bf16 zero lands on the
   smallest subnormal, ~9e-41, which cannot move any product. T2[8] is the case
   that matters -- it holds exactly 1.0, so its residual is 0, and the repair for
   the input that reads it has to come from the T1 side.

   This is a known completeness gap, not just an optimisation: an entry with a
   zero second limb cannot be moved by this search at all, so the search is a
   neighbourhood search around the canonical split rather than a search over all
   bf16 limb pairs. It fails CLOSED -- total_mism() below uses this same
   accessor, so a repair this restriction blocks shows up as a nonzero unsolved
   count and aborts the write. A failure here would therefore not be evidence
   that no 2x2 table exists. */
static float nudge (float v, int n) { return (v == 0.0f || n == 0) ? v : bstep (v, n); }

static float t1_at (int i) {
  return isfinite (c1a[i]) ? c1a[i] + nudge (c1b[i], d1[i]) : c1a[i];
}
static float t2_at (int i) { return c2a[i] + nudge (c2b[i], d2[i]); }

static int mism (b16 got, uint16_t ref) {
  if ((got.u & 0x7fff) > 0x7f80 && (ref & 0x7fff) > 0x7f80) return 0;  /* both NaN */
  return got.u != ref;
}

/* A change to one entry can only move inputs that read it, so scoring that
   entry's own list is exactly equivalent to scoring the whole path -- which is
   what keeps the descent cheap and its per-entry minimisation exact. */
static int score1 (int i) {
  int bad = 0; float a = t1_at (i);
  for (int k = 0; k < n_use1[i]; k++) {
    b16 g; g.f = (__bf16) (a * t2_at (use1_i2[i][k]));
    bad += mism (g, use1_ref[i][k]);
  }
  return bad;
}
static int score2 (int i) {
  int bad = 0; float b = t2_at (i);
  for (int k = 0; k < n_use[i]; k++) {
    b16 g; g.f = (__bf16) (t1_at (use_i1[i][k]) * b);
    bad += mism (g, use_ref[i][k]);
  }
  return bad;
}
static int total_mism (void) {
  int s = 0;
  for (int i = 0; i < T1_N; i++) s += score1 (i);
  return s;
}

/* One coordinate: scan a +/-TUNE_WIN window on the entry's second limb, keep the
   smallest displacement that minimises its own input list. */
static int descend (int *d, int i, int (*score) (int)) {
  int base = (*score) (i);
  if (!base) return 0;
  int best = base, bd = d[i];
  for (int k = -TUNE_WIN; k <= TUNE_WIN; k++) {
    d[i] = k;
    int v = (*score) (i);
    if (v < best || (v == best && abs (k) < abs (bd))) { best = v; bd = k; }
  }
  d[i] = bd;
  return base - best;
}

/* bf16 values print exactly as %a. +Inf has no hex-float literal, so the
   overflow-cap slots are emitted as a builtin instead. */
static void emit_value (FILE *o, float v) {
  if (isinf (v)) fprintf (o, "%s__builtin_inff()", v < 0 ? "-" : "");
  else           fprintf (o, "%af", (double) v);
}

static void emit_table (FILE *o, const char *name, const char *comment,
                        float (*limb)[3], int rows, int nlimb) {
  fprintf (o, "%s\n", comment);
  fprintf (o, "static const __bf16 %s[%d][%d] = {\n", name, rows, nlimb);
  for (int i = 0; i < rows; i++) {
    fprintf (o, "  {");
    for (int k = 0; k < nlimb; k++) {
      if (k) fprintf (o, ", ");
      emit_value (o, limb[i][k]);
    }
    fprintf (o, "},%s", (i % 2) ? "\n" : "");
  }
  if (rows % 2) fprintf (o, "\n");
  fprintf (o, "};\n\n");
}

int main (int argc, char **argv) {
  const char *out_file = (argc > 1) ? argv[1]
                                    : "implementations/expbf16-limb.h";

  static float t1[T1_N][3], t2x[T2_N][3], t1m[T1_N][3], t2m[T2_N][3];

  for (int i = 0; i < T1_N; i++) split_limbs (T1[i], T1_LIMBS,   t1[i]);
  for (int i = 0; i < T2_N; i++) split_limbs (T2[i], T2_LIMBS_X, t2x[i]);

  collect_usage ();

  /* Canonical two-limb split of BOTH factors, then repair what it misrounds. */
  for (int i = 0; i < T1_N; i++) {
    float l[3]; split_limbs (T1[i], T1_LIMBS_M, l); c1a[i] = l[0]; c1b[i] = l[1];
  }
  for (int i = 0; i < T2_N; i++) {
    float l[3]; split_limbs (T2[i], T2_LIMBS_M, l); c2a[i] = l[0]; c2b[i] = l[1];
  }

  int canonical_bad = total_mism (), passes = 0;
  for (int pass = 0; pass < DESCENT_PASSES; pass++) {
    int improved = 0;
    for (int i = 0; i < T2_N; i++) improved += descend (d2, i, score2);
    for (int i = 0; i < T1_N; i++) improved += descend (d1, i, score1);
    passes = pass + 1;
    if (!improved || !total_mism ()) break;
  }

  int tuned = 0;
  int tune_log[T1_N + T2_N][3]; int n_tune_log = 0;
  for (int i = 0; i < T1_N; i++) {
    t1m[i][0] = c1a[i]; t1m[i][1] = nudge (c1b[i], d1[i]); t1m[i][2] = 0.0f;
    if (d1[i]) { tune_log[n_tune_log][0] = 1; tune_log[n_tune_log][1] = i;
                 tune_log[n_tune_log][2] = d1[i]; n_tune_log++; tuned++; }
  }
  for (int i = 0; i < T2_N; i++) {
    t2m[i][0] = c2a[i]; t2m[i][1] = nudge (c2b[i], d2[i]); t2m[i][2] = 0.0f;
    if (d2[i]) { tune_log[n_tune_log][0] = 2; tune_log[n_tune_log][1] = i;
                 tune_log[n_tune_log][2] = d2[i]; n_tune_log++; tuned++; }
  }

  /* The table is not written on a promise. Descent minimises per entry; this
     re-scores every table-path input against the values actually about to be
     emitted, so a table that misrounds anything never reaches the header. */
  int unsolved = total_mism ();

  FILE *o = fopen (out_file, "w");
  if (!o) { fprintf (stderr, "exp-limb-gen: cannot write %s\n", out_file); return 1; }

  fprintf (o,
    "/* Generated by log-research/exp-limb-gen.c -- do not edit by hand.\n"
    "\n"
    "   bf16 limb tables for cr_exp_bf16_limb. Each entry of CORE-MATH's\n"
    "   float32 T1/T2 is stored as a sum of bf16 limbs, so the table needs no\n"
    "   float32 storage. The limbs of a factor are summed in float32 to\n"
    "   rebuild that factor, and the two factors are then multiplied once --\n"
    "   the product is never expanded into cross terms.\n"
    "\n"
    "   T1L/T2L (%dx%d) is exact: three bf16 limbs carry 24 significand bits,\n"
    "   so each reconstructed factor is bit-for-bit the shipped float32 and\n"
    "   the result is identical to cr_exp_bf16 on every input.\n"
    "\n"
    "   T1M/T2M (%dx%d) is minimal, and costs no storage at all: %d B against\n"
    "   the %d B of float32 tables it replaces. Two limbs on BOTH factors means\n"
    "   neither reconstruction is exact, so the significand-width argument does\n"
    "   not apply here -- the canonical split misrounds three inputs, and %d\n"
    "   entries carry a small second-limb adjustment to repair them. Those three\n"
    "   inputs read three disjoint entries, which is why the repairs compose.\n"
    "   Correctness rests on the exhaustive MPFR check, not on construction.\n"
    "*/\n\n"
    "#pragma once\n\n"
    "/* Every literal below is an exact bf16 value, so the initialiser loses\n"
    "   nothing. C++ still warns on the double->__bf16 conversion rank, which\n"
    "   is noise here -- silence it for the tables only. */\n"
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic push\n"
    "#pragma GCC diagnostic ignored \"-Wconversion\"\n"
    "#endif\n\n"
    "#define EXPBF16_T1_LIMBS       %d\n"
    "#define EXPBF16_T2_LIMBS_EXACT %d\n"
    "#define EXPBF16_T1_LIMBS_MIN   %d\n"
    "#define EXPBF16_T2_LIMBS_MIN   %d\n\n",
    T1_LIMBS, T2_LIMBS_X,
    T1_LIMBS_M, T2_LIMBS_M,
    (T1_N * T1_LIMBS_M + T2_N * T2_LIMBS_M) * 2, (T1_N + T2_N) * 4, tuned,
    T1_LIMBS, T2_LIMBS_X, T1_LIMBS_M, T2_LIMBS_M);

  emit_table (o, "T1L",
    "/* T1L[i1] sums to CORE-MATH's T1[i1]. The +Inf entries are the overflow\n"
    "   cap: FLT_MAX is not a bf16 value, and +Inf is what it rounds to. */",
    t1, T1_N, T1_LIMBS);
  emit_table (o, "T2L",
    "/* T2L[i2] sums to CORE-MATH's T2[i2], exactly. */",
    t2x, T2_N, T2_LIMBS_X);
  emit_table (o, "T1M",
    "/* T1M[i1] is the two-limb T1: canonical round-and-subtract, except for the\n"
    "   tuned entries listed in the generator's report. The +Inf entries are the\n"
    "   overflow cap, as in T1L. */",
    t1m, T1_N, T1_LIMBS_M);
  emit_table (o, "T2M",
    "/* T2M[i2] is the two-limb T2: canonical round-and-subtract, except\n"
    "   for the tuned entries listed in the generator's report. */",
    t2m, T2_N, T2_LIMBS_M);

  fprintf (o,
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic pop\n"
    "#endif\n");
  fclose (o);

  long f32_B = (long) (T1_N + T2_N) * 4;
  long ex_B  = (long) (T1_N * T1_LIMBS   + T2_N * T2_LIMBS_X) * 2;
  long mn_B  = (long) (T1_N * T1_LIMBS_M + T2_N * T2_LIMBS_M) * 2;

  printf ("exp-limb-gen: %dx%d (exact) and %dx%d (minimal) -> %s\n",
          T1_LIMBS, T2_LIMBS_X, T1_LIMBS_M, T2_LIMBS_M, out_file);
  printf ("  storage: float32 %ld B | %dx%d %ld B (%+.0f%%) | %dx%d %ld B (%+.0f%%)\n",
          f32_B, T1_LIMBS, T2_LIMBS_X, ex_B, 100.0 * (ex_B - f32_B) / f32_B,
          T1_LIMBS_M, T2_LIMBS_M, mn_B, 100.0 * (mn_B - f32_B) / f32_B);
  printf ("  %dx%d descent: %d misrounded inputs at the canonical split, "
          "%d pass%s, %d tuned entries\n",
          T1_LIMBS_M, T2_LIMBS_M, canonical_bad, passes,
          passes == 1 ? "" : "es", tuned);
  for (int i = 0; i < n_tune_log; i++)
    printf ("    T%d[%d] limb1 %+d ULP\n",
            tune_log[i][0], tune_log[i][1], tune_log[i][2]);
  if (unsolved) {
    printf ("  UNSOLVED: %d table-path inputs still misround -- the %dx%d table "
            "is NOT correctly rounded\n", unsolved, T1_LIMBS_M, T2_LIMBS_M);
    return 1;
  }
  int path_inputs = 0;
  for (int i = 0; i < T1_N; i++) path_inputs += n_use1[i];
  printf ("  %dx%d verified: 0 misroundings over all %d table-path inputs\n",
          T1_LIMBS_M, T2_LIMBS_M, path_inputs);
  return 0;
}
