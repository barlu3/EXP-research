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
#define T2_LIMBS_X   3   /* exact configuration  */
#define T2_LIMBS_M   2   /* minimal configuration */
#define TUNE_WIN    12   /* bf16 ULPs searched either side of a limb */

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

static float sum_limbs (const float *limb, int n) {
  float s = 0.0f;
  for (int j = n - 1; j >= 0; j--) s += limb[j];   /* smallest first */
  return s;
}

/* Inputs each T2 index serves, with the correctly-rounded result it must
   produce. The reference is the shipped float32 product -- cr_exp_bf16 is
   verified correctly rounded on every bfloat16 input, so matching it is
   equivalent to matching MPFR, and needs no MPFR here. */
static int      n_use[T2_N];
static uint16_t use_i1[T2_N][64];
static uint16_t use_ref[T2_N][64];

static void collect_usage (void) {
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) continue;   /* not the table path */
    uint16_t i1 = ((u >> 15) << 8) + (au >> 3) - 0x760;
    uint16_t i2 = ((u >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0;
    b16 ref; ref.f = T1[i1] * T2[i2];
    use_i1[i2][n_use[i2]]  = i1;
    use_ref[i2][n_use[i2]] = ref.u;
    n_use[i2]++;
  }
}

/* Choose T2[i2]'s two limbs. Tries the canonical round-and-subtract split
   first and only searches outward if it misrounds one of the inputs the entry
   serves; the scan is ordered by total ULP displacement so the reported fix is
   the smallest one. Returns the displacement actually used. */
static int tune_t2 (int i2, const float *t1v, float *limb, int *d0, int *d1) {
  float a0 = bf16_of (T2[i2]);
  float b0 = bf16_of (T2[i2] - a0);
  int best = 1 << 30, ba = 0, bb = 0, found = 0;

  for (int da = -TUNE_WIN; da <= TUNE_WIN; da++)
    for (int db = -TUNE_WIN; db <= TUNE_WIN; db++) {
      int d = abs (da) + abs (db);
      if (found && d >= best) continue;
      float t2v = bstep (a0, da) + bstep (b0, db);
      int ok = 1;
      for (int k = 0; k < n_use[i2]; k++) {
        b16 g; g.f = (__bf16) (t1v[use_i1[i2][k]] * t2v);
        if (g.u != use_ref[i2][k]) { ok = 0; break; }
      }
      if (ok) { found = 1; best = d; ba = da; bb = db; }
    }

  limb[0] = bstep (a0, ba);
  limb[1] = bstep (b0, bb);
  *d0 = ba; *d1 = bb;
  return found;
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

  static float t1[T1_N][3], t2x[T2_N][3], t2m[T2_N][3];
  static float t1v[T1_N];

  for (int i = 0; i < T1_N; i++) {
    split_limbs (T1[i], T1_LIMBS, t1[i]);
    t1v[i] = sum_limbs (t1[i], T1_LIMBS);
  }
  for (int i = 0; i < T2_N; i++)
    split_limbs (T2[i], T2_LIMBS_X, t2x[i]);

  collect_usage ();

  int tuned = 0, unsolved = 0;
  int tune_log[T2_N][3]; int n_tune_log = 0;
  for (int i = 0; i < T2_N; i++) {
    if (!n_use[i]) {                       /* unreachable: canonical is fine */
      split_limbs (T2[i], T2_LIMBS_M, t2m[i]);
      continue;
    }
    int d0, d1;
    if (!tune_t2 (i, t1v, t2m[i], &d0, &d1)) {
      unsolved++;
      split_limbs (T2[i], T2_LIMBS_M, t2m[i]);
      fprintf (stderr, "exp-limb-gen: T2[%d] has no 2-limb value within "
                       "+/-%d ULP that serves all %d of its inputs\n",
               i, TUNE_WIN, n_use[i]);
    } else if (d0 || d1) {
      tuned++;
      tune_log[n_tune_log][0] = i;
      tune_log[n_tune_log][1] = d0;
      tune_log[n_tune_log][2] = d1;
      n_tune_log++;
    }
  }

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
    "   T1L/T2M (%dx%d) is minimal. Two limbs cost ~3e-6 relative on T2, which\n"
    "   misrounds two inputs; %d entries carry a one-ULP adjustment on the\n"
    "   second limb to repair that. Since T1 stays exact the entries are\n"
    "   independent, so each was chosen on its own.\n"
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
    "#define EXPBF16_T2_LIMBS_MIN   %d\n\n",
    T1_LIMBS, T2_LIMBS_X, T1_LIMBS, T2_LIMBS_M, tuned,
    T1_LIMBS, T2_LIMBS_X, T2_LIMBS_M);

  emit_table (o, "T1L",
    "/* T1L[i1] sums to CORE-MATH's T1[i1]. The +Inf entries are the overflow\n"
    "   cap: FLT_MAX is not a bf16 value, and +Inf is what it rounds to. */",
    t1, T1_N, T1_LIMBS);
  emit_table (o, "T2L",
    "/* T2L[i2] sums to CORE-MATH's T2[i2], exactly. */",
    t2x, T2_N, T2_LIMBS_X);
  emit_table (o, "T2M",
    "/* T2M[i2] is the two-limb table: canonical round-and-subtract, except\n"
    "   for the tuned entries listed in the generator's report. */",
    t2m, T2_N, T2_LIMBS_M);

  fprintf (o,
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic pop\n"
    "#endif\n");
  fclose (o);

  printf ("exp-limb-gen: T1 %dx%d limbs, T2 %dx%d (exact) and %dx%d (minimal) -> %s\n",
          T1_N, T1_LIMBS, T2_N, T2_LIMBS_X, T2_N, T2_LIMBS_M, out_file);
  printf ("  storage: float32 %d B | %dx%d %d B | %dx%d %d B\n",
          (T1_N + T2_N) * 4,
          T1_LIMBS, T2_LIMBS_X, (T1_N * T1_LIMBS + T2_N * T2_LIMBS_X) * 2,
          T1_LIMBS, T2_LIMBS_M, (T1_N * T1_LIMBS + T2_N * T2_LIMBS_M) * 2);
  printf ("  %dx%d tuned entries: %d", T1_LIMBS, T2_LIMBS_M, tuned);
  for (int i = 0; i < n_tune_log; i++)
    printf ("%s T2[%d](%+d,%+d)", i ? "," : "",
            tune_log[i][0], tune_log[i][1], tune_log[i][2]);
  printf ("\n");
  if (unsolved) {
    printf ("  UNSOLVED entries: %d -- the %dx%d table is NOT correctly rounded\n",
            unsolved, T1_LIMBS, T2_LIMBS_M);
    return 1;
  }
  return 0;
}
