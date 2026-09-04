/* Generate the bf16 limb tables for cr_sin_bf16_limb.

   Third member of the family after limb-gen.c (ln) and exp-limb-gen.c (exp).
   sin is the most structured of the three because cr_sin_bf16 has three
   regimes, and only two of them touch a table.

     |x| <= 0x1.dp-4   sin(x) rounds to x; no table, nothing to split.
     |x| <  4096       fma(S1[i1], C2[i2], C1[i1] * S2[i2]) -- a sum of two
                       products, so it inherits exp's problem twice over.
     |x| >= 4096       an angle-addition chain over S3/C3, up to eight table
                       entries multiplied together per input.

   As with exp, the products are never expanded into cross terms: each FACTOR
   is rebuilt from its limbs in float32 first, and the shipped expression then
   runs unchanged. Cost is (limbs per entry) adds per lookup plus the original
   arithmetic, at any limb count.

   Two configurations are emitted.

     exact    3 limbs on all six tables. Three bf16 limbs carry 24 significand
              bits, exactly float32's, so every reconstructed factor is the
              shipped float32 and the output is identical to cr_sin_bf16 by
              construction.

     minimal  3 limbs on S1/C1, 2 on S2/C2, 3 on S3/C3. The mid path mirrors
              the 3x2 split that solves ln and exp: holding S1/C1 exact keeps
              the residual error on one side of each product, which decouples
              the entries and lets each i2 be chosen on its own. Three of the
              128 index pairs need a one-ULP nudge on a second limb.

   S3/C3 stay at three limbs, and that is a real limit rather than a choice.
   The large path chains up to eight lookups per input and every S3/C3 entry
   is reused across many inputs, so the errors compound and the per-entry
   decoupling that rescues the mid path does not hold. Two limbs there misround
   70 inputs; see log-research/limb-config-sweep.c for the measurement.

   Usage: sin-limb-gen [OUT_FILE]   (default implementations/sin/sinbf16-limb.h)
   Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "implementations/sin/inria-sinbf16.c"

#define S1_N 256
#define S2_N 128
#define S3_N 123
#define LIMBS_X   3   /* exact configuration            */
#define LIMBS_M   2   /* minimal configuration, S1/C1   */
#define TUNE_WIN  8   /* bf16 ULPs searched either side */

typedef union { __bf16 f; uint16_t u; } b16g;

static float bf16_of (float v) { b16g t; t.f = (__bf16) v; return (float) t.f; }

/* Monotone sign-magnitude key over the bf16 grid. sin tables are signed and
   straddle zero, so stepping must go through this rather than the raw bits. */
static int32_t key_of (float v) {
  b16g t; t.f = (__bf16) v;
  return (t.u & 0x8000) ? -(int32_t) (t.u & 0x7fff) : (int32_t) (t.u & 0x7fff);
}
static float from_key (int32_t k) {
  b16g t;
  t.u = (k < 0) ? (uint16_t) (0x8000 | (uint32_t) (-k)) : (uint16_t) k;
  return (float) t.f;
}
static float bstep (float v, int n) { return from_key (key_of (v) + n); }

/* Every sin table value is in [-1, 1] and no smaller in magnitude than
   0x1.22p-8, so bf16's 2^-133 subnormal floor is never approached and the
   guard exp-limb-gen.c needs for FLT_MAX has no analogue here. */
static void split_limbs (float v, int n, float *limb) {
  float r = v;
  for (int j = 0; j < n; j++) { limb[j] = bf16_of (r); r -= limb[j]; }
}
static float sum_limbs (const float *limb, int n) {
  float s = 0.0f;
  for (int j = n - 1; j >= 0; j--) s += limb[j];
  return s;
}

/* Mid-path inputs served by each i1, with the correctly-rounded result each
   must produce. Each i1 covers 8 consecutive encodings and both signs, so 16
   inputs at most. cr_sin_bf16 is verified correctly rounded on every bfloat16
   input, so matching it is equivalent to matching MPFR. */
#define MAX_USE 16
static int      n_use[S1_N];
static uint16_t use_i2[S1_N][MAX_USE];
static uint16_t use_ref[S1_N][MAX_USE];
static uint8_t  use_neg[S1_N][MAX_USE];

static void collect_usage (void) {
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3de8 || au >= 0x4580) continue;   /* not the mid path */
    uint16_t i1 = (au - 0x3d80) >> 3;
    uint16_t i2 = ((((au - 0x3d80) >> 7) << 3) | (au & 0x7));
    b16g in; in.u = u;
    b16g ref; ref.f = cr_sin_bf16 (in.f);
    use_i2[i1][n_use[i1]]  = i2;
    use_ref[i1][n_use[i1]] = ref.u;
    use_neg[i1][n_use[i1]] = (uint8_t) (u >> 15);
    n_use[i1]++;
  }
}

/* Choose the two-limb values for S1[i1] and C1[i1] together -- both appear in
   the same fma, so they cannot be picked independently of each other, though
   they are independent of every other index because S2/C2 are exact. Canonical
   is tried first and the scan is ordered by total displacement, so a reported
   nudge is the smallest one that works.

   At the shipped configuration the canonical split already serves every input
   and nothing is tuned. The search is kept because it is what makes that a
   checked property rather than an assumption: if the CORE-MATH tables are ever
   regenerated, this reports whether two limbs still suffice and by how much. */
static int tune_pair (int i1, const float *s2v, const float *c2v,
                      float *slimb, float *climb, int *ds_out, int *dc_out) {
  float sa = bf16_of (S1[i1]), sb = bf16_of (S1[i1] - sa);
  float ca = bf16_of (C1[i1]), cb = bf16_of (C1[i1] - ca);
  int best = 1 << 30, bs = 0, bc = 0, found = 0;

  for (int ds = -TUNE_WIN; ds <= TUNE_WIN; ds++)
    for (int dc = -TUNE_WIN; dc <= TUNE_WIN; dc++) {
      int d = abs (ds) + abs (dc);
      if (found && d >= best) continue;
      float s1v = sa + bstep (sb, ds), c1v = ca + bstep (cb, dc);
      int ok = 1;
      for (int k = 0; k < n_use[i1]; k++) {
        int i2 = use_i2[i1][k];
        float sgn = use_neg[i1][k] ? -1.0f : 1.0f;
        b16g g;
        g.f = (__bf16) (sgn * __builtin_fmaf (s1v, c2v[i2], c1v * s2v[i2]));
        if (g.u != use_ref[i1][k]) { ok = 0; break; }
      }
      if (ok) { found = 1; best = d; bs = ds; bc = dc; }
    }

  slimb[0] = sa; slimb[1] = bstep (sb, bs);
  climb[0] = ca; climb[1] = bstep (cb, bc);
  *ds_out = bs; *dc_out = bc;
  return found;
}

static void emit_table (FILE *o, const char *name, const char *comment,
                        float (*limb)[3], int rows, int nlimb) {
  fprintf (o, "%s\n", comment);
  fprintf (o, "static const __bf16 %s[%d][%d] = {\n", name, rows, nlimb);
  for (int i = 0; i < rows; i++) {
    fprintf (o, "  {");
    for (int k = 0; k < nlimb; k++)
      fprintf (o, "%s%af", k ? ", " : "", (double) limb[i][k]);
    fprintf (o, "},%s", (i % 2) ? "\n" : "");
  }
  if (rows % 2) fprintf (o, "\n");
  fprintf (o, "};\n\n");
}

int main (int argc, char **argv) {
  const char *out_file = (argc > 1) ? argv[1]
                                    : "implementations/sin/sinbf16-limb.h";

  static float s1[S1_N][3], c1[S1_N][3], s2[S2_N][3], c2[S2_N][3];
  static float s3[S3_N][3], c3[S3_N][3], s1m[S1_N][3], c1m[S1_N][3];
  static float s2v[S2_N], c2v[S2_N];

  for (int i = 0; i < S1_N; i++) {
    split_limbs (S1[i], LIMBS_X, s1[i]);
    split_limbs (C1[i], LIMBS_X, c1[i]);
  }
  for (int i = 0; i < S2_N; i++) {
    split_limbs (S2[i], LIMBS_X, s2[i]); s2v[i] = sum_limbs (s2[i], LIMBS_X);
    split_limbs (C2[i], LIMBS_X, c2[i]); c2v[i] = sum_limbs (c2[i], LIMBS_X);
  }
  for (int i = 0; i < S3_N; i++) {
    split_limbs (S3[i], LIMBS_X, s3[i]);
    split_limbs (C3[i], LIMBS_X, c3[i]);
  }

  collect_usage ();

  int tuned = 0, unsolved = 0, log_i[S1_N][3], nlog = 0;
  for (int i = 0; i < S1_N; i++) {
    if (!n_use[i]) {          /* index outside the mid path: canonical is fine */
      split_limbs (S1[i], LIMBS_M, s1m[i]);
      split_limbs (C1[i], LIMBS_M, c1m[i]);
      continue;
    }
    int ds, dc;
    if (!tune_pair (i, s2v, c2v, s1m[i], c1m[i], &ds, &dc)) {
      unsolved++;
      split_limbs (S1[i], LIMBS_M, s1m[i]);
      split_limbs (C1[i], LIMBS_M, c1m[i]);
      fprintf (stderr, "sin-limb-gen: i1=%d has no 2-limb (S1,C1) pair within "
                       "+/-%d ULP serving all %d of its inputs\n",
               i, TUNE_WIN, n_use[i]);
    } else if (ds || dc) {
      tuned++;
      log_i[nlog][0] = i; log_i[nlog][1] = ds; log_i[nlog][2] = dc; nlog++;
    }
  }

  FILE *o = fopen (out_file, "w");
  if (!o) { fprintf (stderr, "sin-limb-gen: cannot write %s\n", out_file); return 1; }

  fprintf (o,
    "/* Generated by log-research/sin-limb-gen.c -- do not edit by hand.\n"
    "\n"
    "   bf16 limb tables for cr_sin_bf16_limb. Each entry of CORE-MATH's six\n"
    "   float32 sin/cos tables is stored as a sum of bf16 limbs. A factor is\n"
    "   rebuilt by summing its limbs in float32; the shipped expressions --\n"
    "   fma(S1,C2, C1*S2) on the mid path, the angle-addition chain on the\n"
    "   large one -- then run unchanged, with no cross-term expansion.\n"
    "\n"
    "   The 3-limb tables are exact: three bf16 limbs carry 24 significand\n"
    "   bits, exactly float32's, so the reconstruction is bit-for-bit the\n"
    "   shipped value and the result matches cr_sin_bf16 by construction.\n"
    "\n"
    "   S1M/C1M are the two-limb mid-path tables, %d of the 256 index pairs\n"
    "   carrying a one-ULP adjustment on a second limb. S2/C2 stay exact,\n"
    "   which is what keeps the index pairs independent of one another. The\n"
    "   third limb sits on S2/C2 rather than S1/C1 because those tables are\n"
    "   half the size, so it is 512 bytes cheaper there.\n"
    "*/\n\n"
    "#pragma once\n\n"
    "/* Every literal below is an exact bf16 value, so the initialiser loses\n"
    "   nothing. C++ still warns on the double->__bf16 conversion rank, which\n"
    "   is noise here -- silence it for the tables only. */\n"
    "#ifdef __cplusplus\n"
    "#pragma GCC diagnostic push\n"
    "#pragma GCC diagnostic ignored \"-Wconversion\"\n"
    "#endif\n\n"
    "#define SINBF16_LIMBS_EXACT %d\n"
    "#define SINBF16_LIMBS_MIN   %d\n\n",
    tuned, LIMBS_X, LIMBS_M);

  emit_table (o, "S1L", "/* S1L[i1] sums to S1[i1] = sin(2^(i1/8) band), exactly. */", s1, S1_N, LIMBS_X);
  emit_table (o, "C1L", "/* C1L[i1] sums to C1[i1], exactly. */", c1, S1_N, LIMBS_X);
  emit_table (o, "S2L", "/* S2L[i2] sums to S2[i2], exactly. */", s2, S2_N, LIMBS_X);
  emit_table (o, "C2L", "/* C2L[i2] sums to C2[i2], exactly. */", c2, S2_N, LIMBS_X);
  emit_table (o, "S3L", "/* S3L[i] sums to S3[i] = sin(2^(5+i)), exactly. */", s3, S3_N, LIMBS_X);
  emit_table (o, "C3L", "/* C3L[i] sums to C3[i] = cos(2^(5+i)), exactly. */", c3, S3_N, LIMBS_X);
  emit_table (o, "S1M", "/* S1M[i1]: two limbs, tuned where the canonical split misrounds. */", s1m, S1_N, LIMBS_M);
  emit_table (o, "C1M", "/* C1M[i1]: two limbs, tuned alongside S1M -- both feed one fma. */", c1m, S1_N, LIMBS_M);

  fprintf (o, "#ifdef __cplusplus\n#pragma GCC diagnostic pop\n#endif\n");
  fclose (o);

  long f32 = (long) (S1_N * 2 + S2_N * 2 + S3_N * 2) * 4;
  long ex  = (long) (S1_N * 2 + S2_N * 2 + S3_N * 2) * LIMBS_X * 2;
  long mn  = (long) (S1_N * 2 * LIMBS_M + S2_N * 2 * LIMBS_X + S3_N * 2 * LIMBS_X) * 2;
  printf ("sin-limb-gen: S1/C1 %d, S2/C2 %d, S3/C3 %d entries -> %s\n",
          S1_N, S2_N, S3_N, out_file);
  printf ("  storage: float32 %ld B | exact %ld B (%+.0f%%) | minimal %ld B (%+.0f%%)\n",
          f32, ex, 100.0 * (ex - f32) / f32, mn, 100.0 * (mn - f32) / f32);
  printf ("  mid-path S1/C1 2-limb tuned index pairs: %d", tuned);
  for (int i = 0; i < nlog; i++)
    printf ("%s i1=%d(S1%+d,C1%+d)", i ? "," : "", log_i[i][0], log_i[i][1], log_i[i][2]);
  printf ("\n");
  if (unsolved) {
    printf ("  UNSOLVED index pairs: %d -- the minimal table is NOT correctly rounded\n",
            unsolved);
    return 1;
  }
  return 0;
}
