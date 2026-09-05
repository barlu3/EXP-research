/* Limb-configuration frontier for exp() and sin() bf16 tables.

   The generators emit two configurations each and assert that both are
   correctly rounded. This tool is the evidence for those choices: it measures
   every limb configuration, so the claim "3x3 is exact and 3x2 is the minimum"
   is reproducible rather than asserted.

   It answers three questions.

   1. How many inputs does each (n1, n2) configuration misround, with the
      plain round-and-subtract split? This is the frontier EXP-LIMB-PLAN.md
      estimated at 7 for exp from a reconstructed-input probe; measured
      against the shipped tables it is 2 at 3x2 and 0 at 3x3.

   2. Does per-entry tuning close the gap at the minimal configuration? For
      exp and for sin's mid path it does, because holding the first factor
      exact confines the residual error to one side of each product and the
      entries become independent -- 256 (resp. 128) separate searches instead
      of one bilinear model. This is the point EXP-LIMB-PLAN.md task 7 called
      intractable; it is intractable only if BOTH factors are inexact.

   3. Does the same trick rescue sin's large path? It does not, and this is
      the honest negative result of the exercise. That path chains up to eight
      lookup pairs and reuses each entry across many inputs, so the errors
      compound and no per-entry choice fixes them. The tool runs a bounded
      greedy coordinate descent over the entries and reports the floor it
      reaches, rather than claiming the failure is structural without trying.

   The reference throughout is the shipped float32 implementation, which is
   verified correctly rounded on every bfloat16 input -- so matching it is
   equivalent to matching MPFR, and no MPFR is needed here.

   Usage: limb-config-sweep        Run from the repo root.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* Both CORE-MATH sources declare a file-local `b16u16_u`, so pulling them into
   one translation unit needs one of them renamed. The macro is confined to the
   second include and does not leak into anything either file exposes. */
#include "implementations/inria-expbf16.c"
#define b16u16_u b16u16_u_sin
#include "implementations/sin/inria-sinbf16.c"
#undef b16u16_u

typedef union { __bf16 f; uint16_t u; } b16s;

static float bf16_of (float v) { b16s t; t.f = (__bf16) v; return (float) t.f; }
static int32_t key_of (float v) {
  b16s t; t.f = (__bf16) v;
  return (t.u & 0x8000) ? -(int32_t) (t.u & 0x7fff) : (int32_t) (t.u & 0x7fff);
}
static float from_key (int32_t k) {
  b16s t;
  t.u = (k < 0) ? (uint16_t) (0x8000 | (uint32_t) (-k)) : (uint16_t) k;
  return (float) t.f;
}
static float bstep (float v, int n) { return from_key (key_of (v) + n); }

/* Guarded round-and-subtract; see exp-limb-gen.c for why the guard exists. */
static float reconstruct (float v, int n) {
  float r = v, limb[4];
  for (int j = 0; j < n; j++) {
    if (!isfinite (r) || r == 0.0f) { limb[j] = (j == 0) ? r : 0.0f; r = 0.0f; }
    else { limb[j] = bf16_of (r); r -= limb[j]; }
  }
  float s = 0.0f;
  for (int j = n - 1; j >= 0; j--) s += limb[j];
  return s;
}

/* ── exp ─────────────────────────────────────────────────────────────────── */

static int exp_path_inputs = 0, sin_mid_inputs = 0;

static int exp_mismatches (int n1, int n2) {
  int bad = 0;
  exp_path_inputs = 0;
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) continue;
    exp_path_inputs++;
    uint16_t i1 = ((u >> 15) << 8) + (au >> 3) - 0x760;
    uint16_t i2 = ((u >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0;
    b16s got; got.f = (__bf16) (reconstruct (T1[i1], n1) * reconstruct (T2[i2], n2));
    b16s ref; ref.f = T1[i1] * T2[i2];
    if ((got.u & 0x7fff) > 0x7f80 && (ref.u & 0x7fff) > 0x7f80) continue;
    if (got.u != ref.u) bad++;
  }
  return bad;
}

#define TUNE_WIN 12

static void exp_tune_report (void) {
  static int      n_use[256];
  static uint16_t use_i1[256][64], use_ref[256][64];
  static float    t1v[512];

  for (int i = 0; i < 512; i++) t1v[i] = reconstruct (T1[i], 3);
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) continue;
    uint16_t i1 = ((u >> 15) << 8) + (au >> 3) - 0x760;
    uint16_t i2 = ((u >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0;
    b16s ref; ref.f = T1[i1] * T2[i2];
    use_i1[i2][n_use[i2]] = i1; use_ref[i2][n_use[i2]] = ref.u; n_use[i2]++;
  }

  int canon = 0, tuned = 0, unsolved = 0;
  for (int i2 = 0; i2 < 256; i2++) {
    if (!n_use[i2]) continue;
    float a0 = bf16_of (T2[i2]), b0 = bf16_of (T2[i2] - a0);
    int best = 1 << 30, ba = 0, bb = 0, found = 0;
    for (int da = -TUNE_WIN; da <= TUNE_WIN; da++)
      for (int db = -TUNE_WIN; db <= TUNE_WIN; db++) {
        int d = abs (da) + abs (db);
        if (found && d >= best) continue;
        float t2v = bstep (a0, da) + bstep (b0, db);
        int ok = 1;
        for (int k = 0; k < n_use[i2]; k++) {
          b16s g; g.f = (__bf16) (t1v[use_i1[i2][k]] * t2v);
          if (g.u != use_ref[i2][k]) { ok = 0; break; }
        }
        if (ok) { found = 1; best = d; ba = da; bb = db; }
      }
    if (!found) unsolved++;
    else if (!ba && !bb) canon++;
    else { tuned++; printf("      T2[%d] tuned: limb0 %+d ULP, limb1 %+d ULP\n", i2, ba, bb); }
  }
  printf ("    3x2 with per-entry tuning: %d canonical, %d tuned, %d unsolvable\n",
          canon, tuned, unsolved);
}

/* ───────────────────────── Can 2x2 be repaired, with BOTH factors inexact? ─────────────────────────

   exp_tune_report above works because T1 is exact, which confines the residual
   error to one side of the product and makes the T2 entries independent. That
   is gone at 2x2: the correctness rows are bilinear, and the per-entry search
   does not apply. What replaces it is bounded coordinate descent over both
   factors' second limbs -- the same tool that FAILS on sin's large path below.

   The difference is not the tool, it is the problem. Only three inputs break,
   and they read three disjoint entries, so no repair can cost another, and
   descent converges in one pass. Where entries are shared across many inputs,
   as in sin's angle-addition chain, the same descent stalls. */

#define D_WIN 12

static float q1a[512], q1b[512], q2a[256], q2b[256];
static int   e1[512], e2[256];
static int   m_use1[512], m_use2[256];
static uint16_t m1_i2[512][16], m1_ref[512][16];
static uint16_t m2_i1[256][64], m2_ref[256][64];

static float mnudge (float v, int n) { return (v == 0.0f || n == 0) ? v : bstep (v, n); }
static float q1 (int i) { return isfinite (q1a[i]) ? q1a[i] + mnudge (q1b[i], e1[i]) : q1a[i]; }
static float q2 (int i) { return q2a[i] + mnudge (q2b[i], e2[i]); }

static int q_bad (b16s g, uint16_t r) {
  if ((g.u & 0x7fff) > 0x7f80 && (r & 0x7fff) > 0x7f80) return 0;
  return g.u != r;
}
static int qs1 (int i) {
  int b = 0; float a = q1 (i);
  for (int k = 0; k < m_use1[i]; k++) {
    b16s g; g.f = (__bf16) (a * q2 (m1_i2[i][k])); b += q_bad (g, m1_ref[i][k]);
  }
  return b;
}
static int qs2 (int i) {
  int b = 0; float v = q2 (i);
  for (int k = 0; k < m_use2[i]; k++) {
    b16s g; g.f = (__bf16) (q1 (m2_i1[i][k]) * v); b += q_bad (g, m2_ref[i][k]);
  }
  return b;
}
static int q_total (void) { int t = 0; for (int i = 0; i < 512; i++) t += qs1 (i); return t; }

static int q_descend (int *d, int i, int (*sc) (int)) {
  int base = (*sc) (i); if (!base) return 0;
  int best = base, bd = d[i];
  for (int k = -D_WIN; k <= D_WIN; k++) {
    d[i] = k; int v = (*sc) (i);
    if (v < best || (v == best && abs (k) < abs (bd))) { best = v; bd = k; }
  }
  d[i] = bd; return base - best;
}

static void exp_tune_report_2x2 (void) {
  for (int i = 0; i < 512; i++) {
    q1a[i] = bf16_of (T1[i]);
    float r = T1[i] - q1a[i];
    q1b[i] = (isfinite (q1a[i]) && isfinite (r)) ? bf16_of (r) : 0.0f;
    e1[i] = 0; m_use1[i] = 0;
  }
  for (int i = 0; i < 256; i++) {
    q2a[i] = bf16_of (T2[i]); q2b[i] = bf16_of (T2[i] - q2a[i]);
    e2[i] = 0; m_use2[i] = 0;
  }
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) continue;
    uint16_t i1 = ((u >> 15) << 8) + (au >> 3) - 0x760;
    uint16_t i2 = ((u >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0;
    b16s ref; ref.f = T1[i1] * T2[i2];
    /* Same fail-fast as exp-limb-gen.c's collect_usage. The true maxima are 8
       per T1 entry and 16 per T2, so these bounds hold with headroom -- but a
       change to the index-bit layout would otherwise overrun into the adjacent
       statics silently instead of stopping. */
    if (m_use1[i1] >= 16 || m_use2[i2] >= 64) {
      fprintf (stderr, "limb-config-sweep: usage list overflow (i1=%u i2=%u)\n",
               i1, i2);
      exit (1);
    }
    m1_i2[i1][m_use1[i1]] = i2; m1_ref[i1][m_use1[i1]] = ref.u; m_use1[i1]++;
    m2_i1[i2][m_use2[i2]] = i1; m2_ref[i2][m_use2[i2]] = ref.u; m_use2[i2]++;
  }

  printf ("    canonical split                   : %d mismatches\n", q_total ());
  for (int pass = 0; pass < 8; pass++) {
    int imp = 0;
    for (int i = 0; i < 256; i++) imp += q_descend (e2, i, qs2);
    for (int i = 0; i < 512; i++) imp += q_descend (e1, i, qs1);
    printf ("    descent pass %d                    : %d mismatches (%d repaired)\n",
            pass + 1, q_total (), imp);
    if (!imp || !q_total ()) break;
  }
  int nt = 0;
  for (int i = 0; i < 512; i++) if (e1[i]) { printf ("      T1[%d] limb1 %+d ULP\n", i, e1[i]); nt++; }
  for (int i = 0; i < 256; i++) if (e2[i]) { printf ("      T2[%d] limb1 %+d ULP\n", i, e2[i]); nt++; }
  printf ("    %d tuned entries; %d mismatches remain\n", nt, q_total ());
  {
    long f32 = (512 + 256) * 4, a = (512 * 2 + 256 * 2) * 2, b = (512 * 3 + 256 * 2) * 2;
    printf ("    storage: float32 %ld B | 2x2 %ld B (%+.0f%%) | 3x2 %ld B (%+.0f%%)\n",
            f32, a, 100.0 * (a - f32) / f32, b, 100.0 * (b - f32) / f32);
  }
}

/* ── sin, mid path ───────────────────────────────────────────────────────── */

static int sin_mid_mismatches (int n1, int n2) {
  int bad = 0;
  sin_mid_inputs = 0;
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3de8 || au >= 0x4580) continue;
    sin_mid_inputs++;
    uint16_t i1 = (au - 0x3d80) >> 3;
    uint16_t i2 = ((((au - 0x3d80) >> 7) << 3) | (au & 0x7));
    float sg = (u >> 15) ? -1.0f : 1.0f;
    b16s in; in.u = u; b16s ref; ref.f = cr_sin_bf16 (in.f);
    b16s got;
    got.f = (__bf16) (sg * __builtin_fmaf (reconstruct (S1[i1], n1),
                                           reconstruct (C2[i2], n2),
                                           reconstruct (C1[i1], n1)
                                         * reconstruct (S2[i2], n2)));
    if (got.u != ref.u) bad++;
  }
  return bad;
}

/* Mid-path counterpart to exp_tune_report. S2[i2] and C2[i2] feed the same
   fma, so they are chosen jointly; S1/C1 stay exact, which is what keeps one
   index pair independent of every other. */
static void sin_mid_tune_report (void) {
  static int      n_use[128];
  static uint16_t use_i1[128][96], use_ref[128][96];
  static uint8_t  use_neg[128][96];
  static float    s1v[256], c1v[256];

  for (int i = 0; i < 256; i++) { s1v[i] = reconstruct (S1[i], 3);
                                  c1v[i] = reconstruct (C1[i], 3); }
  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t u = (uint16_t) b, au = u & 0x7fff;
    if (au <= 0x3de8 || au >= 0x4580) continue;
    uint16_t i1 = (au - 0x3d80) >> 3;
    uint16_t i2 = ((((au - 0x3d80) >> 7) << 3) | (au & 0x7));
    b16s in; in.u = u; b16s ref; ref.f = cr_sin_bf16 (in.f);
    use_i1[i2][n_use[i2]] = i1; use_ref[i2][n_use[i2]] = ref.u;
    use_neg[i2][n_use[i2]] = (uint8_t) (u >> 15); n_use[i2]++;
  }

  int canon = 0, tuned = 0, unsolved = 0;
  for (int i2 = 0; i2 < 128; i2++) {
    if (!n_use[i2]) continue;
    float sa = bf16_of (S2[i2]), sb = bf16_of (S2[i2] - sa);
    float ca = bf16_of (C2[i2]), cb = bf16_of (C2[i2] - ca);
    int best = 1 << 30, bs = 0, bc = 0, found = 0;
    for (int ds = -8; ds <= 8; ds++)
      for (int dc = -8; dc <= 8; dc++) {
        int d = abs (ds) + abs (dc);
        if (found && d >= best) continue;
        float s2v = sa + bstep (sb, ds), c2v = ca + bstep (cb, dc);
        int ok = 1;
        for (int k = 0; k < n_use[i2]; k++) {
          int i1 = use_i1[i2][k];
          float sg = use_neg[i2][k] ? -1.0f : 1.0f;
          b16s g;
          g.f = (__bf16) (sg * __builtin_fmaf (s1v[i1], c2v, c1v[i1] * s2v));
          if (g.u != use_ref[i2][k]) { ok = 0; break; }
        }
        if (ok) { found = 1; best = d; bs = ds; bc = dc; }
      }
    if (!found) unsolved++;
    else if (!bs && !bc) canon++;
    else { tuned++; printf ("      i2=%d tuned: S2 limb1 %+d ULP, C2 limb1 %+d ULP\n", i2, bs, bc); }
  }
  printf ("    3x2 with per-entry tuning: %d canonical, %d tuned, %d unsolvable\n",
          canon, tuned, unsolved);
}

/* ── sin, large path ─────────────────────────────────────────────────────── */

/* Per-entry limb-1 displacements, indexed by S3/C3 entry. */
static int ds3[123], dc3[123];

static float s3_at (int m, int n) {
  if (n >= 3) return reconstruct (S3[m], n);
  float a = bf16_of (S3[m]);
  return (n == 1) ? a : a + bstep (bf16_of (S3[m] - a), ds3[m]);
}
static float c3_at (int m, int n) {
  if (n >= 3) return reconstruct (C3[m], n);
  float a = bf16_of (C3[m]);
  return (n == 1) ? a : a + bstep (bf16_of (C3[m] - a), dc3[m]);
}

/* Mismatches over large-path inputs whose au>>7 lies in [hi_lo, hi_hi].
   Passing the full range scores the whole path; a narrow range scores only
   the inputs one S3/C3 entry can affect, which is what makes the greedy
   pass below cheap. */
static int sin_large_mismatches (int n3, int hi_lo, int hi_hi) {
  int bad = 0;
  for (int hi = hi_lo; hi <= hi_hi; hi++) {
    if (hi < 0x8b || hi > 0xfe) continue;
    for (int lo = 0; lo < 128; lo++)
      for (int sgn = 0; sgn < 2; sgn++) {
        uint16_t au = (uint16_t) ((hi << 7) | lo);
        uint16_t u  = (uint16_t) (au | (sgn << 15));
        b16s in; in.u = u; b16s ref; ref.f = cr_sin_bf16 (in.f);
        int k = (au >> 7) - 0x8b;
        float s = s3_at (k + 7, n3), c = c3_at (k + 7, n3);
        for (int j = 0; j < 7; j++)
          if ((au >> j) & 1) {
            float sj = s3_at (k + j, n3), cj = c3_at (k + j, n3);
            float t = __builtin_fmaf (s, cj, c * sj);
            c = __builtin_fmaf (c, cj, -s * sj);
            s = t;
          }
        b16s got; got.f = (__bf16) ((sgn ? -1.0f : 1.0f) * s);
        if (got.u != ref.u) bad++;
      }
  }
  return bad;
}

#define GREEDY_WIN   4
#define GREEDY_PASSES 6

/* Bounded coordinate descent over the S3/C3 limb-1 displacements. Entry m is
   read by inputs with au>>7 in [m+0x8b-7, m+0x8b], so a change to it can only
   move mismatches inside that window -- scoring the window is equivalent to
   scoring the whole path, and 123x25 window evaluations stay cheap. */
static int sin_large_greedy (void) {
  for (int m = 0; m < 123; m++) { ds3[m] = 0; dc3[m] = 0; }
  int total = sin_large_mismatches (2, 0x8b, 0xfe);
  printf ("    2 limbs, canonical split          : %d mismatches\n", total);

  for (int pass = 0; pass < GREEDY_PASSES; pass++) {
    int improved = 0;
    for (int m = 0; m < 123; m++) {
      int lo = m + 0x8b - 7, hi = m + 0x8b;
      int base = sin_large_mismatches (2, lo, hi);
      if (!base) continue;
      int bs = ds3[m], bc = dc3[m], best = base;
      for (int a = -GREEDY_WIN; a <= GREEDY_WIN; a++)
        for (int b = -GREEDY_WIN; b <= GREEDY_WIN; b++) {
          ds3[m] = a; dc3[m] = b;
          int v = sin_large_mismatches (2, lo, hi);
          if (v < best) { best = v; bs = a; bc = b; }
        }
      ds3[m] = bs; dc3[m] = bc;
      if (best < base) improved += base - best;
    }
    total = sin_large_mismatches (2, 0x8b, 0xfe);
    printf ("    greedy pass %d                     : %d mismatches (%d repaired)\n",
            pass + 1, total, improved);
    if (!improved || !total) break;
  }
  return total;
}

int main (void) {
  printf ("\n=== bf16 limb-configuration frontier ===\n");
  printf ("Reference: the shipped float32 implementations, which are correctly\n"
          "rounded on every bfloat16 input. Mismatch counts are therefore\n"
          "wrong roundings, not merely differences.\n");

  printf ("\n-- exp: T1 limbs x T2 limbs, canonical split --\n");
  int exp_grid[4][4];
  for (int n1 = 1; n1 <= 3; n1++)
    for (int n2 = 1; n2 <= 3; n2++) exp_grid[n1][n2] = exp_mismatches (n1, n2);
  printf ("    %-8s mismatches (of %d table-path inputs)\n",
          "config", exp_path_inputs);
  for (int n1 = 1; n1 <= 3; n1++)
    for (int n2 = 1; n2 <= 3; n2++)
      printf ("    %d x %d    %6d%s\n", n1, n2, exp_grid[n1][n2],
              exp_grid[n1][n2] ? "" : "   <-- exact");
  printf ("\n-- exp: can 3x2 be repaired per entry? --\n");
  printf ("    (T1 exact confines the error to one factor, so the 256 T2\n"
          "     entries are independent and each is searched on its own)\n");
  exp_tune_report ();

  printf ("\n-- exp: can 2x2 be repaired? (this is what ships as _min) --\n");
  printf ("    (both factors inexact, so the entries couple -- coordinate\n"
          "     descent over both sides rather than a per-entry search)\n");
  exp_tune_report_2x2 ();

  printf ("\n-- sin mid path: S1/C1 limbs x S2/C2 limbs, canonical split --\n");
  int sin_grid[4][4];
  for (int n1 = 1; n1 <= 3; n1++)
    for (int n2 = 1; n2 <= 3; n2++) sin_grid[n1][n2] = sin_mid_mismatches (n1, n2);
  printf ("    %-8s mismatches (of %d mid-path inputs)\n",
          "config", sin_mid_inputs);
  for (int n1 = 1; n1 <= 3; n1++)
    for (int n2 = 1; n2 <= 3; n2++)
      printf ("    %d x %d    %6d%s\n", n1, n2, sin_grid[n1][n2],
              sin_grid[n1][n2] ? "" : "   <-- exact");

  /* Two arrangements reach zero; they are not equivalent. S1/C1 have 256
     entries each against S2/C2's 128, so the third limb is half the price on
     S2/C2 -- and 2x3 needs no tuning, while its mirror needs three adjusted
     index pairs. 2x3 is what sin-limb-gen.c emits. */
  {
    long f32 = (long) (256 * 2 + 128 * 2 + 123 * 2) * 4;
    long a   = (long) (256 * 2 * 2 + 128 * 2 * 3 + 123 * 2 * 3) * 2;  /* 2x3 */
    long b   = (long) (256 * 2 * 3 + 128 * 2 * 2 + 123 * 2 * 3) * 2;  /* 3x2 */
    printf ("\n    both 2x3 and 3x2 can reach zero, but they cost differently:\n");
    printf ("      2x3 (shipped)  %ld B  %+.1f%% vs float32   0 tuned pairs\n",
            a, 100.0 * (a - f32) / f32);
    printf ("      3x2 (mirror)   %ld B  %+.1f%% vs float32   3 tuned pairs\n",
            b, 100.0 * (b - f32) / f32);
  }

  printf ("\n-- sin mid path: the 3x2 mirror, repaired per entry --\n");
  printf ("    (not shipped -- 512 B larger than 2x3 and needs these nudges;\n"
          "     kept to show the repair works from either side of the product)\n");
  sin_mid_tune_report ();

  printf ("\n-- sin large path: S3/C3 limbs --\n");
  for (int n3 = 1; n3 <= 3; n3++) {
    for (int m = 0; m < 123; m++) { ds3[m] = 0; dc3[m] = 0; }
    int m = sin_large_mismatches (n3, 0x8b, 0xfe);
    printf ("    %d limb%s  %6d%s\n", n3, n3 > 1 ? "s " : "  ", m, m ? "" : "   <-- exact");
  }
  printf ("\n-- sin large path: does greedy tuning rescue 2 limbs? --\n");
  int floor_ = sin_large_greedy ();
  if (floor_)
    printf ("    floor reached: %d mismatches, from 70. The chain reuses each entry\n"
            "    across many inputs, so per-entry choices trade one failure for\n"
            "    another -- unlike the mid path, where holding S1/C1 exact makes the\n"
            "    index pairs independent and the same search finishes at zero.\n"
            "    Coordinate descent is not exhaustive, so this is evidence that two\n"
            "    limbs do not suffice on the large path, not a proof. Ruling it out\n"
            "    would need a joint search over the whole chain.\n",
            floor_);
  else
    printf ("    greedy tuning reached 0 -- 2 limbs are viable on the large path.\n");

  printf ("\n");
  return 0;
}
