/* Exhaustive MPFR verification of the bf16-only limb implementations.

   Companion to cross-eval/log/verify-limb.c, which does the same job for ln.
   The target function is selected at compile time, matching verify_mpfr.c:

     (default)          cr_exp_bf16_limb / cr_exp_bf16_limb_min
     -DVERIFY_SIN_LIMB  cr_sin_bf16_limb / cr_sin_bf16_limb_min

   Both variants of the selected function are checked in one run against a
   correctly-rounded MPFR reference over all 65536 bfloat16 bit patterns. The
   exact variant should be bit-identical to CORE-MATH by construction; the
   minimal one is the claim that actually needs testing, since it stores one
   fewer limb on the second factor and repairs the resulting misroundings with
   per-entry ULP adjustments.

   NaN inputs are counted but their payloads are not compared, matching
   verify_mpfr.c.

   Build (from the repo root):
     gcc -O2 -std=c11 -I implementations cross-eval/verify-limb.c \
         implementations/inria-expbf16-limb.c -lmpfr -lgmp -lm \
         -o cross-eval/verify-exp-limb
*/

#include <stdio.h>
#include <stdint.h>
#include <mpfr.h>

typedef union { float f; uint32_t u; } f32u32_vl;
typedef union { __bf16 f; uint16_t u; } b16u16_vl;

#ifdef VERIFY_SIN_LIMB
extern __bf16 cr_sin_bf16_limb (__bf16 x);
extern __bf16 cr_sin_bf16_limb_min (__bf16 x);
#define FN_EXACT   cr_sin_bf16_limb
#define FN_MIN     cr_sin_bf16_limb_min
#define FN_NAME    "sin"
#define OUT_FILE   "cross-eval/sin/MPFR-result-limb-sin.txt"
#define CONFIG_TXT "3 limbs on S1/C1/S3/C3, 3 (exact) or 2 (minimal) on S2/C2"
#else
extern __bf16 cr_exp_bf16_limb (__bf16 x);
extern __bf16 cr_exp_bf16_limb_min (__bf16 x);
#define FN_EXACT   cr_exp_bf16_limb
#define FN_MIN     cr_exp_bf16_limb_min
#define FN_NAME    "exp"
#define OUT_FILE   "cross-eval/MPFR-result-limb-exp.txt"
#define CONFIG_TXT "3 limbs on T1, 3 (exact) or 2 (minimal) on T2"
#endif

static float bf16_to_float (uint16_t b) {
  f32u32_vl v; v.u = (uint32_t) b << 16; return v.f;
}

/* Round a signed MPFR value to a bfloat16 bit pattern. Lifted unchanged from
   cross-eval/log/verify-limb.c -- see the commentary in verify_mpfr.c for why
   this does not use mpfr_subnormalize and why it rounds 256 -> 8 in one step. */
static uint16_t round_mpfr_to_bf16 (mpfr_t mp) {
  if (mpfr_nan_p (mp)) return 0x7fc0;
  uint16_t sign = mpfr_signbit (mp) ? 0x8000 : 0x0000;
  if (mpfr_inf_p (mp))  return sign | 0x7f80;
  if (mpfr_zero_p (mp)) return sign;

  mpfr_t a; mpfr_init2 (a, 256); mpfr_abs (a, mp, MPFR_RNDN);
  mpfr_t res; mpfr_init2 (res, 8);
  mpfr_set (res, a, MPFR_RNDN);
  mpfr_exp_t e = mpfr_get_exp (res);

  uint16_t mag;
  if (mpfr_inf_p (res) || e > 128) {
    mag = 0x7f80;
  } else if (e >= -125) {
    float f = mpfr_get_flt (res, MPFR_RNDN);
    f32u32_vl v; v.f = f;
    mag = (uint16_t) (v.u >> 16) & 0x7fff;
  } else {
    mpfr_t scaled; mpfr_init2 (scaled, 256);
    mpfr_mul_2exp (scaled, a, 133, MPFR_RNDN);
    mpfr_rint (scaled, scaled, MPFR_RNDN);
    long k = mpfr_get_si (scaled, MPFR_RNDN);
    mpfr_clear (scaled);
    if (k <= 0)        mag = 0x0000;
    else if (k >= 128) mag = 0x0080;
    else               mag = (uint16_t) k;
  }
  mpfr_clear (res); mpfr_clear (a);
  return sign | mag;
}

static uint16_t mpfr_ref_bf16 (uint16_t bits) {
  uint16_t au = bits & 0x7fff;
  if (au > 0x7f80) return 0x7fc0;                    /* NaN in -> qNaN */

  mpfr_t x, y;
  mpfr_init2 (x, 256); mpfr_init2 (y, 256);
  uint16_t r;

#ifdef VERIFY_SIN_LIMB
  if (au == 0x7f80) { r = 0x7fc0; goto done; }       /* sin(+-Inf) = NaN */
  if (au == 0)      { r = bits;   goto done; }       /* sin(+-0) = +-0   */
  mpfr_set_flt (x, bf16_to_float (bits), MPFR_RNDN);
  mpfr_sin (y, x, MPFR_RNDN);
#else
  if (bits == 0x7f80) { r = 0x7f80; goto done; }     /* exp(+Inf) = +Inf */
  if (bits == 0xff80) { r = 0x0000; goto done; }     /* exp(-Inf) = +0   */
  mpfr_set_flt (x, bf16_to_float (bits), MPFR_RNDN);
  mpfr_exp (y, x, MPFR_RNDN);
#endif
  r = round_mpfr_to_bf16 (y);
done:
  mpfr_clear (x); mpfr_clear (y);
  return r;
}

struct tally { const char *label; int discrepancies; };

int main (void) {
  FILE *out = fopen (OUT_FILE, "w");
  if (!out) { fprintf (stderr, "cannot write %s\n", OUT_FILE); return 1; }

  fprintf (out,
    "MPFR vs limb-table bfloat16 %s() -- exhaustive verification\n"
    "Implementation: %s\n"
    "Configuration : %s\n"
    "                float32 accumulation per factor, single final rounding;\n"
    "                the product is never expanded into cross terms.\n"
    "MPFR version  : %s\n"
    "Checking all 65536 bfloat16 bit patterns (0x0000-0xFFFF)...\n\n",
    FN_NAME, "cr_" FN_NAME "_bf16_limb and cr_" FN_NAME "_bf16_limb_min",
    CONFIG_TXT, mpfr_get_version ());

  struct tally t[2] = { { "EXACT", 0 }, { "MIN", 0 } };
  int nan_pairs = 0;

  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t bits = (uint16_t) b;
    b16u16_vl in; in.u = bits;
    uint16_t ref = mpfr_ref_bf16 (bits);
    int ref_nan = (ref & 0x7fff) > 0x7f80;

    b16u16_vl got[2];
    got[0].f = FN_EXACT (in.f);
    got[1].f = FN_MIN   (in.f);

    if (ref_nan && (got[0].u & 0x7fff) > 0x7f80
                && (got[1].u & 0x7fff) > 0x7f80) { nan_pairs++; continue; }

    for (int v = 0; v < 2; v++) {
      if ((got[v].u & 0x7fff) > 0x7f80 && ref_nan) continue;
      if (got[v].u == ref) continue;
      t[v].discrepancies++;
      fprintf (out, "DISCREPANCY #%d (%s variant):\n",
               t[v].discrepancies, t[v].label);
      fprintf (out, "  Input     : 0x%04X  (%+.8e as float)\n",
               bits, (double) bf16_to_float (bits));
      fprintf (out, "  LIMB      : 0x%04X  (%+.8e as float)\n",
               got[v].u, (double) bf16_to_float (got[v].u));
      fprintf (out, "  MPFR      : 0x%04X  (%+.8e as float)\n",
               ref, (double) bf16_to_float (ref));
#ifdef VERIFY_SIN_LIMB
      uint16_t au = bits & 0x7fff;
      if (au <= 0x3de8)      fprintf (out, "  path      : small-|x| (no table)\n");
      else if (au < 0x4580)  fprintf (out, "  S1/C1 idx : %d   S2/C2 idx : %d\n",
                                      (au - 0x3d80) >> 3,
                                      ((((au - 0x3d80) >> 7) << 3) | (au & 0x7)));
      else                   fprintf (out, "  S3/C3 base: %d   (angle-addition chain)\n",
                                      (au >> 7) - 0x8b);
#else
      uint16_t au = bits & 0x7fff;
      if (au <= 0x3b00 || au >= 0x42ba)
        fprintf (out, "  path      : special case (no table)\n");
      else
        fprintf (out, "  T1 index  : %d   T2 index : %d\n",
                 ((bits >> 15) << 8) + (au >> 3) - 0x760,
                 ((bits >> 15) << 7) + (((au >> 7) << 3) | (au & 0x7)) - 0x3b0);
#endif
      fprintf (out, "\n");
    }
  }

  fprintf (out,
    "=== SUMMARY ===\n"
    "Total bfloat16 inputs   : 65536\n"
    "NaN-in/NaN-out pairs    : %d  (payload not compared)\n"
    "Discrepancies (exact)   : %d\n"
    "Discrepancies (minimal) : %d\n",
    nan_pairs, t[0].discrepancies, t[1].discrepancies);

  int total = t[0].discrepancies + t[1].discrepancies;
  if (total == 0)
    fprintf (out, "RESULT: PASS -- both limb variants match MPFR for all "
                  "non-NaN bfloat16 inputs.\n");
  else
    fprintf (out, "RESULT: FAIL -- %d incorrectly-rounded results.\n", total);
  fclose (out);

  printf ("discrepancies: %d  (exact %d, minimal %d)  (log: %s)\n",
          total, t[0].discrepancies, t[1].discrepancies, OUT_FILE);
  return total == 0 ? 0 : 1;
}
