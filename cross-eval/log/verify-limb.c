/* Exhaustive MPFR verification of cr_log_bf16_limb (bf16-only limb tables).

   Iterates all 65536 bfloat16 bit patterns and compares the limb
   implementation against a correctly-rounded MPFR reference. Output format
   matches the other cross-eval logs so the runs can be diffed directly.

   NaN inputs are counted but their payloads are not compared, matching
   verify_mpfr.c.

   Build (from the repo root):
     gcc -O2 -std=c11 -I implementations/log cross-eval/log/verify-limb.c \
         implementations/log/inria-logbf16-limb.c -lmpfr -lgmp -lm \
         -o cross-eval/log/verify-limb
*/

#include <stdio.h>
#include <stdint.h>
#include <mpfr.h>

typedef union { float f; uint32_t u; } f32u32;
typedef union { __bf16 f; uint16_t u; } b16u16_v;

extern __bf16 cr_log_bf16_limb (__bf16 x);

#define OUT_FILE "cross-eval/log/MPFR-result-limb-log.txt"

static float bf16_to_float (uint16_t b) {
  f32u32 v; v.u = (uint32_t) b << 16; return v.f;
}

/* Round a signed MPFR value to a bfloat16 bit pattern. Lifted unchanged from
   cross-eval/verify_mpfr.c -- see the commentary there for why this does not
   use mpfr_subnormalize and why it rounds 256 -> 8 in a single step. */
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
    f32u32 v; v.f = f;
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
  if (au > 0x7f80) return 0x7fc0;              /* NaN in -> qNaN */
  if (bits == 0x0000) return 0xff80;           /* log(+0) = -Inf */
  if (bits == 0x8000) return 0xff80;           /* log(-0) = -Inf */
  if (bits >> 15) return 0x7fc0;               /* x < 0 -> qNaN  */
  if (bits == 0x7f80) return 0x7f80;           /* log(+Inf)      */

  mpfr_t x, y;
  mpfr_init2 (x, 256); mpfr_init2 (y, 256);
  mpfr_set_flt (x, bf16_to_float (bits), MPFR_RNDN);
  mpfr_log (y, x, MPFR_RNDN);
  uint16_t r = round_mpfr_to_bf16 (y);
  mpfr_clear (x); mpfr_clear (y);
  return r;
}

int main (void) {
  FILE *out = fopen (OUT_FILE, "w");
  if (!out) { fprintf (stderr, "cannot write %s\n", OUT_FILE); return 1; }

  fprintf (out,
    "MPFR vs limb-table bfloat16 log() — exhaustive verification\n"
    "Implementation: cr_log_bf16_limb (3x bf16 limbs for T1, 2x for T2,\n"
    "                1x for T3; float32 accumulation, single final rounding)\n"
    "MPFR version: %s\n"
    "Checking all 65536 bfloat16 bit patterns (0x0000–0xFFFF)...\n\n",
    mpfr_get_version ());

  int discrepancies = 0, nan_pairs = 0;

  for (uint32_t b = 0; b <= 0xFFFF; b++) {
    uint16_t bits = (uint16_t) b;
    b16u16_v in; in.u = bits;
    b16u16_v got; got.f = cr_log_bf16_limb (in.f);
    uint16_t ref = mpfr_ref_bf16 (bits);

    int got_nan = (got.u & 0x7fff) > 0x7f80;
    int ref_nan = (ref & 0x7fff) > 0x7f80;
    if (got_nan && ref_nan) { nan_pairs++; continue; }

    if (got.u != ref) {
      discrepancies++;
      fprintf (out, "DISCREPANCY #%d:\n", discrepancies);
      fprintf (out, "  Input     : 0x%04X  (%+.8e as float)\n",
               bits, (double) bf16_to_float (bits));
      fprintf (out, "  LIMB      : 0x%04X  (%+.8e as float)\n",
               got.u, (double) bf16_to_float (got.u));
      fprintf (out, "  MPFR      : 0x%04X  (%+.8e as float)\n",
               ref, (double) bf16_to_float (ref));

      int i1 = bits >> 7, i2 = bits & 0x7f;
      if (i1 == 0) {
        fprintf (out, "  T3 index  : %d   (subnormal band, direct lookup)\n", i2);
      } else {
        fprintf (out, "  T1 index  : %d   T2 index : %d\n", i1, i2);
      }
      fprintf (out, "\n");
    }
  }

  fprintf (out,
    "=== SUMMARY ===\n"
    "Total bfloat16 inputs   : 65536\n"
    "NaN-in/NaN-out pairs    : %d  (payload not compared)\n"
    "Discrepancies           : %d\n",
    nan_pairs, discrepancies);
  if (discrepancies == 0)
    fprintf (out, "RESULT: PASS — cr_log_bf16_limb matches MPFR for all "
                  "non-NaN bfloat16 inputs.\n");
  else
    fprintf (out, "RESULT: FAIL — %d inputs produced incorrectly-rounded "
                  "results.\n", discrepancies);
  fclose (out);

  printf ("discrepancies: %d  (log: %s)\n", discrepancies, OUT_FILE);
  return discrepancies == 0 ? 0 : 1;
}
