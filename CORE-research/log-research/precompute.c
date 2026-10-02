/* Precompute ln(x) for every valid bfloat16 input x, using MPFR as ground truth.

   Step 1 of the T1/T2 linear-programming workflow (see bound-calc.cc and the
   emitted lp-constraints.txt). For each of the 65536 bf16 bit patterns u we
   decode x, compute ln(x) in high precision, and record both the exact
   high-precision ln(x) and the correctly-rounded bf16 value y = round_bf16(ln x).

   Valid domain: x finite and strictly positive. x = +0 gives ln = -Inf (a valid,
   exactly-representable bf16 result). Excluded: sign bit set (-0 / negative),
   and the exp==0xFF patterns (+Inf / NaN), where ln has no finite target.

   CORE-MATH (see cross-eval/log/inria-log16bitp.c) splits x = x1 * x2 with
       i1 = u >> 7    (top 9 bits = biased-exponent field; sign is 0 here)
       i2 = u & 0x7f  (7 mantissa bits)
       x1 = 2^(i1 - 127),   x2 = 1 + i2/128
   so ln(x) = ln(x1) + ln(x2) = T1[i1] + T2[i2] for normal x. i1/i2 are emitted so
   bound-calc.cc can group LP constraints by table index without re-deriving it.

   Output (stdout -> log-research/ln-precompute.txt), CSV after '#' headers:
       u,i1,i2,subnormal,x_hex,lnx_exact_hex,y_hex,y_bits
   lnx_exact_hex is ln(x) at WORK_PREC (the source of truth for the bounds);
   y_hex / y_bits are that value rounded to bf16 (RNDN) and its 16-bit pattern. */

#include <stdint.h>
#include <stdio.h>
#include <mpfr.h>

/* bf16: 1 sign, 8 exponent, 7 mantissa. __bf16 carries the float32 exponent range. */
typedef union { __bf16 f; uint16_t u; } b16u16;

#define BF16_PREC 8       /* significand bits: 1 implicit + 7 stored */
#define BF16_EMIN (-133)  /* smallest exponent including bf16 subnormals */
#define BF16_EMAX 128     /* largest bf16 exponent */
#define WORK_PREC 200     /* working precision for ln(x), far above bf16 */

/* y = round_bf16(value) under RNDN, honoring bf16 exponent range; also return the
   16-bit pattern in *bits. value stays at its own precision. */
static void bf16_round (mpfr_t value, mpfr_t y, uint16_t *bits) {
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (BF16_EMIN);
  mpfr_set_emax (BF16_EMAX);
  mpfr_set_prec (y, BF16_PREC);
  int inex = mpfr_set (y, value, MPFR_RNDN);
  mpfr_subnormalize (y, inex, MPFR_RNDN);
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
  b16u16 v;
  v.f = (__bf16) mpfr_get_d (y, MPFR_RNDN); /* y already fits bf16, so exact */
  *bits = v.u;
}

int main (void) {
  mpfr_set_emin (BF16_EMIN);
  mpfr_set_emax (BF16_EMAX);

  mpfr_t x, lnx, y;
  mpfr_init2 (x, WORK_PREC);
  mpfr_init2 (lnx, WORK_PREC);
  mpfr_init2 (y, BF16_PREC);

  printf ("# ln(x) precompute for all valid bfloat16 inputs (MPFR %s)\n",
          mpfr_get_version ());
  printf ("# decomposition: i1=u>>7 (exp field), i2=u&0x7f (mantissa)\n");
  printf ("# x = x1*x2, x1=2^(i1-127), x2=1+i2/128; ln(x)=T1[i1]+T2[i2]\n");
  printf ("# work precision = %d bits, bf16 significand = %d bits, RNDN\n",
          WORK_PREC, BF16_PREC);
  printf ("# columns: u,i1,i2,subnormal,x_hex,lnx_exact_hex,y_hex,y_bits\n");

  long count = 0;
  for (uint32_t uu = 0; uu <= 0xFFFF; uu++) {
    uint16_t u = (uint16_t) uu;
    b16u16 v; v.u = u;

    uint16_t exp = (u >> 7) & 0xFF;
    uint16_t mant = u & 0x7f;
    if (u >> 15) continue;      /* sign set: -0 or negative */
    if (exp == 0xFF) continue;  /* +Inf or NaN */

    uint16_t i1 = u >> 7;       /* == exp, sign is 0 */
    uint16_t i2 = mant;
    int subnormal = (exp == 0); /* exp==0: +0 (mant 0) or subnormal x */

    mpfr_set_d (x, (double) (float) v.f, MPFR_RNDN); /* bf16 fits double exactly */

    uint16_t ybits;
    char lnx_hex[80], y_hex[80];
    if (u == 0x0000) {          /* x = +0 -> ln = -Inf, exact bf16 target */
      snprintf (lnx_hex, sizeof lnx_hex, "-inf");
      snprintf (y_hex, sizeof y_hex, "-inf");
      ybits = 0xFF80;
    } else {
      mpfr_log (lnx, x, MPFR_RNDN);
      bf16_round (lnx, y, &ybits);
      mpfr_snprintf (lnx_hex, sizeof lnx_hex, "%Ra", lnx);
      mpfr_snprintf (y_hex, sizeof y_hex, "%Ra", y);
    }

    char x_hex[80];
    mpfr_snprintf (x_hex, sizeof x_hex, "%Ra", x);

    printf ("%u,%u,%u,%d,%s,%s,%s,0x%04X\n",
            u, i1, i2, subnormal, x_hex, lnx_hex, y_hex, ybits);
    count++;
  }

  fprintf (stderr, "precompute: %ld valid inputs written\n", count);
  mpfr_clear (x);
  mpfr_clear (lnx);
  mpfr_clear (y);
  return 0;
}
