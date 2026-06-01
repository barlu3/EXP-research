/*
 * Exhaustive bfloat16 verification of CORE-MATH against MPFR ground truth.
 *
 * The function under test is selected at compile time:
 *   (default)      exp — includes inria-exp16bitp.c, writes MPFR-result16bitp.txt
 *   -DVERIFY_SIN   sin — includes inria-sinbf16.c,    writes sin/MPFR-result24bitp-sin.txt
 *   -DVERIFY_LOG   log — includes inria-logbf16.c,    writes log/MPFR-result24bitp-log.txt
 *
 * For all 65536 bfloat16 bit patterns, compare the CORE-MATH result against
 * the correctly-rounded MPFR result.  Every discrepancy is logged with input,
 * both outputs, and the lookup-table indices used by CORE-MATH (T1/T2 for exp,
 * S1/C1/S2/C2 for sin, T1/T2 or T3 for log).
 *
 * Compile (from cross-eval/):
 *   gcc -O2 -std=c11              verify_mpfr.c -lmpfr -lgmp -lm -o verify_mpfr
 *   gcc -O2 -std=c11 -DVERIFY_SIN verify_mpfr.c -lmpfr -lgmp -lm -o verify_sin
 *   gcc -O2 -std=c11 -DVERIFY_LOG verify_mpfr.c -lmpfr -lgmp -lm -o verify_log
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>

/* Pull in the CORE-MATH implementation and pick its entry point + output. */
#if defined(VERIFY_SIN)
#include "sin/inria-sin16bitp.c"
// Uncomment to verify the unmodified Inria tables instead of the 16-bit-rounded ones.
// #include "../implementations/sin/inria-sinbf16.c"
#define CR_FUNC   cr_sin_bf16
#define FUNC_NAME "sin"
#define OUT_FILE "sin/MPFR-result16bitp-sin.txt"
// #define OUT_FILE  "sin/MPFR-result24bitp-sin.txt"
#elif defined(VERIFY_LOG)
# include "log/inria-log16bitp.c"
// Uncomment to verify the unmodified Inria tables instead of the 16-bit-rounded ones.
// #include "../implementations/log/inria-logbf16.c"
#define CR_FUNC   cr_log_bf16
#define FUNC_NAME "log"
#define OUT_FILE "log/MPFR-result16bitp-log.txt"
// #define OUT_FILE  "log/MPFR-result24bitp-log.txt"
#else
/* exp: the table source under test (16-bit rounded, or original 24-bit).
   Uncomment the original to verify the unmodified Inria tables instead. */
#include "inria-exp16bitp.c"
// #include "../implementations/inria-expbf16.c"
#define CR_FUNC   cr_exp_bf16
#define FUNC_NAME "exp"
#define OUT_FILE  "MPFR-result16bitp.txt"
// #define OUT_FILE "MPFR-result24bitp.txt"
#endif

/* ── helpers ───────────────────────────────────────────────────────── */

typedef union { float f; uint32_t u; } f32u32;
typedef union { __bf16 f; uint16_t u; } bf16u16;

static float bf16_to_float(uint16_t b)
{
    f32u32 v; v.u = (uint32_t)b << 16; return v.f;
}

/* ── round a signed MPFR value to a bfloat16 bit pattern ──────────────── */
/*
 * bfloat16: 1 sign + 8 exponent + 7 mantissa bits; same exponent range as
 * float32 (bias 127).  In MPFR's significand-in-[0.5,1) convention the
 * normal exponent runs -125..128 with 8 significant bits.
 *
 * We avoid mpfr_subnormalize (buggy for subnormals in this MPFR build) and
 * re-round from the high-precision result manually.  Sign is handled here so
 * the same routine serves exp (positive), sin and log (both signed).
 */
static uint16_t round_mpfr_to_bf16(mpfr_t mp)
{
    if (mpfr_nan_p(mp)) return 0x7fc0;                  /* qNaN              */
    uint16_t sign = mpfr_signbit(mp) ? 0x8000 : 0x0000;
    if (mpfr_inf_p(mp))  return sign | 0x7f80;          /* +-Inf             */
    if (mpfr_zero_p(mp)) return sign;                   /* +-0               */

    /* Work on the magnitude; re-apply the sign at the end. */
    mpfr_t a; mpfr_init2(a, 256); mpfr_abs(a, mp, MPFR_RNDN);

    mpfr_t res; mpfr_init2(res, 8);
    mpfr_set(res, a, MPFR_RNDN);                        /* round to 8 sig bits */
    mpfr_exp_t e = mpfr_get_exp(res);

    uint16_t mag;
    if (mpfr_inf_p(res) || e > 128) {
        mag = 0x7f80;                                   /* overflow -> Inf   */
    } else if (e >= -125) {
        /* Normal: the 8-bit result is exact in float32 (8 <= 24 sig bits);
           the top 16 bits are the bfloat16 magnitude. */
        float f = mpfr_get_flt(res, MPFR_RNDN);
        f32u32 v; v.f = f;
        mag = (uint16_t)(v.u >> 16) & 0x7fff;
    } else {
        /* Subnormal: bfloat16 subnormals are k * 2^(-133), k = 1..127.
           Re-round from the high-precision magnitude to avoid double rounding. */
        mpfr_t scaled; mpfr_init2(scaled, 256);
        mpfr_mul_2exp(scaled, a, 133, MPFR_RNDN);       /* exact bit-shift   */
        mpfr_rint(scaled, scaled, MPFR_RNDN);           /* ties-to-even      */
        long k = mpfr_get_si(scaled, MPFR_RNDN);
        mpfr_clear(scaled);
        if (k <= 0)        mag = 0x0000;                /* underflow         */
        else if (k >= 128) mag = 0x0080;               /* smallest normal   */
        else               mag = (uint16_t)k;
    }

    mpfr_clear(res); mpfr_clear(a);
    return sign | mag;
}

/* ── correctly-rounded bfloat16 reference via MPFR ───────────────────── */
static uint16_t mpfr_ref_bf16(uint16_t bits)
{
    uint16_t au = bits & 0x7fff;
    if (au > 0x7f80) return 0x7fc0;                     /* NaN in -> qNaN    */

    float x = bf16_to_float(bits);
    mpfr_t mp; mpfr_init2(mp, 256);
    uint16_t r;

#if defined(VERIFY_SIN)
    if (au == 0x7f80) { mpfr_clear(mp); return 0x7fc0; }            /* sin(+-Inf)=NaN */
    mpfr_set_flt(mp, x, MPFR_RNDN);
    mpfr_sin(mp, mp, MPFR_RNDN);
#elif defined(VERIFY_LOG)
    if (au == 0x7f80) { mpfr_clear(mp); return (bits >> 15) ? 0x7fc0 : 0x7f80; } /* log(-Inf)=NaN, log(+Inf)=+Inf */
    if (bits >> 15)   { mpfr_clear(mp); return (bits == 0x8000) ? 0xff80 : 0x7fc0; } /* log(-0)=-Inf, log(x<0)=NaN */
    if (au == 0)      { mpfr_clear(mp); return 0xff80; }            /* log(+0) = -Inf */
    mpfr_set_flt(mp, x, MPFR_RNDN);
    mpfr_log(mp, mp, MPFR_RNDN);
#else
    if (au == 0x7f80) { mpfr_clear(mp); return (bits >> 15) ? 0x0000 : 0x7f80; }  /* exp(-Inf)=0, exp(+Inf)=+Inf */
    mpfr_set_flt(mp, x, MPFR_RNDN);
    mpfr_exp(mp, mp, MPFR_RNDN);
#endif

    r = round_mpfr_to_bf16(mp);
    mpfr_clear(mp);
    return r;
}

#if !defined(VERIFY_SIN) && !defined(VERIFY_LOG)
/* ── exp T1/T2 index helper (for discrepancy annotation) ─────────────── */
static void get_indices(uint16_t bits, int *i1_out, int *i2_out)
{
    uint16_t au = bits & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) { *i1_out = *i2_out = -1; return; }
    *i1_out = (int)(((bits >> 15) << 8) + (au >> 3) - 0x760u);
    *i2_out = (int)(((bits >> 15) << 7)
                    + (((au >> 7) << 3) | (au & 0x7u))
                    - 0x3b0u);
}
#endif

/* ── main ──────────────────────────────────────────────────────────── */
int main(void)
{
    FILE *out = fopen(OUT_FILE, "w");
    if (!out) { perror("fopen " OUT_FILE); return 1; }

    fprintf(out,
        "MPFR vs CORE-MATH bfloat16 %s() — exhaustive verification\n"
        "MPFR version: %s\n"
        "Checking all 65536 bfloat16 bit patterns (0x0000–0xFFFF)...\n\n",
        FUNC_NAME, mpfr_get_version());

    int discrepancies = 0;
    int nan_pairs     = 0;

    for (uint32_t u = 0; u <= 0xFFFFu; u++) {
        uint16_t bits = (uint16_t)u;

        bf16u16 in_v, out_v;
        in_v.u  = bits;
        out_v.f = CR_FUNC(in_v.f);
        uint16_t core_bits = out_v.u;

        uint16_t mpfr_bits = mpfr_ref_bf16(bits);

        int core_nan = (core_bits & 0x7fff) > 0x7f80;
        int mpfr_nan = (mpfr_bits  & 0x7fff) > 0x7f80;
        if (core_nan && mpfr_nan) { ++nan_pairs; continue; }

        if (core_bits != mpfr_bits) {
            ++discrepancies;
            float x_f    = bf16_to_float(bits);
            float core_f = bf16_to_float(core_bits);
            float mpfr_f = bf16_to_float(mpfr_bits);

            fprintf(out, "DISCREPANCY #%d:\n", discrepancies);
            fprintf(out, "  Input     : 0x%04X  (%+.8e as float)\n",
                    bits, (double)x_f);
            fprintf(out, "  CORE-MATH : 0x%04X  (%+.8e as float)\n",
                    core_bits, (double)core_f);
            fprintf(out, "  MPFR      : 0x%04X  (%+.8e as float)\n",
                    mpfr_bits, (double)mpfr_f);
#if defined(VERIFY_SIN)
            /* reduction path: sin = sgn * fma(S1[i1], C2[i2], C1[i1]*S2[i2]) */
            {
                uint16_t au = bits & 0x7fff;
                if (au > 0x3de8 && au < 0x4580) {
                    int i1 = (int)((au - 0x3d80) >> 3);
                    int i2 = (int)((((au - 0x3d80) >> 7) << 3) | (au & 0x7));
                    fprintf(out, "  S1 index  : %d   S1[%d] = %.8e\n",
                            i1, i1, (double)S1[i1]);
                    fprintf(out, "  C1 index  : %d   C1[%d] = %.8e\n",
                            i1, i1, (double)C1[i1]);
                    fprintf(out, "  S2 index  : %d   S2[%d] = %.8e\n",
                            i2, i2, (double)S2[i2]);
                    fprintf(out, "  C2 index  : %d   C2[%d] = %.8e\n",
                            i2, i2, (double)C2[i2]);
                    fprintf(out, "  fma(S1,C2,C1*S2) (float32) = %.8e\n",
                            (double)__builtin_fmaf(S1[i1], C2[i2],
                                                   C1[i1] * S2[i2]));
                } else if (au >= 0x4580 && (au >> 7) != 0xff) {
                    /* large-arg path: |x| >= 4096 reconstructs sin via the
                       S3/C3 tables over indices k..k+7 (k=0 at |x|=4096). */
                    int k = (int)((au >> 7) - 0x8b);
                    fprintf(out, "  S3/C3 base : k=%d   S3[%d] = %.8e   C3[%d] = %.8e\n",
                            k, k, (double)S3[k], k, (double)C3[k]);
                    fprintf(out, "  (large-arg path uses S3/C3[k..k+7])\n");
                } else {
                    fprintf(out, "  (special-case path — sin(x) rounds to x, no table)\n");
                }
            }
#elif defined(VERIFY_LOG)
            /* normal: log = T1[i1] + T2[i2];  subnormal (i1==0): log = T3[i2] */
            if ((bits >> 15) == 0 && (bits & 0x7fff) < 0x7f80) {
                int i1 = (int)(bits >> 7);
                int i2 = (int)(bits & 0x7f);
                if (i1 == 0) {
                    fprintf(out, "  T3 index  : %d   T3[%d] = %.8e\n",
                            i2, i2, (double)T3[i2].f);
                } else {
                    fprintf(out, "  T1 index  : %d   T1[%d] = %.8e\n",
                            i1, i1, (double)T1[i1]);
                    fprintf(out, "  T2 index  : %d   T2[%d] = %.8e\n",
                            i2, i2, (double)T2[i2]);
                    fprintf(out, "  T1[i1]+T2[i2] (float32) = %.8e\n",
                            (double)(T1[i1] + T2[i2]));
                }
            } else {
                fprintf(out, "  (special-case path — no table lookup)\n");
            }
#else
            int i1, i2;
            get_indices(bits, &i1, &i2);
            if (i1 >= 0) {
                fprintf(out, "  T1 index  : %d   T1[%d] = %.8e\n",
                        i1, i1, (double)T1[i1]);
                fprintf(out, "  T2 index  : %d   T2[%d] = %.8e\n",
                        i2, i2, (double)T2[i2]);
                fprintf(out, "  T1[i1]*T2[i2] (float32) = %.8e\n",
                        (double)(T1[i1] * T2[i2]));
            } else {
                fprintf(out, "  (special-case path — no table lookup)\n");
            }
#endif
            fprintf(out, "\n");
        }
    }

    fprintf(out,
        "=== SUMMARY ===\n"
        "Total bfloat16 inputs   : 65536\n"
        "NaN-in/NaN-out pairs    : %d  (payload not compared)\n"
        "Discrepancies           : %d\n",
        nan_pairs, discrepancies);

    if (discrepancies == 0)
        fprintf(out,
            "RESULT: PASS — cr_%s_bf16 matches MPFR for all "
            "non-NaN bfloat16 inputs.\n", FUNC_NAME);
    else
        fprintf(out,
            "RESULT: FAIL — %d inputs produced incorrectly-rounded "
            "results.\n", discrepancies);

    fclose(out);
    printf("[%s] Done. Discrepancies: %d  (see %s)\n",
           FUNC_NAME, discrepancies, OUT_FILE);
    return 0;
}
