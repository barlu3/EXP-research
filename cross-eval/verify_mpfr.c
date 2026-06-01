/*
 * Exhaustive bfloat16 exp() verification: CORE-MATH (cr_exp_bf16) vs MPFR.
 *
 * For all 65536 bfloat16 bit patterns, compare the CORE-MATH result against
 * the correctly-rounded MPFR result.  For each discrepancy, prints the input,
 * both outputs, and the T1/T2 table indices used by CORE-MATH.
 * Results are written to MPFR-result.txt.
 *
 * Compile:
 *   gcc -O2 -std=c11 verify_mpfr.c -lmpfr -lgmp -lm -o verify_mpfr
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>

/* Pull in T1[], T2[], cr_exp_bf16(), exp_bf16() from CORE-MATH source. */
// Uncomment line to find discrepancy in rounded 16-bit precision indices.
// Make sure to uncomment the corresponding line in main() to write results to MPFR-result16bitp.txt instead of MPFR-result24bitp.txt.
#include "inria-16bitp.c"
// #include "../implementations/inria-expbf16.c"

/* ── helpers ───────────────────────────────────────────────────────── */

typedef union { float f; uint32_t u; } f32u32;
typedef union { __bf16 f; uint16_t u; } bf16u16;

static float bf16_to_float(uint16_t b)
{
    f32u32 v; v.u = (uint32_t)b << 16; return v.f;
}

/* ── correctly-rounded bfloat16 exp via MPFR ──────────────────────── */
/*
 * bfloat16: 1 sign + 8 exponent + 7 mantissa bits; same exponent range as
 * float32 (bias 127; normal exponent -126..127; 8 significant bits total).
 *
 * In MPFR's convention (significand in [0.5,1)):
 *   emin = -125  (smallest normal 2^-126 = 0.5 * 2^-125)
 *   emax = 128   (largest  normal < 2^128)
 *   precision = 8 bits
 *
 * We avoid mpfr_subnormalize (which has a bug in this MPFR build for
 * subnormal cases) and instead re-round from the 256-bit result to the
 * correct subnormal precision manually.
 */
static uint16_t mpfr_exp_bf16(uint16_t bits)
{
    uint16_t au = bits & 0x7fff;

    /* ── IEEE special values ─────────────────────────────────────────── */
    if (au > 0x7f80) return bits | 0x0040;        /* NaN → quiet NaN   */
    if (au == 0x7f80)
        return (bits == 0x7f80) ? 0x7f80 : 0x0000; /* ±Inf              */

    float x = bf16_to_float(bits);

    /* ── compute exp(x) at 256-bit precision (no emin/emax clamp) ───── */
    mpfr_t mp;
    mpfr_init2(mp, 256);
    mpfr_set_flt(mp, x, MPFR_RNDN);   /* exact: x is a float32 value   */
    mpfr_exp(mp, mp, MPFR_RNDN);

    /* ── round to 8 bits (still no emin/emax, so subnormals are kept) ─ */
    mpfr_t res;
    mpfr_init2(res, 8);
    mpfr_set(res, mp, MPFR_RNDN);
    mpfr_exp_t e = mpfr_zero_p(res) ? (mpfr_exp_t)-9999 : mpfr_get_exp(res);

    uint16_t result;

    if (mpfr_inf_p(res) || e > 128) {
        /* Overflow → bfloat16 +Inf */
        result = 0x7f80;

    } else if (e >= -125) {
        /* Normal bfloat16 value.
         * The 8-bit result is exact for this range; convert to float32
         * (lossless since 8 ≤ 24 sig bits) and take the top 16 bits. */
        float f = mpfr_get_flt(res, MPFR_RNDN);
        f32u32 v; v.f = f;
        result = (uint16_t)(v.u >> 16);

    } else {
        /*
         * Subnormal or underflow (e < -125).
         *
         * bfloat16 subnormals have values k * 2^(-133) for k = 1..127.
         * Round the true value to the nearest such multiple by scaling:
         *   k = round_nearest_even(mp * 2^133)
         * k=0 → underflow (0x0000), k=1..127 → subnormal (0x000k).
         *
         * The 8-bit intermediate (res) is only used for the exponent
         * triage above; we always re-round from the 256-bit mp to avoid
         * any double-rounding error.
         *
         * Edge case e=-133: the 8-bit result sits exactly at 2^(-134)
         * (the threshold between 0x0000 and 0x0001); the scale approach
         * handles this correctly via ties-to-even (0 is even → 0x0000
         * for the exact midpoint; otherwise k=1 → 0x0001).
         */
        mpfr_t scaled;
        mpfr_init2(scaled, 256);
        /* Multiply by 2^133 (exact bit-shift on a 256-bit MPFR value). */
        mpfr_mul_2exp(scaled, mp, 133, MPFR_RNDN);
        /* Round to nearest integer with ties-to-even. */
        mpfr_rint(scaled, scaled, MPFR_RNDN);
        long k = mpfr_get_si(scaled, MPFR_RNDN);
        mpfr_clear(scaled);

        if (k <= 0)
            result = 0x0000;          /* underflow */
        else if (k >= 128)
            result = 0x0080;          /* rounds to smallest normal (safety clamp) */
        else
            result = (uint16_t)k;     /* bfloat16 subnormal mantissa field */
    }

    mpfr_clear(res);
    mpfr_clear(mp);
    return result;
}

/* ── T1/T2 index helper ────────────────────────────────────────────── */
static void get_indices(uint16_t bits, int *i1_out, int *i2_out)
{
    uint16_t au = bits & 0x7fff;
    if (au <= 0x3b00 || au >= 0x42ba) { *i1_out = *i2_out = -1; return; }
    *i1_out = (int)(((bits >> 15) << 8) + (au >> 3) - 0x760u);
    *i2_out = (int)(((bits >> 15) << 7)
                    + (((au >> 7) << 3) | (au & 0x7u))
                    - 0x3b0u);
}

/* ── main ──────────────────────────────────────────────────────────── */
int main(void)
{
    // Uncomment the line below to write results for 16-bit precision
    FILE *out = fopen("MPFR-result16bitp.txt", "w");
    // Since the original bf16 indices were computer with 24-bit precision (32-bit)
    // FILE *out = fopen("MPFR-result24bitp.txt", "w");
    if (!out) { perror("fopen MPFR-result.txt"); return 1; }

    fprintf(out,
        "MPFR vs CORE-MATH bfloat16 exp() — exhaustive verification\n"
        "MPFR version: %s\n"
        "Checking all 65536 bfloat16 bit patterns (0x0000–0xFFFF)...\n\n",
        mpfr_get_version());

    int discrepancies = 0;
    int nan_pairs     = 0;

    for (uint32_t u = 0; u <= 0xFFFFu; u++) {
        uint16_t bits = (uint16_t)u;

        /* CORE-MATH result */
        bf16u16 in_v, out_v;
        in_v.u  = bits;
        out_v.f = cr_exp_bf16(in_v.f);
        uint16_t core_bits = out_v.u;

        /* MPFR result */
        uint16_t mpfr_bits = mpfr_exp_bf16(bits);

        /* Both NaN → skip payload comparison */
        int core_nan = (core_bits & 0x7fff) > 0x7f80;
        int mpfr_nan = (mpfr_bits  & 0x7fff) > 0x7f80;
        if (core_nan && mpfr_nan) { ++nan_pairs; continue; }

        if (core_bits != mpfr_bits) {
            ++discrepancies;
            float x_f    = bf16_to_float(bits);
            float core_f = bf16_to_float(core_bits);
            float mpfr_f = bf16_to_float(mpfr_bits);
            int i1, i2;
            get_indices(bits, &i1, &i2);

            fprintf(out, "DISCREPANCY #%d:\n", discrepancies);
            fprintf(out, "  Input     : 0x%04X  (%+.8e as float)\n",
                    bits, (double)x_f);
            fprintf(out, "  CORE-MATH : 0x%04X  (%+.8e as float)\n",
                    core_bits, (double)core_f);
            fprintf(out, "  MPFR      : 0x%04X  (%+.8e as float)\n",
                    mpfr_bits, (double)mpfr_f);
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
            "RESULT: PASS — cr_exp_bf16 matches MPFR for all "
            "non-NaN bfloat16 inputs.\n");
    else
        fprintf(out,
            "RESULT: FAIL — %d inputs produced incorrectly-rounded "
            "results.\n", discrepancies);

    fclose(out);
    printf("Done. Discrepancies: %d  (see MPFR-result.txt)\n", discrepancies);
    return 0;
}
