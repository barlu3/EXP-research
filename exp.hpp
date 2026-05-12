/**
 * @file  fast_exp.hpp
 * @brief Speed-optimised exp() for float32, float64, AVX2 (x8 float, x4 double).
 *
 * ─── ALGORITHM (all variants) ────────────────────────────────────────────────
 *
 *  Step 1 — Range reduction (Cody-Waite 2-part)
 *    k  = round( x · log₂e )         integer exponent
 *    r  = x − k·ln2_hi − k·ln2_lo    reduced arg, |r| ≤ ln2/2 ≈ 0.347
 *
 *  Step 2 — Minimax polynomial approximation of eʳ
 *    float  : degree-7, Cephes coefficients    → < 2 ULP
 *    double : degree-12 Taylor (1/n!)          → ≈ 1.7 × 10⁻¹⁶ (< ε_mach)
 *
 *  Step 3 — Reconstruction via IEEE 754 bit manipulation
 *    result = P(r) · 2^k
 *    2^k built by writing (k + bias) into the exponent field directly,
 *    avoiding ldexp() overhead.
 *
 * ─── VALID INPUT RANGES ──────────────────────────────────────────────────────
 *  float  : [-88.376, 88.376]   (clamped; outside → 0.0f or clamps to max)
 *  double : [-708.0,  709.0]    (clamped; deliberately avoids k=1024 overflow;
 *                                exp(709.0)≈8.68e307, < DBL_MAX≈1.80e308)
 *
 * ─── COMPILATION ─────────────────────────────────────────────────────────────
 *  g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench
 *
 * ─── REFERENCES ──────────────────────────────────────────────────────────────
 *  [1] Cephes Math Library   — https://www.netlib.org/cephes/
 *  [2] avx_mathfun (Garberoglio) — http://software-lisc.fbk.eu/avx_mathfun/
 *  [3] Musl libc exp.c       — https://git.musl-libc.org/cgit/musl/tree/src/math/exp.c
 *  [4] Cody & Waite, "Software Manual for the Elementary Functions", 1980
 */

#pragma once

#include <immintrin.h>   // AVX2, FMA3
#include <cstdint>
#include <cmath>
#include <bit>           // std::bit_cast  (C++20)
#include <algorithm>     // std::clamp

// ═════════════════════════════════════════════════════════════════════════════
//  SECTION 1 — Compile-time constants
// ═════════════════════════════════════════════════════════════════════════════

namespace fexp::detail {

// ── float32 ──────────────────────────────────────────────────────────────────

/// log₂(e) — multiplier for computing k = round(x · log₂e)
inline constexpr float LOG2EF    =  1.44269504088896341f;
/// ln(2) high part (exact in float) — Cody-Waite decomposition
inline constexpr float LN2H_F    =  6.93145751953125e-1f;
/// ln(2) low part = ln(2) − LN2H_F — corrects range-reduction error
inline constexpr float LN2L_F    =  1.42860682030941723212e-6f;
/// Upper clamp: k_max = 127, scale = 2^127 ≈ 1.7e38 (just below FLT_MAX)
inline constexpr float EXP_HI_F  =  88.3762626647950f;
/// Lower clamp: k_min = -126 → scale = 2^-126 = FLT_MIN (avoids scale=0)
/// = -126 * ln2 ≈ -87.337.  Values below this clamp return exp(-87.337) ≈ 5.6e-39.
inline constexpr float EXP_LO_F  = -87.33654f;
/// IEEE 754 float exponent bias
inline constexpr int   BIAS_F    = 127;

/**
 * Polynomial P for e^r = 1 + r + r² · P(r), degree-5 in r, r ∈ [-ln2/2, ln2/2]
 * Cephes expf coefficients [1].
 * Evaluated as Horner:  ((((P0·r + P1)·r + P2)·r + P3)·r + P4)·r + P5
 * P5 ≈ 1/2!, P4 ≈ 1/3!, P3 ≈ 1/4!, P2 ≈ 1/5!, P1 ≈ 1/6!, P0 ≈ 1/7!
 * Small deviations from exact 1/n! are the minimax adjustment.
 */
inline constexpr float P0F = 1.9875691500E-4f;  //  ≈ 1/7!
inline constexpr float P1F = 1.3981999507E-3f;  //  ≈ 1/6!
inline constexpr float P2F = 8.3334519073E-3f;  //  ≈ 1/5!
inline constexpr float P3F = 4.1665795894E-2f;  //  ≈ 1/4!
inline constexpr float P4F = 1.6666665459E-1f;  //  ≈ 1/3!
inline constexpr float P5F = 5.0000001201E-1f;  //  ≈ 1/2!

// ── float64 ──────────────────────────────────────────────────────────────────

inline constexpr double LOG2E    =  1.4426950408889634073599;
inline constexpr double LN2H_D   =  6.93147180369123816490e-1;
inline constexpr double LN2L_D   =  1.90821492927058770002e-10;
/// Safe upper bound: avoids k = 1024 which would set exponent field to 2047 (±INF)
inline constexpr double EXP_HI_D =  709.0;
/// Safe lower bound: k_min = −1022 → scale = 2^−1022 = DBL_MIN (no denormals)
inline constexpr double EXP_LO_D = -708.0;
inline constexpr int64_t BIAS_D  = 1023LL;

/**
 * Taylor coefficients D[n] = 1/n! for n = 0..12.
 * Degree-12 Horner: p = D12; p = p·r + D11; ... ; p = p·r + D0
 * Truncation error for |r| ≤ ln2/2:
 *   |r|^13 / 13! ≤ (0.347)^13 / 6227020800 ≈ 1.67 × 10⁻¹⁶  (< ε_mach = 2.22e-16)
 */
inline constexpr double D0  = 1.0;
inline constexpr double D1  = 1.0;
inline constexpr double D2  = 5.000000000000000000e-1;   // 1/2!
inline constexpr double D3  = 1.666666666666666667e-1;   // 1/3!
inline constexpr double D4  = 4.166666666666666667e-2;   // 1/4!
inline constexpr double D5  = 8.333333333333333333e-3;   // 1/5!
inline constexpr double D6  = 1.388888888888888889e-3;   // 1/6!
inline constexpr double D7  = 1.984126984126984127e-4;   // 1/7!
inline constexpr double D8  = 2.480158730158730159e-5;   // 1/8!
inline constexpr double D9  = 2.755731922398589065e-6;   // 1/9!
inline constexpr double D10 = 2.755731922398589065e-7;   // 1/10!
inline constexpr double D11 = 2.505210838544171878e-8;   // 1/11!
inline constexpr double D12 = 2.087675698786809897e-9;   // 1/12!

} // namespace fexp::detail


// ═════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — Scalar float32
// ═════════════════════════════════════════════════════════════════════════════

namespace fexp {

/**
 * @brief  Scalar single-precision fast_expf
 * @param  x  Input angle (radians). Clamped to [EXP_LO_F, EXP_HI_F].
 * @return    Approximation of eˣ, < 2 ULP error.
 *
 * @note   Reconstruction:
 *           IEEE 754 float layout: [sign:1][exponent:8][mantissa:23]
 *           2^k = reinterpret_cast<float>((k + 127) << 23)
 *           k range after clamp: [-127, 127] → exponent field [0, 254], no special vals.
 */
[[nodiscard]] inline float fast_expf(float x) noexcept {
    using namespace detail;

    // 1. Clamp ─────────────────────────────────────────────────────────────────
    x = std::clamp(x, EXP_LO_F, EXP_HI_F);

    // 2. Range reduction ───────────────────────────────────────────────────────
    float kf = std::roundf(x * LOG2EF);
    int   k  = static_cast<int>(kf);

    // Cody-Waite 2-part: r = x − k·ln2_hi − k·ln2_lo
    // Two subtractions prevent catastrophic cancellation near multiples of ln2.
    float r  = x - kf * LN2H_F - kf * LN2L_F;

    // 3. Polynomial: e^r = 1 + r + r² · P(r) ──────────────────────────────────
    // Horner from highest degree down:
    float p = P0F;
    p = p * r + P1F;
    p = p * r + P2F;
    p = p * r + P3F;
    p = p * r + P4F;
    p = p * r + P5F;
    // p now = P5 + P4·r + P3·r² + P2·r³ + P1·r⁴ + P0·r⁵

    float r2 = r * r;
    p = p * r2 + r + 1.0f;
    // p = 1 + r + r²·(P5 + P4·r + ... + P0·r⁵) ≈ eʳ

    // 4. Reconstruct 2^k via bit manipulation ──────────────────────────────────
    // Avoids ldexpf() function-call overhead.
    uint32_t bits = static_cast<uint32_t>(k + BIAS_F) << 23;
    float    scale = std::bit_cast<float>(bits);

    return p * scale;
}


// ═════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — Scalar float64
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief  Scalar double-precision fast_exp
 * @param  x  Input. Clamped to [EXP_LO_D, EXP_HI_D] = [-708, 709].
 * @return    Approximation of eˣ.
 *
 * @note   Uses degree-12 Taylor polynomial. Truncation error ≈ 1.67e-16 < ε_mach.
 *           Reconstruction:
 *           IEEE 754 double layout: [sign:1][exponent:11][mantissa:52]
 *           2^k = reinterpret_cast<double>((k + 1023) << 52)
 *           k range after clamp: [-1022, 1023] → exponent field [1, 2046], no INF/NaN.
 */
[[nodiscard]] inline double fast_exp(double x) noexcept {
    using namespace detail;

    // 1. Clamp
    x = std::clamp(x, EXP_LO_D, EXP_HI_D);

    // 2. Range reduction
    double kd = std::round(x * LOG2E);
    int64_t k = static_cast<int64_t>(kd);
    double r  = x - kd * LN2H_D - kd * LN2L_D;

    // 3. Polynomial: degree-12 Horner (12 FMAs with -O3 -mfma)
    double p = D12;
    p = p * r + D11;
    p = p * r + D10;
    p = p * r + D9;
    p = p * r + D8;
    p = p * r + D7;
    p = p * r + D6;
    p = p * r + D5;
    p = p * r + D4;
    p = p * r + D3;
    p = p * r + D2;
    p = p * r + D1;
    p = p * r + D0;
    // p ≈ eʳ

    // 4. Reconstruct 2^k
    uint64_t bits = static_cast<uint64_t>(k + BIAS_D) << 52;
    double   scale = std::bit_cast<double>(bits);

    return p * scale;
}


// ═════════════════════════════════════════════════════════════════════════════
//  SECTION 4 — AVX2 float32 (8-wide)
// ═════════════════════════════════════════════════════════════════════════════

#ifdef __AVX2__

/**
 * @brief  AVX2 single-precision exp — processes 8 floats per call.
 * @param  x  __m256 register with 8 input values.
 * @return    __m256 register with 8 approximations of eˣ.
 *
 * @note   Same algorithm as fast_expf; all operations vectorised.
 *           _mm256_fnmadd_ps(a,b,c) = c − a·b  (negated FMA, for Cody-Waite)
 *           _mm256_fmadd_ps(a,b,c)  = a·b + c  (FMA, for Horner)
 *           Reconstruction: add BIAS to integer k, shift left 23 into exponent field.
 */
[[nodiscard]] inline __m256 fast_expf_avx2(__m256 x) noexcept {
    using namespace detail;

    const __m256 vLOG2EF = _mm256_set1_ps(LOG2EF);
    const __m256 vLN2H   = _mm256_set1_ps(LN2H_F);
    const __m256 vLN2L   = _mm256_set1_ps(LN2L_F);
    const __m256 vHI     = _mm256_set1_ps(EXP_HI_F);
    const __m256 vLO     = _mm256_set1_ps(EXP_LO_F);
    const __m256i vBIAS  = _mm256_set1_epi32(BIAS_F);

    // 1. Clamp
    x = _mm256_min_ps(_mm256_max_ps(x, vLO), vHI);

    // 2. Range reduction
    __m256 kf = _mm256_round_ps(
        _mm256_mul_ps(x, vLOG2EF),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    // r = x − kf·ln2_hi − kf·ln2_lo  (Cody-Waite 2-part)
    __m256 r = _mm256_fnmadd_ps(kf, vLN2H, x);
    r        = _mm256_fnmadd_ps(kf, vLN2L, r);

    // 3. Polynomial: Horner with FMA
    __m256 p = _mm256_set1_ps(P0F);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(P1F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(P2F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(P3F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(P4F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(P5F));

    __m256 r2 = _mm256_mul_ps(r, r);
    p = _mm256_fmadd_ps(p, r2, r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    // 4. Reconstruct: build 8 floats = 2^k[i]
    // Convert rounded kf to int32, add bias, shift into exponent field.
    __m256i ki = _mm256_cvtps_epi32(kf);          // float→int32 (round-to-nearest)
    ki = _mm256_add_epi32(ki, vBIAS);
    ki = _mm256_slli_epi32(ki, 23);               // place in exponent bits [30:23]
    __m256 scale = _mm256_castsi256_ps(ki);

    return _mm256_mul_ps(p, scale);
}


// ═════════════════════════════════════════════════════════════════════════════
//  SECTION 5 — AVX2 float64 (4-wide)
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief  AVX2 double-precision exp — processes 4 doubles per call.
 * @param  x  __m256d register with 4 input values.
 * @return    __m256d register with 4 approximations of eˣ.
 *
 * @note   Reconstruction detail (AVX2 has no _mm256_cvtpd_epi64 before AVX-512):
 *           1. _mm256_cvttpd_epi32(kd)  → __m128i with 4 × int32
 *           2. _mm256_cvtepi32_epi64()  → __m256i with 4 × int64 (sign-extended)
 *           3. _mm256_add_epi64(., bias)
 *           4. _mm256_slli_epi64(., 52) → 4 × 2^k[i] in double IEEE bit layout
 */
[[nodiscard]] inline __m256d fast_exp_avx2(__m256d x) noexcept {
    using namespace detail;

    const __m256d vLOG2E = _mm256_set1_pd(LOG2E);
    const __m256d vLN2H  = _mm256_set1_pd(LN2H_D);
    const __m256d vLN2L  = _mm256_set1_pd(LN2L_D);
    const __m256d vHI    = _mm256_set1_pd(EXP_HI_D);
    const __m256d vLO    = _mm256_set1_pd(EXP_LO_D);

    // 1. Clamp
    x = _mm256_min_pd(_mm256_max_pd(x, vLO), vHI);

    // 2. Range reduction
    __m256d kd = _mm256_round_pd(
        _mm256_mul_pd(x, vLOG2E),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    __m256d r = _mm256_fnmadd_pd(kd, vLN2H, x);
    r         = _mm256_fnmadd_pd(kd, vLN2L, r);

    // 3. Polynomial: degree-12 Horner with FMA
    __m256d p = _mm256_set1_pd(D12);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D11));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D10));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D9));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D8));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D7));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D6));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D5));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D4));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D3));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D2));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D1));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(D0));

    // 4. Reconstruct 2^k[i] (AVX2 workaround — no cvtpd_epi64 until AVX-512)
    __m128i ki32 = _mm256_cvttpd_epi32(kd);          // 4×double → 4×int32 in __m128i
    __m256i ki64 = _mm256_cvtepi32_epi64(ki32);       // sign-extend to 4×int64
    ki64 = _mm256_add_epi64(ki64, _mm256_set1_epi64x(BIAS_D));
    ki64 = _mm256_slli_epi64(ki64, 52);               // place in double exponent bits
    __m256d scale = _mm256_castsi256_pd(ki64);

    return _mm256_mul_pd(p, scale);
}

#endif // __AVX2__

} // namespace fexp