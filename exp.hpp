/**
 * @file  exp.hpp
 * @brief Faithful C++ port of glibc-2.43 exp() — baseline for benchmarking.
 *
 * float32: follows sysdeps/ieee754/flt-32/e_expf.c
 *   N=32 table of 2^(j/N), degree-3 poly, all arithmetic in double.
 *
 * float64: follows sysdeps/ieee754/dbl-64/e_exp.c
 *   N=128 table of (tail, H) pairs, degree-4 poly (C2..C5),
 *   Shift-trick integer rounding, Cody-Waite 2-part range reduction.
 *
 * AVX2 symbols are scalar-loop stubs so benchmark.cpp compiles unchanged.
 *
 * Compile:
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench
 */

#pragma once

#include <immintrin.h>
#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>

// ─── bit_cast portability shim ────────────────────────────────────────────────
// Apple Clang ≤ 14 ships libc++ without std::bit_cast even with -std=c++20.
// __builtin_bit_cast is available in Clang ≥ 9 and GCC ≥ 11, covering all
// relevant compilers.  Fall back to a memcpy template only on exotic toolchains.
#if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
#  define FEXP_BIT_CAST(T, x) __builtin_bit_cast(T, x)
#elif defined(__cpp_lib_bit_cast)
#  define FEXP_BIT_CAST(T, x) std::bit_cast<T>(x)
#else
#  include <cstring>
namespace fexp_compat {
template<typename To, typename From>
static inline To bit_cast(From from) noexcept {
    static_assert(sizeof(To) == sizeof(From));
    To r; std::memcpy(&r, &from, sizeof(r)); return r;
}
}
#  define FEXP_BIT_CAST(T, x) fexp_compat::bit_cast<T>(x)
#endif

namespace fexp {
namespace detail {

static inline uint32_t asuint(float f)     { return FEXP_BIT_CAST(uint32_t, f); }
static inline uint64_t asuint64(double f)  { return FEXP_BIT_CAST(uint64_t, f); }
static inline double   asdouble(uint64_t i){ return FEXP_BIT_CAST(double, i); }

// Top 12 bits of a float  (sign + exponent + top-3 mantissa).
static inline uint32_t top12f(float x)  { return asuint(x) >> 20; }
// Top 12 bits of a double (sign + exponent).
static inline uint32_t top12d(double x) { return static_cast<uint32_t>(asuint64(x) >> 52); }

// ─── flt-32 data (glibc sysdeps/ieee754/flt-32/e_exp2f_data.c) ──────────────

static constexpr int    EXP2F_BITS = 5;
static constexpr int    EXP2F_N    = 1 << EXP2F_BITS;  // 32

// N / ln2  (= log2e * N)
static constexpr double EXPF_INV_LN2N = 0x1.71547652b82fep+0 * EXP2F_N;
// Rounding magic: adding this to a double in [-150*N, 128*N] yields an integer
// in the lower bits without a branch.
static constexpr double EXPF_SHIFT    = 0x1.8p52;

// T[j] = asuint64(2^(j/N)) - (j << (52 - EXP2F_BITS))
// Adding (ki << (52-5)) back reconstructs the IEEE bits of 2^(k/N).
static constexpr uint64_t EXPF_T[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};

// poly_scaled[k] = coeff_k / N^(k+1), approximates 2^(r/N) for r in [-1/2, 1/2]
// Evaluation: z = C[0]*r + C[1]; r2 = r*r; y = C[2]*r + 1; y = z*r2 + y
static constexpr double EXPF_C[3] = {
    0x1.c6af84b912394p-5 / ((double)EXP2F_N * EXP2F_N * EXP2F_N),
    0x1.ebfce50fac4f3p-3 / ((double)EXP2F_N * EXP2F_N),
    0x1.62e42ff0c52d6p-1 /  (double)EXP2F_N,
};

// ─── dbl-64 data (glibc sysdeps/ieee754/dbl-64/e_exp_data.c) ────────────────

static constexpr int    EXP_BITS = 7;
static constexpr int    EXP_N    = 1 << EXP_BITS;  // 128

static constexpr double EXP_INV_LN2N     =  0x1.71547652b82fep0 * EXP_N;
static constexpr double EXP_NEG_LN2HI_N  = -0x1.62e42fefa0000p-8;   // -ln2_hi / N
static constexpr double EXP_NEG_LN2LO_N  = -0x1.cf79abc9e3b3ap-47;  // -ln2_lo / N
static constexpr double EXP_SHIFT        =  0x1.8p52;

// Minimax poly coefficients C2..C5  (abs error 1.555*2^-66, ulp error 0.509)
// Used as: tmp = tail + r + r2*(C2 + r*C3) + r2*r2*(C4 + r*C5)
static constexpr double EXP_C2 = 0x1.ffffffffffdbdp-2;
static constexpr double EXP_C3 = 0x1.555555555543cp-3;
static constexpr double EXP_C4 = 0x1.55555cf172b91p-5;
static constexpr double EXP_C5 = 0x1.1111167a4d017p-7;

// 128 pairs: EXP_TAB[2j]   = asuint64(tail_j)  (small error correction)
//            EXP_TAB[2j+1] = asuint64(H_j) - (j << (52 - EXP_BITS))
// H_j * 2^(ki >> EXP_BITS) approximates 2^(k/N).
static constexpr uint64_t EXP_TAB[256] = {
    0x0,                0x3ff0000000000000,
    0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335,
    0xbc7160139cd8dc5d, 0x3fefec9a3e778061,
    0xbc905e7a108766d1, 0x3fefe315e86e7f85,
    0x3c8cd2523567f613, 0x3fefd9b0d3158574,
    0xbc8bce8023f98efa, 0x3fefd06b29ddf6de,
    0x3c60f74e61e6c861, 0x3fefc74518759bc8,
    0x3c90a3e45b33d399, 0x3fefbe3ecac6f383,
    0x3c979aa65d837b6d, 0x3fefb5586cf9890f,
    0x3c8eb51a92fdeffc, 0x3fefac922b7247f7,
    0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2,
    0xbc6a033489906e0b, 0x3fef9b66affed31b,
    0xbc9556522a2fbd0e, 0x3fef9301d0125b51,
    0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc,
    0xbc91c923b9d5f416, 0x3fef829aaea92de0,
    0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51,
    0xbc801b15eaa59348, 0x3fef72b83c7d517b,
    0xbc8f1ff055de323d, 0x3fef6af9388c8dea,
    0x3c8b898c3f1353bf, 0x3fef635beb6fcb75,
    0xbc96d99c7611eb26, 0x3fef5be084045cd4,
    0x3c9aecf73e3a2f60, 0x3fef54873168b9aa,
    0xbc8fe782cb86389d, 0x3fef4d5022fcd91d,
    0x3c8a6f4144a6c38d, 0x3fef463b88628cd6,
    0x3c807a05b0e4047d, 0x3fef3f49917ddc96,
    0x3c968efde3a8a894, 0x3fef387a6e756238,
    0x3c875e18f274487d, 0x3fef31ce4fb2a63f,
    0x3c80472b981fe7f2, 0x3fef2b4565e27cdd,
    0xbc96b87b3f71085e, 0x3fef24dfe1f56381,
    0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1,
    0xbc3d219b1a6fbffa, 0x3fef187fd0dad990,
    0x3c8b3782720c0ab4, 0x3fef1285a6e4030b,
    0x3c6e149289cecb8f, 0x3fef0cafa93e2f56,
    0x3c834d754db0abb6, 0x3fef06fe0a31b715,
    0x3c864201e2ac744c, 0x3fef0170fc4cd831,
    0x3c8fdd395dd3f84a, 0x3feefc08b26416ff,
    0xbc86a3803b8e5b04, 0x3feef6c55f929ff1,
    0xbc924aedcc4b5068, 0x3feef1a7373aa9cb,
    0xbc9907f81b512d8e, 0x3feeecae6d05d866,
    0xbc71d1e83e9436d2, 0x3feee7db34e59ff7,
    0xbc991919b3ce1b15, 0x3feee32dc313a8e5,
    0x3c859f48a72a4c6d, 0x3feedea64c123422,
    0xbc9312607a28698a, 0x3feeda4504ac801c,
    0xbc58a78f4817895b, 0x3feed60a21f72e2a,
    0xbc7c2c9b67499a1b, 0x3feed1f5d950a897,
    0x3c4363ed60c2ac11, 0x3feece086061892d,
    0x3c9666093b0664ef, 0x3feeca41ed1d0057,
    0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0,
    0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de,
    0x3c7690cebb7aafb0, 0x3feebfdad5362a27,
    0x3c931dbdeb54e077, 0x3feebcb299fddd0d,
    0xbc8f94340071a38e, 0x3feeb9b2769d2ca7,
    0xbc87deccdc93a349, 0x3feeb6daa2cf6642,
    0xbc78dec6bd0f385f, 0x3feeb42b569d4f82,
    0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f,
    0x3c93350518fdd78e, 0x3feeaf4736b527da,
    0x3c7b98b72f8a9b05, 0x3feead12d497c7fd,
    0x3c9063e1e21c5409, 0x3feeab07dd485429,
    0x3c34c7855019c6ea, 0x3feea9268a5946b7,
    0x3c9432e62b64c035, 0x3feea76f15ad2148,
    0xbc8ce44a6199769f, 0x3feea5e1b976dc09,
    0xbc8c33c53bef4da8, 0x3feea47eb03a5585,
    0xbc845378892be9ae, 0x3feea34634ccc320,
    0xbc93cedd78565858, 0x3feea23882552225,
    0x3c5710aa807e1964, 0x3feea155d44ca973,
    0xbc93b3efbf5e2228, 0x3feea09e667f3bcd,
    0xbc6a12ad8734b982, 0x3feea012750bdabf,
    0xbc6367efb86da9ee, 0x3fee9fb23c651a2f,
    0xbc80dc3d54e08851, 0x3fee9f7df9519484,
    0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74,
    0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174,
    0xbc8619321e55e68a, 0x3fee9feb564267c9,
    0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f,
    0xbc7b32dcb94da51d, 0x3feea11473eb0187,
    0x3c94ecfd5467c06b, 0x3feea1ed0130c132,
    0x3c65ebe1abd66c55, 0x3feea2f336cf4e62,
    0xbc88a1c52fb3cf42, 0x3feea427543e1a12,
    0xbc9369b6f13b3734, 0x3feea589994cce13,
    0xbc805e843a19ff1e, 0x3feea71a4623c7ad,
    0xbc94d450d872576e, 0x3feea8d99b4492ed,
    0x3c90ad675b0e8a00, 0x3feeaac7d98a6699,
    0x3c8db72fc1f0eab4, 0x3feeace5422aa0db,
    0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c,
    0x3c7bf68359f35f44, 0x3feeb1ae99157736,
    0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6,
    0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5,
    0xbc6c23f97c90b959, 0x3feeba44cbc8520f,
    0xbc92434322f4f9aa, 0x3feebd829fde4e50,
    0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba,
    0x3c71affc2b91ce27, 0x3feec49182a3f090,
    0x3c6dd235e10a73bb, 0x3feec86319e32323,
    0xbc87c50422622263, 0x3feecc667b5de565,
    0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33,
    0xbc91bbd1d3bcbb15, 0x3feed503b23e255d,
    0x3c90cc319cee31d2, 0x3feed99e1330b358,
    0x3c8469846e735ab3, 0x3feede6b5579fdbf,
    0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a,
    0x3c8c1a7792cb3387, 0x3feee89f995ad3ad,
    0xbc907b8f4ad1d9fa, 0x3feeee07298db666,
    0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb,
    0xbc90a40e3da6f640, 0x3feef9728de5593a,
    0xbc68d6f438ad9334, 0x3feeff76f2fb5e47,
    0xbc91eee26b588a35, 0x3fef05b030a1064a,
    0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2,
    0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09,
    0x3c736eae30af0cb3, 0x3fef199bdd85529c,
    0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a,
    0x3c84e08fd10959ac, 0x3fef27f12e57d14b,
    0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5,
    0x3c676b2c6c921968, 0x3fef3720dcef9069,
    0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa,
    0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c,
    0xbc900dae3875a949, 0x3fef4f87080d89f2,
    0x3c74a385a63d07a7, 0x3fef5818dcfba487,
    0xbc82919e2040220f, 0x3fef60e316c98398,
    0x3c8e5a50d5c192ac, 0x3fef69e603db3285,
    0x3c843a59ac016b4b, 0x3fef7321f301b460,
    0xbc82d52107b43e1f, 0x3fef7c97337b9b5f,
    0xbc892ab93b470dc9, 0x3fef864614f5a129,
    0x3c74b604603a88d3, 0x3fef902ee78b3ff6,
    0x3c83c5ec519d7271, 0x3fef9a51fbc74c83,
    0xbc8ff7128fd391f0, 0x3fefa4afa2a490da,
    0xbc8dae98e223747d, 0x3fefaf482d8e67f1,
    0x3c8ec3bc41aa2008, 0x3fefba1bee615a27,
    0x3c842b94c3a9eb32, 0x3fefc52b376bba97,
    0x3c8a64a931d185ee, 0x3fefd0765b6e4540,
    0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14,
    0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8,
    0x3c5305c14160cc89, 0x3feff3c22b8f71f1,
};

} // namespace detail


// ─── Scalar float32  (glibc flt-32/e_expf.c) ─────────────────────────────────

[[nodiscard]] inline float expf(float x) noexcept {
    using namespace detail;

    // top12f strips sign; &0x7ff gives exp(8 bits) + top-3 mantissa bits.
    // top12f(88.0f) = 0x42B.  Inputs |x| >= 88 need special handling.
    const uint32_t abstop = top12f(x) & 0x7ffu;

    if (__builtin_expect(abstop >= 0x42Bu, 0)) {
        if (asuint(x) == asuint(-std::numeric_limits<float>::infinity()))
            return 0.0f;
        if (abstop >= 0x7f8u)           // NaN or +Inf
            return x + x;
        if (x > 0x1.62e42ep+6f)         // x > ln(2^128) ≈ 88.72
            return std::numeric_limits<float>::infinity();
        if (x < -0x1.9fe368p+6f)        // x < ln(2^-150) ≈ -103.97
            return 0.0f;
    }

    // x * N/ln2 = k + r,  r ∈ [-1/2, 1/2],  k integer
    double xd = static_cast<double>(x);
    double z  = EXPF_INV_LN2N * xd;

    // Shift trick: kd = round(z) as a double, ki encodes k in its low bits
    double   kd = z + EXPF_SHIFT;
    uint64_t ki = asuint64(kd);
    kd -= EXPF_SHIFT;

    double r = z - kd;

    // Reconstruct 2^(k/N) from table:
    //   EXPF_T[ki%N] + (ki << (52-5))  =  asuint64(2^(k/N))
    uint64_t t = EXPF_T[ki % (uint64_t)EXP2F_N];
    t += ki << (52 - EXP2F_BITS);
    double s = asdouble(t);

    // Degree-3 polynomial approximating 2^(r/N)
    double z2 = EXPF_C[0] * r + EXPF_C[1];
    double r2 = r * r;
    double y  = EXPF_C[2] * r + 1.0;
    y = z2 * r2 + y;
    y = y * s;

    return static_cast<float>(y);
}


// ─── Scalar float64  (glibc dbl-64/e_exp.c) ──────────────────────────────────

[[nodiscard]] inline double exp(double x) noexcept {
    using namespace detail;

    // top12d(0x1p-54) = 0x3C9,  top12d(512.0) = 0x408
    // Unsigned trick: catches abstop < 0x3C9 (tiny) and >= 0x408 (large/special).
    uint32_t abstop = top12d(x) & 0x7ffu;

    if (__builtin_expect(abstop - 0x3C9u >= 0x408u - 0x3C9u, 0)) {
        if (abstop - 0x3C9u >= 0x80000000u)
            return 1.0 + x;            // |x| tiny: exp(x) ≈ 1 + x

        if (abstop >= 0x409u) {        // |x| >= 1024, or NaN/Inf
            if (asuint64(x) == asuint64(-std::numeric_limits<double>::infinity()))
                return 0.0;
            if (abstop >= 0x7ffu)      // NaN or +Inf
                return x + x;
            if (asuint64(x) >> 63)
                return 0.0;            // large negative → underflow
            return std::numeric_limits<double>::infinity();  // overflow
        }
        // |x| in [512, 1024): computation is valid but sbits exponent may have
        // wrapped; set abstop=0 to trigger specialcase reconstruction below.
        abstop = 0;
    }

    // x * N/ln2 = k + r,  |r| <= ln2/(2N)
    double z  = EXP_INV_LN2N * x;

    double   kd = z + EXP_SHIFT;
    uint64_t ki = asuint64(kd);
    kd -= EXP_SHIFT;

    // Cody-Waite 2-part: r = x - k*(ln2/N),  negated constants pre-multiplied
    double r = x + kd * EXP_NEG_LN2HI_N + kd * EXP_NEG_LN2LO_N;

    // 2^(k/N) = scale * (1 + tail),  fetched from paired table
    uint64_t idx   = 2 * (ki % (uint64_t)EXP_N);
    uint64_t top   = ki << (52 - EXP_BITS);
    double   tail  = asdouble(EXP_TAB[idx]);
    uint64_t sbits = EXP_TAB[idx + 1] + top;

    // exp(r) - 1 ≈ r + C2*r^2 + C3*r^3 + C4*r^4 + C5*r^5
    double r2  = r * r;
    double tmp = tail + r + r2 * (EXP_C2 + r * EXP_C3)
                       + r2 * r2 * (EXP_C4 + r * EXP_C5);

    // specialcase: sbits exponent overflowed or underflowed
    if (__builtin_expect(abstop == 0, 0)) {
        if ((ki & 0x80000000u) == 0) {
            // k > 0: exponent in sbits overflowed by up to 460; correct by offset
            sbits -= 1009ull << 52;
            double sc = asdouble(sbits);
            return 0x1p1009 * (sc + sc * tmp);
        }
        // k < 0: underflow territory; scale up then down to avoid subnormals
        sbits += 1022ull << 52;
        double sc = asdouble(sbits);
        return 0x1p-1022 * (sc + sc * tmp);
    }

    double scale = asdouble(sbits);
    return scale + scale * tmp;
}


// ─── AVX2 stubs (scalar loop — not vectorised) ───────────────────────────────
// Kept so benchmark.cpp compiles without modification.

#ifdef __AVX2__

[[nodiscard]] inline __m256 fast_expf_avx2(__m256 x) noexcept {
    float in[8], out[8];
    _mm256_storeu_ps(in, x);
    for (int i = 0; i < 8; ++i) out[i] = expf(in[i]);
    return _mm256_loadu_ps(out);
}

[[nodiscard]] inline __m256d fast_exp_avx2(__m256d x) noexcept {
    double in[4], out[4];
    _mm256_storeu_pd(in, x);
    for (int i = 0; i < 4; ++i) out[i] = exp(in[i]);
    return _mm256_loadu_pd(out);
}

#endif // __AVX2__


// ─── Polynomial + range-reduction helpers (for isolated benchmarking) ─────────

// Range-reduce x to the fractional z-kd used by the float polynomial path.
[[nodiscard]] inline double expf_reduce(float x) noexcept {
    using namespace detail;
    double z  = EXPF_INV_LN2N * static_cast<double>(x);
    double kd = z + EXPF_SHIFT;
    kd       -= EXPF_SHIFT;
    return z - kd;
}

// Degree-3 polynomial approximating 2^(r/N_f).  Expects r from expf_reduce().
[[nodiscard]] inline float expf_poly(double r) noexcept {
    using namespace detail;
    double z2 = EXPF_C[0] * r + EXPF_C[1];
    double r2 = r * r;
    double y  = EXPF_C[2] * r + 1.0;
    return static_cast<float>(z2 * r2 + y);
}

// Range-reduce x to the Cody-Waite r used by the double polynomial path.
[[nodiscard]] inline double exp_reduce(double x) noexcept {
    using namespace detail;
    double z  = EXP_INV_LN2N * x;
    double kd = z + EXP_SHIFT;
    kd       -= EXP_SHIFT;
    return x + kd * EXP_NEG_LN2HI_N + kd * EXP_NEG_LN2LO_N;
}

// Degree-4 polynomial approximating exp(r)-1.  Expects r from exp_reduce().
[[nodiscard]] inline double exp_poly(double r) noexcept {
    using namespace detail;
    double r2 = r * r;
    return r + r2 * (EXP_C2 + r * EXP_C3) + r2 * r2 * (EXP_C4 + r * EXP_C5);
}

} // namespace fexp
