/* The input clusters and correctly rounded reference for float32 1/sqrt(x),
   shared by the scalar benchmark (benchmark-rsqrt.cpp) and its vector
   counterpart (../../vector/vrsqrts/benchmark-rsqrt-vec.cpp), so the two time the
   same inputs and are checked against the same answer. */

#ifndef RSQRT_REF_HPP
#define RSQRT_REF_HPP

#include "approx-bench.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <mpfr.h>

namespace rsqrt_ref {

// 1/sqrt is only real for x >= 0, so every cluster is non-negative. FRSQRTE
// reads the exponent's parity along with the top mantissa bits, so [1,4) spans
// both halves of its table. Subnormals are where the 12-bit mode flushes, and
// x >= 2^126 is where the textbook Newton grouping would have gone subnormal.
inline std::uint32_t non_negative(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x7fffffffu; if ((b >> 23) != 0xff) return b; }
}
inline std::uint32_t two_binades(std::mt19937& r) {   // exponent 127 or 128: [1, 4)
    return ((std::uint32_t)r() & 0x007fffffu) | ((127u + ((std::uint32_t)r() & 1u)) << 23);
}
inline std::uint32_t subnormal(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x007fffffu; if (b) return b; }
}
inline std::uint32_t huge(std::mt19937& r) {          // exponent 253 or 254
    return ((std::uint32_t)r() & 0x007fffffu) | ((253u + ((std::uint32_t)r() & 1u)) << 23);
}

inline const std::vector<approx::Cluster> CLUSTERS = {
    { "any x >= 0",     non_negative },
    { "x in [1,4)",     two_binades  },
    { "subnormal x",    subnormal    },
    { "x >= 2^126",     huge         },
};

// Correctly rounded 1/sqrt(x). In double it is within 2^-52 of the truth (two
// roundings of 2^-53), so rounding that to float is right unless it lies within
// that distance of a float rounding boundary. The rare inputs that do -- about
// a hundred of the 2^31 positive floats -- go to MPFR, which rounds correctly
// by construction. Checked against MPFR outright on 2.1M inputs spread over
// the whole positive range: no disagreement. Special values follow
// 1.0f / sqrtf(x): +-0 -> +-inf, x < 0 -> NaN.
inline constexpr const char* REFERENCE_NOTE =
    "correctly rounded 1/sqrt(x) (double, MPFR near rounding boundaries)";
inline void reference(const float* in, float* out, std::size_t n) {
    thread_local struct Mp {
        mpfr_t v;
        Mp()  { mpfr_init2(v, 24); }
        ~Mp() { mpfr_clear(v); }
    } mp;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        if (std::isnan(x) || x < 0.0f) { out[i] = NAN; continue; }
        if (x == 0.0f)                 { out[i] = std::copysign(INFINITY, x); continue; }
        if (std::isinf(x))             { out[i] = 0.0f; continue; }
        const double r = 1.0 / std::sqrt((double)x);
        float y = (float)r;
        if ((double)y != r) {
            const float  nb  = std::nextafter(y, r > y ? INFINITY : 0.0f);
            const double mid = ((double)y + (double)nb) * 0.5;
            if (std::fabs(r - mid) <= 0x1p-50 * r) {
                mpfr_set_flt(mp.v, x, MPFR_RNDN);
                mpfr_rec_sqrt(mp.v, mp.v, MPFR_RNDN);
                y = mpfr_get_flt(mp.v, MPFR_RNDN);
            }
        }
        out[i] = y;
    }
}

// Positive normal x; 1/sqrt(x) is then normal too.
inline constexpr const char* NORMAL_NOTE = "normal x > 0";
inline bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return (hi & 0x8000u) == 0 && e >= 1 && e <= 254;
}

// sqrt in double is off by at most 2^-53, far below anything reported.
inline double rel_error(float x, float y) { return std::fabs((double)y * std::sqrt((double)x) - 1.0); }

}  // namespace rsqrt_ref

#endif  // RSQRT_REF_HPP
