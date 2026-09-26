/* Benchmark: ways to compute 1/sqrt(x) in float32 on AArch64, in cycles.

   The reciprocal square root counterpart of ../benchmark-recip.cpp, asking the
   same question: is the estimate pipeline (FRSQRTE, then Newton steps) cheaper
   than the exact instruction sequence, at what accuracy, over every real
   float32 input?

   There is no exact reciprocal-square-root instruction to compare against.
   The exact route is FSQRT then FDIV -- what clang emits for 1.0f / sqrtf(x) --
   and it rounds twice, so it is faithful rather than correctly rounded. That
   makes the accuracy reference a correctly rounded 1/sqrt computed here, not
   the baseline kernel. Checked exhaustively over every finite float against it:

     FSQRT + FDIV                    within 1 ulp; 26% of positive inputs are
                                     1 ulp off
     FRSQRTE (8-bit)                 2^-8.25 relative on normals
     ... + 1 Newton step             2^-15.9, up to 195 ulp; +0 -> -inf
     ... + 2 Newton steps (est2)     within 2 ulp everywhere, NaN for x < 0,
                                     +-inf at +-0 -- the float32 contender
     FRSQRTE (12-bit, FPCR.AH)       2^-11.58 relative on normals
     ... + 1 Newton step             up to 4 ulp on normals; under AH every
                                     subnormal input is flushed, so positive
                                     subnormals come back +inf

   The Newton step's grouping is not the textbook one; rsqrt-kernels.S explains
   why (e*e overflows or underflows at the ends of the range, and 0*inf needs
   FMULX).

   The baseline is held to 1 ulp and est2 to 2; the rest are reported, never
   checked. Timing, epochs, the report and the accuracy pass are shared with
   the reciprocal benchmark in ../approx-bench.hpp; this file describes only the
   kernels, the input clusters and the reference. */

#include "approx-bench.hpp"

#include <cmath>
#include <cstdint>
#include <random>

#include <mpfr.h>

extern "C" {
void  rsqrt_tput_sqdiv(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_call(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est0(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est1(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est2(const float* in, float* out, std::uint64_t n);
float rsqrt_lat_sqdiv(const float* in, std::uint64_t n);
float rsqrt_lat_call(const float* in, std::uint64_t n);
float rsqrt_lat_est0(const float* in, std::uint64_t n);
float rsqrt_lat_est1(const float* in, std::uint64_t n);
float rsqrt_lat_est2(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

// 1/sqrt is only real for x >= 0, so every cluster is non-negative. FRSQRTE
// reads the exponent's parity along with the top mantissa bits, so [1,4) spans
// both halves of its table. Subnormals are where the 12-bit mode flushes, and
// x >= 2^126 is where the textbook Newton grouping would have gone subnormal.
std::uint32_t non_negative(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x7fffffffu; if ((b >> 23) != 0xff) return b; }
}
std::uint32_t two_binades(std::mt19937& r) {   // exponent 127 or 128: [1, 4)
    return ((std::uint32_t)r() & 0x007fffffu) | ((127u + ((std::uint32_t)r() & 1u)) << 23);
}
std::uint32_t subnormal(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x007fffffu; if (b) return b; }
}
std::uint32_t huge(std::mt19937& r) {          // exponent 253 or 254
    return ((std::uint32_t)r() & 0x007fffffu) | ((253u + ((std::uint32_t)r() & 1u)) << 23);
}

// Correctly rounded 1/sqrt(x). In double it is within 2^-52 of the truth (two
// roundings of 2^-53), so rounding that to float is right unless it lies within
// that distance of a float rounding boundary. The rare inputs that do -- about
// a hundred of the 2^31 positive floats -- go to MPFR, which rounds correctly
// by construction. Checked against MPFR outright on 2.1M inputs spread over
// the whole positive range: no disagreement. Special values follow
// 1.0f / sqrtf(x): +-0 -> +-inf, x < 0 -> NaN.
void reference(const float* in, float* out, std::size_t n) {
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
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return (hi & 0x8000u) == 0 && e >= 1 && e <= 254;
}

// sqrt in double is off by at most 2^-53, far below anything reported.
double rel_error(float x, float y) { return std::fabs((double)y * std::sqrt((double)x) - 1.0); }

const approx::Spec SPEC = {
    .title       = "float32 1/sqrt(x) on AArch64 -- FSQRT+FDIV vs FRSQRTE (8-, 12-bit)",
    .kernels     = "ARM-approx-research/benchmarks/recip-sqrt/rsqrt-kernels.S",
    .report_path = "output/bench_results_rsqrt.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "fsqrt + fdiv",   "sqdiv",    1,         false, rsqrt_tput_sqdiv, rsqrt_lat_sqdiv },
        { "sqrt+div call",  "call",     1,         false, rsqrt_tput_call,  rsqrt_lat_call  },
        { "frsqrte8 only",  "fe8",      UNCHECKED, false, rsqrt_tput_est0,  rsqrt_lat_est0  },
        { "frsqrte8+1 NR",  "fe8+1NR",  UNCHECKED, false, rsqrt_tput_est1,  rsqrt_lat_est1  },
        { "frsqrte8+2 NR",  "fe8+2NR",  2,         false, rsqrt_tput_est2,  rsqrt_lat_est2  },
        { "frsqrte12 only", "fe12",     UNCHECKED, true,  rsqrt_tput_est0,  rsqrt_lat_est0  },
        { "frsqrte12+1 NR", "fe12+1NR", UNCHECKED, true,  rsqrt_tput_est1,  rsqrt_lat_est1  },
    },
    .clusters = {
        { "any x >= 0",     non_negative },
        { "x in [1,4)",     two_binades  },
        { "subnormal x",    subnormal    },
        { "x >= 2^126",     huge         },
    },
    .reference_note = "correctly rounded 1/sqrt(x) (double, MPFR near rounding boundaries)",
    .reference      = reference,
    .normal_block   = normal_block,
    .normal_note    = "normal x > 0",
    .rel_error      = rel_error,
};

}  // namespace

int main(int argc, char** argv) { return approx::run_main(SPEC, argc, argv); }
