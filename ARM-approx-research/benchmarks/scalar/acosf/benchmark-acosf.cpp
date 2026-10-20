/* Benchmark: FRSQRTE as the square root inside float32 acos(x), in cycles.

   ../frecpe/ found FDIV as fast as FRECPE on Apple M5, which rules the
   estimate out wherever it would stand in for a division. ../frsqrte/ found
   the other side: FRSQRTE plus two Newton steps reaches 1/sqrt(x) at 1.84
   cycles per result, against 3.0 for FSQRT + FDIV. So the functions left to
   ask about are the ones that need a square root. acos and asin are the
   plainest: for |x| > 1/2 both reduce to sqrt((1 - |x|)/2), and below that
   they are a polynomial.

   But acos needs sqrt(z), not 1/sqrt(z): one FSQRT, against FRSQRTE's
   estimate, two Newton steps and z * rsqrt(z). The question: inside acosf, at
   full 1-ulp accuracy, is that cheaper than FSQRT? The two hand-written kernels
   (acosf-kernels.S) share everything except the square root, so any timing
   difference is the square root alone. On |x| <= 1/2 they run the same
   instructions, so that cluster is a control: the two must time the same. The
   system libm's acosf is the baseline every ratio divides by.

   All three are held to 1 ulp of the correctly rounded acos over every finite
   float32. Timing, epochs, the report and the accuracy pass are the shared
   driver's, ../../approx-bench.hpp; this file describes the kernels, the input
   clusters and the reference. */

#include "approx-bench.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <mpfr.h>

extern "C" {
void  acosf_tput_call(const float* in, float* out, std::uint64_t n);
void  acosf_tput_fsqrt(const float* in, float* out, std::uint64_t n);
void  acosf_tput_frsqrte(const float* in, float* out, std::uint64_t n);
float acosf_lat_call(const float* in, std::uint64_t n);
float acosf_lat_fsqrt(const float* in, std::uint64_t n);
float acosf_lat_frsqrte(const float* in, std::uint64_t n);
}

namespace {

using approx::to_bits;

// Uniform reals, not uniform bit patterns, which would be almost all tiny |x|.
// [-1, 1) takes either path at random, but the 4096-element buffer repeats and
// the branch predictor learns it, so that row shows no mispredicts
// (docs/BENCHMARKING.md, section 10.8). The other two stay on one path each.
// All three use both signs.
std::uint32_t whole_domain(std::mt19937& r) {        // [-1, 1) in steps of 2^-24
    return to_bits(std::ldexp((float)((std::int32_t)r() >> 7), -24));
}
std::uint32_t polynomial_only(std::mt19937& r) {     // [-1/2, 1/2) in steps of 2^-24
    return to_bits(std::ldexp((float)((std::int32_t)r() >> 8), -24));
}
std::uint32_t needs_sqrt(std::mt19937& r) {          // |x| in [1/2, 1): one binade
    return ((std::uint32_t)r() & 0x807fffffu) | 0x3f000000u;
}

const std::vector<approx::Cluster> CLUSTERS = {
    { "x in [-1,1)",     whole_domain    },
    { "x in [-1/2,1/2)", polynomial_only },
    { "|x| in [1/2,1)",  needs_sqrt      },
};

// Correctly rounded acos(x). In double it is within 2^-52 of the truth, so
// rounding that to float is right unless it lies within that distance of a
// float rounding boundary; those inputs go to MPFR, at 24 bits so its own
// rounding is the float one. acos is at least 2^-11.5 for any float below 1,
// so no result is subnormal. Special values follow C: |x| > 1 and NaN -> NaN,
// acos(1) = +0.
constexpr const char* REFERENCE_NOTE =
    "correctly rounded acos(x) (double, MPFR near rounding boundaries)";
void reference(const float* in, float* out, std::size_t n) {
    thread_local struct Mp {
        mpfr_t v;
        Mp()  { mpfr_init2(v, 24); }
        ~Mp() { mpfr_clear(v); }
    } mp;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        if (!(std::fabs(x) <= 1.0f)) { out[i] = NAN; continue; }
        const double r = std::acos((double)x);
        float y = (float)r;
        if ((double)y != r) {
            const float  nb  = std::nextafter(y, r > y ? INFINITY : -INFINITY);
            const double mid = ((double)y + (double)nb) * 0.5;
            if (std::fabs(r - mid) <= 0x1p-50 * std::fabs(r)) {
                mpfr_set_flt(mp.v, x, MPFR_RNDN);
                mpfr_acos(mp.v, mp.v, MPFR_RNDN);
                y = mpfr_get_flt(mp.v, MPFR_RNDN);
            }
        }
        out[i] = y;
    }
}

// Normal x with |x| < 1. acos(x) is then normal too, and never 0.
constexpr const char* NORMAL_NOTE = "normal x, |x| < 1";
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return e >= 1 && e <= 126;
}

// acos in double is off by at most 2^-52 relative, far below anything reported.
double rel_error(float x, float y) { return std::fabs((double)y / std::acos((double)x) - 1.0); }

const approx::Spec SPEC = {
    .title       = "float32 acos(x) on AArch64 -- FSQRT vs FRSQRTE for its sqrt",
    .kernels     = "ARM-approx-research/benchmarks/scalar/acosf/acosf-kernels.S",
    .report_name = "bench_results_acosf.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "acosf (libm)",   "libm",    1, false, acosf_tput_call,    acosf_lat_call    },
        { "fsqrt",          "fsqrt",   1, false, acosf_tput_fsqrt,   acosf_lat_fsqrt   },
        { "frsqrte + 2 NR", "frsqrte", 1, false, acosf_tput_frsqrte, acosf_lat_frsqrte },
    },
    .clusters       = CLUSTERS,
    .reference_note = REFERENCE_NOTE,
    .reference      = reference,
    .normal_block   = normal_block,
    .normal_note    = NORMAL_NOTE,
    .rel_error      = rel_error,
};

}  // namespace

int main(int argc, char** argv) {
    return approx::run_main(SPEC, argc, argv);
}
