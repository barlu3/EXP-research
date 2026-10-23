/* Benchmark: FRECPE as the range reduction of the log inside float32 asinh(x),
   acosh(x) and log1p(x) = log(1 + x), in cycles.

   ../logf/ found FRECPE beats a table holding its own values as log's range
   reduction, because the table's index has to go from a float register to an
   integer one before the load, while FRECPE indexes on the FP side. There the
   log's argument is the input. In asinh and acosh it is computed, w = |x| +
   sqrt(x^2 +- 1) in double, so the question is whether the win survives a
   reduction on a double that arrives mid-function, behind a square root, with
   both kernels moving it to an integer register anyway for the other table.
   log1p is the plainest case of a computed argument: w = 1 + x, one add.

   The two hand-written kernels (asinhf-acoshf-kernels.S) share everything
   except how ehat(i) is fetched and return bit-identical results -- the
   accuracy table must show identical rows -- so any timing difference is the
   fetch alone. The system libm's function is the baseline every ratio divides
   by. All three are held to 1 ulp of the correctly rounded result over every
   finite float32.

   CMakeLists.txt builds this file three times: bench-asinhf, and with
   AHYP_ACOSH or AHYP_LOG1P defined, bench-acoshf and bench-log1pf. All link
   the one kernel file. Timing, epochs, the
   report and the accuracy pass are the shared driver's, ../../approx-bench.hpp;
   this file describes the kernels, their tables, the input clusters and the
   reference. */

#include "approx-bench.hpp"

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <mpfr.h>

extern "C" {
// Read by asinhf-acoshf-kernels.S; filled by build_tables() before any kernel runs.
double ahyp_recip[256];   // ehat(i)/2: the tab kernels' copy of FRECPE
double ahyp_logc[256];    // -log(ehat(i)), correctly rounded to double

void  asinhf_tput_call(const float* in, float* out, std::uint64_t n);
void  asinhf_tput_tab(const float* in, float* out, std::uint64_t n);
void  asinhf_tput_frecpe(const float* in, float* out, std::uint64_t n);
float asinhf_lat_call(const float* in, std::uint64_t n);
float asinhf_lat_tab(const float* in, std::uint64_t n);
float asinhf_lat_frecpe(const float* in, std::uint64_t n);
void  acoshf_tput_call(const float* in, float* out, std::uint64_t n);
void  acoshf_tput_tab(const float* in, float* out, std::uint64_t n);
void  acoshf_tput_frecpe(const float* in, float* out, std::uint64_t n);
float acoshf_lat_call(const float* in, std::uint64_t n);
float acoshf_lat_tab(const float* in, std::uint64_t n);
float acoshf_lat_frecpe(const float* in, std::uint64_t n);
void  log1pf_tput_call(const float* in, float* out, std::uint64_t n);
void  log1pf_tput_tab(const float* in, float* out, std::uint64_t n);
void  log1pf_tput_frecpe(const float* in, float* out, std::uint64_t n);
float log1pf_lat_call(const float* in, std::uint64_t n);
float log1pf_lat_tab(const float* in, std::uint64_t n);
float log1pf_lat_frecpe(const float* in, std::uint64_t n);
}

#if defined(AHYP_ACOSH)
#define AHYP_FN  "acosh"
#define REF_FN   std::acosh
#define MPFR_FN  mpfr_acosh
#define KERNEL(k) acoshf_##k
#elif defined(AHYP_LOG1P)
#define AHYP_FN  "log1p"
#define REF_FN   std::log1p
#define MPFR_FN  mpfr_log1p
#define KERNEL(k) log1pf_##k
#else
#define AHYP_FN  "asinh"
#define REF_FN   std::asinh
#define MPFR_FN  mpfr_asinh
#define KERNEL(k) asinhf_##k
#endif

namespace {

using approx::to_bits;

// ehat(i) is double FRECPE of 1 + i/256, taken from the instruction: the same
// 256 values as single FRECPE's, indexed the same way.
void build_tables() {
    mpfr_t v;
    mpfr_init2(v, 53);
    for (std::uint32_t i = 0; i < 256; ++i) {
        const double ehat = vrecped_f64(1.0 + i / 256.0);
        ahyp_recip[i] = ehat * 0.5;
        mpfr_set_d(v, ehat, MPFR_RNDN);
        mpfr_log(v, v, MPFR_RNDN);
        ahyp_logc[i] = -mpfr_get_d(v, MPFR_RNDN);
    }
    mpfr_clear(v);
}

// Uniform reals in [-2^e, 2^e), in steps of 2^(e-31): both signs, all paths.
template <int E> std::uint32_t uniform(std::mt19937& r) {
    return to_bits(std::ldexp((float)(std::int32_t)r(), E - 31));
}

#if defined(AHYP_ACOSH)
// Wide x exercises k*ln2 over w up to 2^22; [1, 2) is where results are
// smallest on the main path; [1, 1 + 2^-18) is the near-1 path, where neither
// kernel fetches anything, so the two must time the same -- a control. It has
// only 32 distinct inputs.
std::uint32_t wide(std::mt19937& r) {                 // 1 <= x < 2^21
    return (127u + (std::uint32_t)r() % 21u) << 23 | ((std::uint32_t)r() & 0x007fffffu);
}
std::uint32_t one_binade(std::mt19937& r) {           // 1 <= x < 2
    return 0x3f800000u | ((std::uint32_t)r() & 0x007fffffu);
}
std::uint32_t near_one(std::mt19937& r) {
    return 0x3f800000u | ((std::uint32_t)r() & 0x1fu);
}
const std::vector<approx::Cluster> CLUSTERS = {
    { "x in [1,2^21)",   wide       },
    { "x in [1,2)",      one_binade },
    { "x-1 < 2^-18",     near_one   },
};
#elif defined(AHYP_LOG1P)
// Wide x exercises k*ln2 over w up to 2^20; [-1, 1) is where log1p is used
// most, w from 0 to 2; |x| < 2^-8 is the polynomial on x out of line, where
// neither kernel fetches anything, so the two must time the same -- a control.
std::uint32_t wide(std::mt19937& r) {                 // 2^-8 <= x < 2^20
    return (119u + (std::uint32_t)r() % 28u) << 23 | ((std::uint32_t)r() & 0x007fffffu);
}
const std::vector<approx::Cluster> CLUSTERS = {
    { "2^-8<=x<2^20",   wide        },
    { "x in [-1,1)",    uniform<0>  },
    { "|x| < 2^-8",     uniform<-8> },
};
#else
// Wide |x| exercises k*ln2 over w up to 2^21; [-2, 2) is where most of the
// interesting results lie, both signs at random; |x| < 2^-8 is the polynomial
// out of line, where neither kernel fetches anything, so the two must time the
// same -- a control.
std::uint32_t wide(std::mt19937& r) {                 // 2^-8 <= |x| < 2^20
    return ((std::uint32_t)r() & 0x80000000u) | (119u + (std::uint32_t)r() % 28u) << 23 |
           ((std::uint32_t)r() & 0x007fffffu);
}
const std::vector<approx::Cluster> CLUSTERS = {
    { "2^-8<=|x|<2^20", wide        },
    { "x in [-2,2)",    uniform<1>  },
    { "|x| < 2^-8",     uniform<-8> },
};
#endif

// Correctly rounded f(x). libm's double asinh, acosh and log1p are within an
// ulp or two of the truth, so rounding to float is right unless the double lies
// within 2^-46 of a float rounding boundary; those go to MPFR, at 24 bits so
// its own rounding is the float one. Results are never subnormal near a
// boundary: asinh or log1p of a tiny x is x itself in double. Special values follow C.
constexpr const char* REFERENCE_NOTE =
    "correctly rounded " AHYP_FN "(x) (double, MPFR near rounding boundaries)";
void reference(const float* in, float* out, std::size_t n) {
    thread_local struct Mp {
        mpfr_t v;
        Mp()  { mpfr_init2(v, 24); }
        ~Mp() { mpfr_clear(v); }
    } mp;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = REF_FN((double)in[i]);
        float y = (float)r;
        if (std::isfinite(r) && (double)y != r) {
            const float  nb  = std::nextafter(y, r > y ? INFINITY : -INFINITY);
            const double mid = ((double)y + (double)nb) * 0.5;
            if (std::fabs(r - mid) <= 0x1p-46 * std::fabs(r)) {
                mpfr_set_flt(mp.v, in[i], MPFR_RNDN);
                MPFR_FN(mp.v, mp.v, MPFR_RNDN);
                y = mpfr_get_flt(mp.v, MPFR_RNDN);
            }
        }
        out[i] = y;
    }
}

#if defined(AHYP_ACOSH)
constexpr const char* NORMAL_NOTE = "normal x >= 1";
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return (hi & 0x8000u) == 0 && e >= 127 && e <= 254;
}
#elif defined(AHYP_LOG1P)
constexpr const char* NORMAL_NOTE = "normal x > -1";
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return e >= 1 && (hi & 0x8000u ? e <= 126 : e <= 254);
}
#else
constexpr const char* NORMAL_NOTE = "normal x";
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return e >= 1 && e <= 254;
}
#endif

// asinh, acosh and log1p in double are off by a few 2^-53 relative, far below anything reported.
double rel_error(float x, float y) { return std::fabs((double)y / REF_FN((double)x) - 1.0); }

const approx::Spec SPEC = {
    .title       = "float32 " AHYP_FN "(x) on AArch64 -- FRECPE as the log's range reduction",
    .kernels     = "ARM-approx-research/benchmarks/scalar/asinhf-acoshf/asinhf-acoshf-kernels.S",
    .report_name = "bench_results_" AHYP_FN "f.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { AHYP_FN "f (libm)", "libm",   1, false, KERNEL(tput_call),   KERNEL(lat_call)   },
        { "table + frecpx",   "tab",    1, false, KERNEL(tput_tab),    KERNEL(lat_tab)    },
        { "frecpe",           "frecpe", 1, false, KERNEL(tput_frecpe), KERNEL(lat_frecpe) },
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
    build_tables();
    return approx::run_main(SPEC, argc, argv);
}
