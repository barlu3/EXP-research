/* Benchmark: FRECPE as the range reduction inside float32 log(x), in cycles.

   ../frecpe/ asked whether FRECPE plus Newton steps beats FDIV as a division,
   and on Apple M5 it does not: FDIV is 7 cycles of latency and one per cycle,
   the same throughput as the bare estimate. That settles every elementary
   function that would use FRECPE as a division -- tan and cot as sin/cos, sec
   and csc, tanh, the e^-x inside sinh and cosh -- because the rest of the
   function is the same either way. sin and cos never divide at all.

   log is the one place the estimate is wanted for what it is rather than as a
   stand-in for 1/x. A table-driven log reduces x = 2^k * m against a reciprocal
   1/c from a table indexed by m's top bits, and needs only a few bits of it.
   FRECPE's 8-bit estimate is such a table in hardware, measured exhaustively
   here: for every normal x it is exactly 2^-k * ehat(i), i being x's top eight
   fraction bits, and x*FRECPE(x) - 1 is exact with magnitude at most 2^-8.45.
   It even applies the 2^-k, which a software table has to undo first.

   So the question: inside logf, is FRECPE cheaper than loading the same
   reciprocal from memory? The two hand-written kernels (logf-kernels.S) share
   everything else and return bit-identical results -- the accuracy table must
   show identical rows -- so any timing difference is the fetch alone. glibc
   2.44's logf, built from its own source in ../../gnu-libm/, is the baseline
   every ratio divides by, so the ratios do not move with this Mac's
   libSystem, whose logf stays as a row for comparison.

   All are held to 1 ulp of the correctly rounded log over every finite
   float32, and glibc's log10f to 0: it is CORE-MATH's, correctly rounded, so
   that row also checks the vendored build. Timing, epochs, the report and the
   accuracy pass are the shared driver's, ../../approx-bench.hpp, as is the
   --cache mode that times the kernels with the caches evicted between bursts;
   this file describes the kernels, their tables, the input clusters and the
   reference.

   CMakeLists.txt builds this file and logf-kernels.S three times: as they are
   for log, and with LOGF_BASE=2 and 10 for log2 and log10 (bench-log2f,
   bench-log10f), which the same reduction serves with a different table and
   two constants. Below, log means log in the build's base. */

#include "approx-bench.hpp"

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <mpfr.h>

#if !defined(LOGF_BASE)
#define LOGF_FN  "log"
#define REF_LOG  std::log
#define MPFR_LOG mpfr_log
#define GNU_ULP  1   // glibc's own bound: 0.818 ulp
#elif LOGF_BASE == 2
#define LOGF_FN  "log2"
#define REF_LOG  std::log2
#define MPFR_LOG mpfr_log2
#define GNU_ULP  1   // 0.752 ulp
#else
#define LOGF_FN  "log10"
#define REF_LOG  std::log10
#define MPFR_LOG mpfr_log10
#define GNU_ULP  0   // CORE-MATH: correctly rounded
#endif

extern "C" {
// Read by logf-kernels.S; filled by build_tables() before any kernel runs.
float  logf_recip[256];   // ehat(i)/2: the tab kernel's copy of FRECPE
double logf_logc[256];    // -log(ehat(i)), correctly rounded to double
double logf_base[2];      // log(2) and 1/ln(base), correctly rounded to double

void  logf_tput_gnu(const float* in, float* out, std::uint64_t n);
void  logf_tput_call(const float* in, float* out, std::uint64_t n);
void  logf_tput_tab(const float* in, float* out, std::uint64_t n);
void  logf_tput_frecpe(const float* in, float* out, std::uint64_t n);
float logf_lat_gnu(const float* in, std::uint64_t n);
float logf_lat_call(const float* in, std::uint64_t n);
float logf_lat_tab(const float* in, std::uint64_t n);
float logf_lat_frecpe(const float* in, std::uint64_t n);
}

namespace {

using approx::from_bits;

// ehat(i) is FRECPE of 1 + i/256: the estimate, scaled by 2^-k, for every x
// whose top fraction bits are i. Taken from the instruction itself, in the
// default FP mode the kernels run in. ehat(255) = 1/2 exactly, so its -log is
// log(2) rounded once -- the same double as logf_base[0], which is what makes
// y0 exactly 0 just below 1. 1/ln(base) is log(e), from e to 200 bits.
void build_tables() {
    mpfr_t v, e;
    mpfr_init2(v, 53);
    mpfr_init2(e, 200);
    for (std::uint32_t i = 0; i < 256; ++i) {
        const float ehat = vrecpes_f32(from_bits(0x3f800000u | i << 15));
        logf_recip[i] = ehat * 0.5f;
        mpfr_set_flt(v, ehat, MPFR_RNDN);
        MPFR_LOG(v, v, MPFR_RNDN);
        logf_logc[i] = -mpfr_get_d(v, MPFR_RNDN);
    }
    mpfr_set_ui(v, 2, MPFR_RNDN);
    MPFR_LOG(v, v, MPFR_RNDN);
    logf_base[0] = mpfr_get_d(v, MPFR_RNDN);
    mpfr_set_ui(e, 1, MPFR_RNDN);
    mpfr_exp(e, e, MPFR_RNDN);
    MPFR_LOG(v, e, MPFR_RNDN);
    logf_base[1] = mpfr_get_d(v, MPFR_RNDN);
    mpfr_clears(v, e, (mpfr_ptr)nullptr);
}

// Regions whose handling differs inside the kernels. Wide exponents exercise
// k*ln2; [0.5, 2) is where log changes sign and the result is smallest;
// [1, 1 + 2^-8) is the hand kernels' out-of-line path near 1; subnormals are
// scaled and rejoin the main path.
std::uint32_t positive_normal(std::mt19937& r) {
    return (1u + (std::uint32_t)r() % 254u) << 23 | ((std::uint32_t)r() & 0x007fffffu);
}
std::uint32_t two_binades(std::mt19937& r) {          // exponent 126 or 127
    return ((std::uint32_t)r() & 0x007fffffu) | ((126u + ((std::uint32_t)r() & 1u)) << 23);
}
std::uint32_t just_above_one(std::mt19937& r) {
    return 0x3f800000u | ((std::uint32_t)r() & 0x7fffu);
}
std::uint32_t subnormal(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x007fffffu; if (b) return b; }
}

const std::vector<approx::Cluster> CLUSTERS = {
    { "x > 0 normal",    positive_normal },
    { "x in [0.5,2)",    two_binades     },
    { "x in [1,1+2^-8)", just_above_one  },
    { "subnormal x",     subnormal       },
};

// Correctly rounded log(x). In double it is within 2^-52 of the truth, so
// rounding that to float is right unless it lies within that distance of a
// float rounding boundary; those inputs go to MPFR, at 24 bits so its own
// rounding is the float one. Special values follow C: +-0 -> -inf, x < 0 ->
// NaN, +inf -> +inf.
constexpr const char* REFERENCE_NOTE =
    "correctly rounded " LOGF_FN "(x) (double, MPFR near rounding boundaries)";
void reference(const float* in, float* out, std::size_t n) {
    thread_local struct Mp {
        mpfr_t v;
        Mp()  { mpfr_init2(v, 24); }
        ~Mp() { mpfr_clear(v); }
    } mp;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        if (std::isnan(x) || x < 0.0f) { out[i] = NAN; continue; }
        if (x == 0.0f)                 { out[i] = -INFINITY; continue; }
        if (std::isinf(x))             { out[i] = INFINITY; continue; }
        const double r = REF_LOG((double)x);
        float y = (float)r;
        if ((double)y != r) {
            const float  nb  = std::nextafter(y, r > y ? INFINITY : -INFINITY);
            const double mid = ((double)y + (double)nb) * 0.5;
            if (std::fabs(r - mid) <= 0x1p-50 * std::fabs(r)) {
                mpfr_set_flt(mp.v, x, MPFR_RNDN);
                MPFR_LOG(mp.v, mp.v, MPFR_RNDN);
                y = mpfr_get_flt(mp.v, MPFR_RNDN);
            }
        }
        out[i] = y;
    }
}

// Positive normal x. log(x) is then normal too, or 0 at x = 1.
constexpr const char* NORMAL_NOTE = "normal x > 0";
bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return (hi & 0x8000u) == 0 && e >= 1 && e <= 254;
}

// log in double is off by at most 2^-52 relative, far below anything reported.
double rel_error(float x, float y) { return std::fabs((double)y / REF_LOG((double)x) - 1.0); }

const approx::Spec SPEC = {
    .title       = "float32 " LOGF_FN "(x) on AArch64 -- FRECPE as the range reduction",
    .kernels     = "ARM-approx-research/benchmarks/scalar/logf/logf-kernels.S",
    .report_name = "bench_results_" LOGF_FN "f.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { LOGF_FN "f (glibc)", "glibc", GNU_ULP, false, logf_tput_gnu, logf_lat_gnu },
        { LOGF_FN "f (libm)", "libm",   1, false, logf_tput_call,   logf_lat_call   },
        { "table + frecpx", "tab",    1, false, logf_tput_tab,    logf_lat_tab    },
        { "frecpe",         "frecpe", 1, false, logf_tput_frecpe, logf_lat_frecpe },
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
