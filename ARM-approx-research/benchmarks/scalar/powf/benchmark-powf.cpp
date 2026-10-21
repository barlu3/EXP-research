/* Benchmark: FRECPE as the range reduction inside float32 pow(x, y), in cycles.

   ../logf/ found FRECPE beats a table holding its own values as log's range
   reduction: about 8 cycles of latency, because the table's index has to go
   from a float register to an integer one before the load, while FRECPE
   indexes on the FP side and applies the 2^-k itself. exp, sin and cos have no
   reciprocal in their reduction, so pow = exp2(y * log2 x) is where that win
   could carry into a second function.

   pow needs log2(x) to more precision than logf: y * log2(x) multiplies its
   error by up to 150 on any result that rounds to neither 0 nor inf. So the
   kernels (powf-kernels.S) compute log2 in double to a relative 2^-34.3, which
   FRECPE's exact float32 remainder r = x*FRECPE(x) - 1 makes cheap: the only
   errors are the polynomial's truncation and the table's rounding. The two
   hand-written kernels share everything except how ehat(i) is fetched and
   return bit-identical results -- the accuracy tables must show identical rows
   -- so any timing difference is the fetch alone. The system libm's powf is
   the baseline every ratio divides by.

   Two arguments, one Spec: y is fixed per cluster (Cluster::enter writes
   powf_y, which every kernel reads once per call). That is the smallest change
   to the shared driver -- one hook, no new kernel signature or loop shape --
   and costs pow nothing it needs: the reduction under test depends on x
   alone, and a constant exponent is the common case (gamma 2.2, cube roots).
   Its price is coverage of y, so the accuracy sample is every finite x against
   each cluster's y (4 x 4,278,190,080 pairs), plus a pre-flight check of
   y = +-0, +-inf, NaN and the integer edges against every special x and 2^16
   sampled ones, which have nothing to time. All three variants are held to
   1 ulp of the correctly rounded result throughout.

   Timing, epochs, the report and the accuracy pass are the shared driver's,
   ../../approx-bench.hpp; this file describes the kernels, their tables, the
   input clusters and the reference. */

#include "approx-bench.hpp"

#include <arm_neon.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <mpfr.h>

extern "C" {
// Read by powf-kernels.S; filled by build_tables() and the clusters' enter.
float         powf_y;             // the second argument, fixed per kernel call
float         powf_recip[256];    // ehat(i)/2: the tab kernel's copy of FRECPE
double        powf_log2c[256];    // -log2(ehat(i)), correctly rounded to double
std::uint64_t powf_exp2[32];      // bits(2^(j/32)) - (j << 47)
double        powf_poly[7];       // c1-c4: log2(1+r); e1-e3: 2^(rr/32) - 1

void  powf_tput_call(const float* in, float* out, std::uint64_t n);
void  powf_tput_tab(const float* in, float* out, std::uint64_t n);
void  powf_tput_frecpe(const float* in, float* out, std::uint64_t n);
float powf_lat_call(const float* in, std::uint64_t n);
float powf_lat_tab(const float* in, std::uint64_t n);
float powf_lat_frecpe(const float* in, std::uint64_t n);
}

namespace {

using approx::from_bits;
using approx::to_bits;

// ehat(i) as in ../logf/: FRECPE of 1 + i/256, taken from the instruction.
// ehat(255) = 1/2, so -log2 of it is exactly 1 and y0 = k + 1 = 0 just below 1.
// The polynomial coefficients are Taylor's, rounded once from 200 bits.
void build_tables() {
    mpfr_t v, w, ln2;
    mpfr_inits2(200, w, ln2, (mpfr_ptr)nullptr);
    mpfr_init2(v, 53);
    for (std::uint32_t i = 0; i < 256; ++i) {
        const float ehat = vrecpes_f32(from_bits(0x3f800000u | i << 15));
        powf_recip[i] = ehat * 0.5f;
        mpfr_set_flt(v, ehat, MPFR_RNDN);
        mpfr_log2(v, v, MPFR_RNDN);
        powf_log2c[i] = -mpfr_get_d(v, MPFR_RNDN);
    }
    for (std::uint32_t j = 0; j < 32; ++j) {
        mpfr_set_ui(w, j, MPFR_RNDN);
        mpfr_div_2ui(w, w, 5, MPFR_RNDN);
        mpfr_exp2(v, w, MPFR_RNDN);
        const double e = mpfr_get_d(v, MPFR_RNDN);
        std::uint64_t b;
        std::memcpy(&b, &e, 8);
        powf_exp2[j] = b - ((std::uint64_t)j << 47);
    }
    mpfr_const_log2(ln2, MPFR_RNDN);
    for (int j = 1; j <= 4; ++j) {                      // (-1)^(j+1) / (j ln2)
        mpfr_mul_ui(w, ln2, j, MPFR_RNDN);
        mpfr_si_div(w, j % 2 ? 1 : -1, w, MPFR_RNDN);
        powf_poly[j - 1] = mpfr_get_d(w, MPFR_RNDN);
    }
    mpfr_div_2ui(ln2, ln2, 5, MPFR_RNDN);               // ln2/32
    mpfr_set_ui(w, 1, MPFR_RNDN);
    for (int j = 1; j <= 3; ++j) {                      // (ln2/32)^j / j!
        mpfr_mul(w, w, ln2, MPFR_RNDN);
        mpfr_div_ui(w, w, j, MPFR_RNDN);
        powf_poly[3 + j] = mpfr_get_d(w, MPFR_RNDN);
    }
    mpfr_clears(v, w, ln2, (mpfr_ptr)nullptr);
}

// One y per cluster, and x where that y keeps pow on the path worth timing.
// y = 2.2 over wide x: the main path, results from 2^-125 to 2^127.6.
// y = -1/3 over [0.5, 2): where log2 changes sign, results near 1.
// y = 3 over x < 0: an odd integer, so every element takes the negating copy.
// y = -20000 over [1, 1 + 2^-8): the out-of-line path near 1, where neither
// hand kernel fetches anything, so the two must time the same -- a control.
// Its accuracy pass also drives log2's worst relative error, near 1, to the
// largest amplification a finite result allows.
void y_gamma()   { powf_y = 2.2f; }
void y_cbrt()    { powf_y = -1.0f / 3.0f; }
void y_cube()    { powf_y = 3.0f; }
void y_steep()   { powf_y = -20000.0f; }

std::uint32_t wide_positive(std::mt19937& r) {       // 2^-57 <= x < 2^58
    return (70u + (std::uint32_t)r() % 115u) << 23 | ((std::uint32_t)r() & 0x007fffffu);
}
std::uint32_t two_binades(std::mt19937& r) {         // exponent 126 or 127
    return ((std::uint32_t)r() & 0x007fffffu) | ((126u + ((std::uint32_t)r() & 1u)) << 23);
}
std::uint32_t wide_negative(std::mt19937& r) {       // -2^41 < x <= -2^-40
    return 0x80000000u | (87u + (std::uint32_t)r() % 81u) << 23 | ((std::uint32_t)r() & 0x007fffffu);
}
std::uint32_t just_above_one(std::mt19937& r) {
    return 0x3f800000u | ((std::uint32_t)r() & 0x7fffu);
}

const std::vector<approx::Cluster> CLUSTERS = {
    { "y=2.2, x>0",    wide_positive,  y_gamma },
    { "y=-1/3,[.5,2)", two_binades,    y_cbrt  },
    { "y=3, x<0",      wide_negative,  y_cube  },
    { "y=-20000, x~1", just_above_one, y_steep },
};

// Correctly rounded pow(x, y). libm's double pow is within an ulp of the
// truth and follows C's special cases, so rounding it to float is right unless
// it lies within 2^-45 of a float rounding boundary, the overflow threshold
// among them; those go to MPFR at 200 bits, rounded once to float, subnormals
// included. A double that is NaN, +-inf or +-0 rounds the same as the truth.
constexpr const char* REFERENCE_NOTE =
    "correctly rounded pow(x, y) (double, MPFR near rounding boundaries)";
void reference(const float* in, float* out, std::size_t n) {
    thread_local struct Mp {
        mpfr_t x, y, r;
        Mp()  { mpfr_inits2(200, x, y, r, (mpfr_ptr)nullptr); }
        ~Mp() { mpfr_clears(x, y, r, (mpfr_ptr)nullptr); }
    } mp;
    const float y = powf_y;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = std::pow((double)in[i], (double)y);
        float f = (float)r;
        if (std::isfinite(r) && r != 0.0) {
            bool near = std::fabs(std::fabs(r) - 0x1.ffffffp127) <= 0x1p-45 * std::fabs(r);
            if (!near && (double)f != r && std::isfinite(f)) {
                const float  nb  = std::nextafter(f, r > f ? INFINITY : -INFINITY);
                const double mid = ((double)f + (double)nb) * 0.5;
                near = std::fabs(r - mid) <= 0x1p-45 * std::fabs(r);
            }
            if (near) {
                mpfr_set_flt(mp.x, in[i], MPFR_RNDN);
                mpfr_set_flt(mp.y, y, MPFR_RNDN);
                mpfr_pow(mp.r, mp.x, mp.y, MPFR_RNDN);
                f = mpfr_get_flt(mp.r, MPFR_RNDN);
            }
        }
        out[i] = f;
    }
}

// Positive normal x whose whole binade maps to normal results under this y.
constexpr const char* NORMAL_NOTE = "normal x > 0 whose binade gives normal results";
bool normal_block(std::uint32_t hi) {
    const int e = (int)((hi >> 7) & 0xff);
    if ((hi & 0x8000u) || e < 1 || e > 254) return false;
    const double a = (e - 127) * (double)powf_y, b = (e - 126) * (double)powf_y;
    return std::min(a, b) >= -125.0 && std::max(a, b) <= 127.0;
}

// pow in double is off by at most 2^-52 relative, far below anything reported.
double rel_error(float x, float y) {
    return std::fabs((double)y / std::pow((double)x, (double)powf_y) - 1.0);
}

const approx::Spec SPEC = {
    .title       = "float32 pow(x, y) on AArch64 -- FRECPE as log2's range reduction",
    .kernels     = "ARM-approx-research/benchmarks/scalar/powf/powf-kernels.S",
    .report_name = "bench_results_powf.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "powf (libm)",    "libm",   1, false, powf_tput_call,   powf_lat_call   },
        { "table + frecpx", "tab",    1, false, powf_tput_tab,    powf_lat_tab    },
        { "frecpe",         "frecpe", 1, false, powf_tput_frecpe, powf_lat_frecpe },
    },
    .clusters       = CLUSTERS,
    .reference_note = REFERENCE_NOTE,
    .reference      = reference,
    .normal_block   = normal_block,
    .normal_note    = NORMAL_NOTE,
    .rel_error      = rel_error,
};

// The y no cluster can time: each against every special x and 2^16 sampled
// bit patterns, to the same bounds as the accuracy pass. Run before anything
// is timed; a failure names the first wrong result per variant and y.
bool special_y_ok() {
    const float YS[] = { 0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 0.5f,
                         0x1p24f, 0x1.fffffep23f, -0x1p25f, 0x1p-149f, -0x1p127f };
    const float XS[] = { 0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f,
                         0x1p-149f, -0x1p-149f, FLT_MAX, -FLT_MAX, 0.5f, -0.5f, 2.0f, -2.0f };
    std::vector<float> in(1u << 16), ref(in.size()), out(in.size());
    std::mt19937 rng(7);
    for (auto& x : in) x = from_bits((std::uint32_t)rng());
    std::copy(std::begin(XS), std::end(XS), in.begin());
    bool ok = true;
    for (const float y : YS) {
        powf_y = y;
        reference(in.data(), ref.data(), in.size());
        for (std::size_t v = 1; v < SPEC.variants.size(); ++v) {
            SPEC.variants[v].tput(in.data(), out.data(), in.size());
            for (std::size_t i = 0; i < in.size(); ++i) {
                if (std::isnan(out[i]) && std::isnan(ref[i])) continue;
                const std::int64_t d = approx::ordered(to_bits(out[i])) - approx::ordered(to_bits(ref[i]));
                if (std::isnan(out[i]) || std::isnan(ref[i]) || std::llabs(d) > SPEC.variants[v].max_ulp) {
                    std::fprintf(stderr, "special y %a: %s gives %a for x = %a, want %a\n",
                                 y, SPEC.variants[v].name, out[i], in[i], ref[i]);
                    ok = false;
                    break;
                }
            }
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    build_tables();
    if (!bench::is_epoch_child(argc, argv) && !special_y_ok()) return 1;
    return approx::run_main(SPEC, argc, argv);
}
