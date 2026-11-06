/* The correctly rounded reference for float32 log, log2 and log10 (LOGF_BASE
   unset, 2 or 10), shared by the benchmark (benchmark-logf.cpp) and the table
   tolerance study (../../../table-tolerance/), so the two are held to the same
   answer. Below, log means log in the build's base. */

#ifndef LOGF_REF_HPP
#define LOGF_REF_HPP

#include <cmath>
#include <cstddef>

#include <mpfr.h>

#if !defined(LOGF_BASE)
#define LOGF_FN  "log"
#define REF_LOG  std::log
#define MPFR_LOG mpfr_log
#elif LOGF_BASE == 2
#define LOGF_FN  "log2"
#define REF_LOG  std::log2
#define MPFR_LOG mpfr_log2
#else
#define LOGF_FN  "log10"
#define REF_LOG  std::log10
#define MPFR_LOG mpfr_log10
#endif

namespace logf_ref {

// Correctly rounded log(x). In double it is within 2^-52 of the truth, so
// rounding that to float is right unless it lies within that distance of a
// float rounding boundary; those inputs go to MPFR, at 24 bits so its own
// rounding is the float one. Special values follow C: +-0 -> -inf, x < 0 ->
// NaN, +inf -> +inf.
constexpr const char* REFERENCE_NOTE =
    "correctly rounded " LOGF_FN "(x) (double, MPFR near rounding boundaries)";
inline void reference(const float* in, float* out, std::size_t n) {
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

}  // namespace logf_ref

#endif  // LOGF_REF_HPP
