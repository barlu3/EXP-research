/* The correctly rounded reference for float32 asinh, acosh and log1p, shared by
   the benchmark (benchmark-asinhf-acoshf.cpp) and the table tolerance study
   (../../../table-tolerance/), so the two are held to the same answer. */

#ifndef AHYP_REF_HPP
#define AHYP_REF_HPP

#include <cmath>
#include <cstddef>

#include <mpfr.h>

namespace ahyp_ref {

// Correctly rounded f(x), f being `ref` in double and `mp` in MPFR. libm's
// double asinh, acosh and log1p are within an ulp or two of the truth, so
// rounding to float is right unless the double lies within 2^-46 of a float
// rounding boundary; those go to MPFR, at 24 bits so its own rounding is the
// float one. Results are never subnormal near a boundary: asinh or log1p of a
// tiny x is x itself in double. Special values follow C.
template <class Ref, class Mp>
inline void reference(Ref ref, Mp mp, const float* in, float* out, std::size_t n) {
    thread_local struct Mpv {
        mpfr_t v;
        Mpv()  { mpfr_init2(v, 24); }
        ~Mpv() { mpfr_clear(v); }
    } m;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = ref((double)in[i]);
        float y = (float)r;
        if (std::isfinite(r) && (double)y != r) {
            const float  nb  = std::nextafter(y, r > y ? INFINITY : -INFINITY);
            const double mid = ((double)y + (double)nb) * 0.5;
            if (std::fabs(r - mid) <= 0x1p-46 * std::fabs(r)) {
                mpfr_set_flt(m.v, in[i], MPFR_RNDN);
                mp(m.v, m.v, MPFR_RNDN);
                y = mpfr_get_flt(m.v, MPFR_RNDN);
            }
        }
        out[i] = y;
    }
}

}  // namespace ahyp_ref

#endif  // AHYP_REF_HPP
