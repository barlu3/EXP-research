/* How much error glibc 2.44's log tables tolerate, entry by entry, and whether
   FRECPE's error fits inside that tolerance. One function per build: logf,
   log2f and log10f (LOGF_BASE unset, 2 or 10) and log1pf (TOL_LOG1P), each
   writing table_tolerance_<fn>f.txt beside its binary.

   glibc reads correctly rounded entries: a reciprocal near 1/c, c near the
   middle of the entry's cell, and the log of it. FRECPE's estimate is
   inaccurate -- 8 bits -- but a FRECPE log takes r = m * FRECPE(m) - 1 exactly
   and tabulates the log of the estimate itself, so the inaccuracy only widens
   |r|. Two tolerances follow, each against 1 ulp of the correctly rounded
   result over every input that reads the entry:

     matched    the reciprocal drifts by 1 + eps, on its own bit grid, and its
                log entry is recomputed to match. FRECPE's error is this kind:
                only |r| grows, and glibc's polynomial must absorb it.
     unmatched  the log entry alone moves by delta. The error goes straight
                into the result.

   Then FRECPE in the same terms: its 256 estimates cell by cell (B1); whether
   its |r| sits inside the |r| each glibc entry tolerated (B2); glibc's
   algorithm run with FRECPE as its reciprocal over every float (B3); and the
   matched tolerance of FRECPE's own entries in that algorithm (B4).

   Self-checks fail the run (exit 1): each replica is bit for bit gnu-libm's
   function over every float; glibc's own table passes; every reported bound
   passes and the drift found past it fails. The FRECPE verdict is a finding,
   not a check, and never changes the exit status. */

#include "approx-bench.hpp"
#include "replicas.h"

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mpfr.h>

#if defined(TOL_LOG1P)
#include "ahyp-ref.hpp"
#define FN        "log1p"
#define GLIBC_SRC "s_log1pf.c (CORE-MATH)"
#define TOL       tol_log1pf
#define FRECPE    frecpe_log1pf
#define GNU       gnu_log1pf
#define ENTRY     tol_entry_log1pf
#define TABLE     tol_table_log1pf
#define FR_LG     fr_log1pf_lg
#define FR_ENTRY  fr_entry_log1pf
#define GRID_BITS 23
#define LOG_OFF   TOL_LOG1PF_OFF
#define MPFR_LOGB mpfr_log
#define HAS_NEAR1 0
#else
#include "logf-ref.hpp"
#if !defined(LOGF_BASE)
#define GLIBC_SRC "e_logf.c"
#define TOL       tol_logf
#define FRECPE    frecpe_logf
#define GNU       gnu_logf
#define ENTRY     tol_entry_logf
#define TABLE     tol_table_logf
#define FR_LG     fr_logf_lg
#define FR_ENTRY  fr_entry_logf
#define GRID_BITS 53
#define LOG_OFF   0.0
#elif LOGF_BASE == 2
#define GLIBC_SRC "e_log2f.c"
#define TOL       tol_log2f
#define FRECPE    frecpe_log2f
#define GNU       gnu_log2f
#define ENTRY     tol_entry_log2f
#define TABLE     tol_table_log2f
#define FR_LG     fr_log2f_lg
#define FR_ENTRY  fr_entry_log2f
#define GRID_BITS 53
#define LOG_OFF   0.0
#else
#define GLIBC_SRC "e_log10f.c (CORE-MATH)"
#define TOL       tol_log10f
#define FRECPE    frecpe_log10f
#define GNU       gnu_log10f
#define ENTRY     tol_entry_log10f
#define TABLE     tol_table_log10f
#define FR_LG     fr_log10f_lg
#define FR_ENTRY  fr_entry_log10f
#define GRID_BITS 29
#define LOG_OFF   TOL_LOG10F_OFF
#endif
#define FN        LOGF_FN
#define MPFR_LOGB MPFR_LOG
#define HAS_NEAR1 1
#endif

// FRECPE's entries drift on a grid that keeps r = m * chat - 1 exact: 24-bit
// significands times 29 bits, or log1pf's w = 1 + x, at most 30 bits, times 23.
#if defined(TOL_LOG1P)
#define FR_GRID_BITS 23
#else
#define FR_GRID_BITS 29
#endif

extern "C" float GNU(float);   // gnu-libm: glibc's own build, the gate's other side

namespace {

using approx::from_bits;
using approx::to_bits;

#if defined(TOL_LOG1P)
constexpr const char* REFERENCE_NOTE =
    "correctly rounded log1p(x) (double, MPFR near rounding boundaries)";
void reference(const float* in, float* out, std::size_t n) {
    ahyp_ref::reference([](double v) { return std::log1p(v); },
                        [](mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t d) { return mpfr_log1p(r, a, d); },
                        in, out, n);
}
#else
using logf_ref::REFERENCE_NOTE;
using logf_ref::reference;
#endif

// ─── Plumbing ───────────────────────────────────────────────────────────────

const unsigned NT = std::max(1u, std::thread::hardware_concurrency());
constexpr std::uint64_t ALL = 1ull << 32;

// body(t, lo, hi) over [0, n), one contiguous slice per thread, in order.
template <class F> void parallel(std::uint64_t n, F body) {
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < NT; ++t)
        pool.emplace_back([&body, n, t] { body(t, n * t / NT, n * (t + 1) / NT); });
    for (auto& th : pool) th.join();
}

// Distance from the correctly rounded result in floats, as the benchmarks'
// accuracy pass counts it; a NaN on one side only is infinitely far.
std::uint64_t ulps(float y, float w) {
    if (std::isnan(y) || std::isnan(w)) return std::isnan(y) && std::isnan(w) ? 0 : UINT64_MAX;
    const std::int64_t d = approx::ordered(to_bits(y)) - approx::ordered(to_bits(w));
    return (std::uint64_t)(d < 0 ? -d : d);
}

double bits(double v) { return v == 0 ? -INFINITY : std::log2(std::fabs(v)); }

// ─── One entry's inputs ─────────────────────────────────────────────────────

// Every float that reads entry e, ascending, with its correctly rounded result.
struct Cell { std::vector<std::uint32_t> x; std::vector<float> ref; };

// The cells of entries [first, first + n) by entry(), in one pass over every
// float: FRECPE's 256 entries would take 256 passes one at a time.
std::vector<Cell> gather(int first, int n, int (*entry)(std::uint32_t)) {
    std::vector<std::vector<std::vector<std::uint32_t>>> part(
        NT, std::vector<std::vector<std::uint32_t>>(n));
    parallel(ALL, [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
        for (std::uint64_t b = lo; b < hi; ++b) {
            const unsigned e = (unsigned)(entry((std::uint32_t)b) - first);
            if (e < (unsigned)n) part[t][e].push_back((std::uint32_t)b);
        }
    });
    std::vector<Cell> cells(n);
    for (int e = 0; e < n; ++e) {
        Cell& c = cells[e];
        for (auto& p : part) {
            c.x.insert(c.x.end(), p[e].begin(), p[e].end());
            std::vector<std::uint32_t>().swap(p[e]);
        }
        c.ref.resize(c.x.size());
        parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
            std::vector<float> in(4096);
            for (std::uint64_t k = lo; k < hi; k += 4096) {
                const std::size_t m = (std::size_t)std::min<std::uint64_t>(4096, hi - k);
                for (std::size_t j = 0; j < m; ++j) in[j] = from_bits(c.x[k + j]);
                reference(in.data(), &c.ref[k], m);
            }
        });
    }
    return cells;
}
Cell gather(int e) { return std::move(gather(e, 1, ENTRY)[0]); }

// Whether every input of the cell is within 1 ulp through f with the tables as
// they stand. Stops at the first that is not.
bool passes(const Cell& c, float (*f)(float) = TOL) {
    std::atomic<bool> bad{false};
    parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
        for (std::uint64_t k = lo; k < hi; ++k) {
            if ((k & 4095) == 0 && bad.load(std::memory_order_relaxed)) return;
            if (ulps(f(from_bits(c.x[k])), c.ref[k]) > 1) { bad = true; return; }
        }
    });
    return !bad;
}

// The smallest input of the cell past 1 ulp, for the report.
struct Breach { bool any = false; std::uint32_t x = 0; float y = 0, want = 0; };
Breach breach(const Cell& c, float (*f)(float) = TOL) {
    std::vector<Breach> per(NT);
    parallel(c.x.size(), [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
        for (std::uint64_t k = lo; k < hi; ++k) {
            const float y = f(from_bits(c.x[k]));
            if (ulps(y, c.ref[k]) > 1) { per[t] = { true, c.x[k], y, c.ref[k] }; return; }
        }
    });
    for (const Breach& b : per)
        if (b.any) return b;   // slices are in order, so the first found is the smallest
    return {};
}

// ─── Entries, in MPFR ───────────────────────────────────────────────────────

// rec * (1 + eps), rounded to the entry's own grid of grid significant bits,
// so the drifted reciprocal keeps whatever exactness its algorithm built on it.
double on_grid(double rec, double eps, int grid = GRID_BITS) {
    mpfr_t t;
    mpfr_init2(t, 256);
    mpfr_set_d(t, eps, MPFR_RNDN);
    mpfr_add_ui(t, t, 1, MPFR_RNDN);
    mpfr_mul_d(t, t, rec, MPFR_RNDN);
    mpfr_prec_round(t, grid, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

// -log_b(rec) + offset at 256 bits: the exact log entry for a reciprocal.
void exact_log_entry(mpfr_t t, double rec) {
    mpfr_set_d(t, rec, MPFR_RNDN);
    MPFR_LOGB(t, t, MPFR_RNDN);
    mpfr_neg(t, t, MPFR_RNDN);
    mpfr_add_d(t, t, LOG_OFF, MPFR_RNDN);
}
double log_entry(double rec) {          // glibc's convention: that, rounded once
    mpfr_t t;
    mpfr_init2(t, 256);
    exact_log_entry(t, rec);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}
double log_entry_error(double rec, double lg) {   // stored minus exact
    mpfr_t t;
    mpfr_init2(t, 256);
    exact_log_entry(t, rec);
    mpfr_d_sub(t, lg, t, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

// ─── The two searches ───────────────────────────────────────────────────────

// Signed drifts: pass is the farthest tolerated, fail the nearest found beyond
// it that is not; fail is NaN when nothing failed within the search's range.
struct Bound { double pass = 0, fail = NAN; };

// The matched search, over eps (a positive drift grows the reciprocal). It
// scans |eps| outward from 2^-24 by factors of sqrt(2) and stops at the first
// failure, so a drift counts as tolerated only if every grid point below it
// passed too: the polynomial's error is not monotone in |r|, and a passing
// island past a failure proves nothing. Between the last pass and the first
// failure it bisects, in log scale, to a factor of 2^(1/64), assuming the
// result monotone within that one step. Replace it to treat islands otherwise.
Bound tolerance_search(int dir, const std::function<bool(double)>& pass) {
    double lo = 0, hi = NAN;
    for (double m = 0x1p-24; m <= 0.5; m *= 1.4142135623730951) {
        if (!pass(dir * m)) { hi = m; break; }
        lo = m;
    }
    if (std::isnan(hi)) return { dir * lo, NAN };
    if (lo == 0) return { 0, dir * hi };   // below 2^-24: report it as nothing
    while (hi / lo > 0x1.03p0) {
        const double mid = std::sqrt(lo * hi);
        if (pass(dir * mid)) lo = mid; else hi = mid;
    }
    return { dir * lo, dir * hi };
}

// Doubles in order as integers, -0 and +0 together.
std::int64_t key(double d) {
    std::int64_t b;
    std::memcpy(&b, &d, 8);
    return b < 0 ? -(b & INT64_MAX) : b;
}
double unkey(std::int64_t k) {
    const std::int64_t b = k < 0 ? (-k) | INT64_MIN : k;
    double d;
    std::memcpy(&d, &b, 8);
    return d;
}

// The unmatched search, over the log entry itself: a bisection over its bit
// pattern, exact because every result is monotone in the entry, so the entries
// that keep one input within 1 ulp form an interval and so do all of them.
// Returns deltas from base; fail is the next double past pass.
Bound entry_search(int dir, double base, const std::function<bool(double)>& pass) {
    std::int64_t lo = key(base), hi = 0;
    bool found = false;
    for (double s = 0x1p-60; s <= 1.0; s *= 16) {
        const double v = base + dir * s;
        if (v == base) continue;
        if (!pass(v)) { hi = key(v); found = true; break; }
        lo = key(v);
    }
    if (!found) return { unkey(lo) - base, NAN };
    while ((hi > lo ? hi - lo : lo - hi) > 1) {
        const std::int64_t mid = lo + (hi - lo) / 2;
        if (pass(unkey(mid))) lo = mid; else hi = mid;
    }
    return { unkey(lo) - base, unkey(hi) - base };
}

// ─── FRECPE in glibc's terms ────────────────────────────────────────────────

float g_chat[256];   // FRECPE of 1 + i/256: its estimate for every m in that cell

// The significands glibc multiplies entry i's reciprocal by, [lo, hi).
std::pair<double, double> cell_of(int i) {
#if defined(TOL_LOG1P)
    return { 1 + i / 32.0, 1 + (i + 1) / 32.0 };                 // w = 1 + x
#elif defined(LOGF_BASE) && LOGF_BASE == 10
    return { std::max(1.0, 1 + (i - 0.5) / 64), std::min(2.0, 1 + (i + 0.5) / 64) };
#else
    return { from_bits(0x3f330000u + ((std::uint32_t)i << 19)),  // z in [OFF, 2 OFF)
             from_bits(0x3f330000u + ((std::uint32_t)(i + 1) << 19)) };
#endif
}

// FRECPE's r = m * chat - 1 over m in [a, b), within [1, 2].
std::pair<double, double> frecpe_r(double a, double b) {
    double lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < 256; ++i) {
        const double u = std::max(a, 1 + i / 256.0), v = std::min(b, 1 + (i + 1) / 256.0);
        if (u >= v) continue;
        lo = std::min(lo, u * g_chat[i] - 1);
        hi = std::max(hi, v * g_chat[i] - 1);   // a supremum: v itself is outside
    }
    return { lo, hi };
}
// The same over glibc's significands, which for logf and log2f start below 1:
// there FRECPE sees the significand of 2z.
std::pair<double, double> frecpe_r_over(double a, double b) {
    if (b <= 1) return frecpe_r(2 * a, 2 * b);
    if (a >= 1) return frecpe_r(a, b);
    const auto p = frecpe_r(2 * a, 2.0), q = frecpe_r(1.0, b);
    return { std::min(p.first, q.first), std::max(p.second, q.second) };
}

// Every float through f against the reference: how many land past 1 ulp, the
// worst distance and its smallest input, and the slow paths taken.
struct Sweep { std::uint64_t bad = 0, max_ulp = 0, slow = 0; std::uint32_t worst = 0; };
Sweep sweep(float (*f)(float)) {
    std::vector<Sweep> per(NT);
    parallel(ALL, [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
        std::vector<float> in(4096), ref(4096);
        Sweep& s = per[t];
        const unsigned long long slow0 = tol_slow_count();
        for (std::uint64_t b = lo; b < hi; b += 4096) {
            const std::size_t n = (std::size_t)std::min<std::uint64_t>(4096, hi - b);
            for (std::size_t k = 0; k < n; ++k) in[k] = from_bits((std::uint32_t)(b + k));
            reference(in.data(), ref.data(), n);
            for (std::size_t k = 0; k < n; ++k) {
                const std::uint64_t u = ulps(f(in[k]), ref[k]);
                if (u > 1) ++s.bad;
                if (u > s.max_ulp) { s.max_ulp = u; s.worst = (std::uint32_t)(b + k); }
            }
        }
        s.slow = tol_slow_count() - slow0;
    });
    Sweep all;
    for (const Sweep& s : per) {
        all.bad += s.bad;
        all.slow += s.slow;
        if (s.max_ulp > all.max_ulp) { all.max_ulp = s.max_ulp; all.worst = s.worst; }
    }
    return all;
}

void sweep_line(const char* what, const Sweep& s) {
    char u[24];
    if (s.max_ulp == UINT64_MAX) std::snprintf(u, sizeof u, "NaN");
    else std::snprintf(u, sizeof u, "%llu", (unsigned long long)s.max_ulp);
    bench::logf("  %-34s max %4s ulp, %10llu inputs past 1 ulp", what, u,
                (unsigned long long)s.bad);
    if (s.max_ulp) bench::logf(", worst x = %a", (double)from_bits(s.worst));
    bench::logf("\n");
}

// ─── Per-entry results ──────────────────────────────────────────────────────

struct Entry {
    std::size_t   inputs = 0;
    double        rec = 0, lg = 0;    // glibc's
    double        lg_err = 0;         // stored log entry minus the exact one
    std::int64_t  lg_rn = 0;          // stored minus correctly rounded, in doubles
    bool          base_ok = false;    // glibc's own entry keeps every input within 1 ulp
    Bound         del[2], eps[2];     // [0]: downward, [1]: upward
    std::uint32_t del_x[2] = {}, eps_x[2] = {};   // smallest input breached past each bound
    double        t_lo = 0, t_hi = 0; // the r glibc's polynomial tolerated here
    double        f_lo = 0, f_hi = 0; // FRECPE's r over the same significands
    bool          fits = false;
    double        margin = INFINITY;  // bits between FRECPE's |r| and the tolerated |r|
};

}  // namespace

int main() {
    const auto wall_start = bench::Clock::now();
    tol_reset();
    const std::string report = approx::report_path("table_tolerance_" FN "f.txt");
    bench::open_log(report.c_str());
    approx::give_to_sudo_user(report);
    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { ok = false; bench::logf("  SELF-CHECK FAILED: %s\n", what); }
        return cond;
    };

    const tol_table T = TABLE();
    const int N = T.n, S = T.stride;
    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  float32 " FN "(x): the error glibc 2.44's table entries tolerate,\n");
    bench::logf("  and whether FRECPE's error fits inside it\n");
    bench::logf("  glibc     : gnu-libm/glibc/" GLIBC_SRC ", %d entries, reciprocals\n", N);
    bench::logf("              on a %d-bit grid\n", GRID_BITS);
    bench::logf("  target    : every input reading an entry within 1 ulp of the\n");
    bench::logf("              correctly rounded result\n");
    bench::logf("  reference : %s\n", REFERENCE_NOTE);
    bench::logf("  threads   : %u\n", NT);
    bench::logf("==================================================================\n");

    // -- The gate ---------------------------------------------------------------
    {
        std::vector<std::uint64_t> diff(NT, 0);
        std::vector<std::uint32_t> first(NT, 0);
        parallel(ALL, [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
            for (std::uint64_t b = lo; b < hi; ++b) {
                const float x = from_bits((std::uint32_t)b);
                if (to_bits(TOL(x)) != to_bits(GNU(x)) && diff[t]++ == 0) first[t] = (std::uint32_t)b;
            }
        });
        std::uint64_t n = 0;
        std::uint32_t at = 0;
        for (unsigned t = 0; t < NT; ++t) { if (diff[t] && !n) at = first[t]; n += diff[t]; }
        bench::logf("\n  replica vs gnu-libm's " FN "f, every float: %llu of %llu differ%s\n",
                    (unsigned long long)n, (unsigned long long)ALL, n ? "" : " -- identical");
        if (n) bench::logf("  first at bits 0x%08x\n", at);
        if (!check(n == 0, "the replica is not gnu-libm's function")) {
            bench::close_log();
            return 1;
        }
    }

    // -- FRECPE's estimates, and the log table glibc's convention gives them ------
    int dbl_agree = 0;
    for (int i = 0; i < 256; ++i) {
        g_chat[i] = vrecpes_f32(from_bits(0x3f800000u | (std::uint32_t)i << 15));
        const double lo = 1 + i / 256.0, hi = 1 + (i + 1) / 256.0;
        bool same = true;
        for (double m : { lo, (lo + hi) / 2, std::nextafter(hi, 0.0) })
            same = same && vrecped_f64(m) == (double)g_chat[i];
        dbl_agree += same;
#if defined(TOL_LOG1P)
        FR_LG[i] = log_entry(vrecped_f64(lo));   // log1pf's FRECPE is double, on w
#else
        FR_LG[i] = log_entry((double)g_chat[i]);
#endif
    }

    // -- A: glibc's tolerance, entry by entry -----------------------------------
    std::vector<Entry> E(N);
    for (int i = 0; i < N; ++i) {
        Entry& en = E[i];
        const int o = i * S;
        en.rec = T.base_rec[o];
        en.lg  = T.base_lg[o];
        en.lg_err = log_entry_error(en.rec, en.lg);
        en.lg_rn  = key(en.lg) - key(log_entry(en.rec));
        const Cell c = gather(i);
        en.inputs = c.x.size();
        std::fprintf(stderr, "  entry %2d/%d: %zu inputs\n", i + 1, N, en.inputs);
        if (c.x.empty()) continue;

        auto restore = [&] { T.rec[o] = en.rec; T.lg[o] = en.lg; };
        auto set_matched = [&](double eps) {
            const double r = on_grid(en.rec, eps);
            T.rec[o] = r;
            T.lg[o]  = log_entry(r);
        };
        auto matched = [&](double eps) { set_matched(eps); const bool p = passes(c); restore(); return p; };
        auto unmatched = [&](double lg) { T.lg[o] = lg; const bool p = passes(c); restore(); return p; };

        en.base_ok = passes(c);
        check(en.base_ok, "glibc's own table breaks 1 ulp");
        for (int d = 0; d < 2; ++d) {
            const int dir = d ? 1 : -1;
            en.del[d] = entry_search(dir, en.lg, unmatched);
            check(unmatched(en.lg + en.del[d].pass), "an unmatched bound does not pass");
            if (!std::isnan(en.del[d].fail)) {
                check(!unmatched(en.lg + en.del[d].fail), "the entry past an unmatched bound passes");
                T.lg[o] = en.lg + en.del[d].fail;
                en.del_x[d] = breach(c).x;
                restore();
            }
            en.eps[d] = tolerance_search(dir, matched);
            check(en.eps[d].pass * dir >= 0, "a matched bound has the wrong sign");
            check(matched(en.eps[d].pass), "a matched bound does not pass");
            if (!std::isnan(en.eps[d].fail)) {
                check(std::fabs(en.eps[d].fail) > std::fabs(en.eps[d].pass),
                      "a matched failure lies inside its bound");
                check(!matched(en.eps[d].fail), "the drift past a matched bound passes");
                set_matched(en.eps[d].fail);
                en.eps_x[d] = breach(c).x;
                restore();
            }
        }

        // B2's two ranges for this entry.
        const auto [s_lo, s_hi] = cell_of(i);
        en.t_lo = s_lo * on_grid(en.rec, en.eps[0].pass) - 1;
        en.t_hi = s_hi * on_grid(en.rec, en.eps[1].pass) - 1;
        std::tie(en.f_lo, en.f_hi) = frecpe_r_over(s_lo, s_hi);
        en.fits = en.t_lo <= en.f_lo && en.f_hi <= en.t_hi;
        if (en.f_hi > 0 && en.t_hi > 0) en.margin = std::min(en.margin, std::log2(en.t_hi / en.f_hi));
        if (en.f_lo < 0 && en.t_lo < 0) en.margin = std::min(en.margin, std::log2(en.t_lo / en.f_lo));
    }

    // -- Report A ---------------------------------------------------------------
    approx::box("A1. glibc's entries");
    bench::logf("  The stored log entry against the exact -log_b(reciprocal) + offset:\n"
                "  how much of its own tolerance glibc spends on rounding it.\n\n");
    bench::logf("  %3s %10s  %-24s %-24s %9s %5s\n", "i", "inputs", "reciprocal", "log entry",
                "err bits", "RN?");
    for (int i = 0; i < N; ++i) {
        const Entry& en = E[i];
        bench::logf("  %3d %10zu  %-24a %-24a %9.1f %5s\n", i, en.inputs, en.rec, en.lg,
                    bits(en.lg_err), en.lg_rn == 0 ? "yes" : "no");
    }

    approx::box("A2. unmatched: the log entry alone moves");
    bench::logf("  delta: how far the entry can move, down and up, with every input\n"
                "  still within 1 ulp. 'binds at' is the smallest input that breaks\n"
                "  on the tighter side, one double past the bound.\n\n");
    bench::logf("  %3s %12s %12s %7s   %s\n", "i", "delta down", "delta up", "bits", "binds at");
    int un_i = -1;
    double un_min = INFINITY;
    for (int i = 0; i < N; ++i) {
        const Entry& en = E[i];
        if (!en.inputs) continue;
        const int d = std::fabs(en.del[0].pass) <= std::fabs(en.del[1].pass) ? 0 : 1;
        const double tol = std::fabs(en.del[d].pass);
        if (tol < un_min) { un_min = tol; un_i = i; }
        bench::logf("  %3d %12.3e %12.3e %7.1f   x = %a\n", i, en.del[0].pass, en.del[1].pass,
                    bits(tol), (double)from_bits(en.del_x[d]));
    }

    approx::box("A3. matched: the reciprocal drifts, its log follows");
    bench::logf("  eps: the relative drift tolerated each way, the reciprocal kept on its\n"
                "  %d-bit grid; 'r tolerated' is the r glibc's polynomial then saw over\n"
                "  the cell. '>' marks a side where nothing failed up to 2^-1.\n\n", GRID_BITS);
    bench::logf("  %3s %12s %12s %7s   %-27s %s\n", "i", "eps down", "eps up", "bits",
                "r tolerated", "binds at");
    int m_i = -1;
    double m_min = INFINITY;
    for (int i = 0; i < N; ++i) {
        const Entry& en = E[i];
        if (!en.inputs) continue;
        const int d = std::fabs(en.eps[0].pass) <= std::fabs(en.eps[1].pass) ? 0 : 1;
        const double tol = std::fabs(en.eps[d].pass);
        if (tol < m_min) { m_min = tol; m_i = i; }
        char rr[40];
        std::snprintf(rr, sizeof rr, "[%+.3e, %+.3e]", en.t_lo, en.t_hi);
        bench::logf("  %3d %s%11.3e %s%11.3e %7.1f   %-27s ", i,
                    std::isnan(en.eps[0].fail) ? ">" : " ", en.eps[0].pass,
                    std::isnan(en.eps[1].fail) ? ">" : " ", en.eps[1].pass, bits(tol), rr);
        if (std::isnan(en.eps[d].fail)) bench::logf("none\n");
        else bench::logf("x = %a\n", (double)from_bits(en.eps_x[d]));
    }

    // -- B1 ---------------------------------------------------------------------
    approx::box("B1. FRECPE's estimates, per entry");
    bench::logf("  chat(i) = FRECPE(1 + i/256), its estimate for every m whose top eight\n"
                "  fraction bits are i. 'dev' is chat * centre - 1, its relative error\n"
                "  against the exact reciprocal of the cell's centre; 'max|r|' is FRECPE's\n"
                "  over the cell; 'best' the smallest any one reciprocal could manage.\n\n");
    bench::logf("  %3s  %-14s %4s %11s %9s %9s\n", "i", "chat", "bits", "dev", "max|r|", "best");
    double r_max = 0, dev_max = 0;
    for (int i = 0; i < 256; ++i) {
        const double lo = 1 + i / 256.0, hi = 1 + (i + 1) / 256.0, c = g_chat[i];
        const double dev = c * (lo + hi) / 2 - 1;
        const double mr = std::max(std::fabs(lo * c - 1), std::fabs(hi * c - 1));
        const double best = (hi - lo) / (hi + lo);
        const std::uint32_t m = to_bits(g_chat[i]) & 0x007fffffu;
        const int sig = m ? 24 - __builtin_ctz(m) : 1;
        r_max = std::max(r_max, mr);
        dev_max = std::max(dev_max, std::fabs(dev));
        bench::logf("  %3d  %-14a %4d %+11.3e %9.2f %9.2f\n", i, c, sig, dev, bits(mr), bits(best));
    }
    bench::logf("\n  max |r| over every cell: 2^%.2f   (documented: 2^-8.45)\n", bits(r_max));
    bench::logf("  max |dev|: 2^%.2f\n", bits(dev_max));
    bench::logf("  double FRECPE gives the same 256 values: %d of 256 cells\n", dbl_agree);

    // -- B2 ---------------------------------------------------------------------
    approx::box("B2. FRECPE's r against what each glibc entry tolerated");
    bench::logf("  For the inputs glibc sends to entry i: the r its polynomial tolerated\n"
                "  (A3), and the r FRECPE produces for the same significands. 'margin' is\n"
                "  log2 of tolerated over FRECPE's |r| on the tighter side.\n\n");
    bench::logf("  %3s  %-27s %-27s %4s %7s\n", "i", "r tolerated (glibc)", "r (FRECPE)",
                "fits", "margin");
    bool b2_fit = true;
    double b2_margin = INFINITY;
    for (int i = 0; i < N; ++i) {
        const Entry& en = E[i];
        if (!en.inputs) continue;
        b2_fit = b2_fit && en.fits;
        b2_margin = std::min(b2_margin, en.margin);
        bench::logf("  %3d  [%+.3e, %+.3e]   [%+.3e, %+.3e]   %4s %7.2f\n", i, en.t_lo, en.t_hi,
                    en.f_lo, en.f_hi, en.fits ? "yes" : "NO", en.margin);
    }

    // -- B3 ---------------------------------------------------------------------
    approx::box("B3. glibc's algorithm with FRECPE as its reciprocal");
    bench::logf("  r = m * FRECPE(m) - 1, exact; log entry -log_b(chat) + offset from a\n"
                "  256-entry table; glibc's polynomial, order and rounding tests.%s\n\n",
                HAS_NEAR1 ? " 'c = 1\n  near 1' gives m in [1, 1 + 2^-8) c = 1, as glibc's own table does." : "");
    tol_reset();
    const Sweep glibc_sweep = sweep(TOL);
    fr_near1 = 0;
    const Sweep pure = sweep(FRECPE);
    Sweep near1 = pure;
#if HAS_NEAR1
    fr_near1 = 1;
    near1 = sweep(FRECPE);
    fr_near1 = 0;
#endif
    sweep_line("glibc 2.44, its own table", glibc_sweep);
    sweep_line("FRECPE", pure);
    if (HAS_NEAR1) sweep_line("FRECPE, c = 1 near 1", near1);
#if defined(TOL_LOG1P) || (defined(LOGF_BASE) && LOGF_BASE == 10)
    bench::logf("\n  slow path (fast path's rounding test failed): glibc %llu, FRECPE %llu\n",
                (unsigned long long)glibc_sweep.slow, (unsigned long long)pure.slow);
#endif

    // -- B4: FRECPE's own entries, drifted one at a time -------------------------
    // In the configuration B3's verdict runs: c = 1 for entry 0 where HAS_NEAR1.
    struct FrEntry {
        std::size_t   inputs = 0;
        double        rec = 0, dev = 0;   // the estimate drifted from, and its dev (B1)
        Bound         eps[2];             // [0]: downward, [1]: upward
        std::uint32_t eps_x[2] = {};
    };
    std::vector<FrEntry> F(256);
    fr_near1 = HAS_NEAR1;
    constexpr int BATCH = 16;   // logf: ~134M inputs held at once, as for one glibc entry
    for (int first = 0; first < 256; first += BATCH) {
        const std::vector<Cell> cells = gather(first, BATCH, FR_ENTRY);
        for (int k = 0; k < BATCH; ++k) {
            const int i = first + k;
            const Cell& c = cells[k];
            FrEntry& fe = F[i];
            fe.inputs = c.x.size();
            std::fprintf(stderr, "  FRECPE entry %3d/256: %zu inputs\n", i + 1, fe.inputs);
            if (c.x.empty()) continue;
            fe.rec = HAS_NEAR1 && i == 0 ? 1.0 : (double)g_chat[i];
            fe.dev = fe.rec * (1 + (i + 0.5) / 256) - 1;
            const double lg0 = FR_LG[i];
            auto restore = [&] { fr_cell = -1; FR_LG[i] = lg0; };
            auto set_drift = [&](double eps) {
                fr_cell_rec = on_grid(fe.rec, eps, FR_GRID_BITS);
                FR_LG[i] = log_entry(fr_cell_rec);
                fr_cell = i;
            };
            auto drifted = [&](double eps) {
                set_drift(eps);
                const bool p = passes(c, FRECPE);
                restore();
                return p;
            };

            // Drifted by 0, the entry must be the estimate itself, bit for bit.
            std::vector<std::uint32_t> want(c.x.size());
            parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
                for (std::uint64_t j = lo; j < hi; ++j) want[j] = to_bits(FRECPE(from_bits(c.x[j])));
            });
            set_drift(0);
            std::atomic<bool> diff{false};
            parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
                for (std::uint64_t j = lo; j < hi && !diff.load(std::memory_order_relaxed); ++j)
                    if (to_bits(FRECPE(from_bits(c.x[j]))) != want[j]) diff = true;
            });
            restore();
            check(!diff, "FRECPE's entry drifted by 0 is not the estimate");
            check(drifted(0), "FRECPE's own entry breaks 1 ulp");

            for (int d = 0; d < 2; ++d) {
                const int dir = d ? 1 : -1;
                fe.eps[d] = tolerance_search(dir, drifted);
                check(drifted(fe.eps[d].pass), "a FRECPE drift bound does not pass");
                if (!std::isnan(fe.eps[d].fail)) {
                    check(!drifted(fe.eps[d].fail), "the drift past a FRECPE bound passes");
                    set_drift(fe.eps[d].fail);
                    fe.eps_x[d] = breach(c, FRECPE).x;
                    restore();
                }
            }
        }
    }
    fr_near1 = 0;

    approx::box("B4. FRECPE's own entries, drifted one at a time");
    bench::logf("  Entry i's estimate becomes chat(i) * (1 + eps), rounded to %d bits so\n"
                "  r = m * chat - 1 stays exact, and its log entry is recomputed for the\n"
                "  drifted value; every other entry stays FRECPE's. 'eps' is the drift\n"
                "  each way that keeps every input reading entry i within 1 ulp. 'dev\n"
                "  window' is the same range as an error against the exact reciprocal of\n"
                "  the cell's centre, so FRECPE's own dev (B1) is its eps = 0 point.%s\n\n",
                FR_GRID_BITS,
                HAS_NEAR1 ? "\n  Entry 0 starts from c = 1, as B3's verdict run does." : "");
    bench::logf("  %3s  %-14s %12s %12s %7s   %-27s %s\n", "i", "drifted from", "eps down",
                "eps up", "bits", "dev window", "binds at");
    std::vector<double> fr_tol;
    int fr_i = -1;
    double fr_min = INFINITY;
    for (int i = 0; i < 256; ++i) {
        const FrEntry& fe = F[i];
        if (!fe.inputs) continue;
        const int d = std::fabs(fe.eps[0].pass) <= std::fabs(fe.eps[1].pass) ? 0 : 1;
        const double tol = std::fabs(fe.eps[d].pass);
        fr_tol.push_back(tol);
        if (tol < fr_min) { fr_min = tol; fr_i = i; }
        char win[40];
        std::snprintf(win, sizeof win, "[%+.3e, %+.3e]", (1 + fe.dev) * (1 + fe.eps[0].pass) - 1,
                      (1 + fe.dev) * (1 + fe.eps[1].pass) - 1);
        bench::logf("  %3d  %-14a %s%11.3e %s%11.3e %7.1f   %-27s ", i, fe.rec,
                    std::isnan(fe.eps[0].fail) ? ">" : " ", fe.eps[0].pass,
                    std::isnan(fe.eps[1].fail) ? ">" : " ", fe.eps[1].pass, bits(tol), win);
        if (std::isnan(fe.eps[d].fail)) bench::logf("none\n");
        else bench::logf("x = %a\n", (double)from_bits(fe.eps_x[d]));
    }
    std::sort(fr_tol.begin(), fr_tol.end());
    const double fr_median = fr_tol.empty() ? NAN : fr_tol[fr_tol.size() / 2];
    if (fr_i >= 0)
        bench::logf("\n  tightest: entry %d, tolerating a drift of 2^%.1f; median over entries 2^%.1f\n",
                    fr_i, bits(fr_min), bits(fr_median));
#if HAS_NEAR1
    {
        // FRECPE's own chat(0) is the c = 1 of entry 0, drifted: is it in the window?
        const double own = (double)g_chat[0] - 1;
        const bool inside = F[0].eps[0].pass <= own && own <= F[0].eps[1].pass;
        bench::logf("  entry 0: FRECPE's own chat(0) = %a is c = 1 drifted by %+.3e: %s\n",
                    (double)g_chat[0], own, inside ? "inside its window" : "outside its window");
    }
#endif

    // -- Summary and verdict -----------------------------------------------------
    approx::box("SUMMARY");
    if (un_i >= 0)
        bench::logf("  unmatched: the tightest log entry is %d, tolerating 2^%.1f\n", un_i, bits(un_min));
    if (m_i >= 0)
        bench::logf("  matched:   the tightest reciprocal is %d, tolerating a drift of 2^%.1f\n",
                    m_i, bits(m_min));
    bench::logf("  FRECPE:    max |r| 2^%.2f; inside every entry's tolerated r: %s",
                bits(r_max), b2_fit ? "yes" : "NO");
    if (std::isfinite(b2_margin)) bench::logf(" (tightest margin 2^%.2f)", b2_margin);
    bench::logf("\n");
    if (fr_i >= 0)
        bench::logf("  FRECPE's own entries: the tightest is %d, tolerating a drift of 2^%.1f\n"
                    "             (median 2^%.1f; its estimates' own dev reaches 2^%.2f)\n",
                    fr_i, bits(fr_min), bits(fr_median), bits(dev_max));
    const Sweep& verdict_sweep = HAS_NEAR1 ? near1 : pure;
    const bool fits = b2_fit && verdict_sweep.max_ulp <= 1;
    bench::logf("\n  FRECPE fits: %s\n", fits ? "YES" : "NO");
    bench::logf("    (every entry's tolerated r contains FRECPE's, and glibc's algorithm\n"
                "     with FRECPE%s stays within 1 ulp over every float)\n",
                HAS_NEAR1 ? " (c = 1 near 1)" : "");

    bench::logf("\n  self-checks: %s\n", ok ? "PASS" : "FAIL");
    const double wall = std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("  total wall time: %.1f s\n\n", wall);
    bench::close_log();
    return ok ? 0 : 1;
}
