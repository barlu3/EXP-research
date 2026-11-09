// One FRECPE logf cell's drift edge, c value by c value. B4 (table-tolerance.cpp)
// pins each cell's bound between a drift that passes and one 3% further that
// fails, testing ~40 drifts per side. This takes every c on the same 29-bit grid
// in a window centred on that gap and checks each against every input of the
// cell, so the edge is located exactly and any passing island past it shows.
//
// Usage: edge-scan <cell> <-1|+1> [count]   (count defaults to 2^20)

#include "approx-bench.hpp"
#include "logf-ref.hpp"
#include "replicas.h"

#include <arm_neon.h>
#include <mpfr.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <thread>
#include <vector>

using approx::from_bits;
using approx::to_bits;

namespace {

constexpr int GRID = 29;   // B4's grid for logf: 24-bit m times c exact in double
const unsigned NT = std::max(1u, std::thread::hardware_concurrency());

std::uint64_t ulps(float y, float w) {   // as table-tolerance.cpp counts it
    if (std::isnan(y) || std::isnan(w)) return std::isnan(y) && std::isnan(w) ? 0 : UINT64_MAX;
    const std::int64_t d = approx::ordered(to_bits(y)) - approx::ordered(to_bits(w));
    return (std::uint64_t)(d < 0 ? -d : d);
}

double log_entry(double rec) {   // -log(rec), correctly rounded, as B4 recomputes it
    mpfr_t t;
    mpfr_init2(t, 256);
    mpfr_set_d(t, rec, MPFR_RNDN);
    mpfr_log(t, t, MPFR_RNDN);
    mpfr_neg(t, t, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

double on_grid(double rec, double eps) {   // B4's drifted reciprocal
    mpfr_t t;
    mpfr_init2(t, 256);
    mpfr_set_d(t, eps, MPFR_RNDN);
    mpfr_add_ui(t, t, 1, MPFR_RNDN);
    mpfr_mul_d(t, t, rec, MPFR_RNDN);
    mpfr_prec_round(t, GRID, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

template <class F> void parallel(std::uint64_t n, F body) {
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < NT; ++t)
        pool.emplace_back([&body, n, t] { body(t, n * t / NT, n * (t + 1) / NT); });
    for (auto& th : pool) th.join();
}

// The cell's inputs, smallest |result| first: those break first, so a failing c
// is usually caught in the first chunk. The order changes no verdict.
std::vector<std::uint32_t> X;
std::vector<float> REF;

// Persistent workers, one round per c. Chunks of 4096 go round-robin, so every
// worker starts on the smallest results.
constexpr std::size_t CHUNK = 4096;
std::barrier<> go(NT + 1), done(NT + 1);
std::atomic<bool> quit{false}, bad{false};
std::atomic<std::uint32_t> bad_x{0};

void worker(unsigned t) {
    for (;;) {
        go.arrive_and_wait();
        if (quit) return;
        for (std::size_t c0 = t * CHUNK; c0 < X.size() && !bad.load(std::memory_order_relaxed);
             c0 += NT * CHUNK) {
            const std::size_t c1 = std::min(X.size(), c0 + CHUNK);
            for (std::size_t k = c0; k < c1; ++k)
                if (ulps(frecpe_logf(from_bits(X[k])), REF[k]) > 1) {
                    if (!bad.exchange(true)) bad_x = X[k];
                    break;
                }
        }
        done.arrive_and_wait();
    }
}

bool round_passes() {   // every input within 1 ulp with the table as it stands
    bad = false;
    go.arrive_and_wait();
    done.arrive_and_wait();
    return !bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <cell> <-1|+1> [count]\n", argv[0]); return 2; }
    const int cell = std::atoi(argv[1]), dir = std::atoi(argv[2]) < 0 ? -1 : 1;
    const long count = argc > 3 ? std::atol(argv[3]) : 1L << 20;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    auto check = [&](bool c, const char* what) { if (!c) { ok = false; std::printf("SELF-CHECK FAILED: %s\n", what); } };

    // FRECPE's table as B3/B4 set it up, in B4's configuration (c = 1 near 1).
    float chat[256];
    for (int i = 0; i < 256; ++i) {
        chat[i] = vrecpes_f32(from_bits(0x3f800000u | (std::uint32_t)i << 15));
        fr_logf_lg[i] = log_entry((double)chat[i]);
    }
    fr_near1 = 1;
    const double base = cell == 0 ? 1.0 : (double)chat[cell];
    const double lg0 = fr_logf_lg[cell];

    // Every float that reads the cell, with its correctly rounded log.
    {
        std::vector<std::vector<std::uint32_t>> part(NT);
        parallel(1ull << 32, [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
            for (std::uint64_t b = lo; b < hi; ++b)
                if (fr_entry_logf((std::uint32_t)b) == cell) part[t].push_back((std::uint32_t)b);
        });
        std::vector<std::uint32_t> x;
        for (auto& p : part) x.insert(x.end(), p.begin(), p.end());
        std::vector<float> ref(x.size());
        parallel(x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
            std::vector<float> in(4096);
            for (std::uint64_t k = lo; k < hi; k += 4096) {
                const std::size_t n = (std::size_t)std::min<std::uint64_t>(4096, hi - k);
                for (std::size_t j = 0; j < n; ++j) in[j] = from_bits(x[k + j]);
                logf_ref::reference(in.data(), &ref[k], n);
            }
        });
        std::vector<std::size_t> ord(x.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
            return std::fabs(ref[a]) < std::fabs(ref[b]);
        });
        X.resize(x.size());
        REF.resize(x.size());
        for (std::size_t k = 0; k < x.size(); ++k) { X[k] = x[ord[k]]; REF[k] = ref[ord[k]]; }
    }
    std::printf("cell %d: c = %a, %zu inputs, drift %s, %u threads\n", cell, base, X.size(),
                dir > 0 ? "up" : "down", NT);

    std::vector<std::thread> pool;
    for (unsigned t = 0; t < NT; ++t) pool.emplace_back(worker, t);

    auto set_rec = [&](double c) { fr_cell_rec = c; fr_logf_lg[cell] = log_entry(c); fr_cell = cell; };
    auto restore = [&] { fr_cell = -1; fr_logf_lg[cell] = lg0; };
    auto drifted = [&](double eps) { set_rec(on_grid(base, eps)); const bool p = round_passes(); restore(); return p; };

    // Gate: the writable entry at c itself is FRECPE, bit for bit, and passes.
    {
        std::vector<std::uint32_t> want(X.size());
        parallel(X.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
            for (std::uint64_t k = lo; k < hi; ++k) want[k] = to_bits(frecpe_logf(from_bits(X[k])));
        });
        set_rec(base);
        std::atomic<std::uint64_t> diff{0};
        parallel(X.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
            std::uint64_t d = 0;
            for (std::uint64_t k = lo; k < hi; ++k) d += to_bits(frecpe_logf(from_bits(X[k]))) != want[k];
            diff += d;
        });
        restore();
        std::printf("gate: entry at its own c vs FRECPE: %llu of %zu differ\n",
                    (unsigned long long)diff.load(), X.size());
        check(diff == 0, "the writable entry is not FRECPE");
        check(drifted(0), "FRECPE's own c breaks 1 ulp");
    }

    // B4's search, rerun as table-tolerance.cpp runs it.
    double lo = 0, hi = NAN;
    for (double m = 0x1p-24; m <= 0.5; m *= 1.4142135623730951) {
        if (!drifted(dir * m)) { hi = m; break; }
        lo = m;
    }
    if (std::isnan(hi) || lo == 0) { std::printf("no bound found in range\n"); return 1; }
    while (hi / lo > 0x1.03p0) {
        const double mid = std::sqrt(lo * hi);
        if (drifted(dir * mid)) lo = mid; else hi = mid;
    }
    const double e_pass = dir * lo, e_fail = dir * hi;
    std::printf("B4 search: passes at eps = %+.4e, fails at eps = %+.4e\n", e_pass, e_fail);

    // The window: `count` consecutive grid values centred on the gap, inside out.
    const double c_pass = on_grid(base, e_pass), c_fail = on_grid(base, e_fail);
    const int ex = std::ilogb(c_pass);
    const double step = std::ldexp(1.0, ex - (GRID - 1));
    const std::int64_t k_pass = std::llround(c_pass / step), k_fail = std::llround(c_fail / step);
    const std::int64_t k_mid = (k_pass + k_fail) / 2, k0 = k_mid - dir * (count / 2);
    const double c_first = k0 * step, c_last = (k0 + dir * (count - 1)) * step;
    check(std::ilogb(c_first) == ex && std::ilogb(c_last) == ex, "the window leaves c's binade");
    std::printf("gap: %lld grid values; window: %ld values, eps %+.6e .. %+.6e (step %a)\n\n",
                (long long)((k_fail - k_pass) * dir), count, c_first / base - 1, c_last / base - 1, step);

    std::vector<std::uint8_t> pass(count);
    std::vector<std::uint32_t> who(count, 0);
    for (long j = 0; j < count; ++j) {
        set_rec((k0 + dir * j) * step);
        pass[j] = round_passes();
        if (!pass[j]) who[j] = bad_x;
        if ((j & 0x3ffff) == 0x3ffff)
            std::fprintf(stderr, "  %ld/%ld c values, %.0f s\n", j + 1, count,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    restore();
    quit = true;
    go.arrive_and_wait();
    for (auto& th : pool) th.join();

    // Runs of pass and fail, in scan order (drift growing).
    auto eps_at = [&](long j) { return (k0 + dir * j) * step / base - 1; };
    std::vector<std::pair<long, long>> runs;
    for (long j = 0; j < count; ++j)
        if (j == 0 || pass[j] != pass[j - 1]) runs.push_back({ j, j });
        else runs.back().second = j;
    long n_pass = std::count(pass.begin(), pass.end(), 1);
    std::printf("scanned %ld c values: %ld pass, %ld fail, %zu run%s\n", count, n_pass, count - n_pass,
                runs.size(), runs.size() == 1 ? "" : "s");
    for (std::size_t r = 0; r < runs.size() && r < 20; ++r)
        std::printf("  %s  j %7ld .. %7ld   eps %+.9e .. %+.9e\n", pass[runs[r].first] ? "pass" : "FAIL",
                    runs[r].first, runs[r].second, eps_at(runs[r].first), eps_at(runs[r].second));
    if (runs.size() == 2 && pass[0] && !pass[count - 1]) {
        const long e = runs[0].second;
        std::printf("\nclean edge: last pass eps = %+.9e, first fail eps = %+.9e (adjacent grid values)\n",
                    eps_at(e), eps_at(e + 1));
        check(std::fabs(eps_at(e)) >= std::fabs(e_pass) && std::fabs(eps_at(e + 1)) <= std::fabs(e_fail),
              "the edge lies outside B4's gap");
    } else {
        std::printf("\nNOT a single edge: %zu runs\n", runs.size());
    }

    // Which inputs break the failing c values.
    std::map<std::uint32_t, long> breakers;
    for (long j = 0; j < count; ++j) if (!pass[j]) ++breakers[who[j]];
    std::vector<std::pair<long, std::uint32_t>> top;
    for (auto& [x, n] : breakers) top.push_back({ n, x });
    std::sort(top.rbegin(), top.rend());
    std::printf("\ninputs reported breaking (first found per c): %zu distinct\n", top.size());
    for (std::size_t r = 0; r < top.size() && r < 8; ++r)
        std::printf("  x = %-16a  %7ld c values\n", (double)from_bits(top[r].second), top[r].first);

    std::printf("\nself-checks: %s\nwall: %.1f s\n", ok ? "PASS" : "FAIL",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return ok ? 0 : 1;
}
