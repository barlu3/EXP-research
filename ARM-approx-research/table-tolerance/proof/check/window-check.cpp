// Runs the real compiled frecpe_logf at the proof's windows: for each cell and
// side, c at 1/4, 1/2, 3/4 and all of the proven drift, against every input of
// the cell. The theorem says none can pass 1 ulp; this tests that against the code.
//
// Usage: window-check <eps_star.csv>

#include "approx-bench.hpp"
#include "logf-ref.hpp"
#include "replicas.h"

#include <arm_neon.h>
#include <mpfr.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using approx::from_bits;
using approx::to_bits;

static const unsigned NT = std::max(1u, std::thread::hardware_concurrency());

template <class F> static void parallel(std::uint64_t n, F body) {
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < NT; ++t)
        pool.emplace_back([&body, n, t] { body(t, n * t / NT, n * (t + 1) / NT); });
    for (auto& th : pool) th.join();
}

static std::uint64_t ulps(float y, float w) {
    if (std::isnan(y) || std::isnan(w)) return std::isnan(y) && std::isnan(w) ? 0 : UINT64_MAX;
    const std::int64_t d = approx::ordered(to_bits(y)) - approx::ordered(to_bits(w));
    return (std::uint64_t)(d < 0 ? -d : d);
}

static double log_entry(double rec) {
    mpfr_t t;
    mpfr_init2(t, 256);
    mpfr_set_d(t, rec, MPFR_RNDN);
    mpfr_log(t, t, MPFR_RNDN);
    mpfr_neg(t, t, MPFR_RNDN);
    const double r = mpfr_get_d(t, MPFR_RNDN);
    mpfr_clear(t);
    return r;
}

struct Cell { std::vector<std::uint32_t> x; std::vector<float> ref; };

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s eps_star.csv\n", argv[0]); return 2; }
    std::vector<long> ed(256), eu(256);
    {
        std::ifstream f(argv[1]);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            std::istringstream s(line);
            std::string a, b, c, d;
            std::getline(s, a, ','); std::getline(s, b, ','); std::getline(s, c, ','); std::getline(s, d, ',');
            ed[std::stoi(a)] = std::stol(c);
            eu[std::stoi(a)] = std::stol(d);
        }
    }
    float chat[256];
    for (int i = 0; i < 256; ++i) {
        chat[i] = vrecpes_f32(from_bits(0x3f800000u | (std::uint32_t)i << 15));
        fr_logf_lg[i] = log_entry((double)chat[i]);
    }
    fr_near1 = 1;

    constexpr int BATCH = 16;
    long tested = 0, failed = 0;
    std::uint64_t evals = 0;
    for (int first = 0; first < 256; first += BATCH) {
        std::vector<std::vector<std::vector<std::uint32_t>>> part(NT, std::vector<std::vector<std::uint32_t>>(BATCH));
        parallel(1ull << 32, [&](unsigned t, std::uint64_t lo, std::uint64_t hi) {
            for (std::uint64_t b = lo; b < hi; ++b) {
                const int e = fr_entry_logf((std::uint32_t)b);
                if (e >= first && e < first + BATCH) part[t][e - first].push_back((std::uint32_t)b);
            }
        });
        for (int k = 0; k < BATCH; ++k) {
            const int i = first + k;
            Cell c;
            for (auto& p : part) { c.x.insert(c.x.end(), p[k].begin(), p[k].end()); std::vector<std::uint32_t>().swap(p[k]); }
            c.ref.resize(c.x.size());
            parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
                std::vector<float> in(4096);
                for (std::uint64_t q = lo; q < hi; q += 4096) {
                    const std::size_t n = (std::size_t)std::min<std::uint64_t>(4096, hi - q);
                    for (std::size_t j = 0; j < n; ++j) in[j] = from_bits(c.x[q + j]);
                    logf_ref::reference(in.data(), &c.ref[q], n);
                }
            });
            const double base = i == 0 ? 1.0 : (double)chat[i];
            const double lg0 = fr_logf_lg[i];
            for (int side = -1; side <= 1; side += 2) {
                const double e = (double)(side < 0 ? ed[i] : eu[i]) / 4194304.0;
                for (int frac = 1; frac <= 4; ++frac) {
                    const double rec = base * (1 + side * e * frac / 4);   // exact: few enough bits
                    fr_cell_rec = rec;
                    fr_logf_lg[i] = log_entry(rec);
                    fr_cell = i;
                    std::atomic<std::uint64_t> bad{0};
                    std::atomic<std::uint32_t> at{0};
                    parallel(c.x.size(), [&](unsigned, std::uint64_t lo, std::uint64_t hi) {
                        std::uint64_t b = 0;
                        for (std::uint64_t q = lo; q < hi; ++q)
                            if (ulps(frecpe_logf(from_bits(c.x[q])), c.ref[q]) > 1) { if (!b) at = c.x[q]; ++b; }
                        bad += b;
                    });
                    fr_cell = -1;
                    fr_logf_lg[i] = lg0;
                    ++tested;
                    evals += c.x.size();
                    if (bad) {
                        ++failed;
                        std::printf("FAIL cell %d side %+d frac %d/4: %llu inputs past 1 ulp, first x = %a\n",
                                    i, side, frac, (unsigned long long)bad.load(),
                                    (double)from_bits(at.load()));
                    }
                }
            }
        }
        std::fprintf(stderr, "  cells %d..%d done\n", first, first + BATCH - 1);
    }
    std::printf("%ld window points tested, %llu evaluations, %ld failed\n", tested,
                (unsigned long long)evals, failed);
    return failed ? 1 : 0;
}
