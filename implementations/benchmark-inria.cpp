/**
 * @file  benchmark-inria.cpp
 * @brief Benchmark: Inria cr_exp (CORE-MATH) vs homemade fexp::exp vs std::exp.
 *
 * FLOAT64 only — cr_exp provides double precision only.
 *
 * Three input clusters, 100M iterations each:
 *   Cluster 1 — x near 1     : x ∈ [0.9,   1.1]
 *   Cluster 2 — x near 80    : x ∈ [79.5,  80.5]
 *   Cluster 3 — x near 2e-10 : x ∈ [1e-10, 3e-10]
 *
 * Compile (run from implementations/):
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark-inria.cpp -o output/bench_inria
 * Run:
 *   ./output/bench_inria
 * Capture:
 *   ./output/bench_inria 2>&1 | tee output/bench_results_inria.txt
 */

#include "exp.hpp"
#include "inria-exp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// ─── Dual-output helper ───────────────────────────────────────────────────────

static FILE* g_log = nullptr;

__attribute__((format(printf, 1, 2)))
static void lprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    if (g_log) {
        va_start(ap, fmt);
        std::vfprintf(g_log, fmt, ap);
        va_end(ap);
    }
}

// ─── Anti-optimisation sink ──────────────────────────────────────────────────

static volatile double sink_d = 0.0;

// ─── Error metrics ───────────────────────────────────────────────────────────

static double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

// ─── Benchmark parameters ────────────────────────────────────────────────────

static constexpr int BENCH_ITERS  = 100'000'000;
static constexpr int WARMUP_ITERS =   1'000'000;
static constexpr int ACC_SAMPLES  =     500'000;

// ─── Timing helper ───────────────────────────────────────────────────────────

struct BenchResult { double ns_per_call, total_ms; };

template<typename Fn>
static BenchResult run_bench(Fn fn, const std::vector<double>& inputs) {
    double acc = 0.0;
    for (int i = 0; i < WARMUP_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
    sink_d = acc;

    auto t0 = Clock::now();
    acc = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
    auto t1 = Clock::now();
    sink_d = acc;

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return { (ms * 1e6) / BENCH_ITERS, ms };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    g_log = std::fopen("output/bench_results_inria.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open output/bench_results_inria.txt\n");

    auto wall_start = Clock::now();

    std::mt19937 rng(42);

    // ── Header ────────────────────────────────────────────────────────────────

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  exp() Benchmark — Inria cr_exp vs homemade (exp.hpp) vs stdlib\n");
    lprintf("  Compile flags : -O3 -march=native -mavx2 -mfma -std=c++20\n");
    lprintf("  Iterations    : %'d per variant per cluster\n", BENCH_ITERS);
    lprintf("  Warmup        : %'d iterations (not timed)\n",  WARMUP_ITERS);
    lprintf("  Acc. samples  : %'d random draws per cluster\n", ACC_SAMPLES);
    lprintf("══════════════════════════════════════════════════════════════════\n");

    // ── Value clusters ────────────────────────────────────────────────────────

    struct Cluster { const char* label; double lo, hi; };
    static constexpr Cluster CLUSTERS[] = {
        { "x near 1     ",  0.9,    1.1   },
        { "x near 80    ", 79.5,   80.5   },
        { "x near 2e-10 ", 1e-10,  3e-10  },
    };

    // ═══════════════════════════════════════════════════════════════════════════
    //  float64 (double) — three-way comparison
    //  cr_exp:    CORE-MATH correctly rounded implementation, header-inlined.
    //  fexp::exp: glibc-2.43 algorithm port, header-inlined, no errno.
    //  std::exp:  glibc libm, called via PLT (shared-library ABI).
    // ═══════════════════════════════════════════════════════════════════════════

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  FLOAT64 — cr_exp(x), fexp::exp(x), std::exp(x)               │\n");
    lprintf("│  Inria    : CORE-MATH cr_exp, correctly rounded (≤0.5 ULP)    │\n");
    lprintf("│  Homemade : glibc-2.43 algorithm, header-inlined, no errno    │\n");
    lprintf("│  Stdlib   : glibc libm, called via PLT (shared-library ABI)   │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<double> dist(cl.lo, cl.hi);
        std::vector<double> in(ACC_SAMPLES);
        for (auto& v : in) v = dist(rng);

        double max_rel_fexp  = 0.0;
        double max_rel_inria = 0.0;
        for (auto v : in) {
            max_rel_fexp  = std::max(max_rel_fexp,  rel_err(fexp::exp(v), std::exp(v)));
            max_rel_inria = std::max(max_rel_inria, rel_err(cr_exp(v),    std::exp(v)));
        }

        auto r_std   = run_bench([](double x){ return std::exp(x);  }, in);
        auto r_fexp  = run_bench([](double x){ return fexp::exp(x); }, in);
        auto r_inria = run_bench([](double x){ return cr_exp(x);    }, in);

        double su_fexp  = r_std.total_ms / r_fexp.total_ms;
        double su_inria = r_std.total_ms / r_inria.total_ms;

        lprintf("\n");
        lprintf("  ── Cluster: %s  x ∈ [%.3g, %.3g]  (%d iters)\n",
                cl.label, cl.lo, cl.hi, BENCH_ITERS);
        lprintf("     Accuracy vs std::exp — homemade: %.2e  inria: %.2e%s\n",
                max_rel_fexp, max_rel_inria,
                max_rel_inria == 0.0 ? "  (bit-for-bit identical)" : "");
        lprintf("\n");
        lprintf("     %-42s %9s  %10s  %8s\n",
                "Variant", "ns/call", "total (ms)", "speedup");
        lprintf("     %-42s %9s  %10s  %8s\n",
                "──────────────────────────────────────────",
                "─────────", "──────────", "───────");
        lprintf("     %-42s %9.2f  %10.2f\n",
                "Stdlib   std::exp(x)  [glibc via PLT]",
                r_std.ns_per_call, r_std.total_ms);
        lprintf("     %-42s %9.2f  %10.2f  %7.2fx\n",
                "Homemade fexp::exp(x) [exp.hpp inlined]",
                r_fexp.ns_per_call, r_fexp.total_ms, su_fexp);
        lprintf("     %-42s %9.2f  %10.2f  %7.2fx\n",
                "Inria    cr_exp(x)    [inria-exp.hpp]",
                r_inria.ns_per_call, r_inria.total_ms, su_inria);
    }

    // ── Wall time ─────────────────────────────────────────────────────────────

    auto wall_end = Clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    lprintf("\n\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  Total benchmark wall time: %.3f s\n", wall_s);
    lprintf("══════════════════════════════════════════════════════════════════\n\n");

    if (g_log) {
        std::fclose(g_log);
        std::printf("Results saved to output/bench_results_inria.txt\n");
    }

    return 0;
}
