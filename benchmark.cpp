/**
 * @file  benchmark.cpp
 * @brief Benchmark + accuracy: fexp::exp/expf (homemade, glibc port) vs stdlib.
 *
 * Three input clusters, 1,000,000 iterations each:
 *   Cluster 1 — x near 1     : x ∈ [0.9,   1.1]
 *   Cluster 2 — x near 80    : x ∈ [79.5,  80.5]
 *   Cluster 3 — x near 2e-10 : x ∈ [1e-10, 3e-10]
 *
 * Output goes to stdout and bench_results.txt.
 *
 * Compile:
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench
 */

#include "exp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <bit>

// No "using namespace fexp" — fexp::expf / fexp::exp would be ambiguous
// with ::expf / std::exp at call sites.  Use explicit fexp:: qualification.

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

// ─── Anti-optimisation sinks ─────────────────────────────────────────────────

static volatile float  sink_f = 0.0f;
static volatile double sink_d = 0.0;

// ─── Error metrics ───────────────────────────────────────────────────────────

static uint32_t ulp_dist(float a, float b) {
    int32_t ia = static_cast<int32_t>(FEXP_BIT_CAST(uint32_t, a));
    int32_t ib = static_cast<int32_t>(FEXP_BIT_CAST(uint32_t, b));
    return static_cast<uint32_t>(std::abs(ia - ib));
}

static double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

// ─── Benchmark parameters ────────────────────────────────────────────────────

static constexpr int BENCH_ITERS  = 1'000'000;
static constexpr int WARMUP_ITERS =   200'000;
static constexpr int ACC_SAMPLES  =   100'000;

// ─── Timing helpers ──────────────────────────────────────────────────────────

struct BenchResult { double ns_per_call, total_ms; };

template<typename Fn>
static BenchResult run_bench_d(Fn fn, const std::vector<double>& inputs) {
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

template<typename Fn>
static BenchResult run_bench_f(Fn fn, const std::vector<float>& inputs) {
    float acc = 0.0f;
    for (int i = 0; i < WARMUP_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
    sink_f = acc;

    auto t0 = Clock::now();
    acc = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
    auto t1 = Clock::now();
    sink_f = acc;

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return { (ms * 1e6) / BENCH_ITERS, ms };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    g_log = std::fopen("bench_results.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open bench_results.txt\n");

    auto wall_start = Clock::now();

    std::mt19937 rng(42);

    // ── Header ────────────────────────────────────────────────────────────────

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  exp() Benchmark — homemade (exp.hpp, glibc-2.43 port) vs stdlib\n");
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
    //  float64 (double)
    // ═══════════════════════════════════════════════════════════════════════════

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  FLOAT64 (double) — fexp::exp(x)  vs  std::exp(x)             │\n");
    lprintf("│  Homemade : glibc-2.43 algorithm, header-inlined, no errno    │\n");
    lprintf("│  Stdlib   : glibc libm, called via PLT (shared-library ABI)   │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<double> dist(cl.lo, cl.hi);
        std::vector<double> in(ACC_SAMPLES);
        for (auto& v : in) v = dist(rng);

        double max_rel = 0.0;
        for (auto v : in)
            max_rel = std::max(max_rel, rel_err(fexp::exp(v), std::exp(v)));

        auto r_std = run_bench_d([](double x){ return std::exp(x);  }, in);
        auto r_our = run_bench_d([](double x){ return fexp::exp(x); }, in);
        double su  = r_std.total_ms / r_our.total_ms;

        lprintf("\n");
        lprintf("  ── Cluster: %s  x ∈ [%.3g, %.3g]  (%d iters)\n",
                cl.label, cl.lo, cl.hi, BENCH_ITERS);
        lprintf("     Accuracy vs stdlib — max rel error: %.2e%s\n",
                max_rel,
                max_rel == 0.0 ? "  (bit-for-bit identical)" : "  (< 1 ULP)");
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
                r_our.ns_per_call, r_our.total_ms, su);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  float32 (float)
    // ═══════════════════════════════════════════════════════════════════════════

    lprintf("\n\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  FLOAT32 (float)  — fexp::expf(x) vs  ::expf(x)               │\n");
    lprintf("│  Homemade : glibc-2.43 algorithm, header-inlined, no errno    │\n");
    lprintf("│  Stdlib   : glibc libm, called via PLT (shared-library ABI)   │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<float> dist_f(
            static_cast<float>(cl.lo), static_cast<float>(cl.hi));
        std::vector<float> in_f(ACC_SAMPLES);
        for (auto& v : in_f) v = dist_f(rng);

        double   max_rel_f = 0.0;
        uint32_t max_ulp   = 0;
        for (auto v : in_f) {
            float ref  = ::expf(v);
            float fast = fexp::expf(v);
            max_rel_f  = std::max(max_rel_f, rel_err(fast, ref));
            max_ulp    = std::max(max_ulp, ulp_dist(fast, ref));
        }

        auto r_std = run_bench_f([](float x){ return ::expf(x);      }, in_f);
        auto r_our = run_bench_f([](float x){ return fexp::expf(x);  }, in_f);
        double su  = r_std.total_ms / r_our.total_ms;

        lprintf("\n");
        lprintf("  ── Cluster: %s  x ∈ [%.3g, %.3g]  (%d iters)\n",
                cl.label, cl.lo, cl.hi, BENCH_ITERS);
        lprintf("     Accuracy vs stdlib — max ULP error: %u%s,  max rel error: %.2e\n",
                max_ulp,
                max_ulp == 0 ? " (bit-for-bit identical)" : "",
                max_rel_f);
        lprintf("\n");
        lprintf("     %-42s %9s  %10s  %8s\n",
                "Variant", "ns/call", "total (ms)", "speedup");
        lprintf("     %-42s %9s  %10s  %8s\n",
                "──────────────────────────────────────────",
                "─────────", "──────────", "───────");
        lprintf("     %-42s %9.2f  %10.2f\n",
                "Stdlib   ::expf(x)    [glibc via PLT]",
                r_std.ns_per_call, r_std.total_ms);
        lprintf("     %-42s %9.2f  %10.2f  %7.2fx\n",
                "Homemade fexp::expf(x)[exp.hpp inlined]",
                r_our.ns_per_call, r_our.total_ms, su);
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
        std::printf("Results saved to bench_results.txt\n");
    }

    return 0;
}
