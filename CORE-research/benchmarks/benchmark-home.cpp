/**
 * @file  benchmark-home.cpp
 * @brief Benchmark + accuracy: fexp::exp/expf (homemade, glibc port) vs stdlib.
 *
 * Three sections, each over the three clusters in bench-clusters.hpp:
 *
 * Section 1 -- FLOAT64 full call: times the complete fexp::exp(x) path --
 *   argument check, Cody-Waite 2-part range reduction, 128-entry paired table
 *   lookup, degree-4 polynomial (C2..C5), and final scale-and-add.
 *
 * Section 2 -- FLOAT32 full call: times the complete fexp::expf(x) path --
 *   argument check, shift-trick range reduction, 32-entry table lookup,
 *   degree-3 polynomial (all arithmetic in double), and float cast.
 *
 * Section 3 -- Polynomial isolation: times only the polynomial evaluation step.
 *   Inputs are pre-reduced via fexp::exp_reduce / fexp::expf_reduce (outside the
 *   timed region) so argument checks, table lookups, and final scaling are
 *   excluded. Coefficients and evaluation order match glibc-2.43.
 *
 * Measurement lives in bench-harness.hpp: medians with an IQR taken across
 * separate processes, since code layout is fixed within a process and re-drawn
 * by ASLR across them. Each section carries a control kernel that only touches
 * the input, so the loop's own cost can be subtracted rather than attributed to
 * the function under test.
 *
 * Output goes to stdout and output/bench_results.txt.
 */

#include "exp.hpp"

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int BENCH_ITERS  = 2'000'000;
static constexpr int WARMUP_ITERS =   200'000;
static constexpr int REPS         = bench::DEFAULT_REPS;
static constexpr int EPOCHS       = bench::DEFAULT_EPOCHS;
static constexpr std::size_t BUFFER_N = 16384;

// Cluster ids are offset per section so all four share one epoch table.
static constexpr int F64_BASE   =  0;
static constexpr int F32_BASE   = 10;
static constexpr int POLY64_BASE = 20;
static constexpr int POLY32_BASE = 30;

static std::uint32_t ulp_dist(float a, float b) {
    std::int32_t ia = static_cast<std::int32_t>(FEXP_BIT_CAST(std::uint32_t, a));
    std::int32_t ib = static_cast<std::int32_t>(FEXP_BIT_CAST(std::uint32_t, b));
    return static_cast<std::uint32_t>(std::abs(ia - ib));
}

static double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

static int run_epoch() {
    std::mt19937 rng(42);

    int ci = F64_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto buf = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng, BUFFER_N);
        auto s = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](double x) { return x; },
            [](double x) { return std::exp(x); },
            [](double x) { return fexp::exp(x); });
        for (std::size_t v = 0; v < s.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(s[v]));
        ++ci;
    }

    ci = F32_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto buf = bench::make_real_cycle_buffer<float>(cl.lo, cl.hi, rng, BUFFER_N);
        auto s = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](float x) { return (double)x; },
            [](float x) { return (double)::expf(x); },
            [](float x) { return (double)fexp::expf(x); });
        for (std::size_t v = 0; v < s.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(s[v]));
        ++ci;
    }

    // Reduction happens outside the timed region; only the polynomial is timed.
    ci = POLY64_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto in = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng, BUFFER_N);
        auto red = bench::make_generated_buffer<double>(BUFFER_N, [&](std::size_t i) {
            return fexp::exp_reduce(in.storage[i]);
        });
        auto s = bench::run_interleaved(
            red, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](double r) { return r; },
            [](double r) { return fexp::exp_poly(r); });
        bench::emit_epoch_row(ci, 0, bench::median_of(s[0]));
        bench::emit_epoch_row(ci, 1, bench::median_of(s[1]));
        ++ci;
    }

    ci = POLY32_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto in = bench::make_real_cycle_buffer<float>(cl.lo, cl.hi, rng, BUFFER_N);
        auto red = bench::make_generated_buffer<double>(BUFFER_N, [&](std::size_t i) {
            return fexp::expf_reduce(in.storage[i]);
        });
        auto s = bench::run_interleaved(
            red, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](double r) { return r; },
            [](double r) { return (double)fexp::expf_poly(r); });
        bench::emit_epoch_row(ci, 0, bench::median_of(s[0]));
        bench::emit_epoch_row(ci, 1, bench::median_of(s[1]));
        ++ci;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch();

    bench::open_log("output/bench_results.txt");
    const auto wall_start = bench::Clock::now();

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  exp() Benchmark -- homemade fexp vs stdlib\n");
    bench::print_method_banner(BENCH_ITERS, REPS, WARMUP_ITERS,
        "sampled from each real interval; float64 inputs do not "
        "collapse onto a few values the way bf16 does");
    bench::logf("  Epochs        : %d separate processes; medians taken across them\n", EPOCHS);
    bench::logf("==================================================================\n");

    const bench::EpochTable table = bench::gather_epochs(argv[0], EPOCHS);
    auto at = [&](int c, int v) {
        auto it = table.find({c, v});
        return it == table.end() ? std::vector<double>{} : it->second;
    };

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  FLOAT64 (double) -- fexp::exp(x)  vs  std::exp(x)             |\n");
    bench::logf("|  Homemade : glibc-2.43 algorithm, header-inlined, no errno     |\n");
    bench::logf("|  Stdlib   : libm, called through the shared-library ABI        |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    std::mt19937 rng(42);
    int ci = F64_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto acc_in = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng, 4096);
        double max_rel = 0.0;
        for (std::size_t i = 0; i < acc_in.n; ++i)
            max_rel = std::max(max_rel,
                               rel_err(fexp::exp(acc_in.storage[i]),
                                       std::exp(acc_in.storage[i])));
        bench::logf("\n     accuracy vs std::exp -- max relative error %.2e\n", max_rel);
        if (at(ci, 0).empty() || at(ci, 1).empty() || at(ci, 2).empty())
            bench::logf("  -- %s: incomplete epoch data\n", cl.label);
        else
            bench::report_cluster(cl.label, BUFFER_N, at(ci, 0),
                                  { { "std::exp",  at(ci, 1) },
                                    { "fexp::exp", at(ci, 2) } }, 0);
        ++ci;
    }

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  FLOAT32 (float)  -- fexp::expf(x) vs  ::expf(x)               |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    ci = F32_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto acc_in = bench::make_real_cycle_buffer<float>(cl.lo, cl.hi, rng, 4096);
        double max_rel = 0.0;
        std::uint32_t max_ulp = 0;
        for (std::size_t i = 0; i < acc_in.n; ++i) {
            const float v = acc_in.storage[i];
            const float ref = ::expf(v), fast = fexp::expf(v);
            max_rel = std::max(max_rel, rel_err(fast, ref));
            max_ulp = std::max(max_ulp, ulp_dist(fast, ref));
        }
        bench::logf("\n     accuracy vs ::expf -- max relative error %.2e, max %u ULP\n",
                    max_rel, max_ulp);
        if (at(ci, 0).empty() || at(ci, 1).empty() || at(ci, 2).empty())
            bench::logf("  -- %s: incomplete epoch data\n", cl.label);
        else
            bench::report_cluster(cl.label, BUFFER_N, at(ci, 0),
                                  { { "::expf",     at(ci, 1) },
                                    { "fexp::expf", at(ci, 2) } }, 0);
        ++ci;
    }

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  POLYNOMIAL ISOLATION -- homemade poly step only               |\n");
    bench::logf("|  float64 degree-4: r + r2*(C2+r*C3) + r2*r2*(C4+r*C5)          |\n");
    bench::logf("|  float32 degree-3: (C0*r+C1)*r2 + C2*r + 1  (in double)        |\n");
    bench::logf("|  Reduced args precomputed; table lookup + scaling excluded.    |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    bench::logf("\n  float64 -- fexp::exp_poly(r):\n");
    ci = POLY64_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        if (at(ci, 0).empty() || at(ci, 1).empty())
            bench::logf("  -- %s: incomplete epoch data\n", cl.label);
        else
            bench::report_cluster(cl.label, BUFFER_N, at(ci, 0),
                                  { { "exp_poly", at(ci, 1) } }, 0);
        ++ci;
    }

    bench::logf("\n  float32 -- fexp::expf_poly(r):\n");
    ci = POLY32_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        if (at(ci, 0).empty() || at(ci, 1).empty())
            bench::logf("  -- %s: incomplete epoch data\n", cl.label);
        else
            bench::report_cluster(cl.label, BUFFER_N, at(ci, 0),
                                  { { "expf_poly", at(ci, 1) } }, 0);
        ++ci;
    }

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);

    bench::close_log();
    return 0;
}
