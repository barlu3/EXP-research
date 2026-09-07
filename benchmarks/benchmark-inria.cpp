/**
 * @file  benchmark-inria.cpp
 * @brief Benchmark: Inria cr_exp (CORE-MATH) vs homemade fexp::exp vs std::exp.
 *
 * FLOAT64 only -- cr_exp provides double precision only.
 *
 * Two sections, both over the three clusters in bench-clusters.hpp:
 *   1. Whole-call comparison of the three implementations.
 *   2. The Inria path split into its three stages, with the inputs to each
 *      stage precomputed so only that stage runs inside the timed loop.
 *
 * Measurement lives in bench-harness.hpp. Unlike the bf16 benchmarks these
 * inputs need no special handling -- 500k draws from [0.9, 1.1] land on ~500k
 * distinct doubles, so sampling a real interval does not starve the way it does
 * at 7 mantissa bits. What this file does share with them is the estimator:
 * medians with an IQR taken across separate processes, because code layout is
 * fixed within a process and re-drawn by ASLR across them.
 */

#include "exp.hpp"
#include "inria-exp-seg.hpp"

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct PolyIn   { double th, tl, dx; };
struct StitchIn { double x, fh, fl; i64 ie; b64u64_u ix; };

static constexpr int BENCH_ITERS  = 2'000'000;
static constexpr int WARMUP_ITERS =   200'000;
static constexpr int REPS         = bench::DEFAULT_REPS;
static constexpr int EPOCHS       = bench::DEFAULT_EPOCHS;
static constexpr std::size_t BUFFER_N = 16384;

// Cluster ids are offset so both sections share one epoch table.
static constexpr int SEG_BASE = 10;

static double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

static int run_epoch() {
    std::mt19937 rng(42);
    int ci = 0;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto buf = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng, BUFFER_N);
        auto s = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](double x) { return x; },             // control
            [](double x) { return std::exp(x); },
            [](double x) { return fexp::exp(x); },
            [](double x) { return cr_exp(x); });
        for (std::size_t v = 0; v < s.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(s[v]));
        ++ci;
    }

    std::mt19937 rng2(42);
    ci = SEG_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        auto in = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng2, BUFFER_N);

        auto poly = bench::make_generated_buffer<PolyIn>(BUFFER_N, [&](std::size_t i) {
            double th, tl, dx; i64 ie;
            exp_range_reduce(in.storage[i], &th, &tl, &dx, &ie);
            return PolyIn{th, tl, dx};
        });
        auto stitch = bench::make_generated_buffer<StitchIn>(BUFFER_N, [&](std::size_t i) {
            double th, tl, dx; i64 ie;
            exp_range_reduce(in.storage[i], &th, &tl, &dx, &ie);
            double fh, fl;
            exp_poly_expand(th, tl, dx, &fh, &fl);
            b64u64_u ix = {.f = in.storage[i]};
            return StitchIn{in.storage[i], fh, fl, ie, ix};
        });

        auto s_rr = bench::run_interleaved(
            in, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](double x) { return x; },
            [](double x) { double th, tl, dx; i64 ie;
                           exp_range_reduce(x, &th, &tl, &dx, &ie);
                           return th + tl + dx; });
        auto s_poly = bench::run_interleaved(
            poly, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](const PolyIn& p) { return p.th; },
            [](const PolyIn& p) { double fh, fl;
                                  exp_poly_expand(p.th, p.tl, p.dx, &fh, &fl);
                                  return fh + fl; });
        auto s_st = bench::run_interleaved(
            stitch, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](const StitchIn& s) { return s.x; },
            [](const StitchIn& s) { return exp_stitch(s.x, s.fh, s.fl, s.ie, s.ix); });

        // variant 0 is the control for the stage timings; 1..3 the stages.
        bench::emit_epoch_row(ci, 0, bench::median_of(s_rr[0]));
        bench::emit_epoch_row(ci, 1, bench::median_of(s_rr[1]));
        bench::emit_epoch_row(ci, 2, bench::median_of(s_poly[1]));
        bench::emit_epoch_row(ci, 3, bench::median_of(s_st[1]));
        ++ci;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch();

    bench::open_log("output/bench_results_inria.txt");
    const auto wall_start = bench::Clock::now();

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  float64 exp() Benchmark -- Inria cr_exp vs fexp::exp vs stdlib\n");
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
    bench::logf("|  FLOAT64 -- cr_exp(x), fexp::exp(x), std::exp(x)               |\n");
    bench::logf("|  Inria    : CORE-MATH cr_exp, correctly rounded (<=0.5 ULP)    |\n");
    bench::logf("|  Homemade : glibc-2.43 algorithm, header-inlined, no errno     |\n");
    bench::logf("|  Stdlib   : libm, called through the shared-library ABI        |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    std::mt19937 rng(42);
    int ci = 0;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        // Accuracy is deterministic, so it is checked once here rather than in
        // every epoch.
        auto acc_in = bench::make_real_cycle_buffer<double>(cl.lo, cl.hi, rng, 4096);
        double max_fexp = 0.0, max_inria = 0.0;
        for (std::size_t i = 0; i < acc_in.n; ++i) {
            const double v = acc_in.storage[i];
            max_fexp  = std::max(max_fexp,  rel_err(fexp::exp(v), std::exp(v)));
            max_inria = std::max(max_inria, rel_err(cr_exp(v),    std::exp(v)));
        }
        bench::logf("\n     accuracy vs std::exp -- homemade %.2e  inria %.2e%s\n",
                    max_fexp, max_inria,
                    max_inria == 0.0 ? "  (bit-for-bit identical)" : "");

        const auto ctl = at(ci, 0), std_e = at(ci, 1), fx = at(ci, 2), cr = at(ci, 3);
        if (ctl.empty() || std_e.empty() || fx.empty() || cr.empty()) {
            bench::logf("  -- %s: incomplete epoch data\n", cl.label); ++ci; continue;
        }
        bench::report_cluster(cl.label, BUFFER_N, ctl,
                              { { "std::exp",   std_e },
                                { "fexp::exp",  fx },
                                { "cr_exp",     cr } },
                              /*base_index=*/0);
        ++ci;
    }

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  FLOAT64 -- Inria cr_exp segmented phases                      |\n");
    bench::logf("|  Range Reduce : exp_range_reduce()  (decompose x)              |\n");
    bench::logf("|  Poly Expand  : exp_poly_expand()   (evaluate polynomial)      |\n");
    bench::logf("|  Stitch       : exp_stitch()        (assemble IEEE result)     |\n");
    bench::logf("|  Stage inputs are precomputed, so only the stage is timed.     |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    ci = SEG_BASE;
    for (const auto& cl : bench::REAL_CLUSTERS) {
        const auto ctl = at(ci, 0);
        if (ctl.empty() || at(ci, 1).empty() || at(ci, 2).empty() || at(ci, 3).empty()) {
            bench::logf("  -- %s: incomplete epoch data\n", cl.label); ++ci; continue;
        }
        bench::report_cluster(cl.label, BUFFER_N, ctl,
                              { { "range reduce", at(ci, 1) },
                                { "poly expand",  at(ci, 2) },
                                { "stitch",       at(ci, 3) } },
                              /*base_index=*/0);
        ++ci;
    }

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);

    bench::close_log();
    return 0;
}
