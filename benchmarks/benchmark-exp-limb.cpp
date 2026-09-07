/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for exp().

   Three variants, all correctly rounded on every bfloat16 input (verified
   exhaustively against MPFR by cross-eval/verify-limb.c), so this measures
   cost, not accuracy:

     cr_exp_bf16           2 float32 loads, 1 multiply
     cr_exp_bf16_limb      3x3 limbs: 6 bf16 loads, 4 adds, 1 multiply
     cr_exp_bf16_limb_min  2x2 limbs: 4 bf16 loads, 2 adds, 1 multiply

   Measurement lives in bench-harness.hpp and the input ranges in
   bench-clusters.hpp. Two things there matter for reading these numbers:

     - The sweep clusters stay inside the table path, 2^-9 < |x| < 93. Outside
       it exp returns a constant without a lookup, and roughly 15000 codepoints
       per sign sit below 2^-9 alone -- sweeping "every finite bf16" would have
       measured mostly early exits.

     - The outer unit of replication is a process, not a rep. Code layout is
       fixed within a process and re-drawn by ASLR across them, and it moves
       these ratios further than anything else does.

   As with ln, the storage saving itself is invisible here: every table under
   comparison fits in L1 many times over. What this measures is the arithmetic
   the limb scheme costs to buy it. */

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
__bf16 cr_exp_bf16 (__bf16 x);
__bf16 cr_exp_bf16_limb (__bf16 x);
__bf16 cr_exp_bf16_limb_min (__bf16 x);
}

static constexpr int BENCH_ITERS  = 10'000'000;
static constexpr int WARMUP_ITERS =  1'000'000;
static constexpr int REPS         = bench::DEFAULT_REPS;
static constexpr int EPOCHS       = bench::DEFAULT_EPOCHS;
static constexpr std::size_t MIN_BUFFER = 4096;

static int run_epoch() {
    std::mt19937 rng(42);
    int ci = 0;
    for (const auto& cl : bench::EXP_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        const auto buf = bench::make_bf16_cycle_buffer(cps, rng, MIN_BUFFER);
        auto series = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](__bf16 x) { return (double)(float)x; },                 // control
            [](__bf16 x) { return (double)cr_exp_bf16(x); },           // baseline
            [](__bf16 x) { return (double)cr_exp_bf16_limb(x); },      // 3x3
            [](__bf16 x) { return (double)cr_exp_bf16_limb_min(x); }); // 2x2
        for (std::size_t v = 0; v < series.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(series[v]));
        ++ci;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch();

    bench::open_log("output/bench_results_exp_limb.txt");
    const auto wall_start = bench::Clock::now();

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  bf16 exp() Benchmark -- limb tables vs CORE-MATH float32 tables\n");
    bench::print_method_banner(BENCH_ITERS, REPS, WARMUP_ITERS,
        "the distinct bf16 codepoints in each range, shuffled, "
        "not repeated draws from a real interval");
    bench::logf("  Epochs        : %d separate processes; medians taken across them\n", EPOCHS);
    bench::logf("  All three variants are correctly rounded on all 65536 inputs.\n");
    bench::logf("==================================================================\n");

    // -- Table storage comparison --------------------------------------------
    // T1: 512 entries, T2: 256.
    const long f32_bytes   = (512L * 4) + (256L * 4);
    const long exact_bytes = (512L * 3 * 2) + (256L * 3 * 2);
    const long min_bytes   = (512L * 2 * 2) + (256L * 2 * 2);

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  TABLE STORAGE                                                 |\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("  %-24s %10s %10s %10s\n", "table", "float32", "3x3 limb", "2x2 limb");
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "T1 (512 entries)", 512L*4, 512L*3*2, 512L*2*2);
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "T2 (256 entries)", 256L*4, 256L*3*2, 256L*2*2);
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "total", f32_bytes, exact_bytes, min_bytes);
    bench::logf("  %-24s %10s %+9.1f%% %+9.1f%%\n", "vs float32", "",
                100.0 * (double)(exact_bytes - f32_bytes) / (double)f32_bytes,
                100.0 * (double)(min_bytes   - f32_bytes) / (double)f32_bytes);

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  BFLOAT16 -- cr_exp_bf16 (float32) vs _limb (3x3) vs _min (2x2)|\n");
    bench::logf("|  inria : 2 float32 loads, 1 multiply                           |\n");
    bench::logf("|  3x3   : 6 bf16 loads, 4 adds, 1 multiply                      |\n");
    bench::logf("|  2x2   : 4 bf16 loads, 2 adds, 1 multiply                      |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    const bench::EpochTable table = bench::gather_epochs(argv[0], EPOCHS);

    std::vector<double> r3_sweep, r2_sweep;
    int ci = 0;
    for (const auto& cl : bench::EXP_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        auto at = [&](int v) {
            auto it = table.find({ci, v});
            return it == table.end() ? std::vector<double>{} : it->second;
        };
        const auto ctl = at(0), inria = at(1), x3 = at(2), x2 = at(3);
        if (ctl.empty() || inria.empty() || x3.empty() || x2.empty()) {
            bench::logf("\n  -- %-16s no epoch data (child failed)\n", cl.label);
            ++ci;
            continue;
        }

        bench::report_cluster(cl.label, cps.size(), ctl,
                              { { "inria (f32)", inria },
                                { "3x3 limb",    x3 },
                                { "2x2 limb",    x2 } },
                              /*base_index=*/0);

        if (cl.sweep) {
            r3_sweep.push_back(bench::median_of(bench::paired_ratio(x3, inria)));
            r2_sweep.push_back(bench::median_of(bench::paired_ratio(x2, inria)));
        }
        ++ci;
    }

    bench::logf("\n  table-path sweep vs float32: 3x3 %.2fx, 2x2 %.2fx\n",
                bench::median_of(r3_sweep), bench::median_of(r2_sweep));
    bench::logf("  The sweep rows are the ones to quote; the narrow clusters above\n");
    bench::logf("  keep one or two cache lines hot and measure a corner of the table.\n");

    // -- Exhaustive agreement check -------------------------------------------
    long mism_x = 0, mism_m = 0;
    for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
        const __bf16 x = bench::bf16_from_bits((std::uint16_t)b);
        const std::uint16_t a = bench::bf16_to_bits(cr_exp_bf16(x));
        const std::uint16_t e = bench::bf16_to_bits(cr_exp_bf16_limb(x));
        const std::uint16_t m = bench::bf16_to_bits(cr_exp_bf16_limb_min(x));
        const bool a_nan = (a & 0x7fff) > 0x7f80;
        if (!(a_nan && (e & 0x7fff) > 0x7f80) && a != e) mism_x++;
        if (!(a_nan && (m & 0x7fff) > 0x7f80) && a != m) mism_m++;
    }
    bench::logf("\n  exhaustive agreement over 65536 inputs:\n");
    bench::logf("    3x3 exact vs CORE-MATH : %s (%ld mismatches)\n",
                mism_x == 0 ? "IDENTICAL" : "DIFFER", mism_x);
    bench::logf("    2x2 min   vs CORE-MATH : %s (%ld mismatches)\n",
                mism_m == 0 ? "IDENTICAL" : "DIFFER", mism_m);

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);

    bench::close_log();
    return (mism_x == 0 && mism_m == 0) ? 0 : 1;
}
