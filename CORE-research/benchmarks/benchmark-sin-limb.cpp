/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for sin().

   Three variants, all correctly rounded on every bfloat16 input (verified
   exhaustively against MPFR by cross-eval/verify-limb.c), so this measures
   cost, not accuracy: cr_sin_bf16 against the exact and minimal limb tables.

   sin has three code paths and they cost very different amounts:

     |x| <= 0x1.dp-4   sin(x) rounds to x. No table in any variant, so this
                       cluster is a control: its ratio is the harness noise
                       floor, not a cost of the limb scheme.
     |x| <  4096       mid path, 4 lookups reconstructed per call.
     |x| >= 4096       large path, up to 8 sin/cos pairs.

   The two sweeps are kept separate for that reason -- averaging a 4-lookup
   path with a 16-reconstruction one would hide which the limb tables affect.

   Measurement lives in bench-harness.hpp, input ranges in bench-clusters.hpp.
   The outer unit of replication is a process: code layout is fixed within one
   and re-drawn by ASLR across them, and it moves these ratios more than
   anything else does. */

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
__bf16 cr_sin_bf16 (__bf16 x);
__bf16 cr_sin_bf16_limb (__bf16 x);
__bf16 cr_sin_bf16_limb_min (__bf16 x);
}

static constexpr int BENCH_ITERS  = 10'000'000;
static constexpr int WARMUP_ITERS =  1'000'000;
static constexpr int REPS         = bench::DEFAULT_REPS;
static constexpr int EPOCHS       = bench::DEFAULT_EPOCHS;
static constexpr std::size_t MIN_BUFFER = 4096;

static int run_epoch() {
    std::mt19937 rng(42);
    int ci = 0;
    for (const auto& cl : bench::SIN_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        const auto buf = bench::make_bf16_cycle_buffer(cps, rng, MIN_BUFFER);
        auto series = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](__bf16 x) { return (double)(float)x; },
            [](__bf16 x) { return (double)cr_sin_bf16(x); },
            [](__bf16 x) { return (double)cr_sin_bf16_limb(x); },
            [](__bf16 x) { return (double)cr_sin_bf16_limb_min(x); });
        for (std::size_t v = 0; v < series.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(series[v]));
        ++ci;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch();

    bench::open_log("output/bench_results_sin_limb.txt");
    const auto wall_start = bench::Clock::now();

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  bf16 sin() Benchmark -- limb tables vs CORE-MATH float32 tables\n");
    bench::print_method_banner(BENCH_ITERS, REPS, WARMUP_ITERS,
        "the distinct bf16 codepoints in each range, shuffled, "
        "not repeated draws from a real interval");
    bench::logf("  Epochs        : %d separate processes; medians taken across them\n", EPOCHS);
    bench::logf("  All three variants are correctly rounded on all 65536 inputs.\n");
    bench::logf("==================================================================\n");

    // -- Table storage comparison --------------------------------------------
    // S1/C1: 256 entries each, S2/C2: 128 each, S3/C3: 123 each.
    const long f32_bytes   = (256L*2 + 128L*2 + 123L*2) * 4;
    const long exact_bytes = (256L*2 + 128L*2 + 123L*2) * 3 * 2;
    const long min_bytes   = (256L*2*2 + 128L*2*3 + 123L*2*3) * 2;

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  TABLE STORAGE                                                 |\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("  %-24s %10s %10s %10s\n", "table", "float32", "exact", "minimal");
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "S1+C1 (256 each)", 256L*2*4, 256L*2*3*2, 256L*2*2*2);
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "S2+C2 (128 each)", 128L*2*4, 128L*2*3*2, 128L*2*3*2);
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "S3+C3 (123 each)", 123L*2*4, 123L*2*3*2, 123L*2*3*2);
    bench::logf("  %-24s %9ldB %9ldB %9ldB\n", "total", f32_bytes, exact_bytes, min_bytes);
    bench::logf("  %-24s %10s %+9.1f%% %+9.1f%%\n", "vs float32", "",
                100.0 * (double)(exact_bytes - f32_bytes) / (double)f32_bytes,
                100.0 * (double)(min_bytes   - f32_bytes) / (double)f32_bytes);

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  BFLOAT16 -- cr_sin_bf16 (float32) vs _limb (exact) vs _min    |\n");
    bench::logf("|  small |x| : no table in any variant -- control cluster        |\n");
    bench::logf("|  mid  path : 4 lookups   -> 4 reconstructions per call         |\n");
    bench::logf("|  large path: <=8 pairs   -> <=16 reconstructions per call      |\n");
    bench::logf("+----------------------------------------------------------------+\n");

    const bench::EpochTable table = bench::gather_epochs(argv[0], EPOCHS);

    int ci = 0;
    for (const auto& cl : bench::SIN_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        auto at = [&](int v) {
            auto it = table.find({ci, v});
            return it == table.end() ? std::vector<double>{} : it->second;
        };
        const auto ctl = at(0), inria = at(1), ex = at(2), mn = at(3);
        if (ctl.empty() || inria.empty() || ex.empty() || mn.empty()) {
            bench::logf("\n  -- %-16s no epoch data (child failed)\n", cl.label);
            ++ci;
            continue;
        }
        bench::report_cluster(cl.label, cps.size(), ctl,
                              { { "inria (f32)", inria },
                                { "exact limb",  ex },
                                { "min limb",    mn } },
                              /*base_index=*/0);
        ++ci;
    }

    bench::logf("\n  The two sweep rows are the ones to quote, and they are reported\n");
    bench::logf("  separately: the mid path does 4 reconstructions per call and the\n");
    bench::logf("  large path up to 16, so a combined figure would describe neither.\n");
    bench::logf("  The 'no-table ctl' row takes no table in any variant, so its ratio\n");
    bench::logf("  is the harness noise floor rather than a cost of the limb scheme.\n");

    // -- Exhaustive agreement check -------------------------------------------
    long mism_x = 0, mism_m = 0;
    for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
        const __bf16 x = bench::bf16_from_bits((std::uint16_t)b);
        const std::uint16_t a = bench::bf16_to_bits(cr_sin_bf16(x));
        const std::uint16_t e = bench::bf16_to_bits(cr_sin_bf16_limb(x));
        const std::uint16_t m = bench::bf16_to_bits(cr_sin_bf16_limb_min(x));
        const bool a_nan = (a & 0x7fff) > 0x7f80;
        if (!(a_nan && (e & 0x7fff) > 0x7f80) && a != e) mism_x++;
        if (!(a_nan && (m & 0x7fff) > 0x7f80) && a != m) mism_m++;
    }
    bench::logf("\n  exhaustive agreement over 65536 inputs:\n");
    bench::logf("    exact vs CORE-MATH : %s (%ld mismatches)\n",
                mism_x == 0 ? "IDENTICAL" : "DIFFER", mism_x);
    bench::logf("    min   vs CORE-MATH : %s (%ld mismatches)\n",
                mism_m == 0 ? "IDENTICAL" : "DIFFER", mism_m);

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);

    bench::close_log();
    return (mism_x == 0 && mism_m == 0) ? 0 : 1;
}
