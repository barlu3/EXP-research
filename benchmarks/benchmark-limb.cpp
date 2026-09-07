/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for ln().

   Both implementations are correctly rounded on every bfloat16 input (verified
   exhaustively against MPFR by cross-eval/log/verify-limb.c and
   verify_mpfr.c), so this measures cost, not accuracy.

   What differs. cr_log_bf16 reads two float32 entries and adds them.
   cr_log_bf16_limb reads two bf16 limbs for T1 and two for T2 and sums all
   four in float32. It trades loads and adds for the ability to keep the whole
   table in bf16 storage -- the point of the scheme, since it targets hardware
   with bf16 storage or bf16 MACs and no float32 table path.

   Table sizes are reported alongside and are derived from LOGBF16_T*_LIMBS in
   the generated header, so this report cannot drift from the shipped table:
   the limb scheme matches float32 for T1 and T2 (32 bits/entry) and halves T3
   (16 vs 32), for a 12.5% overall saving.

   Measurement lives in bench-harness.hpp and the input ranges in
   bench-clusters.hpp; see those for what the numbers mean and why the ranges
   are shaped as they are.

   A caveat this benchmark cannot measure away: the two tables are 2036 B and
   1782 B, and any machine this runs on holds both in L1 many times over. The
   storage saving is therefore invisible here by construction. What these
   numbers show is what the limb scheme *costs* in arithmetic to buy it. */

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// Pulled in for LOGBF16_T1_LIMBS / LOGBF16_T2_LIMBS so the storage report and
// the operation-count banner below track the generated table automatically.
// The tables themselves are unused in this translation unit.
#include "logbf16-limb.h"

extern "C" {
__bf16 cr_log_bf16 (__bf16 x);
__bf16 cr_log_bf16_limb (__bf16 x);
}

static constexpr int BENCH_ITERS  = 10'000'000;
static constexpr int WARMUP_ITERS =  1'000'000;
static constexpr int REPS         = bench::DEFAULT_REPS;
static constexpr int EPOCHS       = bench::DEFAULT_EPOCHS;

// Input buffer floor. Clusters with fewer distinct codepoints than this cycle a
// reshuffled permutation to fill it; the full-domain cluster overrides it with
// its own 65536. 4096 bf16 values is 8 KB, small enough that streaming the
// inputs stays L1-resident and the table access is what varies between
// variants.
static constexpr std::size_t MIN_BUFFER = 4096;

// One epoch: measure every cluster in this process and print the per-variant
// medians. The parent aggregates across epochs; see bench-harness.hpp for why
// the outer unit of replication has to be a process.
static int run_epoch() {
    std::mt19937 rng(42);
    int ci = 0;
    for (const auto& cl : bench::LOG_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        const auto buf = bench::make_bf16_cycle_buffer(cps, rng, MIN_BUFFER);
        auto series = bench::run_interleaved(
            buf, BENCH_ITERS, REPS, WARMUP_ITERS,
            [](__bf16 x) { return (double)(float)x; },            // control
            [](__bf16 x) { return (double)cr_log_bf16(x); },      // baseline
            [](__bf16 x) { return (double)cr_log_bf16_limb(x); });
        for (std::size_t v = 0; v < series.size(); ++v)
            bench::emit_epoch_row(ci, (int)v, bench::median_of(series[v]));
        ++ci;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch();

    bench::open_log("output/bench_results_limb.txt");
    const auto wall_start = bench::Clock::now();

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  bf16 log() Benchmark -- limb tables vs CORE-MATH float32 tables\n");
    bench::print_method_banner(BENCH_ITERS, REPS, WARMUP_ITERS,
        "the distinct bf16 codepoints in each range, shuffled, "
        "not repeated draws from a real interval");
    bench::logf("  Epochs        : %d separate processes; medians taken across them\n", EPOCHS);
    bench::logf("  Both variants are correctly rounded on all 65536 inputs.\n");
    bench::logf("==================================================================\n");

    // -- Table storage comparison --------------------------------------------
    // T1: 254 usable entries, T2: 128, T3: 127.
    const long t1_limb_b = 254L * LOGBF16_T1_LIMBS * 2;
    const long t2_limb_b = 128L * LOGBF16_T2_LIMBS * 2;
    const long t3_limb_b = 127L * 1 * 2;
    const long f32_bytes  = (254L * 4) + (128L * 4) + (127L * 4);
    const long limb_bytes = t1_limb_b + t2_limb_b + t3_limb_b;

    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  TABLE STORAGE                                                 |\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("  %-28s %10s %10s\n", "table", "float32", "bf16 limb");
    bench::logf("  %-28s %9ldB %9ldB\n", "T1 (254 entries)", 254L*4, t1_limb_b);
    bench::logf("  %-28s %9ldB %9ldB\n", "T2 (128 entries)", 128L*4, t2_limb_b);
    bench::logf("  %-28s %9ldB %9ldB\n", "T3 (127 entries)", 127L*4, t3_limb_b);
    bench::logf("  %-28s %9ldB %9ldB   (%+.1f%%)\n", "total", f32_bytes, limb_bytes,
                100.0 * (double)(limb_bytes - f32_bytes) / (double)f32_bytes);

    // -- Timing ---------------------------------------------------------------
    bench::logf("\n");
    bench::logf("+----------------------------------------------------------------+\n");
    bench::logf("|  BFLOAT16 -- cr_log_bf16 (float32 tables) vs _limb (bf16)      |\n");
    bench::logf("|  inria : 2 float32 loads, 1 add                                |\n");
    bench::logf("|  limb  : %d bf16 loads, %d adds (float32 accumulation)%*s|\n",
                LOGBF16_T1_LIMBS + LOGBF16_T2_LIMBS,
                LOGBF16_T1_LIMBS + LOGBF16_T2_LIMBS - 1, 12, "");
    bench::logf("+----------------------------------------------------------------+\n");

    // Each child re-runs this binary; ns/call values below are one median per
    // epoch, so the IQR spans process layouts rather than reps inside one.
    const bench::EpochTable table = bench::gather_epochs(argv[0], EPOCHS);

    std::vector<double> gross_ratios, net_ratios;
    double sweep_gross = 0.0, sweep_net = 0.0;
    int ci = 0;

    for (const auto& cl : bench::LOG_CLUSTERS) {
        const auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
        auto at = [&](int v) {
            auto it = table.find({ci, v});
            return it == table.end() ? std::vector<double>{} : it->second;
        };
        const auto ctl = at(0), inria = at(1), limb = at(2);
        if (ctl.empty() || inria.empty() || limb.empty()) {
            bench::logf("\n  -- %-16s no epoch data (child failed)\n", cl.label);
            ++ci;
            continue;
        }

        bench::report_cluster(cl.label, cps.size(), ctl,
                              { { "inria (f32)", inria },
                                { "limb (bf16)", limb } },
                              /*base_index=*/0);

        const double g = bench::median_of(bench::paired_ratio(limb, inria));
        const double n = bench::median_of(bench::paired_net_ratio(limb, inria, ctl));
        gross_ratios.push_back(g);
        net_ratios.push_back(n);
        if (cl.sweep) { sweep_gross = g; sweep_net = n; }
        ++ci;
    }

    bench::logf("\n  domain-sweep limb/inria ratio: %.2fx gross, %.2fx net of control\n",
                sweep_gross, sweep_net);
    bench::logf("  median across clusters       : %.2fx gross, %.2fx net of control\n",
                bench::median_of(gross_ratios), bench::median_of(net_ratios));
    bench::logf("\n  The domain-sweep row is the one to quote. It walks every x > 0 in\n");
    bench::logf("  shuffled order, so the whole table is in play; the narrow clusters\n");
    bench::logf("  above keep one or two cache lines hot and measure a corner of it.\n");
    bench::logf("  x <= 0 is excluded deliberately -- log returns NaN or -Inf there\n");
    bench::logf("  without a lookup, and including it measures the early exit.\n");

    // -- Exhaustive agreement check -------------------------------------------
    // Cheap here and worth doing: a benchmark that silently drifted from the
    // verified build would report meaningless timings.
    long mismatches = 0;
    for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
        const auto u = (std::uint16_t)b;
        const __bf16 x = bench::bf16_from_bits(u);
        const std::uint16_t a = bench::bf16_to_bits(cr_log_bf16(x));
        const std::uint16_t c = bench::bf16_to_bits(cr_log_bf16_limb(x));
        const bool a_nan = (a & 0x7fff) > 0x7f80, c_nan = (c & 0x7fff) > 0x7f80;
        if (a_nan && c_nan) continue;
        if (a != c) mismatches++;
    }
    bench::logf("\n  exhaustive agreement over 65536 inputs: %s (%ld mismatches)\n",
                mismatches == 0 ? "IDENTICAL" : "DIFFER", mismatches);

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);

    bench::close_log();
    return mismatches == 0 ? 0 : 1;
}
