/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for exp().

   Companion to benchmark-limb.cpp, which does the same for ln(). All three
   implementations are correctly rounded on every bfloat16 input -- verified
   exhaustively against MPFR by cross-eval/verify-limb.c -- so this measures
   cost, not accuracy.

   What differs. cr_exp_bf16 reads two float32 entries and multiplies them.
   The limb variants read bf16 limbs, sum each factor in float32, and then do
   the same single multiply. The product is never expanded into cross terms,
   so the cost is n+m adds regardless of limb count -- the point that makes
   exp tractable at all (see log-research/exp-limb-gen.c).

     exact (3x3)   6 bf16 loads, 4 adds, 1 multiply.  Bit-identical to
                   CORE-MATH by construction: 3 bf16 limbs carry 24
                   significand bits, exactly float32's.
     min   (3x2)   5 bf16 loads, 3 adds, 1 multiply.  Two T2 entries carry a
                   one-ULP adjustment to stay correctly rounded.

   Table sizes are reported alongside: both limb schemes are LARGER than the
   float32 tables. The scheme targets hardware with bf16 storage or bf16 MACs
   and no float32 table path, not a space saving.

   Build (from implementations/). The implementations must be compiled as C --
   under g++ they would get C++ linkage and fail to match the extern "C"
   declarations below.
     cc -O3 -march=native -std=c11 -c inria-expbf16.c      -o output/inria-expbf16.o
     cc -O3 -march=native -std=c11 -c inria-expbf16-limb.c -o output/inria-expbf16-limb.o
     c++ -O3 -march=native -std=c++20 benchmark-exp-limb.cpp \
         output/inria-expbf16.o output/inria-expbf16-limb.o -o output/bench_exp_limb
*/

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
__bf16 cr_exp_bf16 (__bf16 x);
__bf16 cr_exp_bf16_limb (__bf16 x);
__bf16 cr_exp_bf16_limb_min (__bf16 x);
}

using Clock = std::chrono::high_resolution_clock;

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

static volatile double sink_d = 0.0;

static constexpr int BENCH_ITERS  = 20'000'000;
static constexpr int WARMUP_ITERS =  1'000'000;
static constexpr int ACC_SAMPLES  =    500'000;

struct BenchResult { double ns_per_call, total_ms; };

// Best of REPEATS. A single timed run of this loop is noisy enough to swing a
// ratio by ~0.4x -- visible directly in the no-table control cluster, whose
// variants execute identical code yet measured 0.54x to 1.06x across runs.
// The minimum is the run least contaminated by scheduling and frequency
// drift, so it is what the table paths are compared on.
static constexpr int REPEATS = 3;

template<typename Fn, typename T>
static BenchResult run_bench(Fn fn, const std::vector<T>& inputs) {
    double acc = 0.0;
    for (int i = 0; i < WARMUP_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
    sink_d = acc;

    double best_ms = 0.0;
    for (int rep = 0; rep < REPEATS; ++rep) {
        auto t0 = Clock::now();
        acc = 0.0;
        for (int i = 0; i < BENCH_ITERS; ++i) acc += fn(inputs[i % inputs.size()]);
        auto t1 = Clock::now();
        sink_d = acc;
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (rep == 0 || ms < best_ms) best_ms = ms;
    }
    return { (best_ms * 1e6) / BENCH_ITERS, best_ms };
}

typedef union { __bf16 f; uint16_t u; } b16u16_b;

static __bf16 bf16_from_float(float v) { b16u16_b t; t.f = (__bf16)v; return t.f; }

int main() {
    g_log = std::fopen("output/bench_results_exp_limb.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open output/bench_results_exp_limb.txt\n");

    auto wall_start = Clock::now();
    std::mt19937 rng(42);

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  bf16 exp() Benchmark — limb tables vs CORE-MATH float32 tables\n");
    lprintf("  Iterations    : %d per variant per cluster\n", BENCH_ITERS);
    lprintf("  Warmup        : %d iterations (not timed)\n",  WARMUP_ITERS);
    lprintf("  Reported      : best of %d timed runs per measurement\n", REPEATS);
    lprintf("  All three variants are correctly rounded on all 65536 inputs.\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");

    // ── Table storage comparison ─────────────────────────────────────────────
    // T1: 512 entries, T2: 256.
    const long f32_bytes   = (512L * 4) + (256L * 4);
    const long exact_bytes = (512L * 3 * 2) + (256L * 3 * 2);
    const long min_bytes   = (512L * 3 * 2) + (256L * 2 * 2);

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  TABLE STORAGE                                                 │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-24s %10s %10s %10s\n", "table", "float32", "3x3 limb", "3x2 limb");
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "T1 (512 entries)", 512L*4, 512L*3*2, 512L*3*2);
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "T2 (256 entries)", 256L*4, 256L*3*2, 256L*2*2);
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "total", f32_bytes, exact_bytes, min_bytes);
    lprintf("  %-24s %10s %+9.1f%% %+9.1f%%\n", "vs float32", "",
            100.0 * (double)(exact_bytes - f32_bytes) / (double)f32_bytes,
            100.0 * (double)(min_bytes   - f32_bytes) / (double)f32_bytes);

    // ── Value clusters, chosen to exercise each code path ────────────────────
    struct Cluster { const char* label; double lo, hi; };
    static constexpr Cluster CLUSTERS[] = {
        { "x near 0.003  ",  0.0025,  0.0035 },  // just past the |x|<=2^-9 exit
        { "x near 1      ",  0.9,     1.1    },  // mid-table
        { "x near 80     ", 79.5,    80.5    },  // near the overflow edge
        { "x near -80    ", -80.5,  -79.5    },  // the negative half of T1/T2
        { "x near 2e-10  ",  1e-10,   3e-10  },  // below the table: rounds to 1
    };

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  BFLOAT16 — cr_exp_bf16 (float32) vs _limb (3x3) vs _min (3x2) │\n");
    lprintf("│  Inria : 2 float32 loads, 1 multiply                           │\n");
    lprintf("│  3x3   : 6 bf16 loads, 4 adds, 1 multiply                      │\n");
    lprintf("│  3x2   : 5 bf16 loads, 3 adds, 1 multiply                      │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-15s %11s %11s %11s %9s %9s\n",
            "cluster", "inria ns", "3x3 ns", "3x2 ns", "3x3/f32", "3x2/f32");

    double sum_x = 0.0, sum_m = 0.0; int nclusters = 0;

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<double> dist(cl.lo, cl.hi);
        std::vector<__bf16> in(ACC_SAMPLES);
        for (auto& v : in) v = bf16_from_float((float)dist(rng));

        auto r_inria = run_bench([](__bf16 x){ return (double)cr_exp_bf16(x); }, in);
        auto r_exact = run_bench([](__bf16 x){ return (double)cr_exp_bf16_limb(x); }, in);
        auto r_min   = run_bench([](__bf16 x){ return (double)cr_exp_bf16_limb_min(x); }, in);

        double rx = r_exact.ns_per_call / r_inria.ns_per_call;
        double rm = r_min.ns_per_call   / r_inria.ns_per_call;
        sum_x += rx; sum_m += rm; nclusters++;
        lprintf("  %-15s %11.4f %11.4f %11.4f %8.2fx %8.2fx\n",
                cl.label, r_inria.ns_per_call, r_exact.ns_per_call,
                r_min.ns_per_call, rx, rm);
    }

    lprintf("\n  mean throughput ratio vs float32 tables: 3x3 %.2fx, 3x2 %.2fx\n",
            sum_x / nclusters, sum_m / nclusters);

    // ── Exhaustive agreement check ───────────────────────────────────────────
    // Cheap here and worth doing: a benchmark that silently drifted from the
    // verified build would report meaningless timings.
    long mism_x = 0, mism_m = 0;
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        b16u16_b x; x.u = (uint16_t)b;
        b16u16_b a; a.f = cr_exp_bf16(x.f);
        b16u16_b e; e.f = cr_exp_bf16_limb(x.f);
        b16u16_b m; m.f = cr_exp_bf16_limb_min(x.f);
        bool a_nan = (a.u & 0x7fff) > 0x7f80;
        if (!(a_nan && (e.u & 0x7fff) > 0x7f80) && a.u != e.u) mism_x++;
        if (!(a_nan && (m.u & 0x7fff) > 0x7f80) && a.u != m.u) mism_m++;
    }
    lprintf("\n  exhaustive agreement over 65536 inputs:\n");
    lprintf("    3x3 exact vs CORE-MATH : %s (%ld mismatches)\n",
            mism_x == 0 ? "IDENTICAL" : "DIFFER", mism_x);
    lprintf("    3x2 min   vs CORE-MATH : %s (%ld mismatches)\n",
            mism_m == 0 ? "IDENTICAL" : "DIFFER", mism_m);

    double wall = std::chrono::duration<double>(Clock::now() - wall_start).count();
    lprintf("\n  total wall time: %.1f s\n\n", wall);

    if (g_log) std::fclose(g_log);
    return (mism_x == 0 && mism_m == 0) ? 0 : 1;
}
