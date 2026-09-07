/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for sin().

   Companion to benchmark-limb.cpp (ln) and benchmark-exp-limb.cpp (exp). All
   three implementations are correctly rounded on every bfloat16 input --
   verified exhaustively against MPFR by cross-eval/verify-limb.c with
   -DVERIFY_SIN_LIMB -- so this measures cost, not accuracy.

   What differs. cr_sin_bf16 has three regimes and the limb cost differs
   sharply between them, which is the main thing this benchmark exists to show:

     |x| <= 0x1.dp-4   no table at all. Identical work in every variant, so
                       this cluster is the control: any difference is noise.
     |x| <  4096       fma(S1, C2, C1 * S2) -- four lookups, so the limb
                       reconstruction cost is paid four times per call.
     |x| >= 4096       an angle-addition chain of up to eight lookup PAIRS,
                       so up to sixteen reconstructions per call. This is
                       where the scheme is most expensive.

   As in the exp version the products are never expanded into cross terms;
   each factor is rebuilt from its limbs and the shipped expression is
   unchanged.

     exact   3 limbs on all six tables. Bit-identical to CORE-MATH by
             construction: 3 bf16 limbs carry 24 significand bits.
     min     2 limbs on S1/C1, 3 on S2/C2 and S3/C3, and no tuned entries at
             all. The third limb goes on S2/C2 because those tables hold 128
             entries against S1/C1's 256, making it 512 B cheaper there; the
             mirror arrangement also reaches zero but costs more and needs
             three hand-adjusted index pairs. S3/C3 cannot drop to two limbs
             -- the large path chains its lookups, so the errors compound.

   Build (from implementations/). The implementations must be compiled as C.
     cc -O3 -march=native -std=c11 -c sin/inria-sinbf16.c      -o output/inria-sinbf16.o
     cc -O3 -march=native -std=c11 -c sin/inria-sinbf16-limb.c -o output/inria-sinbf16-limb.o
     c++ -O3 -march=native -std=c++20 benchmark-sin-limb.cpp \
         output/inria-sinbf16.o output/inria-sinbf16-limb.o -o output/bench_sin_limb
*/

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
__bf16 cr_sin_bf16 (__bf16 x);
__bf16 cr_sin_bf16_limb (__bf16 x);
__bf16 cr_sin_bf16_limb_min (__bf16 x);
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
    g_log = std::fopen("output/bench_results_sin_limb.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open output/bench_results_sin_limb.txt\n");

    auto wall_start = Clock::now();
    std::mt19937 rng(42);

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  bf16 sin() Benchmark — limb tables vs CORE-MATH float32 tables\n");
    lprintf("  Iterations    : %d per variant per cluster\n", BENCH_ITERS);
    lprintf("  Warmup        : %d iterations (not timed)\n",  WARMUP_ITERS);
    lprintf("  Reported      : best of %d timed runs per measurement\n", REPEATS);
    lprintf("  All three variants are correctly rounded on all 65536 inputs.\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");

    // ── Table storage comparison ─────────────────────────────────────────────
    // S1/C1: 256 entries each, S2/C2: 128 each, S3/C3: 123 each.
    const long f32_bytes   = (256L*2 + 128L*2 + 123L*2) * 4;
    const long exact_bytes = (256L*2 + 128L*2 + 123L*2) * 3 * 2;
    const long min_bytes   = (256L*2*2 + 128L*2*3 + 123L*2*3) * 2;

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  TABLE STORAGE                                                 │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-24s %10s %10s %10s\n", "table", "float32", "exact", "minimal");
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "S1+C1 (256 each)", 256L*2*4, 256L*2*3*2, 256L*2*2*2);
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "S2+C2 (128 each)", 128L*2*4, 128L*2*3*2, 128L*2*3*2);
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "S3+C3 (123 each)", 123L*2*4, 123L*2*3*2, 123L*2*3*2);
    lprintf("  %-24s %9ldB %9ldB %9ldB\n", "total", f32_bytes, exact_bytes, min_bytes);
    lprintf("  %-24s %10s %+9.1f%% %+9.1f%%\n", "vs float32", "",
            100.0 * (double)(exact_bytes - f32_bytes) / (double)f32_bytes,
            100.0 * (double)(min_bytes   - f32_bytes) / (double)f32_bytes);

    // ── Value clusters, one per code path ────────────────────────────────────
    struct Cluster { const char* label; double lo, hi; };
    static constexpr Cluster CLUSTERS[] = {
        { "x near 0.05   ",  0.04,   0.06   },  // small-|x|: no table (control)
        { "x near 1      ",  0.9,    1.1    },  // mid path
        { "x near 100    ", 99.0,  101.0    },  // mid path, larger index
        { "x near 1e5    ",  9e4,    1.1e5  },  // large path, short chain
        { "x near 1e30   ",  9e29,   1.1e30 },  // large path, deep chain
    };

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  BFLOAT16 — cr_sin_bf16 (float32) vs _limb (exact) vs _min     │\n");
    lprintf("│  small |x| : no table — control cluster                        │\n");
    lprintf("│  mid  path : 4 lookups   → 4 reconstructions per call          │\n");
    lprintf("│  large path: ≤8 pairs    → ≤16 reconstructions per call        │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-15s %11s %11s %11s %9s %9s\n",
            "cluster", "inria ns", "exact ns", "min ns", "exact/f32", "min/f32");

    double sum_x = 0.0, sum_m = 0.0; int nclusters = 0;

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<double> dist(cl.lo, cl.hi);
        std::vector<__bf16> in(ACC_SAMPLES);
        for (auto& v : in) v = bf16_from_float((float)dist(rng));

        auto r_inria = run_bench([](__bf16 x){ return (double)cr_sin_bf16(x); }, in);
        auto r_exact = run_bench([](__bf16 x){ return (double)cr_sin_bf16_limb(x); }, in);
        auto r_min   = run_bench([](__bf16 x){ return (double)cr_sin_bf16_limb_min(x); }, in);

        double rx = r_exact.ns_per_call / r_inria.ns_per_call;
        double rm = r_min.ns_per_call   / r_inria.ns_per_call;
        sum_x += rx; sum_m += rm; nclusters++;
        lprintf("  %-15s %11.4f %11.4f %11.4f %8.2fx %8.2fx\n",
                cl.label, r_inria.ns_per_call, r_exact.ns_per_call,
                r_min.ns_per_call, rx, rm);
    }

    lprintf("\n  mean throughput ratio vs float32 tables: exact %.2fx, min %.2fx\n",
            sum_x / nclusters, sum_m / nclusters);
    lprintf("  (the 'x near 0.05' row takes no table in any variant, so its ratio\n"
            "   is the harness noise floor rather than a cost of the limb scheme)\n");

    // ── Exhaustive agreement check ───────────────────────────────────────────
    long mism_x = 0, mism_m = 0;
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        b16u16_b x; x.u = (uint16_t)b;
        b16u16_b a; a.f = cr_sin_bf16(x.f);
        b16u16_b e; e.f = cr_sin_bf16_limb(x.f);
        b16u16_b m; m.f = cr_sin_bf16_limb_min(x.f);
        bool a_nan = (a.u & 0x7fff) > 0x7f80;
        if (!(a_nan && (e.u & 0x7fff) > 0x7f80) && a.u != e.u) mism_x++;
        if (!(a_nan && (m.u & 0x7fff) > 0x7f80) && a.u != m.u) mism_m++;
    }
    lprintf("\n  exhaustive agreement over 65536 inputs:\n");
    lprintf("    exact vs CORE-MATH : %s (%ld mismatches)\n",
            mism_x == 0 ? "IDENTICAL" : "DIFFER", mism_x);
    lprintf("    min   vs CORE-MATH : %s (%ld mismatches)\n",
            mism_m == 0 ? "IDENTICAL" : "DIFFER", mism_m);

    double wall = std::chrono::duration<double>(Clock::now() - wall_start).count();
    lprintf("\n  total wall time: %.1f s\n\n", wall);

    if (g_log) std::fclose(g_log);
    return (mism_x == 0 && mism_m == 0) ? 0 : 1;
}
