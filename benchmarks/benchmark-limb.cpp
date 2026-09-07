/* Benchmark: bf16-only limb tables vs CORE-MATH's float32 tables for ln().

   Both implementations are correctly rounded on every bfloat16 input (verified
   exhaustively against MPFR by cross-eval/log/verify-limb.c and
   verify_mpfr.c), so this measures cost, not accuracy.

   What differs. cr_log_bf16 reads two float32 entries and adds them.
   cr_log_bf16_limb reads three bf16 limbs for T1 and two for T2 and sums all
   five in float32. It trades loads and adds for the ability to keep the whole
   table in bf16 storage -- the point of the scheme, since it targets hardware
   with bf16 storage or bf16 MACs and no float32 table path.

   Table sizes are reported alongside: the limb scheme is LARGER for T1
   (48 bits/entry vs 32) and smaller for T3 (16 vs 32).

   Build (from implementations/). The two implementations must be compiled as
   C -- under g++ they would get C++ linkage and fail to match the extern "C"
   declarations below.
     gcc -O3 -march=native -std=c11 -c log/inria-logbf16.c      -o output/inria-logbf16.o
     gcc -O3 -march=native -std=c11 -c log/inria-logbf16-limb.c -o output/inria-logbf16-limb.o
     g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark-limb.cpp \
         output/inria-logbf16.o output/inria-logbf16-limb.o -o output/bench_limb
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
__bf16 cr_log_bf16 (__bf16 x);
__bf16 cr_log_bf16_limb (__bf16 x);
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

static constexpr int BENCH_ITERS  =  20'000'000;
static constexpr int WARMUP_ITERS =   1'000'000;
static constexpr int ACC_SAMPLES  =     500'000;

struct BenchResult { double ns_per_call, total_ms; };

template<typename Fn, typename T>
static BenchResult run_bench(Fn fn, const std::vector<T>& inputs) {
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

typedef union { __bf16 f; uint16_t u; } b16u16_b;

static __bf16 bf16_from_float(float v) { b16u16_b t; t.f = (__bf16)v; return t.f; }

int main() {
    g_log = std::fopen("output/bench_results_limb.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open output/bench_results_limb.txt\n");

    auto wall_start = Clock::now();
    std::mt19937 rng(42);

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  bf16 log() Benchmark — limb tables vs CORE-MATH float32 tables\n");
    lprintf("  Compile flags : -O3 -march=native -mavx2 -mfma -std=c++20\n");
    lprintf("  Iterations    : %'d per variant per cluster\n", BENCH_ITERS);
    lprintf("  Warmup        : %'d iterations (not timed)\n",  WARMUP_ITERS);
    lprintf("  Both variants are correctly rounded on all 65536 inputs.\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");

    // ── Table storage comparison ─────────────────────────────────────────────
    // T1: 254 usable entries, T2: 128, T3: 127.
    const long f32_bytes  = (254L * 4) + (128L * 4) + (127L * 4);
    const long limb_bytes = (254L * 3 * 2) + (128L * 2 * 2) + (127L * 1 * 2);

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  TABLE STORAGE                                                 │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-28s %10s %10s\n", "table", "float32", "bf16 limb");
    lprintf("  %-28s %9ldB %9ldB\n", "T1 (254 entries)", 254L*4, 254L*3*2);
    lprintf("  %-28s %9ldB %9ldB\n", "T2 (128 entries)", 128L*4, 128L*2*2);
    lprintf("  %-28s %9ldB %9ldB\n", "T3 (127 entries)", 127L*4, 127L*1*2);
    lprintf("  %-28s %9ldB %9ldB   (%+.1f%%)\n", "total", f32_bytes, limb_bytes,
            100.0 * (double)(limb_bytes - f32_bytes) / (double)f32_bytes);

    // ── Value clusters, chosen to exercise each table path ───────────────────
    struct Cluster { const char* label; double lo, hi; };
    static constexpr Cluster CLUSTERS[] = {
        { "x near 1      ", 0.9,    1.1    },   // T1_126/127, the hard binade
        { "x near 80     ", 79.5,   80.5   },   // mid-range normals
        { "x near 2e-10  ", 1e-10,  3e-10  },   // small normals
        { "subnormal     ", 1e-40,  1e-38  },   // T3 direct lookup
    };

    lprintf("\n");
    lprintf("┌────────────────────────────────────────────────────────────────┐\n");
    lprintf("│  BFLOAT16 — cr_log_bf16 (float32 tables) vs _limb (bf16)       │\n");
    lprintf("│  Inria : 2 float32 loads, 1 add                                │\n");
    lprintf("│  Limb  : 5 bf16 loads, 4 adds (float32 accumulation)           │\n");
    lprintf("└────────────────────────────────────────────────────────────────┘\n");
    lprintf("  %-15s %12s %12s %10s\n", "cluster", "inria ns", "limb ns", "ratio");

    double sum_ratio = 0.0; int nclusters = 0;

    for (const auto& cl : CLUSTERS) {
        std::uniform_real_distribution<double> dist(cl.lo, cl.hi);
        std::vector<__bf16> in(ACC_SAMPLES);
        for (auto& v : in) v = bf16_from_float((float)dist(rng));

        auto r_inria = run_bench([](__bf16 x){ return (double)cr_log_bf16(x); }, in);
        auto r_limb  = run_bench([](__bf16 x){ return (double)cr_log_bf16_limb(x); }, in);

        double ratio = r_limb.ns_per_call / r_inria.ns_per_call;
        sum_ratio += ratio; nclusters++;
        lprintf("  %-15s %12.4f %12.4f %9.2fx\n",
                cl.label, r_inria.ns_per_call, r_limb.ns_per_call, ratio);
    }

    lprintf("\n  mean limb/inria throughput ratio: %.2fx\n", sum_ratio / nclusters);

    // ── Exhaustive agreement check ───────────────────────────────────────────
    // Cheap here and worth doing: a benchmark that silently drifted from the
    // verified build would report meaningless timings.
    long mismatches = 0;
    for (uint32_t b = 0; b <= 0xFFFF; ++b) {
        b16u16_b x; x.u = (uint16_t)b;
        b16u16_b a; a.f = cr_log_bf16(x.f);
        b16u16_b c; c.f = cr_log_bf16_limb(x.f);
        int a_nan = (a.u & 0x7fff) > 0x7f80, c_nan = (c.u & 0x7fff) > 0x7f80;
        if (a_nan && c_nan) continue;
        if (a.u != c.u) mismatches++;
    }
    lprintf("\n  exhaustive agreement over 65536 inputs: %s (%ld mismatches)\n",
            mismatches == 0 ? "IDENTICAL" : "DIFFER", mismatches);

    double wall = std::chrono::duration<double>(Clock::now() - wall_start).count();
    lprintf("\n  total wall time: %.1f s\n\n", wall);

    if (g_log) std::fclose(g_log);
    return mismatches == 0 ? 0 : 1;
}
