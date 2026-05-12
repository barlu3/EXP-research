/**
 * @file  benchmark.cpp
 * @brief Benchmark + accuracy report for fexp::exp variants vs stdlib.
 *
 * Measures:
 *   - Throughput  (ns/call) for scalar float/double vs stdlib
 *   - Polynomial-only ns/call via explicit std::chrono timing
 *   - Max relative error + max ULP error vs stdlib
 *
 * Results are written to both stdout and bench_results.txt.
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
#include <numeric>
#include <random>
#include <vector>
#include <bit>

// No "using namespace fexp" — fexp::expf / fexp::exp would be ambiguous
// with ::expf / std::exp at call sites.  Use explicit fexp:: qualification.

using Clock = std::chrono::high_resolution_clock;

// ─── Dual-output helper ───────────────────────────────────────────────────────

static FILE* g_log = nullptr;

// Writes to stdout and to g_log (if open) in one call.
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

static volatile float  sink_f = 0.0f;
static volatile double sink_d = 0.0;

template<typename Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = Clock::now();
    fn(iters);
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

uint32_t ulp_dist(float a, float b) {
    int32_t ia = static_cast<int32_t>(std::bit_cast<uint32_t>(a));
    int32_t ib = static_cast<int32_t>(std::bit_cast<uint32_t>(b));
    return static_cast<uint32_t>(std::abs(ia - ib));
}

double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

// ─── Benchmark parameters ────────────────────────────────────────────────────

static constexpr int WARMUP_ITERS = 500'000;
static constexpr int BENCH_ITERS  = 5'000'000;
static constexpr int ACC_SAMPLES  = 200'000;

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    // Open log file — all lprintf output goes here and to stdout.
    g_log = std::fopen("bench_results.txt", "w");
    if (!g_log)
        std::fprintf(stderr, "warning: could not open bench_results.txt\n");

    // ── 1. Generate random inputs ──────────────────────────────────────────────
    std::mt19937 rng(42);
    std::uniform_real_distribution<float>  dist_f(-87.0f, 87.0f);
    std::uniform_real_distribution<double> dist_d(-700.0, 700.0);

    std::vector<float>  in_f(ACC_SAMPLES);
    std::vector<double> in_d(ACC_SAMPLES);
    for (auto& v : in_f) v = dist_f(rng);
    for (auto& v : in_d) v = dist_d(rng);

    // ── 2. Accuracy: scalar float ──────────────────────────────────────────────
    double max_rel_f = 0.0;
    uint32_t max_ulp = 0;
    for (int i = 0; i < ACC_SAMPLES; ++i) {
        float ref  = ::expf(in_f[i]);
        float fast = fexp::expf(in_f[i]);
        max_rel_f  = std::max(max_rel_f, rel_err(fast, ref));
        max_ulp    = std::max(max_ulp,   ulp_dist(fast, ref));
    }

    // ── 3. Accuracy: scalar double ────────────────────────────────────────────
    double max_rel_d = 0.0;
    for (int i = 0; i < ACC_SAMPLES; ++i) {
        double ref  = std::exp(in_d[i]);
        double fast = fexp::exp(in_d[i]);
        max_rel_d   = std::max(max_rel_d, rel_err(fast, ref));
    }

    // ── 4. Accuracy: AVX2 ────────────────────────────────────────────────────
#ifdef __AVX2__
    double max_rel_avxf = 0.0;
    for (int i = 0; i + 8 <= ACC_SAMPLES; i += 8) {
        __m256 vx    = _mm256_loadu_ps(&in_f[i]);
        __m256 vfast = fexp::fast_expf_avx2(vx);
        float  buf[8]; _mm256_storeu_ps(buf, vfast);
        for (int j = 0; j < 8; ++j) {
            float ref = ::expf(in_f[i + j]);
            max_rel_avxf = std::max(max_rel_avxf, rel_err(buf[j], ref));
        }
    }

    double max_rel_avxd = 0.0;
    for (int i = 0; i + 4 <= ACC_SAMPLES; i += 4) {
        __m256d vx    = _mm256_loadu_pd(&in_d[i]);
        __m256d vfast = fexp::fast_exp_avx2(vx);
        double  buf[4]; _mm256_storeu_pd(buf, vfast);
        for (int j = 0; j < 4; ++j) {
            double ref = std::exp(in_d[i + j]);
            max_rel_avxd = std::max(max_rel_avxd, rel_err(buf[j], ref));
        }
    }
#endif

    // ── 5. Throughput: full exp calls ─────────────────────────────────────────

    auto run_scalar_f = [&](bool use_ours) {
        return [&, use_ours](int iters) {
            float acc = 0.0f;
            for (int i = 0; i < iters; ++i) {
                float x = in_f[i % ACC_SAMPLES];
                acc += use_ours ? fexp::expf(x) : ::expf(x);
            }
            sink_f = acc;
        };
    };

    auto run_scalar_d = [&](bool use_ours) {
        return [&, use_ours](int iters) {
            double acc = 0.0;
            for (int i = 0; i < iters; ++i) {
                double x = in_d[i % ACC_SAMPLES];
                acc += use_ours ? fexp::exp(x) : std::exp(x);
            }
            sink_d = acc;
        };
    };

#ifdef __AVX2__
    auto run_avxf = [&](bool use_ours) {
        return [&, use_ours](int iters) {
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < iters; i += 8) {
                __m256 vx = _mm256_loadu_ps(&in_f[(i) % (ACC_SAMPLES - 8)]);
                acc = _mm256_add_ps(acc, use_ours
                    ? fexp::fast_expf_avx2(vx)
                    : _mm256_set_ps(
                        ::expf(in_f[(i+7)%ACC_SAMPLES]), ::expf(in_f[(i+6)%ACC_SAMPLES]),
                        ::expf(in_f[(i+5)%ACC_SAMPLES]), ::expf(in_f[(i+4)%ACC_SAMPLES]),
                        ::expf(in_f[(i+3)%ACC_SAMPLES]), ::expf(in_f[(i+2)%ACC_SAMPLES]),
                        ::expf(in_f[(i+1)%ACC_SAMPLES]), ::expf(in_f[(i+0)%ACC_SAMPLES])));
            }
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            sink_f = tmp[0];
        };
    };

    auto run_avxd = [&](bool use_ours) {
        return [&, use_ours](int iters) {
            __m256d acc = _mm256_setzero_pd();
            for (int i = 0; i < iters; i += 4) {
                __m256d vx = _mm256_loadu_pd(&in_d[(i) % (ACC_SAMPLES - 4)]);
                acc = _mm256_add_pd(acc, use_ours
                    ? fexp::fast_exp_avx2(vx)
                    : _mm256_set_pd(
                        std::exp(in_d[(i+3)%ACC_SAMPLES]), std::exp(in_d[(i+2)%ACC_SAMPLES]),
                        std::exp(in_d[(i+1)%ACC_SAMPLES]), std::exp(in_d[(i+0)%ACC_SAMPLES])));
            }
            double tmp[4]; _mm256_storeu_pd(tmp, acc);
            sink_d = tmp[0];
        };
    };
#endif

    // Warmup
    time_ms(run_scalar_f(false), WARMUP_ITERS);
    time_ms(run_scalar_f(true),  WARMUP_ITERS);

    // Timed runs
    double t_std_expf  = time_ms(run_scalar_f(false), BENCH_ITERS);
    double t_ours_expf = time_ms(run_scalar_f(true),  BENCH_ITERS);
    double t_std_exp   = time_ms(run_scalar_d(false), BENCH_ITERS);
    double t_ours_exp  = time_ms(run_scalar_d(true),  BENCH_ITERS);

#ifdef __AVX2__
    double t_avxf_std  = time_ms(run_avxf(false), BENCH_ITERS);
    double t_avxf_ours = time_ms(run_avxf(true),  BENCH_ITERS);
    double t_avxd_std  = time_ms(run_avxd(false), BENCH_ITERS);
    double t_avxd_ours = time_ms(run_avxd(true),  BENCH_ITERS);
#endif

    // ── 6. Polynomial-only timing (std::chrono) ────────────────────────────────
    std::vector<double> in_rf(ACC_SAMPLES), in_rd(ACC_SAMPLES);
    for (int i = 0; i < ACC_SAMPLES; ++i) {
        in_rf[i] = fexp::expf_reduce(in_f[i]);
        in_rd[i] = fexp::exp_reduce(in_d[i]);
    }

    auto run_poly_f = [&](int iters) {
        float acc = 0.0f;
        for (int i = 0; i < iters; ++i)
            acc += fexp::expf_poly(in_rf[i % ACC_SAMPLES]);
        sink_f = acc;
    };

    auto run_poly_d = [&](int iters) {
        double acc = 0.0;
        for (int i = 0; i < iters; ++i)
            acc += fexp::exp_poly(in_rd[i % ACC_SAMPLES]);
        sink_d = acc;
    };

    run_poly_f(WARMUP_ITERS);
    run_poly_d(WARMUP_ITERS);

    auto poly_f_t0 = Clock::now();
    run_poly_f(BENCH_ITERS);
    auto poly_f_t1 = Clock::now();

    auto poly_d_t0 = Clock::now();
    run_poly_d(BENCH_ITERS);
    auto poly_d_t1 = Clock::now();

    double ns_poly_f = std::chrono::duration<double, std::nano>(poly_f_t1 - poly_f_t0).count()
                       / static_cast<double>(BENCH_ITERS);
    double ns_poly_d = std::chrono::duration<double, std::nano>(poly_d_t1 - poly_d_t0).count()
                       / static_cast<double>(BENCH_ITERS);

    // ── 7. Report ─────────────────────────────────────────────────────────────

    auto ns_per  = [](double ms, int iters) { return (ms * 1e6) / static_cast<double>(iters); };
    auto speedup = [](double base, double fast) { return base / fast; };
    auto pct     = [](double part, double total) { return 100.0 * part / total; };

    lprintf("\n");
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  exp benchmark   (%d calls per variant)\n", BENCH_ITERS);
    lprintf("══════════════════════════════════════════════════════════════════\n");
    lprintf("  %-32s %8s  %8s  %8s\n", "Variant", "ns/call", "ms total", "speedup");
    lprintf("  %-32s %8s  %8s  %8s\n", "--------------------------------",
            "--------", "--------", "-------");

    lprintf("  %-32s %8.2f  %8.1f\n",
            "::expf (stdlib f32)",
            ns_per(t_std_expf, BENCH_ITERS), t_std_expf);
    lprintf("  %-32s %8.2f  %8.1f  %7.2fx\n",
            "fexp::expf (glibc f32)",
            ns_per(t_ours_expf, BENCH_ITERS), t_ours_expf,
            speedup(t_std_expf, t_ours_expf));

    lprintf("  %-32s\n", "");

    lprintf("  %-32s %8.2f  %8.1f\n",
            "std::exp (stdlib f64)",
            ns_per(t_std_exp, BENCH_ITERS), t_std_exp);
    lprintf("  %-32s %8.2f  %8.1f  %7.2fx\n",
            "fexp::exp (glibc f64)",
            ns_per(t_ours_exp, BENCH_ITERS), t_ours_exp,
            speedup(t_std_exp, t_ours_exp));

#ifdef __AVX2__
    lprintf("  %-32s\n", "");

    lprintf("  %-32s %8.2f  %8.1f\n",
            "::expf (AVX2 8×f32, emul)",
            ns_per(t_avxf_std, BENCH_ITERS / 8), t_avxf_std);
    lprintf("  %-32s %8.2f  %8.1f  %7.2fx\n",
            "fexp::expf (AVX2 8×f32, scalar)",
            ns_per(t_avxf_ours, BENCH_ITERS / 8), t_avxf_ours,
            speedup(t_avxf_std, t_avxf_ours));

    lprintf("  %-32s\n", "");

    lprintf("  %-32s %8.2f  %8.1f\n",
            "std::exp (AVX2 4×f64, emul)",
            ns_per(t_avxd_std, BENCH_ITERS / 4), t_avxd_std);
    lprintf("  %-32s %8.2f  %8.1f  %7.2fx\n",
            "fexp::exp (AVX2 4×f64, scalar)",
            ns_per(t_avxd_ours, BENCH_ITERS / 4), t_avxd_ours,
            speedup(t_avxd_std, t_avxd_ours));
#endif

    lprintf("\n");
    lprintf("──────────────────────────────────────────────────────────────────\n");
    lprintf("  Polynomial-only  (%d calls, pre-reduced inputs)\n", BENCH_ITERS);
    lprintf("  Isolates degree-3 (f32) / degree-4 (f64) polynomial kernel.\n");
    lprintf("──────────────────────────────────────────────────────────────────\n");
    lprintf("  %-32s %8s  %8s  %9s\n",
            "Variant", "ns/call", "vs full", "% of full");
    lprintf("  %-32s %8s  %8s  %9s\n",
            "--------------------------------", "--------", "-------", "---------");
    lprintf("  %-32s %8.2f  %8.2f  %8.1f%%\n",
            "expf poly only (f32)",
            ns_poly_f, ns_per(t_ours_expf, BENCH_ITERS),
            pct(ns_poly_f, ns_per(t_ours_expf, BENCH_ITERS)));
    lprintf("  %-32s %8.2f  %8.2f  %8.1f%%\n",
            "exp  poly only (f64)",
            ns_poly_d, ns_per(t_ours_exp, BENCH_ITERS),
            pct(ns_poly_d, ns_per(t_ours_exp, BENCH_ITERS)));

    lprintf("\n");
    lprintf("──────────────────────────────────────────────────────────────────\n");
    lprintf("  Accuracy  (%d samples, ∈ [-87,87] f32 / [-700,700] f64)\n",
            ACC_SAMPLES);
    lprintf("──────────────────────────────────────────────────────────────────\n");
    lprintf("  fexp::expf  scalar — max ULP error : %u\n",   max_ulp);
    lprintf("  fexp::expf  scalar — max rel error : %.2e\n", max_rel_f);
    lprintf("  fexp::exp   scalar — max rel error : %.2e\n", max_rel_d);
#ifdef __AVX2__
    lprintf("  fexp::expf  AVX2   — max rel error : %.2e\n", max_rel_avxf);
    lprintf("  fexp::exp   AVX2   — max rel error : %.2e\n", max_rel_avxd);
#endif
    lprintf("══════════════════════════════════════════════════════════════════\n\n");

    if (g_log) {
        std::fclose(g_log);
        std::printf("results saved to bench_results.txt\n");
    }

    return 0;
}
