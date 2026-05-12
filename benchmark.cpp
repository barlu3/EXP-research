/**
 * @file  benchmark.cpp
 * @brief Benchmark + accuracy report for fast_exp variants vs std::exp.
 *
 * Measures:
 *   - Throughput  (ns/call, calls/sec)
 *   - Max relative error vs std::exp  (float and double)
 *   - Max ULP error (float)
 *
 * Compile:
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++20 benchmark.cpp -o bench
 *
 * Run:
 *   ./bench
 */

#include "exp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <bit>

using namespace fexp;
using Clock = std::chrono::high_resolution_clock;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Prevent dead-code elimination: compiler cannot see through volatile writes.
static volatile float  sink_f  = 0.0f;
static volatile double sink_d  = 0.0;

/**
 * @brief Time a lambda that accepts (int iterations) and returns void.
 *        Returns elapsed milliseconds.
 */
template<typename Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = Clock::now();
    fn(iters);
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/// Compute ULP distance between two floats (same-sign, normal range).
uint32_t ulp_dist(float a, float b) {
    int32_t ia = static_cast<int32_t>(std::bit_cast<uint32_t>(a));
    int32_t ib = static_cast<int32_t>(std::bit_cast<uint32_t>(b));
    return static_cast<uint32_t>(std::abs(ia - ib));
}

/// Compute relative error |fast - ref| / |ref|
double rel_err(double fast, double ref) {
    if (ref == 0.0) return 0.0;
    return std::abs(fast - ref) / std::abs(ref);
}

// ─── Benchmark parameters ────────────────────────────────────────────────────

static constexpr int    WARMUP_ITERS = 500'000;
static constexpr int    BENCH_ITERS  = 5'000'000;
static constexpr int    ACC_SAMPLES  = 200'000;

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    // ── 1. Generate random inputs ──────────────────────────────────────────────
    std::mt19937 rng(42);

    // float inputs within valid range (EXP_LO_F = -87.336, EXP_HI_F = 88.376)
    std::uniform_real_distribution<float>  dist_f(-87.0f, 87.0f);
    // double inputs in [-700, 700]
    std::uniform_real_distribution<double> dist_d(-700.0, 700.0);

    std::vector<float>  in_f(ACC_SAMPLES);
    std::vector<double> in_d(ACC_SAMPLES);
    for (auto& v : in_f) v = dist_f(rng);
    for (auto& v : in_d) v = dist_d(rng);

    // ── 2. Accuracy: scalar float ──────────────────────────────────────────────
    double max_rel_f  = 0.0;
    uint32_t max_ulp  = 0;
    for (int i = 0; i < ACC_SAMPLES; ++i) {
        float ref  = expf(in_f[i]);
        float fast = fast_expf(in_f[i]);
        max_rel_f  = std::max(max_rel_f, rel_err(fast, ref));
        max_ulp    = std::max(max_ulp,   ulp_dist(fast, ref));
    }

    // ── 3. Accuracy: scalar double ────────────────────────────────────────────
    double max_rel_d = 0.0;
    for (int i = 0; i < ACC_SAMPLES; ++i) {
        double ref  = std::exp(in_d[i]);
        double fast = fast_exp(in_d[i]);
        max_rel_d   = std::max(max_rel_d, rel_err(fast, ref));
    }

    // ── 4. Accuracy: AVX2 float ───────────────────────────────────────────────
#ifdef __AVX2__
    double max_rel_avxf = 0.0;
    for (int i = 0; i + 8 <= ACC_SAMPLES; i += 8) {
        __m256 vx    = _mm256_loadu_ps(&in_f[i]);
        __m256 vfast = fast_expf_avx2(vx);
        float  buf[8]; _mm256_storeu_ps(buf, vfast);
        for (int j = 0; j < 8; ++j) {
            float ref = expf(in_f[i + j]);
            max_rel_avxf = std::max(max_rel_avxf, rel_err(buf[j], ref));
        }
    }

    double max_rel_avxd = 0.0;
    for (int i = 0; i + 4 <= ACC_SAMPLES; i += 4) {
        __m256d vx    = _mm256_loadu_pd(&in_d[i]);
        __m256d vfast = fast_exp_avx2(vx);
        double  buf[4]; _mm256_storeu_pd(buf, vfast);
        for (int j = 0; j < 4; ++j) {
            double ref = std::exp(in_d[i + j]);
            max_rel_avxd = std::max(max_rel_avxd, rel_err(buf[j], ref));
        }
    }
#endif

    // ── 5. Throughput benchmarks ──────────────────────────────────────────────

    auto run_scalar_f = [&](bool use_fast) {
        return [&, use_fast](int iters) {
            float acc = 0.0f;
            for (int i = 0; i < iters; ++i) {
                float x = in_f[i % ACC_SAMPLES];
                acc += use_fast ? fast_expf(x) : expf(x);
            }
            sink_f = acc;
        };
    };

    auto run_scalar_d = [&](bool use_fast) {
        return [&, use_fast](int iters) {
            double acc = 0.0;
            for (int i = 0; i < iters; ++i) {
                double x = in_d[i % ACC_SAMPLES];
                acc += use_fast ? fast_exp(x) : std::exp(x);
            }
            sink_d = acc;
        };
    };

#ifdef __AVX2__
    auto run_avxf = [&](bool use_fast) {
        return [&, use_fast](int iters) {
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < iters; i += 8) {
                __m256 vx = _mm256_loadu_ps(&in_f[(i) % (ACC_SAMPLES - 8)]);
                acc = _mm256_add_ps(acc, use_fast ? fast_expf_avx2(vx)
                                                  : _mm256_set_ps(
                                                        expf(in_f[(i+7)%ACC_SAMPLES]),
                                                        expf(in_f[(i+6)%ACC_SAMPLES]),
                                                        expf(in_f[(i+5)%ACC_SAMPLES]),
                                                        expf(in_f[(i+4)%ACC_SAMPLES]),
                                                        expf(in_f[(i+3)%ACC_SAMPLES]),
                                                        expf(in_f[(i+2)%ACC_SAMPLES]),
                                                        expf(in_f[(i+1)%ACC_SAMPLES]),
                                                        expf(in_f[(i+0)%ACC_SAMPLES])));
            }
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            sink_f = tmp[0];
        };
    };

    auto run_avxd = [&](bool use_fast) {
        return [&, use_fast](int iters) {
            __m256d acc = _mm256_setzero_pd();
            for (int i = 0; i < iters; i += 4) {
                __m256d vx = _mm256_loadu_pd(&in_d[(i) % (ACC_SAMPLES - 4)]);
                acc = _mm256_add_pd(acc, use_fast ? fast_exp_avx2(vx)
                                                  : _mm256_set_pd(
                                                        std::exp(in_d[(i+3)%ACC_SAMPLES]),
                                                        std::exp(in_d[(i+2)%ACC_SAMPLES]),
                                                        std::exp(in_d[(i+1)%ACC_SAMPLES]),
                                                        std::exp(in_d[(i+0)%ACC_SAMPLES])));
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
    double t_std_expf    = time_ms(run_scalar_f(false), BENCH_ITERS);
    double t_fast_expf   = time_ms(run_scalar_f(true),  BENCH_ITERS);
    double t_std_exp     = time_ms(run_scalar_d(false), BENCH_ITERS);
    double t_fast_exp    = time_ms(run_scalar_d(true),  BENCH_ITERS);

#ifdef __AVX2__
    double t_avxf_std    = time_ms(run_avxf(false), BENCH_ITERS);
    double t_avxf_fast   = time_ms(run_avxf(true),  BENCH_ITERS);
    double t_avxd_std    = time_ms(run_avxd(false), BENCH_ITERS);
    double t_avxd_fast   = time_ms(run_avxd(true),  BENCH_ITERS);
#endif

    // ── 6. Report ─────────────────────────────────────────────────────────────

    auto ns_per = [](double ms, int iters) {
        return (ms * 1e6) / static_cast<double>(iters);
    };
    auto speedup = [](double base, double fast) {
        return base / fast;
    };

    std::printf("\n");
    std::printf("══════════════════════════════════════════════════════════════════\n");
    std::printf("  fast_exp  benchmark   (%d calls per variant)\n", BENCH_ITERS);
    std::printf("══════════════════════════════════════════════════════════════════\n");
    std::printf("  %-30s %8s  %8s  %8s\n", "Variant", "ns/call", "ms total", "speedup");
    std::printf("  %-30s %8s  %8s  %8s\n", "-------------------------------",
                "--------","--------","-------");

    std::printf("  %-30s %8.2f  %8.1f\n",
                "expf (scalar f32)",
                ns_per(t_std_expf, BENCH_ITERS), t_std_expf);
    std::printf("  %-30s %8.2f  %8.1f  %7.2fx\n",
                "fast_expf (scalar f32)",
                ns_per(t_fast_expf, BENCH_ITERS), t_fast_expf,
                speedup(t_std_expf, t_fast_expf));

    std::printf("  %-30s\n", "");

    std::printf("  %-30s %8.2f  %8.1f\n",
                "std::exp  (scalar f64)",
                ns_per(t_std_exp, BENCH_ITERS), t_std_exp);
    std::printf("  %-30s %8.2f  %8.1f  %7.2fx\n",
                "fast_exp  (scalar f64)",
                ns_per(t_fast_exp, BENCH_ITERS), t_fast_exp,
                speedup(t_std_exp, t_fast_exp));

#ifdef __AVX2__
    std::printf("  %-30s\n", "");

    std::printf("  %-30s %8.2f  %8.1f\n",
                "expf (AVX2 8×f32, emul)",
                ns_per(t_avxf_std, BENCH_ITERS / 8), t_avxf_std);
    std::printf("  %-30s %8.2f  %8.1f  %7.2fx\n",
                "fast_expf (AVX2 8×f32)",
                ns_per(t_avxf_fast, BENCH_ITERS / 8), t_avxf_fast,
                speedup(t_avxf_std, t_avxf_fast));

    std::printf("  %-30s\n", "");

    std::printf("  %-30s %8.2f  %8.1f\n",
                "std::exp  (AVX2 4×f64, emul)",
                ns_per(t_avxd_std, BENCH_ITERS / 4), t_avxd_std);
    std::printf("  %-30s %8.2f  %8.1f  %7.2fx\n",
                "fast_exp  (AVX2 4×f64)",
                ns_per(t_avxd_fast, BENCH_ITERS / 4), t_avxd_fast,
                speedup(t_avxd_std, t_avxd_fast));
#endif

    std::printf("\n");
    std::printf("──────────────────────────────────────────────────────────────────\n");
    std::printf("  Accuracy  (%d samples, input ∈ [-87,87] f32 / [-700,700] f64)\n",
                ACC_SAMPLES);
    std::printf("──────────────────────────────────────────────────────────────────\n");
    std::printf("  fast_expf  scalar  — max ULP error : %u\n",      max_ulp);
    std::printf("  fast_expf  scalar  — max rel error : %.2e\n",    max_rel_f);
    std::printf("  fast_exp   scalar  — max rel error : %.2e\n",    max_rel_d);
#ifdef __AVX2__
    std::printf("  fast_expf  AVX2    — max rel error : %.2e\n",    max_rel_avxf);
    std::printf("  fast_exp   AVX2    — max rel error : %.2e\n",    max_rel_avxd);
#endif
    std::printf("══════════════════════════════════════════════════════════════════\n\n");

    return 0;
}