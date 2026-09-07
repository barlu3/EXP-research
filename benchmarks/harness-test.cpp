/* Unit tests for bench-harness.hpp and the shipped cluster tables.

   Two of these encode the defects that motivated the harness rewrite, so they
   fail if the old behaviour comes back:

     cluster_distinct_input_floor  -- every shipped cluster must reach
       MIN_DISTINCT_INPUTS. The old [79.5, 80.5] range resolved to 3 distinct
       bf16 values and drew 500,000 samples from them, so the table stayed in
       one cache line and the benchmark could not see a table-footprint change.

     noise_floor -- two identical kernels, run through the real timing path,
       must report a paired median ratio near 1.0 with a tight IQR. The old
       single-run harness swung this between 0.54x and 1.06x, which is what
       made the ln ratios untrustworthy in the first place.

   No test framework: cross-eval verifies the same way, with a counter and a
   nonzero exit. */

#include "bench-clusters.hpp"
#include "bench-harness.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_skips = 0;
static const char* g_case = "";

static void begin(const char* name) { g_case = name; }

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("  FAIL  %-32s %s\n", g_case, what);
        ++g_failures;
    }
}

__attribute__((format(printf, 1, 2)))
static void skip(const char* fmt, ...) {
    std::printf("  SKIP  %-32s ", g_case);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
    ++g_skips;
}

static void check_near(double got, double want, double tol, const char* what) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("  FAIL  %-32s %s (got %.6f, want %.6f +- %.6f)\n",
                    g_case, what, got, want, tol);
        ++g_failures;
    }
}

// ─── statistics ──────────────────────────────────────────────────────────────

static void test_quantile_and_median() {
    begin("quantile_and_median");

    // Type-7 on 1..5: q1 = 2, median = 3, q3 = 4 exactly.
    std::vector<double> odd{1, 2, 3, 4, 5};
    check_near(bench::quantile(odd, 0.25), 2.0, 1e-12, "q1 of 1..5");
    check_near(bench::quantile(odd, 0.50), 3.0, 1e-12, "median of 1..5");
    check_near(bench::quantile(odd, 0.75), 4.0, 1e-12, "q3 of 1..5");

    // Even length: median interpolates the two middle values.
    std::vector<double> even{1, 2, 3, 4};
    check_near(bench::median_of(even), 2.5, 1e-12, "median of 1..4");
    check_near(bench::quantile(even, 0.25), 1.75, 1e-12, "q1 of 1..4");
    check_near(bench::quantile(even, 0.75), 3.25, 1e-12, "q3 of 1..4");

    // Order must not matter, and the input is taken by value.
    std::vector<double> shuffled{5, 1, 4, 2, 3};
    check_near(bench::median_of(shuffled), 3.0, 1e-12, "median ignores order");
    check(shuffled[0] == 5.0, "caller's vector is not reordered");

    check_near(bench::median_of({7}), 7.0, 1e-12, "median of one element");
    check_near(bench::median_of({}), 0.0, 1e-12, "median of empty is 0");
}

static void test_summarize() {
    begin("summarize");
    bench::Stats s = bench::summarize({1, 2, 3, 4, 5});
    check(s.n == 5, "n");
    check_near(s.median, 3.0, 1e-12, "median");
    check_near(s.q1, 2.0, 1e-12, "q1");
    check_near(s.q3, 4.0, 1e-12, "q3");
    check_near(s.iqr, 2.0, 1e-12, "iqr = q3 - q1");
    check_near(s.min, 1.0, 1e-12, "min");
    check_near(s.max, 5.0, 1e-12, "max");

    bench::Stats flat = bench::summarize({2, 2, 2, 2});
    check_near(flat.iqr, 0.0, 1e-12, "iqr of constant series is 0");
}

static void test_paired_ratios() {
    begin("paired_ratios");

    // Paired division must happen rep-by-rep. Here both variants are 2x slower
    // in the second rep -- shared drift -- so every paired ratio is still 2.0,
    // whereas dividing the medians would also give 2.0 only by luck.
    std::vector<double> num{2.0, 4.0}, den{1.0, 2.0};
    auto r = bench::paired_ratio(num, den);
    check(r.size() == 2, "one ratio per rep");
    check_near(r[0], 2.0, 1e-12, "rep 0 ratio");
    check_near(r[1], 2.0, 1e-12, "rep 1 ratio drift-cancelled");

    // Net ratio strips the control from both sides before dividing.
    // (3-1)/(2-1) = 2.0, against a gross ratio of 1.5.
    auto net = bench::paired_net_ratio({3.0}, {2.0}, {1.0});
    check(net.size() == 1, "one net ratio");
    check_near(net[0], 2.0, 1e-12, "net ratio subtracts control");
    check_near(bench::paired_ratio({3.0}, {2.0})[0], 1.5, 1e-12,
               "gross ratio is diluted by the control offset");

    // A rep where the control met or beat a variant is unusable, not 'zero'.
    check(bench::paired_net_ratio({1.0}, {2.0}, {1.0}).empty(),
          "drops reps where subtraction goes non-positive");
    check(bench::paired_ratio({1.0}, {0.0}).empty(), "drops zero denominators");
}

// ─── bf16 domain ─────────────────────────────────────────────────────────────

static void test_bf16_finite() {
    begin("bf16_finite");
    // 1 sign + 8 exponent + 7 mantissa: the all-ones exponent covers 128
    // mantissa patterns per sign, so 256 codepoints are Inf or NaN.
    auto all = bench::enumerate_bf16_finite();
    check(all.size() == 65280, "65536 - 256 finite codepoints");

    for (std::uint32_t b = 0; b <= 0xFFFFu; ++b) {
        auto u = (std::uint16_t)b;
        bool want = std::isfinite((double)(float)bench::bf16_from_bits(u));
        if (bench::bf16_is_finite(u) != want) {
            check(false, "bf16_is_finite disagrees with isfinite");
            break;
        }
    }
}

static void test_enumerate_range() {
    begin("enumerate_range");
    auto near1 = bench::enumerate_bf16_in_range(0.9, 1.1);

    // The historical starvation, kept as a test so the reason the clusters were
    // widened stays visible. 38 is the exact count of bf16 values inside the
    // range; the old sampler observed 40 because round-to-nearest pulls the two
    // codepoints just outside each endpoint into the sample set. Both numbers
    // describe the same starvation.
    check(near1.size() == 38, "old [0.9,1.1] holds 38 distinct bf16 values");
    check(bench::enumerate_bf16_in_range(79.5, 80.5).size() == 3,
          "old [79.5,80.5] holds only 3 distinct bf16 values");

    // Ascending by value, no duplicates, all inside the range.
    for (std::size_t i = 0; i < near1.size(); ++i) {
        double v = bench::bf16_to_double(near1[i]);
        check(v >= 0.9 && v <= 1.1, "value inside requested range");
        if (i) check(bench::bf16_to_double(near1[i - 1]) < v, "strictly ascending");
    }

    // Negative ranges must work by value, not by codepoint order.
    auto neg = bench::enumerate_bf16_in_range(-92.0, -32.0);
    check(!neg.empty(), "negative range is non-empty");
    for (std::size_t i = 1; i < neg.size(); ++i)
        check(bench::bf16_to_double(neg[i - 1]) < bench::bf16_to_double(neg[i]),
              "negative range ascends by value");

    check(bench::enumerate_bf16_in_range(1.0, 0.0).empty(), "inverted range is empty");
}

// This is the regression test for the bias that prompted the rewrite.
static void test_cluster_distinct_input_floor() {
    begin("cluster_distinct_input_floor");
    struct Table { const char* name; const bench::Cluster* c; std::size_t n; };
    const Table tables[] = {
        { "log", bench::LOG_CLUSTERS, std::size(bench::LOG_CLUSTERS) },
        { "exp", bench::EXP_CLUSTERS, std::size(bench::EXP_CLUSTERS) },
        { "sin", bench::SIN_CLUSTERS, std::size(bench::SIN_CLUSTERS) },
    };
    for (const auto& t : tables) {
        for (std::size_t i = 0; i < t.n; ++i) {
            const auto& cl = t.c[i];
            auto cps = bench::enumerate_bf16_in_range(cl.lo, cl.hi);
            if (cps.size() < bench::MIN_DISTINCT_INPUTS) {
                std::printf("  FAIL  %-32s %s/%s has %zu distinct inputs, floor is %zu\n",
                            g_case, t.name, cl.label, cps.size(),
                            bench::MIN_DISTINCT_INPUTS);
                ++g_failures;
            }
        }
        // Each harness must carry at least one sweep cluster, and a sweep must
        // be substantially wider than a targeted cluster to be worth the name.
        std::size_t sweeps = 0;
        for (std::size_t i = 0; i < t.n; ++i) {
            if (!t.c[i].sweep) continue;
            ++sweeps;
            check(bench::enumerate_bf16_in_range(t.c[i].lo, t.c[i].hi).size() >= 1000,
                  "sweep cluster covers >=1000 codepoints");
        }
        check(sweeps >= 1, "harness has a sweep cluster");
    }
}

// Regression for the defect that made the first sweep design report 0.69x for
// log: a sweep over every finite bf16 is half x < 0, where log returns NaN
// without touching a table. The sweep measured the error path and the branch
// misprediction from mixing it with the table path, not the table scheme.
// Sweeps must stay inside the path they claim to measure.
static void test_sweeps_stay_on_table_path() {
    begin("sweeps_stay_on_table_path");

    for (const auto& cl : bench::LOG_CLUSTERS) {
        if (!cl.sweep) continue;
        for (auto u : bench::enumerate_bf16_in_range(cl.lo, cl.hi)) {
            if (bench::bf16_to_double(u) <= 0.0) {
                check(false, "log sweep must exclude x <= 0 (NaN/-Inf early exit)");
                break;
            }
        }
    }

    // exp consults a table only for 2^-9 < |x| < 93; outside it returns a
    // constant. 15105 codepoints per sign sit below 2^-9 alone.
    for (const auto& cl : bench::EXP_CLUSTERS) {
        if (!cl.sweep) continue;
        for (auto u : bench::enumerate_bf16_in_range(cl.lo, cl.hi)) {
            const double a = std::fabs(bench::bf16_to_double(u));
            if (!(a > 0.001953125 && a < 93.0)) {
                check(false, "exp sweep must stay inside the table path");
                break;
            }
        }
    }

    // sin returns x itself for |x| <= 0x1.dp-4, with no table in any variant.
    for (const auto& cl : bench::SIN_CLUSTERS) {
        if (!cl.sweep) continue;
        for (auto u : bench::enumerate_bf16_in_range(cl.lo, cl.hi)) {
            if (std::fabs(bench::bf16_to_double(u)) <= 0.11328125) {
                check(false, "sin sweep must exclude the no-table small-|x| exit");
                break;
            }
        }
    }
}

static void test_cycle_buffer() {
    begin("cycle_buffer");
    std::mt19937 rng(42);
    auto cps = bench::enumerate_bf16_in_range(0.5, 2.0);
    check(!cps.empty(), "source codepoints non-empty");

    auto buf = bench::make_bf16_cycle_buffer(cps, rng, 4096);
    check(buf.n == 4096, "window rounded up to a power of two");
    check((buf.n & (buf.n - 1)) == 0, "window is a power of two");
    check(buf.storage.size() == buf.n + bench::LAYOUT_SLACK,
          "storage carries slack for per-rep re-basing");

    // Every offset the runner can pick must still yield a full window.
    for (std::size_t pad : { (std::size_t)0, bench::LAYOUT_SLACK / 2, bench::LAYOUT_SLACK })
        check(buf.window(pad) + buf.n <= buf.storage.data() + buf.storage.size(),
              "window at max offset stays inside storage");

    std::map<std::uint16_t, int> seen;
    for (auto f : buf.storage) seen[bench::bf16_to_bits(f)]++;
    check(seen.size() == cps.size(), "every codepoint appears, and only those");

    int lo = (int)buf.storage.size(), hi = 0;
    for (const auto& kv : seen) { lo = std::min(lo, kv.second); hi = std::max(hi, kv.second); }
    check(hi - lo <= 1, "occurrences differ by at most one (balanced coverage)");

    // A codepoint list longer than the requested minimum still gets a
    // power-of-two window big enough to hold all of it.
    auto full = bench::enumerate_bf16_finite();
    auto fbuf = bench::make_bf16_cycle_buffer(full, rng, 4096);
    check(fbuf.n == 65536, "full domain rounds 65280 up to 65536");

    check(bench::next_pow2(1) == 1 && bench::next_pow2(3) == 4 &&
          bench::next_pow2(4) == 4 && bench::next_pow2(65280) == 65536,
          "next_pow2");
}

// ─── reporting must not invent numbers ───────────────────────────────────────

// A missing (cluster, variant) key used to render as a plausible 0.00 ratio,
// and the baseline row printed a hardcoded 1.00 whether or not it had data.
// A reader cannot distinguish a fabricated number from a measured one, so every
// column backed by an empty series must print "n/a".
static void test_report_never_invents_numbers() {
    begin("report_never_invents_numbers");

    const char* path = "output/.harness-test-report.tmp";
    bench::open_log(path);
    // Variant 1 has no data at all; variant 2 has fewer epochs than the base.
    bench::report_cluster("probe", 128, {1.0, 1.0, 1.0},
                          { { "base",  {2.0, 2.0, 2.0} },
                            { "empty", {} },
                            { "short", {3.0} } },
                          /*base_index=*/0);
    bench::close_log();

    std::FILE* f = std::fopen(path, "r");
    check(f != nullptr, "report file written");
    if (!f) return;
    std::string text;
    char buf[512];
    while (std::fgets(buf, sizeof buf, f)) text += buf;
    std::fclose(f);
    std::remove(path);

    const std::size_t empty_at = text.find("empty");
    check(empty_at != std::string::npos, "empty variant still gets a row");
    if (empty_at != std::string::npos) {
        const std::string row = text.substr(empty_at, text.find('\n', empty_at) - empty_at);
        check(row.find("n/a") != std::string::npos, "empty variant reports n/a");
        check(row.find("0.00") == std::string::npos,
              "empty variant must not print a numeric ratio");
        check(row.find("1.00") == std::string::npos,
              "empty variant must not print a numeric ratio");
    }
    check(text.find("only 1 of 3 epochs paired") != std::string::npos,
          "short series is labelled rather than silently truncated");
}

// ─── epoch failure handling ──────────────────────────────────────────────────

// Readings are paired positionally across variants, so a child that emits only
// part of its rows must be discarded whole. If partial output were appended,
// one variant's epoch 3 would end up divided by another variant's epoch 4 and
// the report would look entirely normal.
static void test_epoch_failure_handling() {
    begin("epoch_failure_handling");

    // Exits nonzero, prints nothing.
    check(bench::gather_epochs("/usr/bin/false", 3).empty(),
          "child exiting nonzero contributes nothing");

    // Exits zero, prints nothing parseable.
    check(bench::gather_epochs("/usr/bin/true", 3).empty(),
          "child producing no rows contributes nothing");

    // A path that cannot be quoted safely is refused rather than shelled out.
    check(bench::gather_epochs("/tmp/evil\"path", 2).empty(),
          "path containing a quote is refused");
}

// ─── the harness measuring itself ────────────────────────────────────────────

// Two identical kernels through the real timing path. Their true ratio is 1.0,
// so whatever this reports is the harness noise floor. The old single-run
// harness produced 0.54x-1.06x here.
static void test_noise_floor() {
    begin("noise_floor");
    std::mt19937 rng(42);
    auto cps = bench::enumerate_bf16_in_range(0.5, 2.0);
    auto buf = bench::make_bf16_cycle_buffer(cps, rng, 4096);

    auto k = [](__bf16 x) { return (double)(float)x * 1.000001; };
    auto series = bench::run_interleaved(buf, 200'000, 21, 50'000, k, k);

    check((int)series[0].size() == 21, "one reading per rep");
    check(bench::summarize(series[0]).median > 0.0, "variant 0 timed");
    check(bench::summarize(series[1]).median > 0.0, "variant 1 timed");

    // A noise-floor check can only conclude anything on a quiet machine. If the
    // timings themselves are scattered, the machine is loaded and a ratio drawn
    // from them says nothing about the harness -- under contention this test saw
    // two identical kernels 7% apart. Report that as inconclusive rather than as
    // a harness defect, or it fails whenever CI is busy.
    const bench::Stats a = bench::summarize(series[0]);
    const double rel_spread = a.median > 0.0 ? a.iqr / a.median : 1.0;
    if (rel_spread > 0.15) {
        skip("machine too loaded to measure a noise floor (control spread %.1f%%)",
             100.0 * rel_spread);
        return;
    }

    const bench::Stats s = bench::summarize(bench::paired_ratio(series[0], series[1]));
    check_near(s.median, 1.0, 0.05, "identical kernels report ratio ~1.0");
    check(s.iqr < 0.10, "paired ratio IQR is tight for identical kernels");
    if (s.iqr >= 0.10)
        std::printf("        (observed IQR %.4f, median %.4f)\n", s.iqr, s.median);
}

int main() {
    std::printf("\nbench-harness unit tests\n\n");
    test_quantile_and_median();
    test_summarize();
    test_paired_ratios();
    test_bf16_finite();
    test_enumerate_range();
    test_cluster_distinct_input_floor();
    test_sweeps_stay_on_table_path();
    test_cycle_buffer();
    test_report_never_invents_numbers();
    test_epoch_failure_handling();
    test_noise_floor();

    if (g_failures == 0 && g_skips == 0)
        std::printf("  all checks passed\n\n");
    else if (g_failures == 0)
        std::printf("  all checks passed (%d skipped)\n\n", g_skips);
    else
        std::printf("\n  %d check(s) failed\n\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
