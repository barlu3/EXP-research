/* Shared measurement core for every benchmark in this directory.

   This header exists because run_bench used to be copy-pasted into all five
   benchmarks and had drifted into three different behaviours: benchmark-limb
   timed a single run, benchmark-exp-limb and benchmark-sin-limb took the best
   of three, and the float64 pair took a single run again. The ln harness also
   still printed a hardcoded -mavx2 -mfma banner that CMake had already dropped
   on Apple Silicon. One copy cannot drift from itself.

   What the core does differently from the old run_bench:

     - Replication with dispersion. Every measurement is REPS timed runs, and
       what gets reported is the median with its interquartile range. Best-of-N
       hides exactly the variance that motivated this rewrite: it reports the
       luckiest run and says nothing about how far the unlucky ones fell.

     - Paired interleaving. Variants are timed A,B,C,A,B,C rather than all of A
       then all of B, and the ratio is the median of the per-rep ratios rather
       than the ratio of the medians. Frequency and thermal drift then cancel
       inside each rep instead of accumulating across a run.

     - A control variant. A kernel that only loads and converts the input costs
       about 0.6 ns/call in this loop, against 0.98 ns for cr_log_bf16 -- 61% of
       what the old harness attributed to the function was loop scaffolding.
       That offset is additive and identical across variants, so it drags every
       ratio toward 1.0. Reporting it lets a net ratio be quoted alongside.

     - Codepoint enumeration instead of interval sampling. bf16 carries 7
       mantissa bits, so a narrow real interval collapses to very few distinct
       values: the old [79.5, 80.5] cluster drew 500,000 samples from exactly 3
       of them. Buffers are now built from the distinct codepoints a range
       actually contains, shuffled, so the table sees the spread of indices it
       would see in use.

   Buffers are power-of-two sized and indexed with a mask. That is tidier than
   i % inputs.size() but is not a performance fix -- the modulo measured 0.031
   ns/iter against the mask, which is noise at this scale. */

#ifndef BENCH_HARNESS_HPP
#define BENCH_HARNESS_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bench {

using Clock = std::chrono::steady_clock;

// Default replication count. 15 is enough for a stable median and a meaningful
// quartile spread without making `make bench` painful: the bf16 harnesses run
// 20M iterations per rep per variant.
inline constexpr int DEFAULT_REPS = 5;

// Reps run and thrown away before the recorded ones. The first rep is
// contaminated for every variant at once -- 1.14 ns against a 0.57 ns steady
// state for the control -- which is the process still settling rather than
// anything about the code under test.
inline constexpr int DISCARD_REPS = 3;

// Every cluster must reach this many distinct inputs to be worth reporting.
// The harness-test asserts it over the shipped cluster tables, so a range that
// starves cannot be added without the test failing.
inline constexpr std::size_t MIN_DISTINCT_INPUTS = 100;

// ─── Statistics ──────────────────────────────────────────────────────────────

// Type-7 quantile (the R and NumPy default): linear interpolation between the
// order statistics bracketing p. Named so the reported IQR is reproducible
// rather than "whatever the implementation happened to do".
inline double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double h = (double)(v.size() - 1) * p;
    const std::size_t lo = (std::size_t)h;
    const std::size_t hi = (lo + 1 < v.size()) ? lo + 1 : lo;
    return v[lo] + (h - (double)lo) * (v[hi] - v[lo]);
}

inline double median_of(std::vector<double> v) { return quantile(std::move(v), 0.5); }

struct Stats {
    int    n      = 0;
    double median = 0.0;
    double q1     = 0.0;
    double q3     = 0.0;
    double iqr    = 0.0;
    double min    = 0.0;
    double max    = 0.0;
};

inline Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n      = (int)v.size();
    s.median = quantile(v, 0.50);
    s.q1     = quantile(v, 0.25);
    s.q3     = quantile(v, 0.75);
    s.iqr    = s.q3 - s.q1;
    s.min    = v.front();
    s.max    = v.back();
    return s;
}

// Ratio taken rep-by-rep, so drift shared by both variants within a rep divides
// out. Taking median(num)/median(den) instead would let a slow patch in one
// variant's runs bias the ratio even when both were slow together.
inline std::vector<double> paired_ratio(const std::vector<double>& num,
                                        const std::vector<double>& den) {
    std::vector<double> r;
    const std::size_t n = std::min(num.size(), den.size());
    r.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (den[i] > 0.0) r.push_back(num[i] / den[i]);
    return r;
}

// As above, with the control's scaffolding cost removed from both sides first.
// Reps where either side lands at or below the control are dropped: the
// subtraction has gone unstable there and the quotient is meaningless.
inline std::vector<double> paired_net_ratio(const std::vector<double>& num,
                                            const std::vector<double>& den,
                                            const std::vector<double>& ctl) {
    std::vector<double> r;
    const std::size_t n = std::min({num.size(), den.size(), ctl.size()});
    r.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = num[i] - ctl[i], b = den[i] - ctl[i];
        if (a > 0.0 && b > 0.0) r.push_back(a / b);
    }
    return r;
}

// ─── bf16 domain ─────────────────────────────────────────────────────────────

union bf16_bits { __bf16 f; std::uint16_t u; };

inline __bf16 bf16_from_bits(std::uint16_t u) { bf16_bits t; t.u = u; return t.f; }
inline std::uint16_t bf16_to_bits(__bf16 f)   { bf16_bits t; t.f = f; return t.u; }
inline double bf16_to_double(std::uint16_t u) { return (double)(float)bf16_from_bits(u); }

// bf16 is 1 sign + 8 exponent + 7 mantissa, so the all-ones exponent covers
// 128 mantissa patterns per sign: 256 non-finite codepoints, 65280 finite.
inline bool bf16_is_finite(std::uint16_t u) { return (u & 0x7f80u) != 0x7f80u; }

// The distinct bf16 values a real interval actually contains, ascending by
// value. This is the whole point of the rewrite: callers describe a range and
// get the codepoints in it, not repeated draws from a distribution that a
// 7-bit mantissa collapses onto a handful of values.
inline std::vector<std::uint16_t> enumerate_bf16_in_range(double lo, double hi) {
    std::vector<std::pair<double, std::uint16_t>> hits;
    for (std::uint32_t b = 0; b <= 0xFFFFu; ++b) {
        const auto u = (std::uint16_t)b;
        if (!bf16_is_finite(u)) continue;
        const double v = bf16_to_double(u);
        if (v >= lo && v <= hi) hits.push_back({v, u});
    }
    std::sort(hits.begin(), hits.end(),
              [](const auto& a, const auto& c) { return a.first < c.first; });
    std::vector<std::uint16_t> out;
    out.reserve(hits.size());
    for (const auto& h : hits) out.push_back(h.second);
    return out;
}

// Every finite bf16, ascending by value. Both zeros are present; they are
// distinct codepoints and the kernels treat them as such.
inline std::vector<std::uint16_t> enumerate_bf16_finite() {
    return enumerate_bf16_in_range(-INFINITY, INFINITY);
}

inline std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Elements of slack allocated past the timed window. Each rep reads from a
// different offset inside that slack, which moves the buffer's base address and
// so its cache-set alignment relative to the tables.
//
// This exists because layout, not scheduling, turned out to be the dominant
// source of variation. Eight runs of the same binary on an idle machine gave
// limb/inria ratios from 0.848 to 1.235 for one cluster, while the inlined
// control held to 0.4% across all of them -- the machine was quiet, but each
// process laid its heap out differently against the tables in .rodata. A single
// fixed buffer makes each process internally consistent (IQR as low as 0.005)
// and consistently wrong, so the reported spread would have hidden a +-20%
// uncertainty rather than shown it. Re-basing per rep samples that distribution
// inside one run, which is what makes the IQR mean something.
inline constexpr std::size_t LAYOUT_SLACK = 4096;

template <typename T>
struct InputBuffer {
    std::vector<T> storage;
    std::size_t    n = 0;   // timed window length, a power of two

    // n is a power of two, so mask() is n-1. Asserted because n == 0 would
    // underflow to SIZE_MAX and turn `i & mask` into an unbounded index --
    // a silent out-of-bounds read rather than a crash at the empty buffer.
    // make_bf16_cycle_buffer returns an empty buffer for an empty codepoint
    // list, and only the cluster-floor test currently stops that reaching here.
    std::size_t mask() const { assert(n > 0 && "InputBuffer window is empty"); return n - 1; }
    const T* window(std::size_t pad) const { return storage.data() + pad; }
};

// Cycles a reshuffled permutation of `cps`. Each pass is shuffled afresh so the
// access order does not repeat with a short period the branch predictor and
// prefetcher can learn -- that would flatter the table paths in a way real call
// sites would not. Every codepoint appears floor(N/k) or ceil(N/k) times.
//
// Storage runs LAYOUT_SLACK elements past the window so any offset in
// [0, LAYOUT_SLACK] yields a full, valid window.
inline InputBuffer<__bf16> make_bf16_cycle_buffer(const std::vector<std::uint16_t>& cps,
                                                  std::mt19937& rng,
                                                  std::size_t min_size) {
    InputBuffer<__bf16> buf;
    if (cps.empty()) return buf;
    buf.n = next_pow2(std::max(min_size, cps.size()));
    const std::size_t total = buf.n + LAYOUT_SLACK;
    buf.storage.reserve(total);
    std::vector<std::uint16_t> pass(cps);
    while (buf.storage.size() < total) {
        std::shuffle(pass.begin(), pass.end(), rng);
        for (std::uint16_t u : pass) {
            if (buf.storage.size() == total) break;
            buf.storage.push_back(bf16_from_bits(u));
        }
    }
    return buf;
}

// float64/float32 inputs do not suffer the collapse bf16 does -- 500k draws
// from [0.9, 1.1] land on ~500k distinct doubles -- so these keep plain
// sampling and only gain the power-of-two size.
template <typename T>
inline InputBuffer<T> make_real_cycle_buffer(double lo, double hi,
                                             std::mt19937& rng, std::size_t min_size) {
    InputBuffer<T> buf;
    buf.n = next_pow2(min_size);
    std::uniform_real_distribution<double> dist(lo, hi);
    buf.storage.reserve(buf.n + LAYOUT_SLACK);
    for (std::size_t i = 0; i < buf.n + LAYOUT_SLACK; ++i)
        buf.storage.push_back((T)dist(rng));
    return buf;
}

// For inputs that are neither bf16 codepoints nor plain reals -- the segmented
// exp phases feed pre-reduced structs, so the timed loop measures one stage
// rather than the whole call. gen(i) supplies element i.
template <typename T, typename Gen>
inline InputBuffer<T> make_generated_buffer(std::size_t min_size, Gen gen) {
    InputBuffer<T> buf;
    buf.n = next_pow2(min_size);
    buf.storage.reserve(buf.n + LAYOUT_SLACK);
    for (std::size_t i = 0; i < buf.n + LAYOUT_SLACK; ++i) buf.storage.push_back(gen(i));
    return buf;
}

// ─── Timing ──────────────────────────────────────────────────────────────────

// Keeps the accumulator from being optimised away without putting a store in
// the loop body.
inline volatile double g_sink = 0.0;

// Iterations run untimed immediately before each timed run. Interleaving means
// the variant that ran last left its own table in cache and evicted this one's,
// so without this the first slice of every timed run pays another variant's
// eviction. That showed up as one variant swinging 0.92-1.41 ns between reps
// while the inlined control held to within 2%.
inline constexpr int PREWARM_ITERS = 200'000;

template <typename Fn, typename T>
inline double time_once_ns(Fn fn, const T* data, std::size_t mask, int iters) {
    double acc = 0.0;
    for (int i = 0; i < PREWARM_ITERS; ++i) acc += fn(data[(std::size_t)i & mask]);
    g_sink = acc;

    acc = 0.0;
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) acc += fn(data[(std::size_t)i & mask]);
    const auto t1 = Clock::now();
    g_sink = acc;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)iters;
}

template <typename Fn, typename T>
inline void warmup(Fn fn, const T* data, std::size_t mask, int iters) {
    double acc = 0.0;
    for (int i = 0; i < iters; ++i) acc += fn(data[(std::size_t)i & mask]);
    g_sink = acc;
}

namespace detail {
// Times the single variant whose index is `v`. The index comparison is a fold
// over the pack and happens outside the timed region, so selecting a variant at
// runtime costs nothing inside the loop.
template <typename T, typename Tuple, std::size_t... I>
inline void time_by_index(std::size_t v,
                          std::array<std::vector<double>, sizeof...(I)>& out,
                          const T* data, std::size_t mask, int iters, Tuple& fns,
                          std::index_sequence<I...>) {
    ((v == I ? (void)out[I].push_back(time_once_ns(std::get<I>(fns), data, mask, iters))
             : (void)0), ...);
}
template <typename T, typename Tuple, std::size_t... I>
inline void warm_all(const T* data, std::size_t mask, int iters, Tuple& fns,
                     std::index_sequence<I...>) {
    ((warmup(std::get<I>(fns), data, mask, iters)), ...);
}
}  // namespace detail

// Times every variant once per rep, in declared order, and returns the raw
// per-rep ns/call series for each. Summarising is left to the caller so the
// same series can feed both a median/IQR row and a paired ratio.
template <typename T, typename... Fns>
inline std::array<std::vector<double>, sizeof...(Fns)>
run_interleaved(const InputBuffer<T>& buf, int iters, int reps, int warm, Fns... fns) {
    std::array<std::vector<double>, sizeof...(Fns)> out;
    auto fn_tuple = std::make_tuple(fns...);
    using Idx = std::make_index_sequence<sizeof...(Fns)>;
    constexpr std::size_t N = sizeof...(Fns);
    // Fixed seed: the offsets are a controlled sweep over layouts, not a source
    // of run-to-run randomness. Every invocation visits the same set.
    std::mt19937 pad_rng(12345);
    std::uniform_int_distribution<std::size_t> pad_dist(0, LAYOUT_SLACK);
    detail::warm_all(buf.window(0), buf.mask(), warm, fn_tuple, Idx{});
    // Rotate the running order by one position each rep. Interleaving alone
    // still pins each variant to a fixed slot, and the slot matters: whichever
    // variant runs first after the gap between reps absorbs the frequency ramp,
    // which showed up as a 10x difference in IQR between two variants whose
    // medians were stable. Rotating gives every variant every position, so that
    // cost is shared instead of assigned to one of them.
    for (int r = 0; r < reps + DISCARD_REPS; ++r) {
        // One offset per rep, shared by every variant in that rep, so the
        // paired ratio still compares variants under identical conditions
        // while the series across reps spans many layouts.
        const std::size_t pad = pad_dist(pad_rng);
        const T* data = buf.window(pad);
        for (std::size_t p = 0; p < N; ++p)
            detail::time_by_index(((std::size_t)r + p) % N, out, data, buf.mask(),
                                  iters, fn_tuple, Idx{});
        if (r == DISCARD_REPS - 1)
            for (auto& v : out) v.clear();
    }
    return out;
}

// ─── Epochs (measurement across processes) ───────────────────────────────────

// Repetition inside one process cannot average out code layout. The kernels are
// reached through a call into another translation unit, and where that code and
// its .rodata tables land relative to each other is fixed for the life of the
// process and re-drawn by ASLR on the next one. Eight runs of one binary put the
// same cluster's ratio anywhere from 0.83 to 1.14 while the inlined control held
// to 0.4%; within any single run the spread was often under 0.01, so the process
// was confidently reporting a different answer each time.
//
// So the outer unit of replication is a process. The parent re-runs its own
// binary with --epoch, each child measures and prints its per-variant medians,
// and the parent takes the median and IQR across those. Everything the child
// does internally -- interleaving, rotation, per-rep re-basing -- still applies;
// epochs add the one axis a single process cannot sample.

inline constexpr int DEFAULT_EPOCHS = 7;

inline bool is_epoch_child(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--epoch") return true;
    return false;
}

// Children speak one line per measurement; anything else they print is ignored.
inline void emit_epoch_row(int cluster, int variant, double ns) {
    std::printf("ROW %d %d %.9f\n", cluster, variant, ns);
}

// key: (cluster, variant) -> one ns/call reading per epoch
using EpochTable = std::map<std::pair<int, int>, std::vector<double>>;

// An epoch is committed only if the child exited cleanly AND produced exactly
// the same set of rows as the first successful epoch. Partial output must be
// discarded rather than appended: the readings are paired positionally, so a
// child that died between emitting variant 0 and variant 2 of a cluster would
// leave the series different lengths and silently pair one variant's epoch 3
// against another's epoch 4.
inline EpochTable gather_epochs(const char* self, int epochs) {
    EpochTable table;
    std::vector<std::pair<int, int>> expected;   // row keys, from the first good epoch
    int committed = 0;

    // popen runs a shell, so a quote in the path would break out of the quoting
    // below. Refuse rather than build a malformed command.
    if (std::string(self).find('"') != std::string::npos) {
        std::fprintf(stderr, "error: executable path contains a quote; cannot spawn epochs\n");
        return table;
    }
    const std::string cmd = std::string("\"") + self + "\" --epoch";

    for (int e = 0; e < epochs; ++e) {
        std::FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::fprintf(stderr, "warning: could not spawn epoch %d\n", e);
            continue;
        }
        std::vector<std::pair<std::pair<int, int>, double>> rows;
        char line[256];
        while (std::fgets(line, sizeof line, pipe)) {
            int c, v; double ns;
            if (std::sscanf(line, "ROW %d %d %lf", &c, &v, &ns) == 3)
                rows.push_back({{c, v}, ns});
        }
        const int status = pclose(pipe);

        if (status != 0) {
            std::fprintf(stderr, "warning: epoch %d exited %d; discarding it\n", e, status);
            continue;
        }
        std::vector<std::pair<int, int>> keys;
        keys.reserve(rows.size());
        for (const auto& r : rows) keys.push_back(r.first);

        if (expected.empty()) {
            if (keys.empty()) {
                std::fprintf(stderr, "warning: epoch %d produced no rows; discarding it\n", e);
                continue;
            }
            expected = keys;
        } else if (keys != expected) {
            std::fprintf(stderr,
                         "warning: epoch %d produced %zu rows, expected %zu; discarding it\n",
                         e, keys.size(), expected.size());
            continue;
        }
        for (const auto& r : rows) table[r.first].push_back(r.second);
        ++committed;
    }

    if (committed == 0)
        std::fprintf(stderr, "error: no epoch completed; timings will be missing\n");
    else if (committed < epochs)
        std::fprintf(stderr, "warning: %d of %d epochs completed\n", committed, epochs);
    return table;
}

// ─── Dual output (stdout + report file) ──────────────────────────────────────

// Was copy-pasted into all five benchmarks as a file-static `lprintf`.
inline std::FILE* g_log = nullptr;

inline void open_log(const char* path) {
    g_log = std::fopen(path, "w");
    if (!g_log) std::fprintf(stderr, "warning: could not open %s\n", path);
}

inline void close_log() {
    if (g_log) { std::fclose(g_log); g_log = nullptr; }
}

__attribute__((format(printf, 1, 2)))
inline void logf(const char* fmt, ...) {
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

// ─── Reporting ───────────────────────────────────────────────────────────────

struct VariantSeries {
    const char*         name;
    std::vector<double> ns;   // per-rep ns/call, as returned by run_interleaved
};

// One block per cluster, one row per variant. A block rather than a single wide
// row because the interesting quantity is now four numbers per variant, and
// because it scales to the three-variant exp and sin harnesses unchanged.
//
// `net` divides out the control before taking the ratio, so it estimates the
// kernels' relative cost rather than the relative cost of (kernel + loop).
// Both are printed: the gross ratio is what a caller in this loop shape would
// see, the net ratio is what the arithmetic actually costs.
inline void report_cluster(const char* label, std::size_t distinct,
                           const std::vector<double>& ctl,
                           const std::vector<VariantSeries>& vs,
                           std::size_t base_index) {
    const Stats c = summarize(ctl);
    logf("\n  -- %-16s %6zu distinct inputs   control %.4f ns/call\n",
         label, distinct, c.median);
    logf("     %-14s %9s %8s %8s %8s %8s\n",
         "variant", "ns med", "ns IQR", "ratio", "IQR", "net");
    logf("     %-14s %9s %8s %8s %8s %8s\n",
         "--------------", "---------", "--------", "--------", "--------", "--------");

    // Every column prints "n/a" rather than a number whenever the series behind
    // it is empty. A missing (cluster, variant) key must not render as a
    // plausible 0.00 ratio or an assumed 1.00 baseline -- a reader cannot tell
    // a fabricated number from a measured one.
    const std::vector<double>& base = vs[base_index].ns;
    const bool base_ok = !base.empty();
    for (std::size_t i = 0; i < vs.size(); ++i) {
        const Stats s = summarize(vs[i].ns);
        if (s.n == 0) {
            logf("     %-14s %9s %8s %8s %8s %8s\n",
                 vs[i].name, "n/a", "n/a", "n/a", "n/a", "n/a");
            continue;
        }
        if (i == base_index) {
            logf("     %-14s %9.4f %8.4f %8s %8s %8s\n",
                 vs[i].name, s.median, s.iqr, "1.00", "--", "1.00");
            continue;
        }
        if (!base_ok) {
            logf("     %-14s %9.4f %8.4f %8s %8s %8s\n",
                 vs[i].name, s.median, s.iqr, "n/a", "n/a", "n/a");
            continue;
        }
        const Stats r = summarize(paired_ratio(vs[i].ns, base));
        const Stats n = summarize(paired_net_ratio(vs[i].ns, base, ctl));
        char ratio_s[16], iqr_s[16], net_s[16];
        if (r.n > 0) { std::snprintf(ratio_s, sizeof ratio_s, "%.2f", r.median);
                       std::snprintf(iqr_s,   sizeof iqr_s,   "%.3f", r.iqr); }
        else         { std::snprintf(ratio_s, sizeof ratio_s, "n/a");
                       std::snprintf(iqr_s,   sizeof iqr_s,   "n/a"); }
        if (n.n > 0)   std::snprintf(net_s, sizeof net_s, "%.2f", n.median);
        else           std::snprintf(net_s, sizeof net_s, "n/a");

        logf("     %-14s %9.4f %8.4f %8s %8s %8s\n",
             vs[i].name, s.median, s.iqr, ratio_s, iqr_s, net_s);

        // A short series means epochs were dropped for this variant but not for
        // the baseline; say so rather than quietly comparing fewer samples.
        if (r.n > 0 && (std::size_t)r.n < base.size())
            logf("     %-14s (only %d of %zu epochs paired)\n", "", r.n, base.size());
    }
}

// Printed once per report so the reader knows how the numbers were produced
// without going to the source.
inline void print_method_banner(int iters, int reps, int warm,
                                const char* input_note) {
    logf("  Iterations    : %d per variant per rep\n", iters);
    logf("  Warmup        : %d iterations (not timed)\n", warm);
    logf("  Replication   : %d timed reps, variants interleaved within each rep\n", reps);
    logf("  Reported      : median ns/call with interquartile range;\n");
    logf("                  ratios are the median of the per-rep paired ratios\n");
    logf("  Control       : a load-and-convert-only kernel, timed per cluster.\n");
    logf("                  'net' is the ratio with that offset removed.\n");
    logf("  Inputs        : %s\n", input_note);
    // CMake probes -mavx2/-mfma and drops them where the target rejects them,
    // so these are read from the predefined macros rather than written out.
    logf("  Compile flags : -O3 -march=native%s%s -std=c++20\n",
#if defined(__AVX2__)
         " -mavx2",
#else
         "",
#endif
#if defined(__FMA__)
         " -mfma"
#else
         ""
#endif
    );
}

}  // namespace bench

#endif  // BENCH_HARNESS_HPP
