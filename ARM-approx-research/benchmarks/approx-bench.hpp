/* Shared driver for the cycle-counted benchmarks of ARM's estimate
   instructions: scalar/frecpe/ (FRECPE against FDIV), scalar/frsqrte/
   (FRSQRTE against FSQRT + FDIV), and the 4-lane vector versions of both in
   vector/vrecpe/ and vector/vrsqrts/.

   Both put one question to a different function -- does an estimate
   instruction plus Newton steps beat the exact instruction sequence, and at
   what accuracy -- so everything except the kernels, the input clusters and the
   correctly rounded reference lives here rather than in two copies. The main
   harness's header records why that matters: run_bench was once copy-pasted
   into five benchmarks and drifted into three different behaviours.

   A benchmark describes itself in a Spec and hands it to run_main, which
   supplies the FPCR.AH scoping for the 12-bit estimates, the interleaved timing
   loop, epochs, the report and the exhaustive accuracy pass. Cycle counts come
   from bench-cycles.hpp; statistics, epochs and report_cluster from the main
   harness, CORE-research/benchmarks/bench-harness.hpp. */

#ifndef APPROX_BENCH_HPP
#define APPROX_BENCH_HPP

#include "bench-cycles.hpp"
#include "bench-harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <climits>
#include <cstdlib>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#include <unistd.h>

// approx-common.S: each loop shape with the arithmetic removed, and the
// cycle-counter calibration chain.
extern "C" {
void          approx_tput_ctrl(const float* in, float* out, std::uint64_t n);
float         approx_lat_ctrl(const float* in, std::uint64_t n);
std::uint64_t approx_calib_chain(std::uint64_t iters);
}

namespace approx {

// 16 KB of inputs and 16 KB of outputs: L1-resident, so the loads and stores
// never wait on memory and the variants differ only in arithmetic.
inline constexpr std::size_t   BUF_N      = 4096;
// Elements per timed run. Long enough that the timer's 42 ns granularity and a
// kperf read's syscall are both far below 0.1% of the run.
inline constexpr std::uint64_t TPUT_ELEMS = 1u << 22;
inline constexpr std::uint64_t LAT_ELEMS  = 1u << 20;
inline constexpr int REPS   = bench::DEFAULT_REPS;
inline constexpr int EPOCHS = bench::DEFAULT_EPOCHS;

// Elements per step of the throughput loop shape: every tput kernel must accept
// any positive multiple of this. The scalar kernels step by 8, the vector ones
// by 32 (eight 4-lane registers).
inline constexpr std::uint64_t TPUT_STEP = 32;

// Variant::max_ulp for rows the accuracy pass reports but holds to nothing:
// they are not meant to be float32 results.
inline constexpr int UNCHECKED = -1;

struct Variant {
    const char* name;      // report row
    const char* short_;    // summary column and instruction-count line
    int         max_ulp;   // worst allowed distance from the correctly rounded
                           // result on any finite input (0 = bit-exact), with
                           // NaN exactly where the reference is NaN; or UNCHECKED
    bool        alt_fp;    // run with FPCR.AH set: the 12-bit estimates
    void  (*tput)(const float*, float*, std::uint64_t);
    float (*lat)(const float*, std::uint64_t);
};

// variants[0] of every Spec: the loop with nothing in it.
inline const Variant CONTROL = { "control", "ctrl", UNCHECKED, false,
                                 approx_tput_ctrl, approx_lat_ctrl };

struct Cluster {
    const char* label;
    std::uint32_t (*gen)(std::mt19937&);
};

struct Spec {
    const char* title;         // banner headline
    const char* kernels;       // where the timed code lives, for the banner
    const char* report_name;   // file name only; written beside the binary,
                               // which CMake builds into the benchmark's directory
    std::vector<Variant> variants;   // [0] is CONTROL, [1] the ratio baseline
    std::vector<Cluster> clusters;

    // The accuracy pass. `reference` must be correctly rounded; it runs in the
    // default FP mode over blocks of 2^16 inputs that share their top 16 bits.
    const char* reference_note;
    void   (*reference)(const float* in, float* out, std::size_t n);
    // Whether a block, named by those top 16 bits (sign and exponent), has
    // both input and result normal. The 'normal' accuracy columns look only
    // there, where an estimate's own error shows without the subnormal and
    // flush-to-zero cases on top.
    bool   (*normal_block)(std::uint32_t hi16);
    const char* normal_note;
    // |y / f(x) - 1| for finite nonzero y, accurate far below 2^-24.
    double (*rel_error)(float x, float y);

    // Inputs consumed per link of the latency chain. 1 for the scalar kernels;
    // 4 for the vector ones, whose links are one 4-lane op each. Latency is
    // reported per link -- the cost of one call on a critical path -- so a
    // vector row reads directly against the scalar report's.
    int lat_lanes = 1;
};

// FPCR.AH set for the scope's lifetime when `on`, then restored. With FEAT_AFP
// and FEAT_RPRES this is what turns single-precision FRECPE and FRSQRTE into
// 12-bit estimates. Scopes wrap whole runs, because a set/restore pair costs
// ~75 cycles. Everything inside a scope is an opaque kernel call, so no float
// arithmetic of the compiler's can be moved into or out of the altered mode.
class AltFp {
public:
    explicit AltFp(bool on) : on_(on) {
        if (!on_) return;
        __asm__ volatile("mrs %0, fpcr" : "=r"(saved_));
        __asm__ volatile("msr fpcr, %0" :: "r"(saved_ | 2u) : "memory");   // AH is bit 1
    }
    ~AltFp() {
        if (on_) __asm__ volatile("msr fpcr, %0" :: "r"(saved_) : "memory");
    }
    AltFp(const AltFp&) = delete;
    AltFp& operator=(const AltFp&) = delete;

private:
    bool          on_;
    std::uint64_t saved_ = 0;
};

// The 12-bit rows are only 12-bit if the core has both features; without them
// AH is ignored or changes nothing, and those rows quietly measure the 8-bit
// estimate. The banner prints these so a report cannot hide that. The
// accuracy table's relative-error column is the direct evidence either way.
inline int arm_feature(const char* name) {
    int v = 0;
    std::size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, nullptr, 0) == 0 ? v : 0;
}

inline float from_bits(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline std::uint32_t to_bits(float f)   { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

enum Mode { LATENCY = 0, THROUGHPUT = 1, NMODES = 2 };

// Epoch rows are keyed (cluster, variant). Mode is folded into the cluster id;
// instruction counts, present only in hardware mode, sit at variant nv + v.
inline int row_id(const Spec& s, int mode, int cluster) {
    return mode * (int)s.clusters.size() + cluster;
}

// Clusters are generated in order from one seed, so the parent rebuilds exactly
// the buffers its children timed.
inline std::vector<std::vector<float>> make_inputs(const Spec& s) {
    std::mt19937 rng(42);
    std::vector<std::vector<float>> all;
    for (const auto& cl : s.clusters) {
        std::vector<float> v(BUF_N);
        for (auto& x : v) x = from_bits(cl.gen(rng));
        all.push_back(std::move(v));
    }
    return all;
}

inline std::size_t distinct(const std::vector<float>& v) {
    std::vector<std::uint32_t> b;
    for (float x : v) b.push_back(to_bits(x));
    std::sort(b.begin(), b.end());
    return (std::size_t)(std::unique(b.begin(), b.end()) - b.begin());
}

inline volatile float g_sinkf = 0.0f;

inline void run(Mode m, const Variant& v, const float* in, float* out, std::uint64_t elems) {
    const AltFp fp(v.alt_fp);
    for (std::uint64_t done = 0; done < elems; done += BUF_N) {
        if (m == THROUGHPUT) v.tput(in, out, BUF_N);
        else                 g_sinkf = v.lat(in, BUF_N);
    }
}

struct Series {
    std::vector<std::vector<double>> cyc, ins;   // [variant][rep], per element
};

// run_interleaved's discipline, with cycles in place of ns: variants
// interleaved and rotated one slot per rep, a prewarm pass before every timed
// run, and the first DISCARD_REPS thrown away.
inline Series measure(const Spec& s, const bench::CycleSource& cs, Mode m,
                      const float* in, float* out) {
    const int nv = (int)s.variants.size();
    Series r{ std::vector<std::vector<double>>(nv), std::vector<std::vector<double>>(nv) };
    const std::uint64_t elems = m == THROUGHPUT ? TPUT_ELEMS : LAT_ELEMS;
    // Throughput is per result; latency per chain link (Spec::lat_lanes).
    const double per = m == THROUGHPUT ? (double)elems : (double)(elems / s.lat_lanes);
    for (const auto& v : s.variants) run(m, v, in, out, elems / 8);
    for (int rep = 0; rep < REPS + bench::DISCARD_REPS; ++rep) {
        // Timer mode converts with a factor taken in the same rep, so a clock
        // change between reps is corrected; one inside a rep is not.
        const double ns_per_cyc = cs.hardware() ? 0.0 : cs.ns_per_cycle();
        for (int p = 0; p < nv; ++p) {
            const int vi = (rep + p) % nv;
            const Variant& v = s.variants[vi];
            run(m, v, in, out, BUF_N);
            const bench::CounterReading a = cs.read();
            run(m, v, in, out, elems);
            const bench::CounterReading b = cs.read();
            const double cyc = cs.hardware() ? (double)(b.cycles - a.cycles)
                                             : (b.ns - a.ns) / ns_per_cyc;
            r.cyc[vi].push_back(cyc / per);
            if (cs.hardware())
                r.ins[vi].push_back((double)(b.instructions - a.instructions) / per);
        }
        if (rep == bench::DISCARD_REPS - 1) {
            for (auto& x : r.cyc) x.clear();
            for (auto& x : r.ins) x.clear();
        }
    }
    return r;
}

inline int run_epoch(const Spec& s) {
    bench::prefer_performance_cores();
    std::string why;
    const bench::CycleSource cs = bench::CycleSource::open(approx_calib_chain, &why);
    const auto inputs = make_inputs(s);
    const int nv = (int)s.variants.size();
    std::vector<float> out(BUF_N);
    for (int c = 0; c < (int)s.clusters.size(); ++c)
        for (int m = 0; m < NMODES; ++m) {
            const Series r = measure(s, cs, (Mode)m, inputs[c].data(), out.data());
            for (int v = 0; v < nv; ++v) {
                bench::emit_epoch_row(row_id(s, m, c), v, bench::median_of(r.cyc[v]));
                if (cs.hardware())
                    bench::emit_epoch_row(row_id(s, m, c), nv + v, bench::median_of(r.ins[v]));
            }
        }
    return 0;
}

// ─── Reporting ──────────────────────────────────────────────────────────────

inline void box(const char* title) {
    bench::logf("\n+----------------------------------------------------------------+\n");
    bench::logf("|  %-62s|\n", title);
    bench::logf("+----------------------------------------------------------------+\n");
}

inline std::vector<double> paired_diff(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> d;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) d.push_back(a[i] - b[i]);
    return d;
}

inline std::string cell(const std::vector<double>& xs) {
    if (xs.empty()) return "n/a";
    char s[32];
    std::snprintf(s, sizeof s, "%.2f", bench::median_of(xs));
    return s;
}

// ─── Exhaustive accuracy ────────────────────────────────────────────────────

struct Accuracy {
    std::uint64_t n = 0, mismatches = 0, nan_mismatches = 0, max_ulp = 0;
    std::uint32_t worst = 0;   // smallest bit pattern reaching max_ulp, for a stable report
    std::uint64_t max_ulp_normal = 0;
    double        max_rel_normal = 0.0;
};

// Distance in representable floats, with -0 and +0 adjacent.
inline std::int64_t ordered(std::uint32_t b) {
    return (b >> 31) ? -(std::int64_t)(b & 0x7fffffffu) : (std::int64_t)b;
}

// Every finite float, through the same throughput kernels that were timed,
// against the Spec's correctly rounded reference. Blocks share their top 16
// bits, which fixes sign and exponent, so a block is all finite or all Inf/NaN
// and never needs padding. A result matches when its bits equal the
// reference's, or both are NaN; a NaN on one side only is a NaN mismatch and
// counts as infinitely far.
inline std::vector<Accuracy> exhaustive_accuracy(const Spec& s) {
    const int nv = (int)s.variants.size();
    std::vector<Accuracy> total(nv);
    std::mutex mu;
    const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nthreads; ++t)
        pool.emplace_back([&, t] {
            std::vector<Accuracy> acc(nv);
            std::vector<float> in(1u << 16), ref(1u << 16), out(1u << 16);
            for (std::uint32_t hi = t; hi < (1u << 16); hi += nthreads) {
                if (((hi >> 7) & 0xff) == 0xff) continue;
                const bool normal = s.normal_block(hi);
                for (std::uint32_t lo = 0; lo < (1u << 16); ++lo) in[lo] = from_bits(hi << 16 | lo);
                s.reference(in.data(), ref.data(), in.size());
                for (int v = 1; v < nv; ++v) {
                    {
                        const AltFp fp(s.variants[v].alt_fp);
                        s.variants[v].tput(in.data(), out.data(), in.size());
                    }
                    Accuracy& a = acc[v];
                    for (std::uint32_t lo = 0; lo < (1u << 16); ++lo) {
                        ++a.n;
                        const float y = out[lo], w = ref[lo];
                        if (normal) {
                            const double rel = (std::isfinite(y) && y != 0.0f)
                                ? s.rel_error(in[lo], y) : INFINITY;
                            a.max_rel_normal = std::max(a.max_rel_normal, rel);
                        }
                        if (std::isnan(y) && std::isnan(w)) continue;
                        const std::uint32_t g = to_bits(y), wb = to_bits(w);
                        if (g == wb) continue;
                        ++a.mismatches;
                        const std::uint32_t xb = hi << 16 | lo;
                        std::uint64_t u;
                        if (std::isnan(y) != std::isnan(w)) { ++a.nan_mismatches; u = UINT64_MAX; }
                        else {
                            const std::int64_t d = ordered(g) - ordered(wb);
                            u = (std::uint64_t)(d < 0 ? -d : d);
                        }
                        if (normal) a.max_ulp_normal = std::max(a.max_ulp_normal, u);
                        if (u > a.max_ulp || (u == a.max_ulp && xb < a.worst)) {
                            a.max_ulp = u;
                            a.worst = xb;
                        }
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mu);
            for (int v = 1; v < nv; ++v) {
                Accuracy& d = total[v];
                const Accuracy& a = acc[v];
                d.n += a.n; d.mismatches += a.mismatches; d.nan_mismatches += a.nan_mismatches;
                d.max_ulp_normal = std::max(d.max_ulp_normal, a.max_ulp_normal);
                d.max_rel_normal = std::max(d.max_rel_normal, a.max_rel_normal);
                if (a.mismatches && (a.max_ulp > d.max_ulp ||
                                     (a.max_ulp == d.max_ulp && a.worst < d.worst))) {
                    d.max_ulp = a.max_ulp;
                    d.worst = a.worst;
                }
            }
        });
    for (auto& th : pool) th.join();
    return total;
}

// ─── The program ────────────────────────────────────────────────────────────

// The report goes beside the executable, which CMake builds into the
// benchmark's own source directory. A path relative to the working directory
// broke every run started anywhere else -- `sudo ./bench-recip` from
// scalar/frecpe/ could not open its report.
inline std::string report_path(const char* name) {
    char exe[PATH_MAX], real[PATH_MAX];
    std::uint32_t sz = sizeof exe;
    if (_NSGetExecutablePath(exe, &sz) != 0 || !realpath(exe, real)) return name;
    std::string path = real;
    return path.substr(0, path.rfind('/') + 1) + name;
}

// Under sudo the report is created owned by root, and a later run without sudo
// could not overwrite it. Hand it back to the user who ran sudo.
inline void give_to_sudo_user(const std::string& path) {
    const char* uid = std::getenv("SUDO_UID");
    const char* gid = std::getenv("SUDO_GID");
    if (geteuid() != 0 || !uid || !gid) return;
    if (chown(path.c_str(), (uid_t)std::strtoul(uid, nullptr, 10),
              (gid_t)std::strtoul(gid, nullptr, 10)) != 0)
        std::fprintf(stderr, "warning: could not chown %s\n", path.c_str());
}

inline int run_main(const Spec& s, int argc, char** argv) {
    if (bench::is_epoch_child(argc, argv)) return run_epoch(s);

    const std::string report = report_path(s.report_name);
    bench::open_log(report.c_str());
    give_to_sudo_user(report);
    const auto wall_start = bench::Clock::now();
    const int nv  = (int)s.variants.size();
    const int ncl = (int)s.clusters.size();

    // The parent opens a source only to learn why the children might fall back
    // to the timer; which mode they were actually in is read from their rows.
    std::string why;
    (void)bench::CycleSource::open(approx_calib_chain, &why);

    bench::logf("\n");
    bench::logf("==================================================================\n");
    bench::logf("  %s\n", s.title);
    bench::logf("  Kernels       : hand-written, %s\n", s.kernels);
    bench::logf("  Per timed run : %llu elements (throughput), %llu (latency),\n",
                (unsigned long long)TPUT_ELEMS, (unsigned long long)LAT_ELEMS);
    bench::logf("                  as passes over a %zu-element L1-resident buffer\n", BUF_N);
    bench::logf("  Replication   : %d reps x %d epochs, variants interleaved and rotated\n",
                REPS, EPOCHS);
    bench::logf("  Reported      : median cycles per result with IQR across epochs;\n");
    bench::logf("                  ratios are medians of per-epoch paired ratios vs %s\n",
                s.variants[1].short_);
    bench::logf("  Control       : the same loop with the arithmetic removed\n");
    if (s.lat_lanes > 1) {
        bench::logf("  Vector width  : %d lanes. Throughput is cycles per result (per lane);\n",
                    s.lat_lanes);
        bench::logf("                  latency is cycles per chain link, one %d-lane op\n",
                    s.lat_lanes);
    }
    if (std::any_of(s.variants.begin(), s.variants.end(), [](const Variant& v) { return v.alt_fp; })) {
        bench::logf("  12-bit rows   : FPCR.AH set around each whole run; this core reports\n");
        bench::logf("                  FEAT_AFP=%d FEAT_RPRES=%d (both 1 for a real 12-bit estimate)\n",
                    arm_feature("hw.optional.arm.FEAT_AFP"),
                    arm_feature("hw.optional.arm.FEAT_RPRES"));
    }

    const bench::EpochTable table = bench::gather_epochs(argv[0], EPOCHS);
    auto at = [&](int id, int v) {
        auto it = table.find({id, v});
        return it == table.end() ? std::vector<double>{} : it->second;
    };
    const bool hw = !at(row_id(s, 0, 0), nv).empty();
    if (hw) {
        bench::logf("  Cycles        : hardware, kperf fixed counter (validated against a\n");
        bench::logf("                  dependent ADD chain: 8 cycles, 14 instructions/iter)\n");
    } else {
        bench::logf("  Cycles        : ESTIMATED -- ns / (ns per cycle of a dependent ADD\n");
        bench::logf("                  chain, measured each rep). No hardware counters:\n");
        bench::logf("                  %s\n", why.c_str());
    }
    bench::logf("==================================================================\n");

    const auto inputs = make_inputs(s);
    static const char* MODE_TITLE[NMODES] = {
        "LATENCY -- a dependent chain through every input",
        "THROUGHPUT -- independent inputs, results stored",
    };
    static const char* MODE_NOTE[NMODES] = {
        "  The control is the chain's AND+ORR link alone. Latencies add along a\n"
        "  chain, so 'net' is the arithmetic's own latency ratio.\n",
        "  The control is load+store+loop. In throughput these overlap with the\n"
        "  arithmetic instead of adding to it, so read 'ratio'; 'net' over-corrects.\n",
    };

    for (int m = 0; m < NMODES; ++m) {
        box(MODE_TITLE[m]);
        bench::logf("%s", MODE_NOTE[m]);
        if (m == LATENCY && s.lat_lanes > 1)
            bench::logf("  Each link is one %d-lane op, so these are cycles per call, not per\n"
                        "  result; divide by %d for the latency share of one lane.\n",
                        s.lat_lanes, s.lat_lanes);
        for (int c = 0; c < ncl; ++c) {
            const int id = row_id(s, m, c);
            std::vector<bench::VariantSeries> vs;
            for (int v = 1; v < nv; ++v) vs.push_back({ s.variants[v].name, at(id, v) });
            bench::report_cluster(s.clusters[c].label, distinct(inputs[c]), at(id, 0), vs,
                                  /*base_index=*/0, "cyc");
        }
        if (hw) {
            // Data-independent, so one cluster says it. A count other than the
            // listing implies means the loop that ran is not the one in the .S.
            bench::logf("\n  retired instructions per element:");
            for (int v = 0; v < nv; ++v)
                bench::logf("  %s %s", s.variants[v].short_, cell(at(row_id(s, m, 0), nv + v)).c_str());
            bench::logf("\n");
        }
    }

    box(s.lat_lanes > 1 ? "SUMMARY -- cycles per link (latency) / result (tput)"
                        : "SUMMARY -- cycles per result, median across epochs");
    // One table per shape, one column per non-control variant. Latency is net
    // of the control, because latencies add along the chain; throughput is
    // gross, because there the control's work overlaps the arithmetic.
    static const char* SUMMARY_TITLE[NMODES] = {
        "latency, net of control", "throughput, gross",
    };
    constexpr int W = 10;
    for (int m = 0; m < NMODES; ++m) {
        bench::logf("\n  %-15s", SUMMARY_TITLE[m]);
        for (int v = 1; v < nv; ++v) bench::logf("%*s", W, s.variants[v].short_);
        bench::logf("\n");
        for (int c = 0; c < ncl; ++c) {
            const int id = row_id(s, m, c);
            const auto ctl = at(id, 0);
            bench::logf("  %-15s", s.clusters[c].label);
            for (int v = 1; v < nv; ++v)
                bench::logf("%*s", W, cell(m == LATENCY ? paired_diff(at(id, v), ctl)
                                                        : at(id, v)).c_str());
            bench::logf("\n");
        }
    }

    // -- Exhaustive accuracy ---------------------------------------------------
    // Run after the timings so the all-core load cannot heat the machine first.
    const auto acc = exhaustive_accuracy(s);
    box("ACCURACY -- every finite float32 vs correctly rounded result");
    bench::logf("  %llu inputs per variant.\n"
                "  reference : %s\n"
                "  'normal'  : only %s, where an estimate's own error shows\n"
                "              without the subnormal and flush-to-zero cases on top\n"
                "  'rel'     : the worst |y/f(x) - 1| there\n\n",
                (unsigned long long)acc[1].n, s.reference_note, s.normal_note);
    bench::logf("  %-16s %11s %9s %11s %11s   %s\n", "variant", "differ", "max ulp",
                "ulp normal", "rel normal", "worst input");
    bool ok = true;
    for (int v = 1; v < nv; ++v) {
        const Accuracy& a = acc[v];
        char ulp[24], ulpn[24], rel[24];
        if (a.nan_mismatches) std::snprintf(ulp, sizeof ulp, "NaN");
        else std::snprintf(ulp, sizeof ulp, "%llu", (unsigned long long)a.max_ulp);
        if (a.max_ulp_normal == UINT64_MAX) std::snprintf(ulpn, sizeof ulpn, "NaN");
        else std::snprintf(ulpn, sizeof ulpn, "%llu", (unsigned long long)a.max_ulp_normal);
        if (a.max_rel_normal == 0.0)            std::snprintf(rel, sizeof rel, "0");
        else if (std::isinf(a.max_rel_normal))  std::snprintf(rel, sizeof rel, "inf");
        else std::snprintf(rel, sizeof rel, "2^%.2f", std::log2(a.max_rel_normal));
        bench::logf("  %-16s %11llu %9s %11s %11s", s.variants[v].name,
                    (unsigned long long)a.mismatches, ulp, ulpn, rel);
        if (a.mismatches) {
            const float x = from_bits(a.worst);
            std::vector<float> in8(TPUT_STEP, x), out8(TPUT_STEP), ref8(TPUT_STEP);
            {
                const AltFp fp(s.variants[v].alt_fp);
                s.variants[v].tput(in8.data(), out8.data(), TPUT_STEP);
            }
            s.reference(in8.data(), ref8.data(), TPUT_STEP);
            bench::logf("   %a -> %a (want %a)", x, out8[0], ref8[0]);
        }
        bench::logf("\n");
        const int bound = s.variants[v].max_ulp;
        if (bound != UNCHECKED && (a.nan_mismatches != 0 || a.max_ulp > (std::uint64_t)bound))
            ok = false;
    }

    // Name what each row was held to, so a PASS never reads as covering the
    // reported-only rows.
    bench::logf("\n  accuracy: %s\n",
                ok ? "PASS" : "FAIL -- timings above are for kernels that are wrong");
    std::vector<int> bounds;
    for (int v = 1; v < nv; ++v) bounds.push_back(s.variants[v].max_ulp);
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    std::rotate(bounds.begin(), std::upper_bound(bounds.begin(), bounds.end(), UNCHECKED),
                bounds.end());   // UNCHECKED last
    for (int b : bounds) {
        char label[32];
        if (b == UNCHECKED) std::snprintf(label, sizeof label, "reported, unchecked");
        else if (b == 0)    std::snprintf(label, sizeof label, "must be exact");
        else                std::snprintf(label, sizeof label, "must be within %d ulp", b);
        std::string names;
        for (int v = 1; v < nv; ++v)
            if (s.variants[v].max_ulp == b) {
                if (!names.empty()) names += ", ";
                names += s.variants[v].name;
            }
        bench::logf("    %-20s: %s\n", label, names.c_str());
    }

    const double wall =
        std::chrono::duration<double>(bench::Clock::now() - wall_start).count();
    bench::logf("\n  total wall time: %.1f s\n\n", wall);
    bench::close_log();
    return ok ? 0 : 1;
}

}  // namespace approx

#endif  // APPROX_BENCH_HPP
