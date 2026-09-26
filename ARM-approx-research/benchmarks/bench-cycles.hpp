/* Cycle counts for hand-written kernels.

   benchmarks/bench-harness.hpp times in nanoseconds, which suits C kernels
   called through a loop the compiler owns. benchmark-recip times AArch64
   loops that cost a handful of cycles per element and asks a question stated
   in cycles, so it reads the core's own counters where it can.

   Two sources, chosen once at startup:

     - Hardware (macOS, root). Apple's private kperf framework exposes two fixed
       counters per thread: core cycles and retired instructions. Cycles do not
       move with clock frequency, so most of the drift the ns harness has to
       fight disappears. kperf refuses every call without root ("Operation not
       permitted" from kpc_set_counting), so this needs sudo.

     - Timer (everything else). Nanoseconds from steady_clock, converted with a
       ns-per-cycle figure the caller measures with the same calibration kernel.
       The macOS timer advances at 24 MHz -- CNTFRQ_EL0 reports 1 GHz, but the
       count moves in steps of ~42 -- so a timed run must last milliseconds for
       the granularity to vanish. Frequency drift within a rep is not removed.

   Hardware counters are not trusted on sight. The calibration kernel retires
   14 instructions per iteration on an 8-cycle critical path, so a working pair
   must read 8 and 14 per iteration; which counter reads which also settles
   their order. Anything else and the source stays in timer mode and says why,
   rather than reporting cycles it cannot vouch for. */

#ifndef BENCH_CYCLES_HPP
#define BENCH_CYCLES_HPP

#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__APPLE__)
#include <dlfcn.h>
#include <pthread.h>
#endif

namespace bench {

// The calibration kernel's contract: per iteration, an 8-cycle dependent chain
// and 14 retired instructions.
using CalibFn = std::uint64_t (*)(std::uint64_t iters);
inline constexpr double        CALIB_CYCLES_PER_ITER = 8.0;
inline constexpr double        CALIB_INSNS_PER_ITER  = 14.0;
inline constexpr std::uint64_t CALIB_ITERS           = 1u << 20;   // ~8M cycles, ~2 ms

// Ask the scheduler for a performance core. macOS offers no affinity API; the
// QoS class is the supported way to keep a thread off the efficiency cluster.
inline void prefer_performance_cores() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

struct CounterReading {
    std::uint64_t cycles       = 0;   // hardware mode only
    std::uint64_t instructions = 0;   // hardware mode only
    double        ns           = 0.0;
};

class CycleSource {
public:
    // Hardware mode if kperf is usable and passes calibration; otherwise timer
    // mode, with the reason written to *why.
    static CycleSource open(CalibFn calib, std::string* why);

    bool hardware() const { return get_ != nullptr; }

    // Exits the process if the counters fail mid-run: an epoch child with a hole
    // in its series must be discarded by the parent, not reported.
    CounterReading read() const;

    // Timer mode's conversion factor, measured now with the calibration kernel.
    double ns_per_cycle() const;

private:
    using GetFn = int (*)(std::uint32_t tid, std::uint32_t n, std::uint64_t* buf);
    static constexpr std::uint32_t KPC_CLASS_FIXED_MASK = 1u << 0;
    static constexpr std::uint32_t KPC_MAX_COUNTERS     = 32;

    static double now_ns() {
        return std::chrono::duration<double, std::nano>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    GetFn   get_   = nullptr;
    CalibFn calib_ = nullptr;
    int     cyc_   = 0;
    int     ins_   = 1;
};

inline CycleSource CycleSource::open(CalibFn calib, std::string* why) {
    CycleSource cs;
    cs.calib_ = calib;
#if defined(__APPLE__)
    void* h = dlopen("/System/Library/PrivateFrameworks/kperf.framework/kperf", RTLD_LAZY);
    if (!h) { *why = "kperf.framework could not be loaded"; return cs; }
    auto set_counting = (int (*)(std::uint32_t))dlsym(h, "kpc_set_counting");
    auto set_thread   = (int (*)(std::uint32_t))dlsym(h, "kpc_set_thread_counting");
    auto count        = (std::uint32_t (*)(std::uint32_t))dlsym(h, "kpc_get_counter_count");
    auto get_thread   = (GetFn)dlsym(h, "kpc_get_thread_counters");
    if (!set_counting || !set_thread || !count || !get_thread) {
        *why = "kperf is missing an expected kpc_* symbol";
        return cs;
    }
    if (count(KPC_CLASS_FIXED_MASK) < 2) {
        *why = "kperf reports fewer than two fixed counters";
        return cs;
    }
    errno = 0;
    if (set_counting(KPC_CLASS_FIXED_MASK) != 0 || set_thread(KPC_CLASS_FIXED_MASK) != 0) {
        *why = std::string("kperf refused (") + std::strerror(errno) +
               "); run under sudo for hardware cycle counts";
        return cs;
    }

    std::uint64_t a[KPC_MAX_COUNTERS] = {}, b[KPC_MAX_COUNTERS] = {};
    calib(CALIB_ITERS / 8);
    const int ra = get_thread(0, KPC_MAX_COUNTERS, a);
    calib(CALIB_ITERS);
    const int rb = get_thread(0, KPC_MAX_COUNTERS, b);
    if (ra != 0 || rb != 0) {
        *why = "kpc_get_thread_counters failed after counting was enabled";
        return cs;
    }
    const double c0 = (double)(b[0] - a[0]) / (double)CALIB_ITERS;
    const double c1 = (double)(b[1] - a[1]) / (double)CALIB_ITERS;
    auto near = [](double x, double want) { return std::fabs(x - want) <= 0.05 * want; };
    if (near(c0, CALIB_CYCLES_PER_ITER) && near(c1, CALIB_INSNS_PER_ITER)) {
        cs.cyc_ = 0; cs.ins_ = 1;
    } else if (near(c1, CALIB_CYCLES_PER_ITER) && near(c0, CALIB_INSNS_PER_ITER)) {
        cs.cyc_ = 1; cs.ins_ = 0;
    } else {
        char msg[200];
        std::snprintf(msg, sizeof msg,
                      "fixed counters read %.2f and %.2f per calibration iteration, "
                      "expected %.0f cycles and %.0f instructions",
                      c0, c1, CALIB_CYCLES_PER_ITER, CALIB_INSNS_PER_ITER);
        *why = msg;
        return cs;
    }
    cs.get_ = get_thread;
#else
    *why = "hardware cycle counters are only wired up for macOS";
#endif
    return cs;
}

inline CounterReading CycleSource::read() const {
    CounterReading r;
    if (get_) {
        std::uint64_t buf[KPC_MAX_COUNTERS] = {};
        if (get_(0, KPC_MAX_COUNTERS, buf) != 0) {
            std::fprintf(stderr, "error: kpc_get_thread_counters failed mid-run\n");
            std::exit(2);
        }
        r.cycles       = buf[cyc_];
        r.instructions = buf[ins_];
    }
    r.ns = now_ns();
    return r;
}

inline double CycleSource::ns_per_cycle() const {
    calib_(CALIB_ITERS / 8);
    const double t0 = now_ns();
    calib_(CALIB_ITERS);
    const double t1 = now_ns();
    return (t1 - t0) / ((double)CALIB_ITERS * CALIB_CYCLES_PER_ITER);
}

}  // namespace bench

#endif  // BENCH_CYCLES_HPP
