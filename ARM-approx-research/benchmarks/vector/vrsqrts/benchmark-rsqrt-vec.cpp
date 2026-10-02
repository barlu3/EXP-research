/* Benchmark: float32 1/sqrt(x) on AArch64, four lanes at a time, in cycles.

   The vector counterpart of ../../scalar/frsqrte/benchmark-rsqrt.cpp. Same
   variants, same input clusters, same MPFR-backed reference and accuracy
   bounds; the only change is that every kernel works on 4-lane .4s vectors
   (rsqrt-vec-kernels.S) instead of scalars.

   Units match ../vrecpe/benchmark-recip-vec.cpp: throughput is cycles per
   result (per lane), latency is cycles per chain link (one 4-lane call), so
   latency reads directly against the scalar report's.

   Each lane computes what the scalar instruction does, so the accuracy table
   should match the scalar report's exactly; a difference means a kernel bug. */

#include "approx-bench.hpp"
#include "scalar/frsqrte/rsqrt-ref.hpp"

#include <cstdint>

extern "C" {
void  vec_tput_ctrl(const float* in, float* out, std::uint64_t n);
float vec_lat_ctrl(const float* in, std::uint64_t n);
void  rsqrtv_tput_sqdiv(const float* in, float* out, std::uint64_t n);
void  rsqrtv_tput_call(const float* in, float* out, std::uint64_t n);
void  rsqrtv_tput_est0(const float* in, float* out, std::uint64_t n);
void  rsqrtv_tput_est1(const float* in, float* out, std::uint64_t n);
void  rsqrtv_tput_est2(const float* in, float* out, std::uint64_t n);
float rsqrtv_lat_sqdiv(const float* in, std::uint64_t n);
float rsqrtv_lat_call(const float* in, std::uint64_t n);
float rsqrtv_lat_est0(const float* in, std::uint64_t n);
float rsqrtv_lat_est1(const float* in, std::uint64_t n);
float rsqrtv_lat_est2(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32x4 1/sqrt(x) on AArch64 -- FSQRT+FDIV vs FRSQRTE, .4s",
    .kernels     = "ARM-approx-research/benchmarks/vector/vrsqrts/rsqrt-vec-kernels.S",
    .report_path = "vector/vrsqrts/bench_results_rsqrt_vec.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        { "control",        "ctrl",     UNCHECKED, false, vec_tput_ctrl,     vec_lat_ctrl     },
        { "fsqrt + fdiv",   "sqdiv",    1,         false, rsqrtv_tput_sqdiv, rsqrtv_lat_sqdiv },
        { "sqrt+div call",  "call",     1,         false, rsqrtv_tput_call,  rsqrtv_lat_call  },
        { "frsqrte8 only",  "fe8",      UNCHECKED, false, rsqrtv_tput_est0,  rsqrtv_lat_est0  },
        { "frsqrte8+1 NR",  "fe8+1NR",  UNCHECKED, false, rsqrtv_tput_est1,  rsqrtv_lat_est1  },
        { "frsqrte8+2 NR",  "fe8+2NR",  2,         false, rsqrtv_tput_est2,  rsqrtv_lat_est2  },
        { "frsqrte12 only", "fe12",     UNCHECKED, true,  rsqrtv_tput_est0,  rsqrtv_lat_est0  },
        { "frsqrte12+1 NR", "fe12+1NR", UNCHECKED, true,  rsqrtv_tput_est1,  rsqrtv_lat_est1  },
    },
    .clusters       = rsqrt_ref::CLUSTERS,
    .reference_note = rsqrt_ref::REFERENCE_NOTE,
    .reference      = rsqrt_ref::reference,
    .normal_block   = rsqrt_ref::normal_block,
    .normal_note    = rsqrt_ref::NORMAL_NOTE,
    .rel_error      = rsqrt_ref::rel_error,
    .lat_lanes      = 4,
};

}  // namespace

int main(int argc, char** argv) { return approx::run_main(SPEC, argc, argv); }
