/* Benchmark: ways to compute 1/sqrt(x) in float32 on x86-64, in cycles. The
   x86-64 counterpart of ARM-approx-research/benchmarks/scalar/frsqrte/
   benchmark-rsqrt.cpp: same question, same driver (approx-bench.hpp), same
   input clusters and correctly rounded reference (rsqrt-ref.hpp), so the two
   reports compare row for row.

   The question: is the estimate pipeline (VRSQRTSS, then FMA Newton steps)
   cheaper than VSQRTSS + VDIVSS, at what accuracy, over every real float32
   input?

   Where x86 differs from AArch64 (rsqrt-kernels.S has the details):
     - VRSQRTSS is always a 12-bit estimate; no 8-bit mode, no FPCR.AH.
     - No FRSQRTS or FMULX, so a Newton step is a multiply and two FMAs, and
       0*inf is NaN: the bare pipeline breaks at +-0, +inf and on subnormals.
       est2 sends those to a scaled slow path behind a per-block range check.
     - gcc keeps errno for sqrtf, so the call variant branches to libm's sqrtf
       for x < 0.

   The baselines are held to 1 ulp and est2 to 2, as on AArch64; the rest are
   reported, never checked. est2u is est2 minus its range check, so the
   difference between those two rows is what the check costs. */

#include "approx-bench.hpp"
#include "rsqrt-ref.hpp"

#include <cstdint>

extern "C" {
void  rsqrt_tput_sqdiv(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_call(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est0(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est1(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est2(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est2u(const float* in, float* out, std::uint64_t n);
float rsqrt_lat_sqdiv(const float* in, std::uint64_t n);
float rsqrt_lat_call(const float* in, std::uint64_t n);
float rsqrt_lat_est0(const float* in, std::uint64_t n);
float rsqrt_lat_est1(const float* in, std::uint64_t n);
float rsqrt_lat_est2(const float* in, std::uint64_t n);
float rsqrt_lat_est2u(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32 1/sqrt(x) on x86-64 -- SQRTSS+DIVSS vs RSQRTSS (12-bit)",
    .kernels     = "x86-approx-research/benchmarks/scalar/rsqrtss/rsqrt-kernels.S",
    .report_name = "bench_results_rsqrt.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "sqrtss + divss",    "sqdiv",     1,         false, rsqrt_tput_sqdiv, rsqrt_lat_sqdiv },
        { "sqrt+div call",     "call",      1,         false, rsqrt_tput_call,  rsqrt_lat_call  },
        { "rsqrtss only",      "rsq",       UNCHECKED, false, rsqrt_tput_est0,  rsqrt_lat_est0  },
        { "rsqrtss + 1 NR",    "rsq+1NR",   UNCHECKED, false, rsqrt_tput_est1,  rsqrt_lat_est1  },
        { "rsqrtss + 2 NR",    "rsq+2NR",   2,         false, rsqrt_tput_est2,  rsqrt_lat_est2  },
        { "rsq+2NR unchecked", "rsq+2NRu",  UNCHECKED, false, rsqrt_tput_est2u, rsqrt_lat_est2u },
    },
    .clusters       = rsqrt_ref::CLUSTERS,
    .reference_note = rsqrt_ref::REFERENCE_NOTE,
    .reference      = rsqrt_ref::reference,
    .normal_block   = rsqrt_ref::normal_block,
    .normal_note    = rsqrt_ref::NORMAL_NOTE,
    .rel_error      = rsqrt_ref::rel_error,
};

}  // namespace

int main(int argc, char** argv) { return approx::run_main(SPEC, argc, argv); }
