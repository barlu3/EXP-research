/* Benchmark: ways to compute 1/sqrt(x) in float32 on AArch64, in cycles.

   The reciprocal square root counterpart of ../frecpe/benchmark-recip.cpp, asking the
   same question: is the estimate pipeline (FRSQRTE, then Newton steps) cheaper
   than the exact instruction sequence, at what accuracy, over every real
   float32 input?

   There is no exact reciprocal-square-root instruction to compare against.
   The exact route is FSQRT then FDIV -- what clang emits for 1.0f / sqrtf(x) --
   and it rounds twice, so it is faithful rather than correctly rounded. That
   makes the accuracy reference a correctly rounded 1/sqrt computed here, not
   the baseline kernel. Checked exhaustively over every finite float against it:

     FSQRT + FDIV                    within 1 ulp; 26% of positive inputs are
                                     1 ulp off
     FRSQRTE (8-bit)                 2^-8.25 relative on normals
     ... + 1 Newton step             2^-15.9, up to 195 ulp; +0 -> -inf
     ... + 2 Newton steps (est2)     within 2 ulp everywhere, NaN for x < 0,
                                     +-inf at +-0 -- the float32 contender
     FRSQRTE (12-bit, FPCR.AH)       2^-11.58 relative on normals
     ... + 1 Newton step             up to 4 ulp on normals; under AH every
                                     subnormal input is flushed, so positive
                                     subnormals come back +inf

   The Newton step's grouping is not the textbook one; rsqrt-kernels.S explains
   why (e*e overflows or underflows at the ends of the range, and 0*inf needs
   FMULX).

   The baseline is held to 1 ulp and est2 to 2; the rest are reported, never
   checked. Timing, epochs, the report and the accuracy pass are shared with
   the reciprocal benchmark in ../../approx-bench.hpp; this file describes only the
   kernels, the input clusters and the reference. */

#include "approx-bench.hpp"
#include "rsqrt-ref.hpp"

#include <cstdint>

extern "C" {
void  rsqrt_tput_sqdiv(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_call(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est0(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est1(const float* in, float* out, std::uint64_t n);
void  rsqrt_tput_est2(const float* in, float* out, std::uint64_t n);
float rsqrt_lat_sqdiv(const float* in, std::uint64_t n);
float rsqrt_lat_call(const float* in, std::uint64_t n);
float rsqrt_lat_est0(const float* in, std::uint64_t n);
float rsqrt_lat_est1(const float* in, std::uint64_t n);
float rsqrt_lat_est2(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32 1/sqrt(x) on AArch64 -- FSQRT+FDIV vs FRSQRTE (8-, 12-bit)",
    .kernels     = "ARM-approx-research/benchmarks/scalar/frsqrte/rsqrt-kernels.S",
    .report_path = "scalar/frsqrte/bench_results_rsqrt.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "fsqrt + fdiv",   "sqdiv",    1,         false, rsqrt_tput_sqdiv, rsqrt_lat_sqdiv },
        { "sqrt+div call",  "call",     1,         false, rsqrt_tput_call,  rsqrt_lat_call  },
        { "frsqrte8 only",  "fe8",      UNCHECKED, false, rsqrt_tput_est0,  rsqrt_lat_est0  },
        { "frsqrte8+1 NR",  "fe8+1NR",  UNCHECKED, false, rsqrt_tput_est1,  rsqrt_lat_est1  },
        { "frsqrte8+2 NR",  "fe8+2NR",  2,         false, rsqrt_tput_est2,  rsqrt_lat_est2  },
        { "frsqrte12 only", "fe12",     UNCHECKED, true,  rsqrt_tput_est0,  rsqrt_lat_est0  },
        { "frsqrte12+1 NR", "fe12+1NR", UNCHECKED, true,  rsqrt_tput_est1,  rsqrt_lat_est1  },
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
