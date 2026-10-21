/* Benchmark: ways to compute 1/x in float32 on x86-64, in cycles. The x86-64
   counterpart of ARM-approx-research/benchmarks/scalar/frecpe/benchmark-recip.cpp:
   same question, same driver (approx-bench.hpp), same input clusters and
   correctly rounded reference (recip-ref.hpp), so the two reports compare row
   for row.

   The question: is the reciprocal-estimate pipeline (VRCPSS, then FMA Newton
   steps) cheaper than VDIVSS, or than calling a function that does the VDIVSS?

   x86 differs from AArch64 in three ways that matter here:
     - VRCPSS is always a 12-bit estimate; there is no 8-bit mode and no
       FPCR.AH, so no variant sets alt_fp.
     - There is no FRECPS. A Newton step is two FMAs (recip-kernels.S).
     - VRCPSS treats subnormal inputs as zero and flushes tiny results, so the
       Newton pipelines break outside the normal range. est2 sends those
       inputs to a scaled slow path (recip-kernels.S has the cases) and is the
       float32 contender: the accuracy pass fails the run if it is ever more
       than 1 ulp off, and if divss or call differ from 1.0f/x anywhere. The
       other estimate rows are reported, never checked.

   Cycle counts: this driver's hardware counters are macOS-only (kperf), so on
   x86 every report is the timer estimate, labelled as such. Under WSL2 the
   timer runs inside a Hyper-V guest; treat small differences with suspicion. */

#include "approx-bench.hpp"
#include "recip-ref.hpp"

#include <cstdint>

extern "C" {
void  recip_tput_fdiv(const float* in, float* out, std::uint64_t n);
void  recip_tput_call(const float* in, float* out, std::uint64_t n);
void  recip_tput_est0(const float* in, float* out, std::uint64_t n);
void  recip_tput_est1(const float* in, float* out, std::uint64_t n);
void  recip_tput_est2(const float* in, float* out, std::uint64_t n);
float recip_lat_fdiv(const float* in, std::uint64_t n);
float recip_lat_call(const float* in, std::uint64_t n);
float recip_lat_est0(const float* in, std::uint64_t n);
float recip_lat_est1(const float* in, std::uint64_t n);
float recip_lat_est2(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32 1/x on x86-64 -- DIVSS vs RCPSS (12-bit)",
    .kernels     = "x86-approx-research/benchmarks/scalar/rcpss/recip-kernels.S",
    .report_name = "bench_results_recip.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "divss (inline)", "divss",   0,         false, recip_tput_fdiv, recip_lat_fdiv },
        { "divss via call", "call",    0,         false, recip_tput_call, recip_lat_call },
        { "rcpss only",     "rcp",     UNCHECKED, false, recip_tput_est0, recip_lat_est0 },
        { "rcpss + 1 NR",   "rcp+1NR", UNCHECKED, false, recip_tput_est1, recip_lat_est1 },
        { "rcpss + 2 NR",   "rcp+2NR", 1,         false, recip_tput_est2, recip_lat_est2 },
    },
    .clusters       = recip_ref::CLUSTERS,
    .reference_note = recip_ref::REFERENCE_NOTE,
    .reference      = recip_ref::reference,
    .normal_block   = recip_ref::normal_block,
    .normal_note    = recip_ref::NORMAL_NOTE,
    .rel_error      = recip_ref::rel_error,
};

}  // namespace

int main(int argc, char** argv) { return approx::run_main(SPEC, argc, argv); }
