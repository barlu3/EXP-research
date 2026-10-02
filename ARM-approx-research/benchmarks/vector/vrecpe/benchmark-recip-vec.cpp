/* Benchmark: float32 1/x on AArch64, four lanes at a time, in cycles.

   The vector counterpart of ../../scalar/frecpe/benchmark-recip.cpp. Same
   variants, same input clusters, same reference and accuracy bounds; the only
   change is that every kernel works on 4-lane .4s vectors
   (recip-vec-kernels.S) instead of scalars.

   Why a separate benchmark: the scalar report shows FRECPE + 1 Newton step
   tying FDIV in throughput (both ~1 cycle per result, both bound at one per
   cycle) and losing in latency. A vector FRECPE produces four estimates per
   instruction, while FDIV's cost per lane is a property of the divider, so the
   ranking can change at full width. This measures whether it does.

   Units, so the two reports read against each other:
     throughput   cycles per result (per lane), as in the scalar report
     latency      cycles per chain link -- one 4-lane call -- so it compares
                  directly with the scalar latency of one call

   The accuracy pass runs every finite float32 through the vector throughput
   kernels. Each lane computes what the scalar instruction does, so the table
   should match the scalar report's exactly; a difference means a kernel bug. */

#include "approx-bench.hpp"
#include "scalar/frecpe/recip-ref.hpp"

#include <cstdint>

extern "C" {
void  vec_tput_ctrl(const float* in, float* out, std::uint64_t n);
float vec_lat_ctrl(const float* in, std::uint64_t n);
void  recipv_tput_fdiv(const float* in, float* out, std::uint64_t n);
void  recipv_tput_call(const float* in, float* out, std::uint64_t n);
void  recipv_tput_est2(const float* in, float* out, std::uint64_t n);
void  recipv_tput_est0(const float* in, float* out, std::uint64_t n);
void  recipv_tput_est1(const float* in, float* out, std::uint64_t n);
float recipv_lat_fdiv(const float* in, std::uint64_t n);
float recipv_lat_call(const float* in, std::uint64_t n);
float recipv_lat_est2(const float* in, std::uint64_t n);
float recipv_lat_est0(const float* in, std::uint64_t n);
float recipv_lat_est1(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32x4 1/x on AArch64 -- FDIV vs FRECPE (8-, 12-bit), .4s",
    .kernels     = "ARM-approx-research/benchmarks/vector/vrecpe/recip-vec-kernels.S",
    .report_name = "bench_results_vrecpe.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        { "control",        "ctrl",     UNCHECKED, false, vec_tput_ctrl,    vec_lat_ctrl    },
        { "fdiv (inline)",  "fdiv",     0,         false, recipv_tput_fdiv, recipv_lat_fdiv },
        { "fdiv via call",  "call",     0,         false, recipv_tput_call, recipv_lat_call },
        { "frecpe8 only",   "fe8",      UNCHECKED, false, recipv_tput_est0, recipv_lat_est0 },
        { "frecpe8 + 1 NR", "fe8+1NR",  UNCHECKED, false, recipv_tput_est1, recipv_lat_est1 },
        { "frecpe8 + 2 NR", "fe8+2NR",  1,         false, recipv_tput_est2, recipv_lat_est2 },
        { "frecpe12 only",  "fe12",     UNCHECKED, true,  recipv_tput_est0, recipv_lat_est0 },
        { "frecpe12+1 NR",  "fe12+1NR", UNCHECKED, true,  recipv_tput_est1, recipv_lat_est1 },
    },
    .clusters       = recip_ref::CLUSTERS,
    .reference_note = recip_ref::REFERENCE_NOTE,
    .reference      = recip_ref::reference,
    .normal_block   = recip_ref::normal_block,
    .normal_note    = recip_ref::NORMAL_NOTE,
    .rel_error      = recip_ref::rel_error,
    .lat_lanes      = 4,
};

}  // namespace

int main(int argc, char** argv) { return approx::run_main(SPEC, argc, argv); }
