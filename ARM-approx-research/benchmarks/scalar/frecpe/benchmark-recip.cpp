/* Benchmark: three ways to compute 1/x in float32 on AArch64, in cycles.

   The question: at full float32 accuracy, over every real float32 input, is
   the reciprocal-estimate pipeline (FRECPE, then Newton steps via FRECPS/FMUL)
   cheaper than FDIV, or than calling a function that does the FDIV?

   "Full float32" fixes the pipeline's length before anything is timed.
   Checked exhaustively over all 4,278,190,080 finite floats against the
   correctly rounded quotient:

     FRECPE + 1 Newton step          up to 129 ulp off -- not float32
     FRECPE + 2 Newton steps         within 1 ulp, except every |x| < 2^-128,
                                     which returns the wrong-signed infinity
     ... + a sign-bit copy  (est2)   within 1 ulp everywhere; 32% of inputs
                                     differ from FDIV by that 1 ulp
     ... + one FMA correction        still not correctly rounded: misses all
                                     504 normals with an all-ones mantissa and
                                     12 inputs with subnormal results, and is
                                     NaN wherever 1/x overflows (x = +-0 and
                                     |x| <= 2^-128)

   So est2 is the float32 contender: faithful, not bit-identical to FDIV.
   Nothing cheap is bit-identical. The accuracy pass at the end repeats the
   exhaustive check on the exact kernels that were timed, and fails the run if
   fdiv or call differ from 1.0f/x anywhere, or est2 is ever more than 1 ulp off.

   Four more variants are timed but are not contenders; their accuracy is
   reported, never checked:

     frecpe8 only      the bare 8-bit estimate. Isolates what FRECPE itself
                       costs, so the Newton steps' share of est2 can be read
                       off by subtraction.
     frecpe8 + 1 NR    the textbook one-step pipeline on the 8-bit estimate:
                       about 2^-16 relative, the cheapest refinement there is.
     frecpe12 only     the same instruction with FPCR.AH set. On cores with
                       FEAT_AFP and FEAT_RPRES (Apple M5 has both) that makes
                       single-precision FRECPE a 12-bit estimate.
     frecpe12 + 1 NR   the textbook one-step pipeline on the 12-bit estimate.

   What AH buys and costs, measured over every finite float:

     estimate error, normal x and 1/x     2^-8.45 -> 2^-11.70 relative: 12
                                          bits, but not a strict 2^-12 bound
     FRECPE latency                       3 -> 8 cycles; throughput unchanged
                                          at 1/cycle. FRECPS and FMUL are
                                          unaffected by AH (timed alone).
     12-bit + one FRECPS/FMUL step        within 3 ulp on normals, so still
                                          not float32; an FMA step instead
                                          reaches 2 ulp but is NaN at +-0
     the flush-to-zero AH forces inside   every subnormal x -> +-inf, every
     FRECPE and FRECPS                    |x| >= 2^126 -> +-0: 46M inputs
                                          (1.1%) that no Newton step repairs

   The driver sets AH around each whole run (approx::AltFp), never per kernel
   call: a set/restore pair costs ~75 cycles, ~2% of a 4096-element pass.

   The timed loops are hand-written in recip-kernels.S, so no compiler sits
   between the instructions and the count. C calls a kernel once per pass over a
   4096-element buffer; at a few cycles per element, that call is a few cycles
   per ~10^4, below anything this reports, and the control kernel pays it too.

   Two loop shapes, because the ranking can differ between them:
     latency     one dependent chain through the inputs -- the cost of a
                 reciprocal on a critical path
     throughput  independent inputs -- the cost of a loop of reciprocals

   This file only describes the benchmark: its kernels, input clusters and
   reference. Timing, epochs, the report and the accuracy pass are shared with
   the reciprocal square root benchmark in approx-bench.hpp. Cycle counts come
   from bench-cycles.hpp: the core's fixed counters when run under sudo,
   otherwise a timer converted with a measured ns-per-cycle and labelled as an
   estimate. Epochs, rotation, discarded reps and the paired ratios are the
   main harness's; see docs/BENCHMARKING.md for why each exists. */

#include "approx-bench.hpp"
#include "recip-ref.hpp"

#include <cstdint>

extern "C" {
void  recip_tput_fdiv(const float* in, float* out, std::uint64_t n);
void  recip_tput_call(const float* in, float* out, std::uint64_t n);
void  recip_tput_est2(const float* in, float* out, std::uint64_t n);
void  recip_tput_est0(const float* in, float* out, std::uint64_t n);
void  recip_tput_est1(const float* in, float* out, std::uint64_t n);
float recip_lat_fdiv(const float* in, std::uint64_t n);
float recip_lat_call(const float* in, std::uint64_t n);
float recip_lat_est2(const float* in, std::uint64_t n);
float recip_lat_est0(const float* in, std::uint64_t n);
float recip_lat_est1(const float* in, std::uint64_t n);
}

namespace {

using approx::UNCHECKED;

const approx::Spec SPEC = {
    .title       = "float32 1/x on AArch64 -- FDIV vs FRECPE (8- and 12-bit)",
    .kernels     = "ARM-approx-research/benchmarks/scalar/frecpe/recip-kernels.S",
    .report_name = "bench_results_recip.txt",
    // Index 0 is the control, 1 the baseline every ratio divides by.
    .variants = {
        approx::CONTROL,
        { "fdiv (inline)",  "fdiv",     0,         false, recip_tput_fdiv, recip_lat_fdiv },
        { "fdiv via call",  "call",     0,         false, recip_tput_call, recip_lat_call },
        { "frecpe8 only",   "fe8",      UNCHECKED, false, recip_tput_est0, recip_lat_est0 },
        { "frecpe8 + 1 NR", "fe8+1NR",  UNCHECKED, false, recip_tput_est1, recip_lat_est1 },
        { "frecpe8 + 2 NR", "fe8+2NR",  1,         false, recip_tput_est2, recip_lat_est2 },
        { "frecpe12 only",  "fe12",     UNCHECKED, true,  recip_tput_est0, recip_lat_est0 },
        { "frecpe12+1 NR",  "fe12+1NR", UNCHECKED, true,  recip_tput_est1, recip_lat_est1 },
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
