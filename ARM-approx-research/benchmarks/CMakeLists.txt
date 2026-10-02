# Cycle-counted benchmarks of ARM's estimate instructions against the exact
# instruction sequences they stand in for. Each lives in its own directory,
# with its sources, its binary and its report:
#   scalar/frecpe/   bench-recip       FRECPE against FDIV
#   scalar/frsqrte/  bench-rsqrt       FRSQRTE against FSQRT + FDIV
#   vector/vrecpe/   bench-recip-vec   bench-recip on 4-lane .4s vectors
#   vector/vrsqrts/  bench-rsqrt-vec   bench-rsqrt on 4-lane .4s vectors
# Each writes its report beside its binary, whatever the working directory.
#
# The kernels are hand-written AArch64 (Mach-O symbol names), so these exist
# only on Apple Silicon. Everything the benchmarks share lives here: the driver
# (approx-bench.hpp), cycle counting (bench-cycles.hpp), the loop-shape macros
# (approx-kernels.inc) and the control and calibration kernels (approx-common.S,
# built once as the approx-common object library). Statistics, epochs and
# reporting come from the main harness in CORE-research/benchmarks/.
#
# -O3 applies to the C++ drivers only. The timed code is assembly, so the
# -march=native probing in CORE-research/benchmarks/CMakeLists.txt would change nothing here.
#
# They count cycles with the kperf fixed counters when run as root, e.g.
#   cd ARM-approx-research/benchmarks/scalar/frecpe && sudo ./bench-recip
# Without root, the run-bench-* targets and CTest fall back to a calibrated
# timer and say so in the report. Each ends with an exhaustive accuracy check
# over every finite float32 and exits nonzero if a kernel is wrong, so each is a
# test too.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  enable_language(ASM)

  add_library(approx-common OBJECT approx-common.S)
  target_include_directories(approx-common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"          # approx-bench.hpp, bench-cycles.hpp, approx-kernels.inc
    "${CMAKE_SOURCE_DIR}/CORE-research/benchmarks")  # bench-harness.hpp

  # The working directory of the run-bench-* targets and CTest. Reports do not
  # depend on it: each is written beside its binary (approx-bench.hpp).
  set(APPROX_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

  # Timing runs must never overlap, so each run-bench-* target depends on the
  # one before: scalar/frecpe, scalar/frsqrte, vector/vrecpe, vector/vrsqrts.
  add_subdirectory(scalar/frecpe)
  add_subdirectory(scalar/frsqrte)
  add_subdirectory(vector)
endif()
