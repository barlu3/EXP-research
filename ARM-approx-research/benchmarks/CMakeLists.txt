# Cycle-counted benchmarks of ARM's estimate instructions against the exact
# instruction sequences they stand in for:
#   bench-recip   FRECPE against FDIV                 (this directory)
#   bench-rsqrt   FRSQRTE against FSQRT + FDIV        (recip-sqrt/)
#
# The kernels are hand-written AArch64 (Mach-O symbol names), so these exist
# only on Apple Silicon. Everything the benchmarks share lives here: the driver
# (approx-bench.hpp), cycle counting (bench-cycles.hpp), the loop-shape macros
# (approx-kernels.inc) and the control and calibration kernels (approx-common.S,
# built once as the approx-common object library). Statistics, epochs and
# reporting come from the main harness in benchmarks/.
#
# -O3 applies to the C++ drivers only. The timed code is assembly, so the
# -march=native probing in benchmarks/CMakeLists.txt would change nothing here.
#
# They count cycles with the kperf fixed counters when run as root, e.g.
#   cd ARM-approx-research/benchmarks && sudo output/bench-recip
# Without root, the run-bench-* targets and CTest fall back to a calibrated
# timer and say so in the report. Each ends with an exhaustive accuracy check
# over every finite float32 and exits nonzero if a kernel is wrong, so each is a
# test too.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  enable_language(ASM)

  add_library(approx-common OBJECT approx-common.S)
  target_include_directories(approx-common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"          # approx-bench.hpp, bench-cycles.hpp, approx-kernels.inc
    "${CMAKE_SOURCE_DIR}/benchmarks")      # bench-harness.hpp

  set(APPROX_OUT "${CMAKE_CURRENT_SOURCE_DIR}/output")

  add_executable(bench-recip benchmark-recip.cpp recip-kernels.S)
  target_link_libraries(bench-recip PRIVATE approx-common)
  target_compile_options(bench-recip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set_target_properties(bench-recip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${APPROX_OUT}")

  add_custom_target(run-bench-recip
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${APPROX_OUT}"
    COMMAND bench-recip
    DEPENDS bench-recip
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running benchmark-recip.cpp"
    USES_TERMINAL)
  # Timing runs must never overlap; join the end of the main benchmarks' chain.
  add_dependencies(run-bench-recip run-bench-sin-limb)

  add_test(NAME bench-recip COMMAND bench-recip WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(bench-recip PROPERTIES TIMEOUT 900 LABELS benchmark)

  add_subdirectory(recip-sqrt)
endif()
