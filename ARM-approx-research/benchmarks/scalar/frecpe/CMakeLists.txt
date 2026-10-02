# FRECPE against FDIV: float32 1/x. ../../CMakeLists.txt enables ASM and
# defines approx-common, the shared driver and kernels. recip-ref.hpp holds the
# input clusters and reference, which ../../vector/vrecpe/ reuses.
#
# The binary and its report land here. Run it from ARM-approx-research/benchmarks
# (APPROX_DIR), which the report path is relative to; as root for hardware
# cycle counts:
#   cd ARM-approx-research/benchmarks && sudo scalar/frecpe/bench-recip
add_executable(bench-recip benchmark-recip.cpp recip-kernels.S)
target_link_libraries(bench-recip PRIVATE approx-common)
target_compile_options(bench-recip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-recip PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-recip
  COMMAND bench-recip
  DEPENDS bench-recip
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running scalar/frecpe/benchmark-recip.cpp"
  USES_TERMINAL)
# Timing runs must never overlap; join the end of the main benchmarks' chain.
add_dependencies(run-bench-recip run-bench-sin-limb)

add_test(NAME bench-recip COMMAND bench-recip WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-recip PROPERTIES TIMEOUT 900 LABELS benchmark)
