# float32x4 1/x: the 4-lane counterpart of ../../scalar/frecpe/, whose
# recip-ref.hpp supplies the input clusters and reference. ../CMakeLists.txt
# defines approx-vec-common (vector loop macros and control loops).
#
# The binary and its report land here. Run it from ARM-approx-research/benchmarks
# (APPROX_DIR), which the report path is relative to; as root for hardware
# cycle counts:
#   cd ARM-approx-research/benchmarks && sudo vector/vrecpe/bench-recip-vec
add_executable(bench-recip-vec benchmark-recip-vec.cpp recip-vec-kernels.S)
target_link_libraries(bench-recip-vec PRIVATE approx-common approx-vec-common)
target_compile_options(bench-recip-vec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-recip-vec PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-recip-vec
  COMMAND bench-recip-vec
  DEPENDS bench-recip-vec
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running vector/vrecpe/benchmark-recip-vec.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-recip-vec run-bench-rsqrt)

add_test(NAME bench-recip-vec COMMAND bench-recip-vec WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-recip-vec PROPERTIES TIMEOUT 900 LABELS benchmark)
