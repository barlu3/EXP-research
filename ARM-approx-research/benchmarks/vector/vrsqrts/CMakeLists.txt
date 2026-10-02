# float32x4 1/sqrt(x): the 4-lane counterpart of ../../scalar/frsqrte/, whose
# rsqrt-ref.hpp supplies the input clusters and reference. ../CMakeLists.txt
# defines approx-vec-common (vector loop macros and control loops).
#
# The binary and its report land here; the report is written beside the binary
# whatever the working directory. Run it as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/vector/vrsqrts && sudo ./bench-rsqrt-vec
add_executable(bench-rsqrt-vec benchmark-rsqrt-vec.cpp rsqrt-vec-kernels.S)
target_link_libraries(bench-rsqrt-vec PRIVATE approx-common approx-vec-common MPFR::MPFR)
target_compile_options(bench-rsqrt-vec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-rsqrt-vec PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-rsqrt-vec
  COMMAND bench-rsqrt-vec
  DEPENDS bench-rsqrt-vec
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running vector/vrsqrts/benchmark-rsqrt-vec.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-rsqrt-vec run-bench-recip-vec)

add_test(NAME bench-rsqrt-vec COMMAND bench-rsqrt-vec WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-rsqrt-vec PROPERTIES TIMEOUT 900 LABELS benchmark)
