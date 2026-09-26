# FRSQRTE against FSQRT + FDIV: the reciprocal square root counterpart of
# bench-recip, built from the same driver and shared kernels (../CMakeLists.txt
# enables ASM and defines approx-common). MPFR decides the rare inputs where the
# accuracy pass's double-precision reference sits too close to a float rounding
# boundary to call.
#
# Run it as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/recip-sqrt && sudo output/bench-rsqrt
set(RSQRT_OUT "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_executable(bench-rsqrt benchmark-rsqrt.cpp rsqrt-kernels.S)
target_link_libraries(bench-rsqrt PRIVATE approx-common MPFR::MPFR)
target_compile_options(bench-rsqrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-rsqrt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${RSQRT_OUT}")

add_custom_target(run-bench-rsqrt
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${RSQRT_OUT}"
  COMMAND bench-rsqrt
  DEPENDS bench-rsqrt
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Running benchmark-rsqrt.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-rsqrt run-bench-recip)

add_test(NAME bench-rsqrt COMMAND bench-rsqrt WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(bench-rsqrt PROPERTIES TIMEOUT 900 LABELS benchmark)
