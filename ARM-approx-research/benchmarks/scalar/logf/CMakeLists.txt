# FRECPE as the range reduction inside float32 log(x), against a table holding
# the same values and against the system libm's logf. ../../CMakeLists.txt
# enables ASM and defines approx-common, the shared driver and kernels. MPFR
# builds the -log(estimate) table and decides the rare inputs where the
# accuracy pass's double-precision reference is too close to a float rounding
# boundary to call.
#
# The binary and its report land here; the report is written beside the binary
# whatever the working directory. Run it as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/scalar/logf && sudo ./bench-logf
add_executable(bench-logf benchmark-logf.cpp logf-kernels.S)
target_link_libraries(bench-logf PRIVATE approx-common MPFR::MPFR)
target_compile_options(bench-logf PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-logf PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-logf
  COMMAND bench-logf
  DEPENDS bench-logf
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running scalar/logf/benchmark-logf.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-logf run-bench-rsqrt-vec)

add_test(NAME bench-logf COMMAND bench-logf WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-logf PROPERTIES TIMEOUT 900 LABELS benchmark)
