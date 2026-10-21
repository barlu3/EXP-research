# FRECPE as the range reduction of log2 inside float32 pow(x, y), against a
# table holding the same values and against the system libm's powf.
# ../../CMakeLists.txt enables ASM and defines approx-common, the shared driver
# and kernels. MPFR builds the kernels' tables and decides the rare inputs where
# the accuracy pass's double-precision reference is too close to a float
# rounding boundary to call.
#
# The binary and its report land here; the report is written beside the binary
# whatever the working directory. Run it as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/scalar/powf && sudo ./bench-powf
add_executable(bench-powf benchmark-powf.cpp powf-kernels.S)
target_link_libraries(bench-powf PRIVATE approx-common MPFR::MPFR)
target_compile_options(bench-powf PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-powf PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-powf
  COMMAND bench-powf
  DEPENDS bench-powf
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running scalar/powf/benchmark-powf.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-powf run-bench-acosf)

add_test(NAME bench-powf COMMAND bench-powf WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-powf PROPERTIES TIMEOUT 900 LABELS benchmark)
