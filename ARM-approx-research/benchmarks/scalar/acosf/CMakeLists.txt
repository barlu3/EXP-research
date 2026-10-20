# FRSQRTE plus Newton steps as the square root inside float32 acos(x), against
# FSQRT in the same kernel and against the system libm's acosf.
# ../../CMakeLists.txt enables ASM and defines approx-common, the shared driver
# and kernels. MPFR decides the rare inputs where the accuracy pass's
# double-precision reference is too close to a float rounding boundary to call.
#
# The binary and its report land here; the report is written beside the binary
# whatever the working directory. Run it as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/scalar/acosf && sudo ./bench-acosf
add_executable(bench-acosf benchmark-acosf.cpp acosf-kernels.S)
target_link_libraries(bench-acosf PRIVATE approx-common MPFR::MPFR)
target_compile_options(bench-acosf PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-acosf PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-acosf
  COMMAND bench-acosf
  DEPENDS bench-acosf
  WORKING_DIRECTORY "${APPROX_DIR}"
  COMMENT "Running scalar/acosf/benchmark-acosf.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-acosf run-bench-logf)

add_test(NAME bench-acosf COMMAND bench-acosf WORKING_DIRECTORY "${APPROX_DIR}")
set_tests_properties(bench-acosf PROPERTIES TIMEOUT 900 LABELS benchmark)
