# RSQRTSS against SQRTSS + DIVSS: float32 1/sqrt(x) on x86-64. ../../CMakeLists.txt
# enables ASM and defines x86-approx-common. The inputs and reference are the
# AArch64 benchmark's own rsqrt-ref.hpp, whose MPFR fallback decides the rare
# inputs a double-precision reference cannot call.
#
# The binary and its report land here:
#   cd x86-approx-research/benchmarks/scalar/rsqrtss && ./bench-rsqrt-x86
add_executable(bench-rsqrt-x86 benchmark-rsqrt.cpp rsqrt-kernels.S)
target_link_libraries(bench-rsqrt-x86 PRIVATE x86-approx-common MPFR::MPFR m)
target_include_directories(bench-rsqrt-x86 PRIVATE
  "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks/scalar/frsqrte")  # rsqrt-ref.hpp
target_compile_options(bench-rsqrt-x86 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-rsqrt-x86 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-rsqrt-x86
  COMMAND bench-rsqrt-x86
  DEPENDS bench-rsqrt-x86
  WORKING_DIRECTORY "${X86_APPROX_DIR}"
  COMMENT "Running x86 scalar/rsqrtss/benchmark-rsqrt.cpp"
  USES_TERMINAL)
# Timing runs must never overlap.
add_dependencies(run-bench-rsqrt-x86 run-bench-recip-x86)

add_test(NAME bench-rsqrt-x86 COMMAND bench-rsqrt-x86 WORKING_DIRECTORY "${X86_APPROX_DIR}")
set_tests_properties(bench-rsqrt-x86 PROPERTIES TIMEOUT 900 LABELS benchmark)
