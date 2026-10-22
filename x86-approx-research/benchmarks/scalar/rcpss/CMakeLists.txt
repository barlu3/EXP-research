# RCPSS against DIVSS: float32 1/x on x86-64. ../../CMakeLists.txt enables ASM
# and defines x86-approx-common. The inputs and reference are the AArch64
# benchmark's own recip-ref.hpp, so both time the same inputs.
#
# The binary and its report land here:
#   cd x86-approx-research/benchmarks/scalar/rcpss && ./bench-recip-x86
add_executable(bench-recip-x86 benchmark-recip.cpp recip-kernels.S)
target_link_libraries(bench-recip-x86 PRIVATE x86-approx-common)
target_include_directories(bench-recip-x86 PRIVATE
  "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks/scalar/frecpe")  # recip-ref.hpp
target_compile_options(bench-recip-x86 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(bench-recip-x86 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(run-bench-recip-x86
  COMMAND bench-recip-x86
  DEPENDS bench-recip-x86
  WORKING_DIRECTORY "${X86_APPROX_DIR}"
  COMMENT "Running x86 scalar/rcpss/benchmark-recip.cpp"
  USES_TERMINAL)
# Timing runs must never overlap; join the end of the main benchmarks' chain.
add_dependencies(run-bench-recip-x86 run-bench-sin-limb)

add_test(NAME bench-recip-x86 COMMAND bench-recip-x86 WORKING_DIRECTORY "${X86_APPROX_DIR}")
set_tests_properties(bench-recip-x86 PROPERTIES TIMEOUT 900 LABELS benchmark)
