# FRECPE as the range reduction inside float32 log(x), against a table holding
# the same values and against the system libm's logf. ../../CMakeLists.txt
# enables ASM and defines approx-common, the shared driver and kernels. MPFR
# builds the -log(estimate) table and decides the rare inputs where the
# accuracy pass's double-precision reference is too close to a float rounding
# boundary to call.
#
# The same two sources build log2f and log10f too, with LOGF_BASE set: only the
# table and two constants change (logf-kernels.S), so the three report on one
# reduction in three functions.
#
# The binaries and their reports land here; each report is written beside its
# binary whatever the working directory. Run as root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/scalar/logf && sudo ./bench-logf
foreach(fn IN ITEMS logf log2f log10f)
  add_executable(bench-${fn} benchmark-logf.cpp logf-kernels.S)
  target_link_libraries(bench-${fn} PRIVATE approx-common MPFR::MPFR)
  target_compile_options(bench-${fn} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set_target_properties(bench-${fn} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(run-bench-${fn}
    COMMAND bench-${fn}
    DEPENDS bench-${fn}
    WORKING_DIRECTORY "${APPROX_DIR}"
    COMMENT "Running scalar/logf/benchmark-logf.cpp as ${fn}"
    USES_TERMINAL)

  add_test(NAME bench-${fn} COMMAND bench-${fn} WORKING_DIRECTORY "${APPROX_DIR}")
  set_tests_properties(bench-${fn} PROPERTIES TIMEOUT 900 LABELS benchmark)
endforeach()
target_compile_definitions(bench-log2f PRIVATE LOGF_BASE=2)
target_compile_definitions(bench-log10f PRIVATE LOGF_BASE=10)

# Timing runs must never overlap. log2f and log10f run after scalar/powf.
add_dependencies(run-bench-logf run-bench-rsqrt-vec)
add_dependencies(run-bench-log2f run-bench-powf)
add_dependencies(run-bench-log10f run-bench-log2f)
