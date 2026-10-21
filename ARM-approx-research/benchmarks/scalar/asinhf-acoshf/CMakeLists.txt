# FRECPE as the range reduction of the log inside float32 asinh(x) and
# acosh(x), against a table holding the same values and against the system
# libm's asinhf and acoshf. ../../CMakeLists.txt enables ASM and defines
# approx-common, the shared driver and kernels. MPFR builds the -log(estimate)
# table and decides the rare inputs where the accuracy pass's double-precision
# reference is too close to a float rounding boundary to call.
#
# One kernel file holds both functions; the driver is built twice, the second
# time with AHYP_ACOSH defined. The binaries and their reports land here; each
# report is written beside its binary whatever the working directory. Run as
# root for hardware cycle counts:
#   cd ARM-approx-research/benchmarks/scalar/asinhf-acoshf && sudo ./bench-asinhf
foreach(fn IN ITEMS asinhf acoshf)
  add_executable(bench-${fn} benchmark-asinhf-acoshf.cpp asinhf-acoshf-kernels.S)
  target_link_libraries(bench-${fn} PRIVATE approx-common MPFR::MPFR)
  target_compile_options(bench-${fn} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set_target_properties(bench-${fn} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(run-bench-${fn}
    COMMAND bench-${fn}
    DEPENDS bench-${fn}
    WORKING_DIRECTORY "${APPROX_DIR}"
    COMMENT "Running scalar/asinhf-acoshf/benchmark-asinhf-acoshf.cpp as ${fn}"
    USES_TERMINAL)

  add_test(NAME bench-${fn} COMMAND bench-${fn} WORKING_DIRECTORY "${APPROX_DIR}")
  set_tests_properties(bench-${fn} PROPERTIES TIMEOUT 900 LABELS benchmark)
endforeach()
target_compile_definitions(bench-acoshf PRIVATE AHYP_ACOSH)

# Timing runs must never overlap.
add_dependencies(run-bench-asinhf run-bench-log10f)
add_dependencies(run-bench-acoshf run-bench-asinhf)
