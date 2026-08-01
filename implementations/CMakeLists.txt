# Throughput benchmarks. These measure performance, not correctness — the
# homemade exp is header-inlined, so -march=native materially changes results.

set(BENCH_FLAGS -O3)
if(EXP_NATIVE_ARCH)
  list(APPEND BENCH_FLAGS -march=native -mavx2 -mfma)
endif()

foreach(variant home inria)
  add_executable(bench-${variant} benchmark-${variant}.cpp)
  target_compile_options(bench-${variant} PRIVATE ${BENCH_FLAGS})
  target_include_directories(bench-${variant} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(bench-${variant} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

  add_custom_target(run-bench-${variant}
    COMMAND bench-${variant}
    DEPENDS bench-${variant}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running benchmark-${variant}"
    USES_TERMINAL)
endforeach()

# Lookup-table generation needs SageMath; tables are never edited by hand.
find_program(SAGE_EXECUTABLE sage)
if(SAGE_EXECUTABLE)
  add_custom_target(tables
    COMMAND "${SAGE_EXECUTABLE}" inria-expbf16.sage
    COMMAND "${SAGE_EXECUTABLE}" inria-expf16.sage
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating T1/T2 lookup tables via SageMath"
    USES_TERMINAL)
else()
  add_custom_target(tables
    COMMAND "${CMAKE_COMMAND}" -E echo
            "SageMath not found — install it to regenerate lookup tables."
    COMMAND "${CMAKE_COMMAND}" -E false
    COMMENT "Lookup-table generation unavailable")
endif()
