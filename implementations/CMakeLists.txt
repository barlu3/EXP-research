# The math kernels themselves. Nothing here is built directly: the CORE-MATH
# sources are compiled into the benchmarks (benchmarks/) and the MPFR verifiers
# (cross-eval/), which name them by path. The only target this file owns is the
# SageMath regeneration of the float16/bfloat16 lookup tables.

# Lookup-table generation needs SageMath; tables are never edited by hand.
find_program(SAGE_EXECUTABLE sage)
if(SAGE_EXECUTABLE)
  add_custom_target(tables
    COMMAND "${SAGE_EXECUTABLE}" inria-expbf16.sage
    COMMAND "${SAGE_EXECUTABLE}" inria-expf16.sage
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/exp"
    COMMENT "Regenerating T1/T2 lookup tables via SageMath"
    USES_TERMINAL)
else()
  add_custom_target(tables
    COMMAND "${CMAKE_COMMAND}" -E echo
            "SageMath not found — install it to regenerate lookup tables."
    COMMAND "${CMAKE_COMMAND}" -E false
    COMMENT "Lookup-table generation unavailable")
endif()
