# MPFR-driven cross-evaluation. Both tools select their target function at
# compile time and #include the table source, so each variant is a separate
# executable. Outputs are written relative to cross-eval/, hence the
# WORKING_DIRECTORY on every run target.

# add_mpfr_tool(<binary> <source> <run-target> <comment> [define])
function(add_mpfr_tool binname source runtarget comment)
  add_executable(${binname} ${source})
  target_compile_options(${binname} PRIVATE -O2)
  target_link_libraries(${binname} PRIVATE MPFR::MPFR)
  target_include_directories(${binname} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  if(ARGC GREATER 4)
    target_compile_definitions(${binname} PRIVATE ${ARGV4})
  endif()
  set_target_properties(${binname} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(${runtarget}
    COMMAND ${binname}
    DEPENDS ${binname}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "${comment}"
    USES_TERMINAL)
endfunction()

set(VERIFY_COMMENT "all 65536 bf16 patterns vs MPFR")
add_mpfr_tool(verify_mpfr verify_mpfr.c verify-exp "Verifying exp: ${VERIFY_COMMENT}")
add_mpfr_tool(verify_sin  verify_mpfr.c verify-sin "Verifying sin: ${VERIFY_COMMENT}" VERIFY_SIN)
add_mpfr_tool(verify_log  verify_mpfr.c verify-log "Verifying log: ${VERIFY_COMMENT}" VERIFY_LOG)

add_mpfr_tool(round-16bitp round-16bitp.c run-round-exp "Rounding exp tables to 16-bit")
add_mpfr_tool(round-sin    round-16bitp.c run-round-sin "Rounding sin tables to 16-bit" ROUND_SIN)
add_mpfr_tool(round-log    round-16bitp.c run-round-log "Rounding log tables to 16-bit" ROUND_LOG)

add_custom_target(round
  DEPENDS run-round-exp run-round-sin run-round-log
  COMMENT "Regenerating all 16-bit-rounded table reports")

# CTest wires the same binaries up as a pass/fail suite. The tools always exit
# 0, so the pass condition comes from the report text: zero discrepancies.
# These currently FAIL (exp 7, sin 158, log 33) — those are open research
# findings in the 16-bit-rounded tables, not build breakage.
add_test(NAME verify-exp COMMAND verify_mpfr WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME verify-sin COMMAND verify_sin  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME verify-log COMMAND verify_log  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set_tests_properties(verify-exp verify-sin verify-log PROPERTIES
  FAIL_REGULAR_EXPRESSION "Discrepancies: *[1-9]")
