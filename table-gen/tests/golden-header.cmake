# CTest driver: the committed limb table must be exactly what the generator
# produces today.
#
# This is the guardrail for every change to the tuning search. The search may be
# restructured, gated, instrumented or sped up freely -- but the bytes it emits
# are the shipped artefact, and cr_log_bf16_limb's correctness rests on the
# specific 14 tuned entries in that file. A refactor that changes the emitted
# table has changed the library, whatever it did to the code.
#
# Regenerating is cheap (~0.1 s, dominated by 32512 MPFR references), so this
# runs as an ordinary test rather than a manual target.
#
#   cmake -DGEN=<limb-gen> -DREF=<committed header> -DTMP=<scratch> -P this

foreach(var GEN REF TMP)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "golden-header.cmake: -D${var}=... is required")
  endif()
endforeach()

execute_process(
  COMMAND "${GEN}" "${TMP}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE  err)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "generator exited ${rc}; it refuses to write a table it has not scored clean.\n"
    "stdout:\n${out}\nstderr:\n${err}")
endif()

# compare_files is byte-exact, which is the point: a one-ULP drift in any entry
# is a different library, and a reordered emission is a different diff to audit.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${TMP}" "${REF}"
  RESULT_VARIABLE differ)

if(NOT differ EQUAL 0)
  message(FATAL_ERROR
    "regenerated table differs from the committed one.\n"
    "  committed : ${REF}\n"
    "  regenerated: ${TMP}\n"
    "Either the generator changed behaviour (fix it, or re-run `make limb-tables`\n"
    "and review the diff entry by entry), or the header was hand-edited.\n"
    "generator said:\n${out}")
endif()

message(STATUS "golden: regenerated table is byte-identical to the committed one")
