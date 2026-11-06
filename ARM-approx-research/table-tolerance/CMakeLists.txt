# How much error glibc 2.44's log tables tolerate, entry by entry, and whether
# FRECPE's error fits inside: table-tolerance.cpp, built once per function as
# table-tolerance-logf, -log2f, -log10f and -log1pf. Each writes
# table_tolerance_<fn>f.txt beside itself. See docs/FRECPE-RANGE-REDUCTION.md,
# "How inaccurate can the entries be?".
#
# glibc's code is reused, not retyped, wherever its tables allow: logf and
# log2f are the vendored e_logf.c and e_log2f.c compiled a second time here,
# with __logf_data renamed to the writable tol_logf_data (and the log2f
# equivalent) and the data files renamed glibc_logf_data, which keeps glibc's
# values. log10f and log1pf keep their tables inside the function, so
# replicas.c carries copies. Every replica is checked bit for bit against
# gnu-libm, over every float, before anything else runs. All of it is built
# with gnu-libm's flags so it rounds as gnu-libm does.
#
# Source file properties are scoped to this directory, so gnu-libm's own build
# of the same files is untouched.
set(GLIBC_SRC "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks/gnu-libm/glibc")
add_library(tol-glibc OBJECT
  "${GLIBC_SRC}/e_logf.c"  "${GLIBC_SRC}/e_logf_data.c"
  "${GLIBC_SRC}/e_log2f.c" "${GLIBC_SRC}/e_log2f_data.c"
  "${GLIBC_SRC}/math_errf.c"
  replicas.c)
target_include_directories(tol-glibc PRIVATE "${GLIBC_SRC}/../shim" "${GLIBC_SRC}")
target_compile_options(tol-glibc PRIVATE -O2 -frounding-math -ffp-contract=fast)
set_source_files_properties("${GLIBC_SRC}/e_logf.c" PROPERTIES
  COMPILE_DEFINITIONS "__logf=tol_logf;__logf_data=tol_logf_data")
set_source_files_properties("${GLIBC_SRC}/e_logf_data.c" PROPERTIES
  COMPILE_DEFINITIONS "__logf_data=glibc_logf_data")
set_source_files_properties("${GLIBC_SRC}/e_log2f.c" PROPERTIES
  COMPILE_DEFINITIONS "__log2f=tol_log2f;__log2f_data=tol_log2f_data")
set_source_files_properties("${GLIBC_SRC}/e_log2f_data.c" PROPERTIES
  COMPILE_DEFINITIONS "__log2f_data=glibc_log2f_data")

# CTest: per function, the run itself (label table-tolerance), which fails only
# when a self-check does, and a finding (label table-tolerance-finding) that
# reads its report and fails when FRECPE does not fit. Both labels stay out of
# the per-push CI run; the weekly-macos job runs the first, gating, and the
# second, reported only. The finding requires the run as a fixture: run it
# alone with -FS 'tol-.*' to read an existing report instead of rerunning.
foreach(fn IN ITEMS logf log2f log10f log1pf)
  add_executable(table-tolerance-${fn} table-tolerance.cpp)
  target_link_libraries(table-tolerance-${fn} PRIVATE tol-glibc gnu-libm approx-common MPFR::MPFR)
  target_include_directories(table-tolerance-${fn} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks/scalar/logf"            # logf-ref.hpp
    "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks/scalar/asinhf-acoshf")  # ahyp-ref.hpp
  target_compile_options(table-tolerance-${fn} PRIVATE -O3)
  set_target_properties(table-tolerance-${fn} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_test(NAME table-tolerance-${fn} COMMAND table-tolerance-${fn})
  set_tests_properties(table-tolerance-${fn} PROPERTIES
    TIMEOUT 3600 LABELS table-tolerance FIXTURES_SETUP tol-${fn})
  add_test(NAME table-tolerance-fits-${fn}
    COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_CURRENT_SOURCE_DIR}/table_tolerance_${fn}.txt")
  set_tests_properties(table-tolerance-fits-${fn} PROPERTIES
    FIXTURES_REQUIRED tol-${fn} FAIL_REGULAR_EXPRESSION "FRECPE fits: NO"
    LABELS table-tolerance-finding)
endforeach()
target_compile_definitions(table-tolerance-log2f  PRIVATE LOGF_BASE=2)
target_compile_definitions(table-tolerance-log10f PRIVATE LOGF_BASE=10)
target_compile_definitions(table-tolerance-log1pf PRIVATE TOL_LOG1P)

# `make table-tolerance`: all four, one after another.
add_custom_target(table-tolerance
  COMMAND table-tolerance-logf
  COMMAND table-tolerance-log2f
  COMMAND table-tolerance-log10f
  COMMAND table-tolerance-log1pf
  DEPENDS table-tolerance-logf table-tolerance-log2f table-tolerance-log10f table-tolerance-log1pf
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Measuring glibc's log table tolerance against FRECPE's error"
  USES_TERMINAL)
