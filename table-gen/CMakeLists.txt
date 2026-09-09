# bf16 limb-table generation for ln, exp and sin, plus the configuration sweep
# that justifies the limb counts each generator ships.
#
# One generator per function, laid out by function:
#   log/limb-gen.c          -> implementations/log/logbf16-limb.h
#   exp/exp-limb-gen.c      -> implementations/exp/expbf16-limb.h
#   sin/sin-limb-gen.c      -> implementations/sin/sinbf16-limb.h
#   sweep/limb-config-sweep -> the evidence behind those configurations
#
# Every one of them writes into the source tree and reads the shipped CORE-MATH
# tables by repo-root-relative path, so each target runs from ${CMAKE_SOURCE_DIR}.
#
# Only the ln generator needs MPFR, and the split is deliberate. limb-gen
# recomputes ln ideals at high precision; the exp and sin generators instead
# split the *shipped* CORE-MATH values, because those tables carry manual ULP
# adjustments and an overflow cap that recomputed ideals would not reproduce.
# Splitting what actually ships is what makes the generated limb tables match
# cr_exp_bf16 / cr_sin_bf16 exactly.

# add_limb_tool(<name> <source> [MPFR])
function(add_limb_tool name source)
  add_executable(${name} ${source})
  target_compile_options(${name} PRIVATE -O2)
  target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}")
  if("MPFR" IN_LIST ARGN)
    target_link_libraries(${name} PRIVATE MPFR::MPFR)
  endif()
  # isfinite/fabsf, plus the expf/sinf in the CORE-MATH sources the exp/sin/sweep
  # tools include. limb-gen needs none of them and did not link m before the
  # move; naming it here is inert and keeps one uniform helper.
  # macOS carries libm in libSystem; glibc needs it named.
  if(NOT APPLE)
    target_link_libraries(${name} PRIVATE m)
  endif()
  # Binaries land beside their source, matching the other tools in this repo.
  get_filename_component(srcdir "${source}" DIRECTORY)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${srcdir}")
endfunction()

add_limb_tool(limb-gen          log/limb-gen.c MPFR)
add_limb_tool(exp-limb-gen      exp/exp-limb-gen.c)
add_limb_tool(sin-limb-gen      sin/sin-limb-gen.c)
add_limb_tool(limb-config-sweep sweep/limb-config-sweep.c)

# Diagnostics behind docs/limb-tuning.tex. These are replicas of the generators'
# split/score/descent code with the reporting extended, so every figure the
# report quotes is reproducible rather than asserted. limb-lp is not a replica:
# it solves the same problem a different way (difference constraints), so it
# doubles as an independent check on the descent's answer.
add_limb_tool(limb-diagnostics analysis/limb-diagnostics.c MPFR)
add_limb_tool(limb-margins     analysis/limb-margins.c     MPFR)
add_limb_tool(limb-lp          analysis/limb-lp.c          MPFR)

# Regenerates implementations/log/logbf16-limb.h (2x bf16 limbs for T1, 2x for
# T2, 1x for T3, with 14 tuned entries -- see docs/LIMB-TUNING.md). Writes into
# the source tree, so it is a manual target.
add_custom_target(ln-limb-tables
  COMMAND limb-gen
  DEPENDS limb-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating bf16 limb tables for cr_log_bf16_limb"
  USES_TERMINAL)

# Regenerate implementations/exp/expbf16-limb.h and sin/sinbf16-limb.h. Each
# emits two configurations: an exact one (3 bf16 limbs carry float32's 24
# significand bits, so the reconstruction is bit-for-bit the shipped value) and
# a minimal one with per-entry ULP adjustments. Both write into the source tree,
# so like ln-limb-tables they are manual targets.
add_custom_target(exp-limb-tables
  COMMAND exp-limb-gen
  DEPENDS exp-limb-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating bf16 limb tables for cr_exp_bf16_limb"
  USES_TERMINAL)

add_custom_target(sin-limb-tables
  COMMAND sin-limb-gen
  DEPENDS sin-limb-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating bf16 limb tables for cr_sin_bf16_limb"
  USES_TERMINAL)

add_custom_target(limb-tables
  DEPENDS ln-limb-tables exp-limb-tables sin-limb-tables
  COMMENT "Regenerating every bf16 limb table")

# The evidence behind those configurations: every limb count measured against
# the shipped float32 implementations, plus the per-entry repair searches.
# Answers why exp is 2x2 and why sin's large path cannot drop to 2 limbs.
# Unit tests for the feasibility certificate: the difference-constraint solver
# and the ln system built on it. Fast (one MPFR model build), so it runs in
# CTest proper rather than as a manual target.
add_limb_tool(limb-cert-test tests/limb-cert-test.c MPFR)
add_test(NAME limb-cert-test COMMAND limb-cert-test
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# The emitted table is the shipped artefact, so it is pinned by a test rather
# than by convention: regenerate and byte-compare against what is committed.
# Guards every change to the tuning search, including the certificate work.
add_test(NAME limb-gen-golden
  COMMAND "${CMAKE_COMMAND}"
    -DGEN=$<TARGET_FILE:limb-gen>
    -DREF=${CMAKE_SOURCE_DIR}/implementations/log/logbf16-limb.h
    -DTMP=${CMAKE_CURRENT_BINARY_DIR}/logbf16-limb.golden.h
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden-header.cmake"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Everything docs/limb-tuning.tex quotes that the generators do not print.
add_custom_target(limb-analysis
  COMMAND limb-diagnostics
  COMMAND limb-margins
  COMMAND limb-lp
  DEPENDS limb-diagnostics limb-margins limb-lp
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Reproducing the limb-tuning report's measurements"
  USES_TERMINAL)

add_custom_target(limb-sweep
  COMMAND limb-config-sweep
  DEPENDS limb-config-sweep
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Sweeping bf16 limb configurations for exp and sin"
  USES_TERMINAL)
