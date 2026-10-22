# Cycle-estimated benchmarks of x86's estimate instructions against the exact
# instruction sequences they stand in for: the x86-64 counterpart of
# ARM-approx-research/benchmarks/, one directory per benchmark with the same
# layout.
#   scalar/rcpss/   bench-recip-x86   RCPSS against DIVSS
#   scalar/rsqrtss/ bench-rsqrt-x86   RSQRTSS against SQRTSS + DIVSS
#
# Only the kernels and the control/calibration loops are x86's own. The driver
# (approx-bench.hpp), cycle source (bench-cycles.hpp) and inputs/reference
# (recip-ref.hpp) are ARM-approx-research's; statistics and reporting are
# CORE-research/benchmarks/bench-harness.hpp. Kernels use ELF symbols and the
# SysV ABI, so this builds on x86-64 Linux only (WSL2 included).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  enable_language(ASM)

  add_library(x86-approx-common OBJECT approx-common.S)
  target_include_directories(x86-approx-common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"                              # approx-kernels.inc (x86)
    "${CMAKE_SOURCE_DIR}/ARM-approx-research/benchmarks"       # approx-bench.hpp, bench-cycles.hpp
    "${CMAKE_SOURCE_DIR}/CORE-research/benchmarks")            # bench-harness.hpp

  set(X86_APPROX_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

  add_subdirectory(scalar/rcpss)
  add_subdirectory(scalar/rsqrtss)
endif()
