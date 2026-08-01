cmake_minimum_required(VERSION 3.20)
project(exp-research LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(cmake/FindMPFR.cmake)
include(cmake/Dependencies.cmake)

# Benchmarks need AVX2/FMA; the MPFR tools are plain -O2 and stay portable.
option(EXP_NATIVE_ARCH "Build benchmarks with -march=native -mavx2 -mfma" ON)

enable_testing()

add_subdirectory(implementations)
add_subdirectory(cross-eval)
add_subdirectory(log-research)

# MPFR presence check — replaces the README's manual `cc -o version version.c`.
add_executable(mpfr-version version.c)
target_link_libraries(mpfr-version PRIVATE MPFR::MPFR)

add_custom_target(check-mpfr
  COMMAND mpfr-version
  DEPENDS mpfr-version
  COMMENT "Reporting MPFR library/header versions")

# `verify` is the pass/fail gate: exhaustive 65536-pattern check, zero
# discrepancies required. Each tool writes its report relative to cross-eval/.
add_custom_target(verify
  DEPENDS verify-exp verify-sin verify-log
  COMMENT "Exhaustive CORE-MATH vs MPFR verification (exp, sin, log)")

add_custom_target(bench
  DEPENDS run-bench-home run-bench-inria
  COMMENT "Running throughput benchmarks")

message(STATUS "")
message(STATUS "exp-research configured (${CMAKE_BUILD_TYPE})")
message(STATUS "  native arch for benchmarks : ${EXP_NATIVE_ARCH}")
message(STATUS "  make verify      - exhaustive MPFR check (pass = 0 discrepancies)")
message(STATUS "  make bench       - build + run benchmarks")
message(STATUS "  make round       - regenerate 16-bit-rounded tables")
message(STATUS "  make glibc       - fetch + extract glibc reference source")
message(STATUS "  make check-mpfr  - print MPFR version")
message(STATUS "")
