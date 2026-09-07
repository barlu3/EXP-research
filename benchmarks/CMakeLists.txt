# Throughput benchmarks. These measure performance, not correctness — the
# homemade exp is header-inlined, so -march=native materially changes results.
#
# The sources live here; the kernels they measure live in implementations/.
# IMPL is the one place that coupling is spelled out.

include(CheckCXXCompilerFlag)

set(IMPL "${CMAKE_SOURCE_DIR}/implementations")
set(BENCH_OUT "${CMAKE_CURRENT_SOURCE_DIR}/output")

# -mavx2/-mfma are x86-only and hard-error on Apple Silicon, so each flag is
# probed rather than assumed. EXP_NATIVE_ARCH=OFF drops all of them and keeps
# the binaries portable.
set(BENCH_FLAGS -O3)
if(EXP_NATIVE_ARCH)
  foreach(flag -march=native -mavx2 -mfma)
    string(MAKE_C_IDENTIFIER "HAVE${flag}" flag_var)
    check_cxx_compiler_flag(${flag} ${flag_var})
    if(${flag_var})
      list(APPEND BENCH_FLAGS ${flag})
    endif()
  endforeach()
endif()

# The float64 exp benchmarks are header-only; they include exp.hpp and
# inria-exp-seg.hpp straight out of implementations/.
foreach(variant home inria)
  add_executable(bench-${variant} benchmark-${variant}.cpp)
  target_compile_options(bench-${variant} PRIVATE ${BENCH_FLAGS})
  target_include_directories(bench-${variant} PRIVATE "${IMPL}/exp")
  set_target_properties(bench-${variant} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BENCH_OUT}")

  add_custom_target(run-bench-${variant}
    COMMAND bench-${variant}
    DEPENDS bench-${variant}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running benchmark-${variant}"
    USES_TERMINAL)
endforeach()

# The bf16 limb benchmarks link the CORE-MATH implementations, which must be
# compiled as C: under a C++ compiler they would get C++ linkage and fail to
# match the benchmarks' extern "C" declarations.
#
# add_limb_bench(<name> <benchmark.cpp> <include dir> <impl sources...>)
function(add_limb_bench name source incdir)
  add_executable(${name} ${source} ${ARGN})
  target_compile_options(${name} PRIVATE ${BENCH_FLAGS})
  target_include_directories(${name} PRIVATE "${incdir}")
  # The CORE-MATH sources carry expf/sinf fallbacks. macOS has libm in
  # libSystem; glibc needs it named.
  if(NOT APPLE)
    target_link_libraries(${name} PRIVATE m)
  endif()
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BENCH_OUT}")

  add_custom_target(run-${name}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${BENCH_OUT}"
    COMMAND ${name}
    DEPENDS ${name}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running ${source}"
    USES_TERMINAL)
endfunction()

add_limb_bench(bench-log-limb benchmark-limb.cpp
  "${IMPL}/log"
  "${IMPL}/log/inria-logbf16.c" "${IMPL}/log/inria-logbf16-limb.c")

add_limb_bench(bench-exp-limb benchmark-exp-limb.cpp
  "${IMPL}/exp"
  "${IMPL}/exp/inria-expbf16.c" "${IMPL}/exp/inria-expbf16-limb.c")

add_limb_bench(bench-sin-limb benchmark-sin-limb.cpp
  "${IMPL}/sin"
  "${IMPL}/sin/inria-sinbf16.c" "${IMPL}/sin/inria-sinbf16-limb.c")

# Force the benchmark runs to happen one at a time.
#
# `make bench` builds the aggregate target with -j <ncpu>, and the five
# run-bench-* targets are independent, so Make was starting all five at once and
# every benchmark measured a machine running four other benchmarks. It showed up
# as interquartile ranges as large as the medians (4.4 ns IQR on a 16.0 ns
# median) and controls at roughly twice their uncontended cost.
#
# USES_TERMINAL would serialise these under Ninja via the console job pool, but
# this project configures with Unix Makefiles, where it does not. An explicit
# chain is the portable way to say "never run these concurrently"; each target
# is still individually invokable.
add_dependencies(run-bench-inria     run-bench-home)
add_dependencies(run-bench-log-limb  run-bench-inria)
add_dependencies(run-bench-exp-limb  run-bench-log-limb)
add_dependencies(run-bench-sin-limb  run-bench-exp-limb)

# Unit tests for the shared measurement core. Pure computation apart from one
# short self-timing check, so it runs in a second and belongs in CTest proper
# rather than alongside the timing targets.
add_executable(bench-harness-test harness-test.cpp)
target_compile_options(bench-harness-test PRIVATE ${BENCH_FLAGS})
set_target_properties(bench-harness-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${BENCH_OUT}")
add_test(NAME bench-harness-test COMMAND bench-harness-test
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# The limb benchmarks end with an exhaustive agreement check against the
# float32 tables and exit nonzero if it fails, so they double as regression
# tests. They are fast enough to run in CTest; the timing loops dominate, so
# each is given room.
add_test(NAME bench-log-limb COMMAND bench-log-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME bench-exp-limb COMMAND bench-exp-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME bench-sin-limb COMMAND bench-sin-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(bench-log-limb bench-exp-limb bench-sin-limb
  PROPERTIES TIMEOUT 900)
