# Throughput benchmarks. These measure performance, not correctness — the
# homemade exp is header-inlined, so -march=native materially changes results.

include(CheckCXXCompilerFlag)

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

set(BENCH_OUT "${CMAKE_CURRENT_SOURCE_DIR}/output")

# The float64 exp benchmarks are header-only.
foreach(variant home inria)
  add_executable(bench-${variant} benchmark-${variant}.cpp)
  target_compile_options(bench-${variant} PRIVATE ${BENCH_FLAGS})
  target_include_directories(bench-${variant} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
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
  "${CMAKE_CURRENT_SOURCE_DIR}/log"
  log/inria-logbf16.c log/inria-logbf16-limb.c)

add_limb_bench(bench-exp-limb benchmark-exp-limb.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}"
  inria-expbf16.c inria-expbf16-limb.c)

add_limb_bench(bench-sin-limb benchmark-sin-limb.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/sin"
  sin/inria-sinbf16.c sin/inria-sinbf16-limb.c)

# The limb benchmarks end with an exhaustive agreement check against the
# float32 tables and exit nonzero if it fails, so they double as regression
# tests. They are fast enough to run in CTest; the timing loops dominate, so
# each is given room.
add_test(NAME bench-log-limb COMMAND bench-log-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME bench-exp-limb COMMAND bench-exp-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME bench-sin-limb COMMAND bench-sin-limb WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(bench-log-limb bench-exp-limb bench-sin-limb
  PROPERTIES TIMEOUT 900)

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
