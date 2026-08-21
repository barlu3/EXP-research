# T1/T2 linear-programming workflow for ln(). Three sequential steps:
#   1. precompute  -> ln-precompute.txt   (ln(x) for every bf16 input)
#   2. bound-calc  -> ln-bounds.txt, lp-constraints.txt
#   3. milp-gen    -> milp-constraints.lp (bf16-grid MILP, solve with highs/glpsol)
# All three write outputs relative to log-research/.

# NOTE: milp-gen.cc is C, not C++ — it is compiled with -xc to match
# inria-log16bitp.c. Letting CMake infer C++ from the extension breaks the build.
set_source_files_properties(milp-gen.cc PROPERTIES LANGUAGE C)

foreach(tool precompute bound-calc)
  add_executable(${tool} ${tool}.c)
  target_compile_options(${tool} PRIVATE -O2)
  target_link_libraries(${tool} PRIVATE MPFR::MPFR)
  set_target_properties(${tool} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

add_executable(milp-gen milp-gen.cc)
target_compile_options(milp-gen PRIVATE -O2)
target_link_libraries(milp-gen PRIVATE MPFR::MPFR)
set_target_properties(milp-gen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Drives milp-gen/split-milp.py/glpsol/check-solution.py as subprocesses, so it
# needs no MPFR of its own. Like the others it must run from the repo root.
add_executable(t2-precision-sweep t2-precision-sweep.c)
target_compile_options(t2-precision-sweep PRIVATE -O2)
set_target_properties(t2-precision-sweep PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# precompute writes its table to stdout; the others manage their own files.
add_custom_target(ln-precompute
  COMMAND sh -c "$<TARGET_FILE:precompute> > log-research/ln-precompute.txt"
  DEPENDS precompute
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Step 1: precomputing ln(x) for all bf16 inputs"
  VERBATIM USES_TERMINAL)

# NOTE: bound-calc and milp-gen hardcode "log-research/"-prefixed output paths
# (LP_FILE / OUT_FILE), so they must run from the repo root, not from here.
add_custom_target(ln-bounds
  COMMAND sh -c "$<TARGET_FILE:bound-calc> > log-research/ln-bounds.txt"
  DEPENDS bound-calc
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Step 2: computing ln bounds + LP constraints"
  VERBATIM USES_TERMINAL)

add_custom_target(ln-milp
  COMMAND milp-gen
  DEPENDS milp-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Step 3: emitting bf16-grid MILP (solve with: glpsol --lp log-research/milp-constraints.lp)"
  USES_TERMINAL)

# Sweeps T2 width and reports the minimum that solves, or that none does.
# Needs glpsol on PATH; runs the whole pipeline once per width, so it is slow.
add_custom_target(ln-t2-sweep
  COMMAND t2-precision-sweep
  DEPENDS t2-precision-sweep milp-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Sweeping T2 significand width for the minimum feasible precision"
  USES_TERMINAL)
