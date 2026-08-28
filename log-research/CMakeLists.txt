# T1/T2 linear-programming workflow for ln(). Three sequential steps:
#   1. precompute  -> ln-precompute.txt   (ln(x) for every bf16 input)
#   2. bound-calc  -> ln-bounds.txt, lp-constraints.txt
#   3. milp-gen    -> milp-constraints.lp (bf16-grid MILP, solve with highs/glpsol)
# All three write outputs relative to log-research/.

# NOTE: milp-gen.cc is C, not C++ — it is compiled with -xc to match
# inria-log16bitp.c. Letting CMake infer C++ from the extension breaks the build.
set_source_files_properties(milp-gen.cc PROPERTIES LANGUAGE C)
set_source_files_properties(milp-limb-gen.cc PROPERTIES LANGUAGE C)

foreach(tool precompute bound-calc limb-gen)
  add_executable(${tool} ${tool}.c)
  target_compile_options(${tool} PRIVATE -O2)
  target_link_libraries(${tool} PRIVATE MPFR::MPFR)
  set_target_properties(${tool} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

foreach(gen milp-gen milp-limb-gen)
  add_executable(${gen} ${gen}.cc)
  target_compile_options(${gen} PRIVATE -O2)
  target_link_libraries(${gen} PRIVATE MPFR::MPFR)
  set_target_properties(${gen} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

# Drives milp-gen/split-milp.py/glpsol/check-solution.py as subprocesses, so it
# needs no MPFR of its own. Like the others it must run from the repo root.
foreach(sweep t2-precision-sweep t1-precision-sweep)
  add_executable(${sweep} ${sweep}.c)
  target_compile_options(${sweep} PRIVATE -O2)
  set_target_properties(${sweep} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

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

# Mirror image: widen T1 with T2 pinned (10 bits by default).
add_custom_target(ln-t1-sweep
  COMMAND t1-precision-sweep
  DEPENDS t1-precision-sweep milp-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Sweeping T1 significand width with T2 pinned"
  USES_TERMINAL)

# Regenerates implementations/log/logbf16-limb.h (3x bf16 limbs for T1, 2x for
# T2, 1x for T3). Writes into the source tree, so it is a manual target.
add_custom_target(ln-limb-tables
  COMMAND limb-gen
  DEPENDS limb-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating bf16 limb tables for cr_log_bf16_limb"
  USES_TERMINAL)

# Per-limb MILP: proves a bf16 limb table can hit every rounding interval.
# Solve the fragments, not the whole file -- glpsol hangs on the full model.
# Never trust a bare "Optimal" from it: the coupling bounds are ~1e-9 apart,
# tighter than glpsol's tolerances. Re-check with check-limb-solution.py.
add_custom_target(ln-limb-milp
  COMMAND milp-limb-gen
  DEPENDS milp-limb-gen
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Emitting per-limb MILP (split with split-milp.py, solve with glpsol)"
  USES_TERMINAL)
