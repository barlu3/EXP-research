# CORE-MATH research: the bf16/float16 kernels, their MPFR verification, the
# limb-table generators, the ln LP/MILP workflow and the throughput benchmarks.
#
# CORE_ROOT is the directory these tools treat as their root. The C sources name
# each other and their outputs by CORE_ROOT-relative path
# (#include "implementations/...", "log-research/lp-constraints.txt"), so every
# target that runs from or includes "the root" uses CORE_ROOT, not
# CMAKE_SOURCE_DIR.
set(CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(implementations)
add_subdirectory(benchmarks)
add_subdirectory(cross-eval)
add_subdirectory(log-research)
add_subdirectory(table-gen)
