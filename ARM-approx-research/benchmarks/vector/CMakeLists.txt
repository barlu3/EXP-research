# 4-lane (.4s) versions of the scalar benchmarks: the same variants, clusters,
# reference and report, with every kernel on float32x4 vectors.
#   vrecpe/    bench-recip-vec   1/x       (reference from ../scalar/frecpe/)
#   vrsqrts/   bench-rsqrt-vec   1/sqrt(x) (reference from ../scalar/frsqrte/)
#
# What both share lives here and is built once as approx-vec-common: the
# vector loop-shape macros (vec-kernels.inc) and the vector control loops
# (vec-common.S). ../CMakeLists.txt enables ASM and defines approx-common.
add_library(approx-vec-common OBJECT vec-common.S)
target_include_directories(approx-vec-common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}")           # vec-kernels.inc

add_subdirectory(vrecpe)
add_subdirectory(vrsqrts)
