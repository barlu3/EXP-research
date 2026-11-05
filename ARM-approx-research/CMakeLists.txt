# Research into ARM's approximate-arithmetic instructions (FRECPE and friends):
# what they cost against the exact instructions they stand in for.
add_subdirectory(benchmarks)
# How much error glibc's log tables tolerate, and whether FRECPE's fits inside.
# It reuses gnu-libm and the benchmarks' driver, so it exists only where they do.
if(TARGET gnu-libm)
  add_subdirectory(table-tolerance)
endif()
