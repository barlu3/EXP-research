# glibc 2.44's float32 log, log2, log10 and log1p, built from glibc's own source:
# the baseline the log benchmarks divide by, so their ratios do not depend on
# which libSystem this Mac ships. glibc/ is sysdeps/ieee754/flt-32/ of the
# glibc-2.44 tarball (sha256 37f600f2bef3c5e8300147059568b2a2e40a7ad6ccc65ce942556d49429cc667),
# copied verbatim with COPYING.LIB; refresh it from
#   dependencies/fetch-glibc.sh 2.44 && tar xf dependencies/glibc-2.44.tar.xz -C dependencies
# shim/ stands in for the glibc-internal headers those files include.
#
# Each function is renamed gnu_* the way glibc's own multiarch builds rename
# theirs (defining __logf also skips e_logf.c's alias block), so nothing here
# exports logf and the like: the kernels' `bl _logf` still binds to libSystem.
#
# SHARED, so a call goes through a dyld stub: the same calling cost as
# libSystem's logf, and as glibc's libm.so through the PLT. -O2 -frounding-math
# is glibc's libm build (Makeconfig +math-flags) and -ffp-contract=fast is
# GCC's default for glibc's gnu11, which clang does not share; spelled out
# because the build tree may be Debug.
add_library(gnu-libm SHARED
  glibc/e_logf.c   glibc/e_logf_data.c
  glibc/e_log2f.c  glibc/e_log2f_data.c
  glibc/e_log10f.c glibc/s_log1pf.c
  glibc/math_errf.c)
target_include_directories(gnu-libm PRIVATE shim)
target_compile_definitions(gnu-libm PRIVATE
  __logf=gnu_logf __log2f=gnu_log2f __log10f=gnu_log10f __log1pf=gnu_log1pf)
target_compile_options(gnu-libm PRIVATE -O2 -frounding-math -ffp-contract=fast)
