# glibc reference source: fetch the tarball, extract it, and confirm the
# baseline exp implementation is present. Replaces the README's manual
# `tar xf` + `find ... -name e_exp.c` steps.

set(GLIBC_VERSION 2.43)
set(GLIBC_DIR "${CMAKE_SOURCE_DIR}/dependencies/glibc-${GLIBC_VERSION}")
set(GLIBC_TARBALL "${CMAKE_SOURCE_DIR}/dependencies/glibc-${GLIBC_VERSION}.tar.xz")

add_custom_command(OUTPUT "${GLIBC_TARBALL}"
  COMMAND "${CMAKE_SOURCE_DIR}/dependencies/fethc-glibc.sh"
  COMMENT "Fetching glibc-${GLIBC_VERSION} source tarball"
  VERBATIM)

add_custom_command(OUTPUT "${GLIBC_DIR}/sysdeps/ieee754/dbl-64/e_exp.c"
  COMMAND "${CMAKE_COMMAND}" -E tar xf "${GLIBC_TARBALL}"
  DEPENDS "${GLIBC_TARBALL}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/dependencies"
  COMMENT "Extracting glibc-${GLIBC_VERSION}"
  VERBATIM)

add_custom_target(glibc
  DEPENDS "${GLIBC_DIR}/sysdeps/ieee754/dbl-64/e_exp.c"
  COMMENT "glibc reference source ready")
