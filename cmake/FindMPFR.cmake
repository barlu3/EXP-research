# Locates MPFR + GMP and exposes them as the MPFR::MPFR interface target.
# Distro packages ship no CMake config, so this is a plain path search.

find_path(MPFR_INCLUDE_DIR mpfr.h
  HINTS ENV MPFR_ROOT
  PATH_SUFFIXES include
  PATHS /usr/include/x86_64-linux-gnu /usr/local/include /opt/local/include /opt/homebrew/include)

find_library(MPFR_LIBRARY NAMES mpfr
  HINTS ENV MPFR_ROOT
  PATH_SUFFIXES lib
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/local/lib /opt/homebrew/lib)

find_library(GMP_LIBRARY NAMES gmp
  HINTS ENV GMP_ROOT
  PATH_SUFFIXES lib
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/local/lib /opt/homebrew/lib)

if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY OR NOT GMP_LIBRARY)
  message(FATAL_ERROR
    "MPFR/GMP not found. Install with:\n"
    "  Debian/Ubuntu : sudo apt install libmpfr-dev libgmp-dev\n"
    "  macOS (brew)  : brew install mpfr gmp\n"
    "  macOS (port)  : sudo port install mpfr gmp\n"
    "Or point MPFR_ROOT / GMP_ROOT at a custom prefix.")
endif()

if(NOT TARGET MPFR::MPFR)
  add_library(MPFR::MPFR INTERFACE IMPORTED)
  set_target_properties(MPFR::MPFR PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${MPFR_LIBRARY};${GMP_LIBRARY};m")
endif()

message(STATUS "Found MPFR: ${MPFR_LIBRARY}")
