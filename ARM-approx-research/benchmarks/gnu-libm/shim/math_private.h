/* Stand-in for glibc's internal math_private.h, plus the parts of
   libc-symbols.h and sys/cdefs.h that every glibc file sees implicitly: only
   what the vendored flt-32 log family in ../glibc/ uses. Symbol aliasing and
   versioning are no-ops; gnu-libm's CMake target renames each function
   instead (-D__logf=gnu_logf, ...).  */
#ifndef GNU_LIBM_MATH_PRIVATE_H
#define GNU_LIBM_MATH_PRIVATE_H

#include <stdbool.h>   /* as glibc's own math_private.h: math_config.h uses bool */

#define __glibc_likely(cond)   __builtin_expect ((cond), 1)
#define __glibc_unlikely(cond) __builtin_expect ((cond), 0)
#define attribute_hidden       __attribute__ ((visibility ("hidden")))
#define NOINLINE               __attribute__ ((noinline))

#define strong_alias(name, aliasname)
#define versioned_symbol(lib, local, symbol, version)

/* math_config.h's roundeven_finite calls C23 roundeven, which the macOS SDK's
   math.h does not declare yet.  */
#define roundeven(x)  __builtin_roundeven (x)
#define roundevenf(x) __builtin_roundevenf (x)

#endif
