/* The macOS SDK has no C23 <stdbit.h>. math_config.h uses only
   stdc_trailing_zeros, and only in its ROUNDEVEN*_INTRINSICS=0 fallbacks.  */
#define stdc_trailing_zeros(x) \
  ((unsigned int) __builtin_ctzg ((x), (int) sizeof (x) * 8))
