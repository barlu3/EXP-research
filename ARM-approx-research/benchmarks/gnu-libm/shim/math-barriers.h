/* glibc's generic barriers: keep a value's computation, and the FP exception
   it raises, from being folded away or moved.  */
#define math_opt_barrier(x) \
  ({ __typeof (x) __x = (x); __asm__ ("" : "+m" (__x)); __x; })
#define math_force_eval(x) \
  ({ __typeof (x) __x = (x); __asm__ __volatile__ ("" : : "m" (__x)); })
