/* No public aliases (logf, log1pf, logp1f, ...): gnu-libm exports only the
   gnu_* names its CMake target renames to, so it never competes with
   libSystem for a symbol.  */
#define libm_alias_float(from, to)
#define libm_alias_float_other(from, to)
