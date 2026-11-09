/* glibc 2.44's logf, log2f, log10f and log1pf with writable tables, and the
   same algorithms with FRECPE as their reciprocal: what table-tolerance.cpp
   perturbs and measures. logf and log2f are the vendored e_logf.c and e_log2f.c
   rebuilt by CMakeLists.txt with their table renamed; log10f and log1pf, whose
   tables live inside the function, are copied in replicas.c. */

#ifndef TABLE_TOLERANCE_REPLICAS_H
#define TABLE_TOLERANCE_REPLICAS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Added to -log(reciprocal) in glibc's log10f and log1pf log entries. log10f's
   is its tl[0] (tr[0] = 1); log1pf's keeps its fast result above the truth. */
#define TOL_LOG10F_OFF (-0x1.2p-46)
#define TOL_LOG1PF_OFF 0x1.7654p-37

/* One function's table. Entry i's reciprocal is rec[i * stride] and its log
   entry lg[i * stride]; base_rec and base_lg hold glibc's values. */
struct tol_table {
    double*       rec;
    double*       lg;
    const double* base_rec;
    const double* base_lg;
    int           n, stride;
};
struct tol_table tol_table_logf(void);
struct tol_table tol_table_log2f(void);
struct tol_table tol_table_log10f(void);
struct tol_table tol_table_log1pf(void);
void tol_reset(void);   /* every table back to glibc's values */

/* glibc's algorithm on the writable table: bit-identical to gnu_<fn>f while
   the table holds glibc's values. */
float tol_logf(float x);
float tol_log2f(float x);
float tol_log10f(float x);
float tol_log1pf(float x);

/* The entry glibc reads for the input with these bits, or -1 if it takes a
   path that reads none. */
int tol_entry_logf(uint32_t ux);
int tol_entry_log2f(uint32_t ux);
int tol_entry_log10f(uint32_t ux);
int tol_entry_log1pf(uint32_t ux);

/* glibc's algorithm with FRECPE as its reciprocal: r = m * FRECPE(m) - 1,
   exact, and log entry fr_<fn>f_lg[i] (filled by the caller) for FRECPE's
   entry i, in place of glibc's. With fr_near1 set, inputs whose significand is
   in [1, 1 + 2^-8) use c = 1 instead, as glibc's own tables do there (logf,
   log2f, log10f; log1pf's table path never gets that close to 1). */
extern double fr_logf_lg[256], fr_log2f_lg[256], fr_log10f_lg[256], fr_log1pf_lg[256];
extern int fr_near1;
float frecpe_logf(float x);
float frecpe_log2f(float x);
float frecpe_log10f(float x);
float frecpe_log1pf(float x);

/* With fr_cell = i >= 0, FRECPE's entry i reads fr_cell_rec in place of the
   estimate (and of fr_near1's c = 1, for i = 0), still with log entry
   fr_<fn>f_lg[i]: one entry drifted at a time. -1, the default, drifts none. */
extern int fr_cell;
extern double fr_cell_rec;

/* The FRECPE entry frecpe_<fn>f reads for the input with these bits, or -1 if
   it takes a path that reads none. */
int fr_entry_logf(uint32_t ux);
int fr_entry_log2f(uint32_t ux);
int fr_entry_log10f(uint32_t ux);
int fr_entry_log1pf(uint32_t ux);

/* Calls that took log10f's or log1pf's slow path so far, on this thread. A
   function rather than the thread-local itself: C++ reads an extern
   thread_local through a wrapper that C does not emit. */
unsigned long long tol_slow_count(void);

#ifdef __cplusplus
}
#endif

#endif  /* TABLE_TOLERANCE_REPLICAS_H */
