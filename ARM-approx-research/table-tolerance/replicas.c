/* glibc 2.44's logf, log2f, log10f and log1pf with writable tables, and with
   FRECPE as their reciprocal (replicas.h). Built with gnu-libm's flags
   (-O2 -frounding-math -ffp-contract=fast) so the replicas round exactly as
   gnu-libm does; table-tolerance.cpp checks that bit for bit, over every float,
   before using them.

   log10f and log1pf below are CORE-MATH's, as vendored in
   ../benchmarks/gnu-libm/glibc/e_log10f.c and s_log1pf.c, copied with their
   tables lifted out of the function and made writable, and each body shared
   between glibc's table (frecpe = 0) and FRECPE's estimate (frecpe = 1).

   log10f: Copyright (c) 2022-2026 Alexei Sibidanov.
   log1pf: Copyright (c) 2023, 2024 Alexei Sibidanov.
   Both are part of the CORE-MATH project (src/binary32/log10/log10f.c revision
   ebff4c43, src/binary32/log1p/log1pf.c revision 24ef43a1), under this notice:

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "math_config.h"
#include "replicas.h"

static _Thread_local unsigned long long tol_slow_paths;
unsigned long long tol_slow_count (void) { return tol_slow_paths; }
int fr_near1;
int fr_cell = -1;
double fr_cell_rec;
double fr_logf_lg[256], fr_log2f_lg[256], fr_log10f_lg[256], fr_log1pf_lg[256];

/* ─── logf, log2f ───────────────────────────────────────────────────────── */

/* e_logf.c and e_log2f.c, rebuilt with __logf_data renamed tol_logf_data
   (CMakeLists.txt), read these. They declare them const, which holds while a
   call runs: the tables change only between calls. e_logf_data.c and
   e_log2f_data.c, rebuilt as glibc_logf_data and glibc_log2f_data, keep
   glibc's values. */
struct logf_data  tol_logf_data;
struct log2f_data tol_log2f_data;
extern const struct logf_data  glibc_logf_data;
extern const struct log2f_data glibc_log2f_data;

/* glibc's split, x = 2^k z with z in [OFF, 2 OFF), and its index: the top four
   bits of z's offset from OFF. Both functions use the same OFF and width. */
#define LOGF_OFF 0x3f330000
static int entry_logf_like (uint32_t ix)
{
  if (ix == 0x3f800000)
    return -1;                          /* log(1) returns 0 before the table */
  if (ix - 0x00800000 >= 0x7f800000 - 0x00800000)
    {
      if (ix * 2 == 0 || ix >= 0x7f800000)
	return -1;                      /* 0, x < 0, inf, nan */
      ix = asuint (asfloat (ix) * 0x1p23f) - (23 << 23);
    }
  return ((ix - LOGF_OFF) >> (23 - LOGF_TABLE_BITS)) % (1 << LOGF_TABLE_BITS);
}
int tol_entry_logf (uint32_t ux)  { return entry_logf_like (ux); }
int tol_entry_log2f (uint32_t ux) { return entry_logf_like (ux); }

/* FRECPE's entry: the top eight fraction bits, of a subnormal once scaled. */
static int fr_entry_logf_like (uint32_t ix)
{
  if (entry_logf_like (ix) < 0)
    return -1;
  if (ix < 0x00800000)
    ix = asuint (asfloat (ix) * 0x1p23f);
  return (ix >> 15) & 0xff;
}
int fr_entry_logf (uint32_t ux)  { return fr_entry_logf_like (ux); }
int fr_entry_log2f (uint32_t ux) { return fr_entry_logf_like (ux); }

struct tol_table tol_table_logf (void)
{
  return (struct tol_table) { &tol_logf_data.tab[0].invc, &tol_logf_data.tab[0].logc,
			      &glibc_logf_data.tab[0].invc, &glibc_logf_data.tab[0].logc,
			      1 << LOGF_TABLE_BITS, 2 };
}
struct tol_table tol_table_log2f (void)
{
  return (struct tol_table) { &tol_log2f_data.tab[0].invc, &tol_log2f_data.tab[0].logc,
			      &glibc_log2f_data.tab[0].invc, &glibc_log2f_data.tab[0].logc,
			      1 << LOG2F_TABLE_BITS, 2 };
}

/* x = 2^k m with m in [1, 2), so FRECPE(x) = 2^-k chat(i), i being m's top
   eight fraction bits, and r = x * FRECPE(x) - 1 = m * chat(i) - 1: 24 bits
   times 9, exact in double. With fr_near1, i = 0 takes c = 1 (r = m - 1) and
   reports i = -1. Entry fr_cell takes fr_cell_rec, before either. k0 corrects
   k for a subnormal scaled by 2^23. */
static double frecpe_reduce (float x, int k0, int *k, int *i)
{
  uint32_t ix = asuint (x);
  double m = asfloat ((ix & 0x007fffff) | 0x3f800000);
  *k = (int) (ix >> 23) - 127 + k0;
  *i = (ix >> 15) & 0xff;
  if (__glibc_unlikely (*i == fr_cell))
    return m * fr_cell_rec - 1;
  if (fr_near1 && *i == 0)
    {
      *i = -1;
      return m - 1;
    }
  return (double) x * (double) vrecpes_f32 (x) - 1;
}

/* glibc's logf and log2f with that reduction: their special cases, their
   polynomial and its evaluation order, a 256-entry log table. */
float frecpe_logf (float x)
{
  uint32_t ix = asuint (x);
  int k0 = 0, k, i;
  if (__glibc_unlikely (ix == 0x3f800000))
    return 0;
  if (__glibc_unlikely (ix - 0x00800000 >= 0x7f800000 - 0x00800000))
    {
      if (ix * 2 == 0)
	return __math_divzerof (1);
      if (ix == 0x7f800000)
	return x;
      if ((ix & 0x80000000) || ix * 2 >= 0xff000000)
	return __math_invalidf (x);
      x *= 0x1p23f;
      k0 = -23;
    }
  const double *A = glibc_logf_data.poly;
  double r = frecpe_reduce (x, k0, &k, &i);
  double y0 = (i < 0 ? 0 : fr_logf_lg[i]) + (double) k * glibc_logf_data.ln2;
  double r2 = r * r;
  double y = A[1] * r + A[2];
  y = A[0] * r2 + y;
  y = y * r2 + (y0 + r);
  return (float) y;
}

float frecpe_log2f (float x)
{
  uint32_t ix = asuint (x);
  int k0 = 0, k, i;
  if (__glibc_unlikely (ix == 0x3f800000))
    return 0;
  if (__glibc_unlikely (ix - 0x00800000 >= 0x7f800000 - 0x00800000))
    {
      if (ix * 2 == 0)
	return __math_divzerof (1);
      if (ix == 0x7f800000)
	return x;
      if ((ix & 0x80000000) || ix * 2 >= 0xff000000)
	return __math_invalidf (x);
      x *= 0x1p23f;
      k0 = -23;
    }
  const double *A = glibc_log2f_data.poly;
  double r = frecpe_reduce (x, k0, &k, &i);
  double y0 = (i < 0 ? 0 : fr_log2f_lg[i]) + (double) k;
  double r2 = r * r;
  double y = A[1] * r + A[2];
  y = A[0] * r2 + y;
  double p = A[3] * r + y0;
  y = y * r2 + p;
  return (float) y;
}

/* ─── log10f (CORE-MATH) ────────────────────────────────────────────────── */

// reciprocal of 1+i/64 (i=0..64) rounded to 29 bits
static const double base_log10f_tr[65] =
  {
    0x1p+0,         0x1.f81f82p-1,  0x1.f07c1fp-1,  0x1.e9131acp-1,
    0x1.e1e1e1ep-1, 0x1.dae6077p-1, 0x1.d41d41dp-1, 0x1.cd85689p-1,
    0x1.c71c71cp-1, 0x1.c0e0704p-1, 0x1.bacf915p-1, 0x1.b4e81b5p-1,
    0x1.af286bdp-1, 0x1.a98ef6p-1,  0x1.a41a41ap-1, 0x1.9ec8e95p-1,
    0x1.999999ap-1, 0x1.948b0fdp-1, 0x1.8f9c19p-1,  0x1.8acb90fp-1,
    0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f41p-1, 0x1.78a4c81p-1,
    0x1.745d174p-1, 0x1.702e05cp-1, 0x1.6c16c17p-1, 0x1.6816817p-1,
    0x1.642c859p-1, 0x1.605816p-1,  0x1.5c9882cp-1, 0x1.58ed231p-1,
    0x1.5555555p-1, 0x1.51d07ebp-1, 0x1.4e5e0a7p-1, 0x1.4afd6ap-1,
    0x1.47ae148p-1, 0x1.446f865p-1, 0x1.4141414p-1, 0x1.3e22cbdp-1,
    0x1.3b13b14p-1, 0x1.3813814p-1, 0x1.3521cfbp-1, 0x1.323e34ap-1,
    0x1.2f684bep-1, 0x1.2c9fb4ep-1, 0x1.29e412ap-1, 0x1.27350b9p-1,
    0x1.2492492p-1, 0x1.21fb781p-1, 0x1.1f7047ep-1, 0x1.1cf06aep-1,
    0x1.1a7b961p-1, 0x1.1811812p-1, 0x1.15b1e5fp-1, 0x1.135c811p-1,
    0x1.1111111p-1, 0x1.0ecf56cp-1, 0x1.0c9715p-1,  0x1.0a6810ap-1,
    0x1.0842108p-1, 0x1.0624dd3p-1, 0x1.041041p-1,  0x1.0204081p-1,
    0.5
  };
// logarithms of the reciprocals with offset
static const double base_log10f_tl[65] =
  {
    -0x1.2p-46,             0x1.b947689310dfap-8, 0x1.b5e909c96d11bp-7,
    0x1.45f4f59ed1e08p-6,   0x1.af5f92cbd8bc1p-6, 0x1.0ba01a606dcdep-5,
    0x1.3ed119b9a29cdp-5,   0x1.714834298ed14p-5, 0x1.a30a9d983564cp-5,
    0x1.d41d512670665p-5,   0x1.02428c0f65442p-4, 0x1.1a23444eecb67p-4,
    0x1.31b30543f4bddp-4,   0x1.48f3ed39bfc2dp-4, 0x1.5fe8049a0e34bp-4,
    0x1.769140a6a9f3p-4,    0x1.8cf1836c98bdbp-4, 0x1.a30a9d55540cap-4,
    0x1.b8de4d1ee8167p-4,   0x1.ce6e4202ca20fp-4, 0x1.e3bc1accacd3p-4,
    0x1.f8c9683b5aafdp-4,   0x1.06cbd68ca9a03p-3, 0x1.11142f19df6c5p-3,
    0x1.1b3e71fa7a913p-3,   0x1.254b4d37a4677p-3, 0x1.2f3b6912cbe9cp-3,
    0x1.390f68311581ap-3,   0x1.42c7e7fffc53dp-3, 0x1.4c65808c78cd1p-3,
    0x1.55e8c50751beap-3,   0x1.5f52445dec36cp-3, 0x1.68a288c3f1195p-3,
    0x1.71da17bdf0cadp-3,   0x1.7af973608af6ep-3, 0x1.84011952a250ep-3,
    0x1.8cf1837a7e9f4p-3,   0x1.95cb2891e436ap-3, 0x1.9e8e7b0f86974p-3,
    0x1.a73beaa5db121p-3,   0x1.afd3e39455867p-3, 0x1.b856cf060d985p-3,
    0x1.c0c5134de1f9p-3,    0x1.c91f1371bc934p-3, 0x1.d1652ffcd3ee8p-3,
    0x1.d997c6f635e09p-3,   0x1.e1b733ab90edp-3,  0x1.e9c3ceadac7ebp-3,
    0x1.f1bdeec43a29ap-3,   0x1.f9a5e7a5fa392p-3, 0x1.00be05ac02ef5p-2,
    0x1.04a054d81a29ep-2,   0x1.087a0835957c5p-2, 0x1.0c4b4570994e1p-2,
    0x1.101431aa1fe1bp-2,   0x1.13d4f08b98da2p-2, 0x1.178da53edb85cp-2,
    0x1.1b3e71e9f9d22p-2,   0x1.1ee777defdeb7p-2, 0x1.2288d7b48e205p-2,
    0x1.2622b0f52e469p-2,   0x1.29b522a4c62dep-2, 0x1.2d404b0e30f4ap-2,
    0x1.30c4478f3fbafp-2,   0x1.34413509f78dfp-2
  };
double tol_log10f_tr[65], tol_log10f_tl[65];

// 10^n
static const union
{
  float f;
  uint32_t u;
} log10f_st[] =
{
  { 0x1.2a05f2p+33 }, { 0x1.4p+3 },     { 0x1.9p+6 },      { 0 },
  { 0x1.f4p+9 },      { 0 },            { 0x1.388p+13 },   { 0x1.86ap+16 },
  { 0 },              { 0x1.e848p+19 }, { 0 },             { 0x1.312dp+23 },
  { 0x1.7d784p+26 },  { 0 },            { 0x1.dcd65p+29 }, { 0x1p+0 }
};
static const double log10f_b[] =
  {
    0x1.bcb7b15d35067p-2, -0x1.bcbb1cd29cbafp-3, 0x1.2870e2624ce4ep-3
  };
static const double log10f_c[] =
  {
    0x1.bcb7b1526e50ep-2,  -0x1.bcb7b1526e53dp-3, 0x1.287a7636f3fa2p-3,
    -0x1.bcb7b146a14b3p-4, 0x1.63c627d5219cbp-4,  -0x1.2880736c8762dp-4,
    0x1.fc1ecf913961ap-5
  };

static __attribute__ ((noinline)) float
log10f_special (float x)
{
  uint32_t ux = asuint (x), ax = ux<<1;
  if (ax == 0u) // x = +/-0.0
    /* -0.0 */
    return __math_divzerof (1);
  if (ux == 0x7f800000u)
    return x; // x=+inf
  if (ax > 0xff000000u)
    return x + x; // x=nan
  return __math_invalidf (x);
}

/* CORE-MATH's __log10f. The one change besides the table: the slow path's
   tl[j] - tl[0] removes the offset, tl[0] being -log10(1) + offset; it is held
   at that constant here, so moving entry 0 moves entry 0 alone. */
static inline __attribute__ ((always_inline)) float
log10f_core (float x, int frecpe)
{
  const double *b = log10f_b, *c = log10f_c;
  // ln(2)/ln(10)
  const double ln10 = 0x1.34413509f79ffp-2,
	       ln10h = 0x1.34413509f7ap-2,
	       ln10l = -0x1.0cee0ed4ca7e9p-54;
  uint32_t ux = asuint (x);
  if (__glibc_unlikely (ux >= 0x7f800000u))
    return log10f_special (x); // <=-0, nan, inf
  if (__glibc_unlikely (ux == log10f_st[(ux >> 24) & 0xf].u))
    { // x = 10^n
      unsigned je = ((int) ux >> 23) - 126;
      je = (je * 0x4d104d4) >> 28;
      return je;
    }
  if (__glibc_unlikely (ux < 0x00800000u))
    {
      if (__glibc_unlikely (ux == 0u))
	return log10f_special (x); // x=+0
      // subnormal
      int n = __builtin_clz (ux) - 8;
      ux <<= n;
      ux -= n << 23;
    }
  int e = ((int) ux >> 23) - 127;
  int64_t m = ux & ((1 << 23) - 1), j = (m + (1 << (23 - 7))) >> (23 - 6);
  double tz = asdouble ((m << (52 - 23)) | (UINT64_C(0x3ff) << 52));
  double trj, tlj;
  if (!frecpe)
    {
      trj = tol_log10f_tr[j];
      tlj = tol_log10f_tl[j];
    }
  else if (__glibc_unlikely (m >> 15 == fr_cell))
    {
      trj = fr_cell_rec;
      tlj = fr_log10f_lg[fr_cell];
    }
  else if (fr_near1 && m >> 15 == 0)
    {
      trj = 1;
      tlj = TOL_LOG10F_OFF;
    }
  else
    {
      trj = vrecpes_f32 ((float) tz);
      tlj = fr_log10f_lg[m >> 15];
    }
  double z = tz * trj - 1, z2 = z * z;
  double r = ((e * ln10 + tlj) + z * b[0]) + z2 * (b[1] + z * b[2]);
  float ub = r, lb = r + 0x1.af23fp-34;
  if (__glibc_unlikely (ub != lb))
    {
      ++tol_slow_paths;
      double f = z
		 * ((c[0] + z * c[1])
		    + z2
			  * ((c[2] + z * c[3])
			     + z2 * (c[4] + z * c[5] + z2 * c[6])));
      f += ln10l * e;
      f += tlj - TOL_LOG10F_OFF;
      double el = e * ln10h;
      r = el + f;
      ub = r;
      tz = r;
      if (__glibc_unlikely (!(asuint64 (tz) & ((1 << 28) - 1))))
	{
	  double dr = (el - r) + f;
	  r += dr * 32;
	  ub = r;
	}
    }
  return ub;
}
float tol_log10f (float x)    { return log10f_core (x, 0); }
float frecpe_log10f (float x) { return log10f_core (x, 1); }

/* The fraction bits log10f_core indexes by, or -1 for a special case. */
static int64_t log10f_frac (uint32_t ux)
{
  if (ux >= 0x7f800000u || ux == 0u || ux == log10f_st[(ux >> 24) & 0xf].u)
    return -1;
  if (ux < 0x00800000u)
    {
      int n = __builtin_clz (ux) - 8;
      ux <<= n;
      ux -= n << 23;
    }
  return ux & ((1 << 23) - 1);
}
int tol_entry_log10f (uint32_t ux)
{
  int64_t m = log10f_frac (ux);
  return m < 0 ? -1 : (m + (1 << (23 - 7))) >> (23 - 6);
}
int fr_entry_log10f (uint32_t ux)
{
  int64_t m = log10f_frac (ux);
  return m < 0 ? -1 : m >> 15;
}

struct tol_table tol_table_log10f (void)
{
  return (struct tol_table) { tol_log10f_tr, tol_log10f_tl,
			      base_log10f_tr, base_log10f_tl, 65, 1 };
}

/* ─── log1pf (CORE-MATH) ────────────────────────────────────────────────── */

/* the reciprocal 1/(1+(j+0.5)/32) is rounded to 23 bits */
static const double base_log1pf_x0[32] =
  { 0x1.f81f80p-1, 0x1.e9131cp-1, 0x1.dae608p-1, 0x1.cd8568p-1,
    0x1.c0e070p-1, 0x1.b4e81cp-1, 0x1.a98ef8p-1, 0x1.9ec8e8p-1,
    0x1.948b10p-1, 0x1.8acb90p-1, 0x1.818180p-1, 0x1.78a4c8p-1,
    0x1.702e04p-1, 0x1.681680p-1, 0x1.605818p-1, 0x1.58ed24p-1,
    0x1.51d080p-1, 0x1.4afd6cp-1, 0x1.446f88p-1, 0x1.3e22ccp-1,
    0x1.381380p-1, 0x1.323e34p-1, 0x1.2c9fb4p-1, 0x1.27350cp-1,
    0x1.21fb78p-1, 0x1.1cf06cp-1, 0x1.181180p-1, 0x1.135c80p-1,
    0x1.0ecf58p-1, 0x1.0a6810p-1, 0x1.0624dcp-1, 0x1.020408p-1
  };
/* the logarithm of the reciprocal is offset by 0x1.7654p-37 so
   log1p_fast(x) - log1p(x) > 0 */
static const double base_log1pf_lix[32] = {
    0x1.fc0b0b1599ce4p-7, 0x1.77457a64a42abp-5, 0x1.341d74627847dp-4,
    0x1.a926d8a568810p-4, 0x1.0d77e8cd667aap-3, 0x1.44d2b38d15679p-3,
    0x1.7ab886a16b2b3p-3, 0x1.af3c9b686996dp-3, 0x1.e27075e30cc37p-3,
    0x1.0a3250a767d98p-2, 0x1.229423bd2662ep-2, 0x1.3a64c596c3292p-2,
    0x1.51aadd530e505p-2, 0x1.686c85e9e0177p-2, 0x1.7eaf7df859caep-2,
    0x1.94793ee2403b3p-2, 0x1.a9cec5a9cf512p-2, 0x1.beb4d3baa086fp-2,
    0x1.d32fe2a03d8b5p-2, 0x1.e744257d97431p-2, 0x1.faf58cf7bdfe7p-2,
    0x1.0723e6d1e5599p-1, 0x1.109f3b52ec2f3p-1, 0x1.19ee6a7693fc5p-1,
    0x1.23130d9c03597p-1, 0x1.2c0e9cc4604f1p-1, 0x1.34e28bd9e5837p-1,
    0x1.3d9028a72cd5fp-1, 0x1.4618b9c1dd52dp-1, 0x1.4e7d825b8d20bp-1,
    0x1.56bf9fab56a02p-1, 0x1.5ee02ab258ccap-1
};
double tol_log1pf_x0[32], tol_log1pf_lix[32];

static const double log1pf_b[] =
  {
    0x1p+0,
    -0x1p-1,
    0x1.5555555556f6bp-2,
    -0x1.00000000029b9p-2,
    0x1.9999988d176e4p-3,
    -0x1.55555418889a7p-3,
    0x1.24adeca50e2bcp-3,
    -0x1.001ba33bf57cfp-3
  };
static const double log1pf_c[] =
  {
    0x1.ffffffe1eac82p-1, -0x1.ffffff7da1724p-2, 0x1.5564d8fa59d0cp-2,
    -0x1.001219d3dba2ap-2
  };

static __attribute__ ((noinline)) float
log1pf_special (float x)
{
  uint32_t t = asuint (x);
  if (t == 0xbf800000u)
    return __math_divzerof (1);
  if (t == 0x7f800000u)
    return x; /* +inf */
  uint32_t ax = t << 1;
  if (ax > 0xff000000u)
    return x + x; /* nan */
  return __math_invalidf (0.0f);
}

/* CORE-MATH's __log1pf. With frecpe, the table path indexes w = 1 + x by its
   top eight fraction bits and takes double FRECPE of its significand. */
static inline __attribute__ ((always_inline)) float
log1pf_core (float x, int frecpe)
{
  const double *b = log1pf_b, *c = log1pf_c;
  double z = x;
  uint32_t ux = asuint (x);
  if (__glibc_unlikely (ux >= 0xbf800000u))
    return log1pf_special (x); // x<=-1, x=-inf, x=-nan
  uint32_t ax = ux & (~0u >> 1);
  if (__glibc_unlikely (ax >= 0x7f800000u))
    return log1pf_special (x); // x=+inf, x=+nan
  if (__glibc_likely (ax < 0x3c880000u))
    { // |x| < 0x1.1p-6
      if (__glibc_unlikely (ax < 0x33000000u))
	{ // |x| < 0x1p-25
	  if (!ax)
	    return x;
	  return fmaf (x, -x, x);
	}
      double z2 = z * z, z4 = z2 * z2;
      double f = z2
		 * ((b[1] + z * b[2]) + z2 * (b[3] + z * b[4])
		    + z4 * (b[5] + z * (b[6] + z * b[7])));
      double r = z + f;
      if (__glibc_unlikely (asuint64 (r) & 0xfffffffll) == 0)
	r += 0x1p14 * (f + (z - r));
      return r;
    }
  else
    {
      uint64_t tp = asuint64 (z + 1.0);
      int e = (tp >> 52) - 0x3ff;
      uint64_t m52 = tp & (~0ull >> 12);
      unsigned j = (tp >> (52 - 5)) & 31;
      double xd = asdouble (m52 | UINT64_C (0x3ff) << 52);
      double x0j, lixj;
      if (!frecpe)
	{
	  x0j = tol_log1pf_x0[j];
	  lixj = tol_log1pf_lix[j];
	}
      else
	{
	  int i = (tp >> (52 - 8)) & 255;
	  x0j = __glibc_unlikely (i == fr_cell) ? fr_cell_rec : vrecped_f64 (xd);
	  lixj = fr_log1pf_lg[i];
	}
      z = xd * x0j - 1; // z is exact for x<0x1.0cp+30
      const double ln2 = 0x1.62e42fefa39efp-1;
      double z2 = z * z,
	     r = (ln2 * e + lixj)
		 + z * ((c[0] + z * c[1]) + z2 * (c[2] + z * c[3]));
      const double eps = 2.1555e-11;
      float ub = r, lb = r - eps;
      if (__glibc_unlikely (ub != lb))
	{
	  ++tol_slow_paths;
	  double z4 = z2 * z2,
		 f = z2
		     * ((b[1] + z * b[2]) + z2 * (b[3] + z * b[4])
			+ z4 * (b[5] + z * (b[6] + z * b[7])));
	  double lj = lixj - TOL_LOG1PF_OFF; // subtract the offset
	  const double ln2l = 0x1.7f7d1cf79abcap-20, ln2h = 0x1.62e4p-1;
	  double Lh = ln2h * e, Ll = ln2l * e;
	  Ll += z;
	  double rh = Lh + lj, rl = ((Lh - rh) + lj) + (Ll + f);
	  float fh = rh + rl;
	  double Fl = (rh - (double) fh) + rl;
	  float fl = Fl, tfl = fl * 2.0f;
	  if ((fh + tfl) - fh == tfl)
	    fl += copysignf (0.5f, (float) (Fl - (double) fl)) * fabsf (fl);
	  ub = fh + fl;
	}
      return ub;
    }
}
float tol_log1pf (float x)    { return log1pf_core (x, 0); }
float frecpe_log1pf (float x) { return log1pf_core (x, 1); }

int tol_entry_log1pf (uint32_t ux)
{
  uint32_t ax = ux & (~0u >> 1);
  if (ux >= 0xbf800000u || ax >= 0x7f800000u || ax < 0x3c880000u)
    return -1;
  return (asuint64 ((double) asfloat (ux) + 1.0) >> (52 - 5)) & 31;
}
int fr_entry_log1pf (uint32_t ux)
{
  if (tol_entry_log1pf (ux) < 0)
    return -1;
  return (asuint64 ((double) asfloat (ux) + 1.0) >> (52 - 8)) & 255;
}

struct tol_table tol_table_log1pf (void)
{
  return (struct tol_table) { tol_log1pf_x0, tol_log1pf_lix,
			      base_log1pf_x0, base_log1pf_lix, 32, 1 };
}

/* ─── All four ──────────────────────────────────────────────────────────── */

__attribute__ ((constructor)) void tol_reset (void)
{
  memcpy (&tol_logf_data, &glibc_logf_data, sizeof tol_logf_data);
  memcpy (&tol_log2f_data, &glibc_log2f_data, sizeof tol_log2f_data);
  memcpy (tol_log10f_tr, base_log10f_tr, sizeof tol_log10f_tr);
  memcpy (tol_log10f_tl, base_log10f_tl, sizeof tol_log10f_tl);
  memcpy (tol_log1pf_x0, base_log1pf_x0, sizeof tol_log1pf_x0);
  memcpy (tol_log1pf_lix, base_log1pf_lix, sizeof tol_log1pf_lix);
}
