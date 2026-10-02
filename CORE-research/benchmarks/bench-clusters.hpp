/* Input clusters for the three bf16 limb benchmarks.

   These live in a header rather than inside each benchmark so harness-test can
   assert over them: every cluster has to reach MIN_DISTINCT_INPUTS distinct
   bf16 values, which is what stops a range like the old [79.5, 80.5] -- three
   distinct inputs, drawn 500,000 times -- from being added again.

   Ranges are chosen to fill binades while staying strictly inside one code
   path, since widening a cluster across a path boundary would change what it
   measures rather than just how much of it it sees. The relevant boundaries,
   read off the CORE-MATH kernels:

     exp  |x| <= 0x1p-9   (0.001953)  rounds to 1, no table
          |x| >= 0x1.74p+6 (93.0)     overflow/underflow constant, no table
          table path is the open interval between them
     sin  |x| <= 0x1.dp-4 (0.113281)  rounds to x, no table -- the control path
          |x| <  4096                 mid path, S1/C1 + S2/C2
          |x| >= 4096                 large path, up to 8 sin/cos pairs
     log  x < 2^-126 (1.17549e-38)    subnormal, T3 direct lookup

   Each table also carries a sweep cluster covering every codepoint that
   actually reaches the table path. That is the headline: it is the only
   cluster whose access pattern puts the whole table in play rather than a
   handful of entries.

   The sweeps deliberately stop at the table path rather than spanning every
   finite bf16. A sweep over all 65536 codepoints measured log's limb tables at
   0.69x -- apparently a large win -- but the breakdown was:

       all finite      n=65280   inria 3.881  limb 2.763   0.71x
       negative (NaN)  n=32639   inria 1.136  limb 0.683   0.60x
       positive        n=32641   inria 1.382  limb 1.385   1.00x

   Half of that domain is x < 0, where log returns NaN without consulting a
   table at all, and limb's early exit happens to be laid out faster. Shuffling
   two paths that differ that much together also mispredicts constantly, which
   is why the mixed row is 3.9 ns against ~1.2 ns for either half alone. The
   result was a headline number that measured error handling and branch
   misprediction rather than the table scheme. Sweeping only the table path
   gives 1.00x, which is the honest answer. */

#ifndef BENCH_CLUSTERS_HPP
#define BENCH_CLUSTERS_HPP

#include "bench-harness.hpp"

#include <limits>

namespace bench {

struct Cluster {
    const char* label;
    double      lo;
    double      hi;
    // True for the clusters that sweep a whole table path. These are the rows
    // to quote: the narrow ones keep one or two cache lines hot and measure a
    // corner of the table rather than the table.
    bool        sweep = false;
};

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// ── log ──────────────────────────────────────────────────────────────────────
// Only the first two needed widening; [1e-10, 3e-10] and the subnormal range
// already cleared the floor at 202 and 109 distinct values.
inline constexpr Cluster LOG_CLUSTERS[] = {
    { "x near 1",      0.5,     2.0     },  // T1_126/127, the hard binade
    { "x in [64,128]", 64.0,    128.0   },  // mid-range normals (was 3 values)
    { "x near 2e-10",  1e-10,   3e-10   },  // small normals
    { "subnormal",     1e-40,   1e-38   },  // T3 direct lookup
    // Every x > 0: log's whole defined domain, normals and subnormals both.
    // 1e-45 is below the smallest bf16 subnormal (9.18e-41), so this excludes
    // +0 and everything negative -- those return -Inf and NaN without a lookup.
    { "domain sweep",  1e-45,   kInf,   true },
};

// ── exp ──────────────────────────────────────────────────────────────────────
// The two large clusters stop at |x| = 92 so they stay below the 0x1.74p+6
// overflow/underflow cutoff; crossing it would swap the table path for a
// constant return and make the cluster measure an early exit.
inline constexpr Cluster EXP_CLUSTERS[] = {
    { "x just >2^-9", 0.00196,  0.0078  },  // just past the |x|<=2^-9 exit
    { "x near 1",     0.5,      2.0     },  // mid-table
    { "x in [32,92]", 32.0,     92.0    },  // below the overflow edge
    { "x in [-92,-32]", -92.0, -32.0    },  // the negative half of T1/T2
    { "x near 2e-10", 1e-10,    3e-10   },  // below the table: rounds to 1
    // The table path, swept in full. Two clusters because it is the open
    // interval 2^-9 < |x| < 93 on each side of zero, and a single [lo,hi]
    // would drag in the ~15000 codepoints per sign below 2^-9 that return 1
    // without a lookup.
    { "sweep +table", 0.00196,  92.9,   true },
    { "sweep -table", -92.9,   -0.00196, true },
};

// ── sin ──────────────────────────────────────────────────────────────────────
// The first cluster stays under 0x1.dp-4 so it remains the no-table control:
// every variant runs identical code there, which makes its ratio a direct read
// of the harness noise floor.
inline constexpr Cluster SIN_CLUSTERS[] = {
    { "no-table ctl", 0.03125,  0.113   },  // small |x|: no table in any variant
    { "x near 1",     0.5,      2.0     },  // mid path
    { "x in [64,128]", 64.0,   128.0    },  // mid path, larger index
    { "x near 1e5",   65536.0,  131072.0 }, // large path, short chain
    { "x near 1e30",  6.3e29,   1.27e30 },  // large path, deep chain
    // The two table paths, each swept in full. sin folds the sign away, so the
    // positive half is representative. Split rather than merged because the mid
    // and large paths cost very different amounts and averaging them would hide
    // which one the limb tables actually affect.
    { "sweep mid",    0.1134,   4095.0, true },
    { "sweep large",  4096.0,   kInf,   true },
};

// ── float64 / float32 ────────────────────────────────────────────────────────
// The float64 benchmarks keep plain sampling. Their inputs do not collapse the
// way bf16 does -- 500k draws from [0.9, 1.1] land on ~500k distinct doubles --
// so the starvation that forced codepoint enumeration for bf16 does not arise,
// and there is no table whose footprint a wider range would exercise.
inline constexpr Cluster REAL_CLUSTERS[] = {
    { "x near 1",     0.9,    1.1   },
    { "x near 80",   79.5,   80.5   },
    { "x near 2e-10", 1e-10,  3e-10 },
};

}  // namespace bench

#endif  // BENCH_CLUSTERS_HPP
