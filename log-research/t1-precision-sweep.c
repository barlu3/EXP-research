/* Find the minimum T1 significand width that admits a correctly-rounded ln(),
   holding T2 at a fixed width (default 10 bits).

   Question. The companion t2-precision-sweep showed that widening T2 alone
   never helps: with T2 at unlimited precision, 95 of 128 T2 indices still have
   an empty feasible interval, because CAND_ULP confines T1 to +/-3 bf16 ULPs of
   its ideal. T1's grid, not T2's, carries the conflict. This sweep tests that
   directly -- pin T2 and widen T1 instead.

   Method. For each width p = MIN_BITS..MAX_BITS this drives the existing
   pipeline, one process per stage, exactly as a human would run it:

     milp-gen T2 FILE p  -> a MILP whose T1 candidates sit on a p-bit grid
     split-milp.py       -> fragments (each keeps all of T2, a subset of T1)
     glpsol --lp         -> per-fragment status
     check-solution.py   -> exact-arithmetic revalidation

   Accepting a width. A width is accepted only if EVERY fragment reports
   INTEGER OPTIMAL *and* the composed assignment survives check-solution.py.
   That second gate is not optional: the coupling bounds sit ~1e-9 apart, which
   is tighter than glpsol's default primal/integer tolerances, so glpsol prints
   OPTIMAL on assignments that violate the source rows by up to ~1e-3. A sweep
   that trusted the bare status would report a minimum that is simply too small.

   Why a ceiling is still a real answer. A finer T1 grid shrinks the ULP that
   CAND_ULP counts, so the +/-3 ULP window narrows in absolute terms as p grows.
   Widening T1 therefore tightens its window even as it adds candidates, and
   there is no guarantee any width succeeds. "none <= MAX_BITS" is a legitimate
   result and this program reports it rather than hunting past the ceiling.

   The search is linear from MIN_BITS upward, not bisection: feasibility is not
   known to be monotone in p. A finer T1 grid moves every candidate, so a width
   that fails tells you nothing rigorous about a narrower one.

   Usage:
     t1-precision-sweep [--min-bits N] [--max-bits N] [--t2-bits N]
                        [--blocks N] [--tmlim SEC] [--keep] [--all-widths]
   Stops at the first accepted width unless --all-widths is given.

   Run from the repo root (the pipeline stages hardcode log-research/ paths):
     cmake --build build --target t1-precision-sweep
     ./log-research/t1-precision-sweep

   Build (compile as C; matches the rest of log-research):
     gcc -O2 -std=c11 log-research/t1-precision-sweep.c -o log-research/t1-precision-sweep
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

#define BF16_PREC 8
#define PREC_MAX 53             /* milp-gen prints candidates through double */
#define T2_BITS_DEFAULT 10      /* T2 held fixed while T1 is swept */
#define MAX_FRAGMENTS 512
#define PATH_MAX_LEN 256
/* Derived paths append a fixed suffix to a PATH_MAX_LEN base, so give them room. */
#define PATH_DERIVED_LEN (PATH_MAX_LEN + 64)
#define CMD_MAX_LEN 2048

/* Named so the bound survives being passed around -- a bare array parameter
   decays to a pointer and the compiler can no longer size-check the snprintfs. */
typedef char frag_name_t[PATH_MAX_LEN];

/* Paths are relative to the repo root, matching milp-gen and build-lp.sh. */
#define DIR_ROOT "log-research"
#define DIR_SWEEP DIR_ROOT "/sweep-t1"
#define SPLIT_PY DIR_ROOT "/split-milp.py"
#define CHECK_PY DIR_ROOT "/check-solution.py"

/* Per-width verdict. INFEASIBLE is a proof (fragment rows are a subset of the
   full model, so infeasibility lifts); OPTIMAL alone is not, which is why
   VIOLATED exists as a distinct outcome from ACCEPTED. */
typedef enum {
  WIDTH_ACCEPTED,     /* all fragments optimal AND exact re-check clean */
  WIDTH_INFEASIBLE,   /* some fragment infeasible -> full model infeasible */
  WIDTH_VIOLATED,     /* all fragments "optimal" but exact re-check found rows violated */
  WIDTH_TIMEOUT,      /* a fragment hit the solver time limit; verdict unknown */
  WIDTH_ERROR         /* a pipeline stage failed to run */
} width_result_t;

static const char *result_name (width_result_t r) {
  switch (r) {
    case WIDTH_ACCEPTED:   return "ACCEPTED";
    case WIDTH_INFEASIBLE: return "INFEASIBLE";
    case WIDTH_VIOLATED:   return "VIOLATED";
    case WIDTH_TIMEOUT:    return "TIMEOUT";
    default:               return "ERROR";
  }
}

static int opt_min_bits = BF16_PREC;
static int opt_max_bits = 24;
static int opt_t2_bits = T2_BITS_DEFAULT;
static int opt_blocks = 16;
static int opt_tmlim = 300;
static int opt_keep = 0;
static int opt_stop_at_first = 1;

/* Run a shell command, sending its output to `log` (appended) when given.
   Returns the exit status, or -1 if the command could not be run at all. */
static int run (const char *cmd, const char *log) {
  char full[CMD_MAX_LEN];
  int n = log ? snprintf (full, sizeof full, "%s >>'%s' 2>&1", cmd, log)
              : snprintf (full, sizeof full, "%s", cmd);
  if (n < 0 || (size_t) n >= sizeof full) {
    fprintf (stderr, "sweep: command too long\n");
    return -1;
  }
  int rc = system (full);
  if (rc == -1) return -1;
  return WIFEXITED (rc) ? WEXITSTATUS (rc) : -1;
}

/* True if `needle` appears anywhere in the file at `path`. Reads in chunks with
   an overlap so a match straddling a chunk boundary is still found. */
static int file_contains (const char *path, const char *needle) {
  FILE *f = fopen (path, "r");
  if (!f) return 0;

  size_t nlen = strlen (needle);
  size_t keep = nlen > 1 ? nlen - 1 : 0;
  char buf[8192];
  size_t have = 0;
  int found = 0;

  while (!found) {
    size_t got = fread (buf + have, 1, sizeof buf - have - 1, f);
    if (got == 0) break;
    have += got;
    buf[have] = '\0';
    if (strstr (buf, needle)) { found = 1; break; }
    if (have > keep) {
      memmove (buf, buf + have - keep, keep);
      have = keep;
    }
  }
  fclose (f);
  return found;
}

/* Collect "<dir>/<name>.lp" basenames into `out`. Returns the count, or -1. */
static int list_fragments (const char *dir, frag_name_t *out, int max) {
  DIR *d = opendir (dir);
  if (!d) return -1;

  int n = 0;
  struct dirent *e;
  while ((e = readdir (d))) {
    const char *ext = strrchr (e->d_name, '.');
    if (!ext || strcmp (ext, ".lp") != 0) continue;
    if (n >= max) { closedir (d); return -1; }
    snprintf (out[n++], PATH_MAX_LEN, "%s", e->d_name);
  }
  closedir (d);

  /* Deterministic order so the printed table is stable run to run. */
  for (int i = 1; i < n; i++) {
    char tmp[PATH_MAX_LEN];
    snprintf (tmp, sizeof tmp, "%s", out[i]);
    int j = i - 1;
    while (j >= 0 && strcmp (out[j], tmp) > 0) {
      snprintf (out[j + 1], PATH_MAX_LEN, "%s", out[j]);
      j--;
    }
    snprintf (out[j + 1], PATH_MAX_LEN, "%s", tmp);
  }
  return n;
}

static void rm_tree (const char *dir) {
  char cmd[CMD_MAX_LEN];
  snprintf (cmd, sizeof cmd, "rm -rf '%s'", dir);
  run (cmd, NULL);
}

/* Generate, split, solve and validate the model at T1 width `bits`, with T2
   pinned to opt_t2_bits.
   `optimal_out`/`total_out` report the fragment tally for the status table. */
static width_result_t try_width (int bits, int *optimal_out, int *total_out) {
  char wdir[PATH_MAX_LEN], cmd[CMD_MAX_LEN];
  char lp[PATH_DERIVED_LEN], fdir[PATH_DERIVED_LEN];
  char sdir[PATH_DERIVED_LEN + 64], log[PATH_DERIVED_LEN + 64];

  snprintf (wdir, sizeof wdir, "%s/p%02d", DIR_SWEEP, bits);
  snprintf (lp,   sizeof lp,   "%s/milp-constraints.lp", wdir);
  snprintf (fdir, sizeof fdir, "%s/fragments", wdir);
  snprintf (sdir, sizeof sdir, "%s/solutions", fdir);
  snprintf (log,  sizeof log,  "%s/sweep.log", wdir);

  *optimal_out = *total_out = 0;

  /* A stale directory from an earlier run would mix fragments across widths. */
  rm_tree (wdir);
  snprintf (cmd, sizeof cmd, "mkdir -p '%s' '%s'", fdir, sdir);
  if (run (cmd, NULL) != 0) {
    fprintf (stderr, "sweep: cannot create %s\n", wdir);
    return WIDTH_ERROR;
  }

  snprintf (cmd, sizeof cmd, "%s/milp-gen %d '%s' %d",
            DIR_ROOT, opt_t2_bits, lp, bits);
  if (run (cmd, log) != 0) {
    fprintf (stderr, "sweep: milp-gen failed at %d bits (see %s)\n", bits, log);
    return WIDTH_ERROR;
  }

  snprintf (cmd, sizeof cmd,
            "python3 '%s' --input '%s' --outdir '%s' --blocks %d",
            SPLIT_PY, lp, fdir, opt_blocks);
  if (run (cmd, log) != 0) {
    fprintf (stderr, "sweep: split-milp.py failed at %d bits (see %s)\n", bits, log);
    return WIDTH_ERROR;
  }

  static frag_name_t frags[MAX_FRAGMENTS];
  int nfrag = list_fragments (fdir, frags, MAX_FRAGMENTS);
  if (nfrag <= 0) {
    fprintf (stderr, "sweep: no fragments produced at %d bits\n", bits);
    return WIDTH_ERROR;
  }
  *total_out = nfrag;

  /* Solve every fragment. One INFEASIBLE is already a proof for the full model,
     but keep going so the log records how widespread the conflict is. */
  int infeasible = 0, timeout = 0, unknown = 0;
  for (int i = 0; i < nfrag; i++) {
    char base[PATH_MAX_LEN], flog[PATH_DERIVED_LEN + 128];
    memcpy (base, frags[i], sizeof base);   /* same width; strips the .lp below */
    base[sizeof base - 1] = '\0';
    char *dot = strrchr (base, '.');
    if (dot) *dot = '\0';
    if (snprintf (flog, sizeof flog, "%s/%s.log", sdir, base) >= (int) sizeof flog) {
      fprintf (stderr, "sweep: log path too long for %s\n", base);
      return WIDTH_ERROR;
    }

    char frag_path[PATH_DERIVED_LEN + PATH_MAX_LEN];
    int plen = snprintf (frag_path, sizeof frag_path, "%s/%.*s",
                         fdir, (int) sizeof frags[i], frags[i]);
    if (plen < 0 || (size_t) plen >= sizeof frag_path) {
      fprintf (stderr, "sweep: fragment path too long: %s\n", frags[i]);
      return WIDTH_ERROR;
    }
    snprintf (cmd, sizeof cmd,
              "glpsol --lp '%s' --tmlim %d --output '%s/%s.sol'",
              frag_path, opt_tmlim, sdir, base);
    run (cmd, flog);

    if (file_contains (flog, "INTEGER OPTIMAL SOLUTION FOUND")) (*optimal_out)++;
    else if (file_contains (flog, "HAS NO INTEGER FEASIBLE SOLUTION")) infeasible++;
    else if (file_contains (flog, "HAS NO PRIMAL FEASIBLE SOLUTION")) infeasible++;
    else if (file_contains (flog, "HAS NO FEASIBLE SOLUTION")) infeasible++;
    else if (file_contains (flog, "TIME LIMIT EXCEEDED")) timeout++;
    else unknown++;
  }

  if (infeasible > 0) return WIDTH_INFEASIBLE;
  if (timeout > 0 || unknown > 0) return WIDTH_TIMEOUT;

  /* Every fragment claims optimal. That is exactly the case the exact re-check
     exists for -- fragments choose T2 independently and glpsol's tolerances are
     looser than the ~1e-9 bound gaps, so believe this only after revalidation. */
  snprintf (cmd, sizeof cmd,
            "python3 '%s' --lp '%s' %s/*.sol", CHECK_PY, lp, sdir);
  int rc = run (cmd, log);
  if (rc < 0) {
    fprintf (stderr, "sweep: check-solution.py failed to run at %d bits\n", bits);
    return WIDTH_ERROR;
  }
  return rc == 0 ? WIDTH_ACCEPTED : WIDTH_VIOLATED;
}

static void usage (const char *argv0) {
  fprintf (stderr,
    "usage: %s [options]   (run from the repo root)\n"
    "  --min-bits N     first T1 width to try (default %d, bf16)\n"
    "  --max-bits N     last T1 width to try (default %d, ceiling %d)\n"
    "  --t2-bits N      T2 width, held fixed (default %d)\n"
    "  --blocks N       T1 fragments per width (default %d)\n"
    "  --tmlim SEC      glpsol time limit per fragment (default %d)\n"
    "  --keep           keep per-width LP/fragment files under %s\n"
    "  --all-widths     keep sweeping after the first accepted width\n",
    argv0, BF16_PREC, 24, PREC_MAX, T2_BITS_DEFAULT, 16, 300, DIR_SWEEP);
}

int main (int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    int has_val = i + 1 < argc;
    if (!strcmp (a, "--min-bits") && has_val) opt_min_bits = atoi (argv[++i]);
    else if (!strcmp (a, "--max-bits") && has_val) opt_max_bits = atoi (argv[++i]);
    else if (!strcmp (a, "--t2-bits") && has_val) opt_t2_bits = atoi (argv[++i]);
    else if (!strcmp (a, "--blocks") && has_val) opt_blocks = atoi (argv[++i]);
    else if (!strcmp (a, "--tmlim") && has_val) opt_tmlim = atoi (argv[++i]);
    else if (!strcmp (a, "--keep")) opt_keep = 1;
    else if (!strcmp (a, "--all-widths")) opt_stop_at_first = 0;
    else { usage (argv[0]); return 2; }
  }

  if (opt_min_bits < BF16_PREC || opt_max_bits > PREC_MAX
      || opt_min_bits > opt_max_bits || opt_tmlim < 1) {
    fprintf (stderr, "sweep: bad options (need %d <= min <= max <= %d)\n",
             BF16_PREC, PREC_MAX);
    return 2;
  }
  if (opt_t2_bits < BF16_PREC || opt_t2_bits > PREC_MAX) {
    fprintf (stderr, "sweep: --t2-bits must be %d..%d\n", BF16_PREC, PREC_MAX);
    return 2;
  }
  /* split-milp.py emits one fragment per block plus the T2-only sanity one. */
  if (opt_blocks < 1 || opt_blocks >= MAX_FRAGMENTS) {
    fprintf (stderr, "sweep: --blocks must be 1..%d\n", MAX_FRAGMENTS - 1);
    return 2;
  }

  /* Fail early and clearly rather than after minutes of solving. */
  struct stat st;
  if (stat (DIR_ROOT "/milp-gen", &st) != 0) {
    fprintf (stderr, "sweep: %s/milp-gen not found -- build it first "
             "(cmake --build build --target milp-gen), and run from the repo root\n",
             DIR_ROOT);
    return 1;
  }

  char cmd[CMD_MAX_LEN];
  snprintf (cmd, sizeof cmd, "mkdir -p '%s'", DIR_SWEEP);
  run (cmd, NULL);

  printf ("T1 precision sweep: widths %d..%d, %d T1 fragments, tmlim %ds\n",
          opt_min_bits, opt_max_bits, opt_blocks, opt_tmlim);
  printf ("T2 pinned at %d bits; correctness bounds stay bf16 rounding intervals.\n\n",
          opt_t2_bits);
  printf ("%-6s %-12s %-14s %s\n", "BITS", "RESULT", "FRAGMENTS", "MEANING");
  fflush (stdout);

  int found = -1;
  for (int bits = opt_min_bits; bits <= opt_max_bits; bits++) {
    int optimal = 0, total = 0;
    width_result_t r = try_width (bits, &optimal, &total);

    const char *meaning;
    switch (r) {
      case WIDTH_ACCEPTED:
        meaning = "correctly-rounded tables exist at this width"; break;
      case WIDTH_INFEASIBLE:
        meaning = "proof: no table pair at this width"; break;
      case WIDTH_VIOLATED:
        meaning = "glpsol optimal but exact re-check found violations"; break;
      case WIDTH_TIMEOUT:
        meaning = "solver did not decide; raise --tmlim"; break;
      default:
        meaning = "pipeline stage failed"; break;
    }

    char tally[32];
    snprintf (tally, sizeof tally, "%d/%d optimal", optimal, total);
    printf ("%-6d %-12s %-14s %s\n", bits, result_name (r), tally, meaning);
    fflush (stdout);

    if (!opt_keep) {
      char wdir[PATH_MAX_LEN];
      snprintf (wdir, sizeof wdir, "%s/p%02d", DIR_SWEEP, bits);
      if (r != WIDTH_ACCEPTED) rm_tree (wdir);
    }

    if (r == WIDTH_ERROR) return 1;
    if (r == WIDTH_ACCEPTED) {
      found = bits;
      if (opt_stop_at_first) break;
    }
  }

  printf ("\n");
  if (found > 0) {
    printf ("minimum T1 precision: %d bits (%+d over bf16), T2 pinned at %d\n",
            found, found - BF16_PREC, opt_t2_bits);
    return 0;
  }

  /* No width worked. Say so as a result, not as a failure -- with CAND_ULP
     fixed, T1's own candidate window can be the binding constraint, and then
     no T2 width can succeed however far the sweep runs. */
  printf ("no T1 width in %d..%d admits a correctly-rounded table (T2 at %d).\n",
          opt_min_bits, opt_max_bits, opt_t2_bits);
  printf ("Widening T1 also shrinks its ULP, so the +/-CAND_ULP window narrows\n"
          "in absolute terms as fast as candidates are added. Raising CAND_ULP\n"
          "(milp-gen.cc) widens that window independently and is the lever that\n"
          "remains untested.\n");
  return 1;
}
