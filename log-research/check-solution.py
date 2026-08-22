#!/usr/bin/env python3
"""Re-check a composed fragment solution against the original LP constraints.

glpsol's default tolerances are looser than this model's coupling bounds, which
sit ~1e-9 apart. It therefore reports OPTIMAL on fragments whose combined
assignment violates the source rows. This script recomputes every coupling row
in exact decimal arithmetic and reports violations, so an all-OPTIMAL fragment
sweep is never mistaken for a certified solution. The single-variable T3 rows
(`s<u>_lo/hi: T3_k >=/<= bound`, the subnormal lookups) are checked the same way.

Values come from the zT*_k_j binaries plus the LP's literal link coefficients,
never from the .sol Activity column -- that column is printed at ~6 significant
digits, an error ~65x the tolerance being tested.

Usage:
    python3 check-solution.py fragments/solutions/fragpin-t1-*.sol
"""

import argparse
import re
import sys
from decimal import Decimal
from pathlib import Path

SOL_BIN_RE = re.compile(r"^\s*\d+\s+z(T[123]_\d+)_(\d+)\s+\*?\s+(\d+)")
COUPLE_RE = re.compile(r"^\s*(c\d+_(?:lo|hi)): (T1_\d+) \+ (T2_\d+) (>=|<=) (-?[\d.]+)\s*$")
SUBNORM_RE = re.compile(r"^\s*(s\d+_(?:lo|hi)): (T3_\d+) (>=|<=) (-?[\d.]+)\s*$")
LINK_RE = re.compile(r"^\s*link(T[123]_\d+): (T[123]_\d+) (.+) = 0\s*$")
TERM_RE = re.compile(r"([-+]) ([\d.]+) z(T[123]_\d+)_(\d+)")


def load_candidates(lp_path):
    """Map (var, candidate index) -> exact coefficient from the link rows.

    The .sol Activity column is printed at ~6 significant digits, which is far
    coarser than the ~1e-9 gaps between coupling bounds. Recovering values from
    the LP's own literal coefficients keeps the check exact.
    """
    cands = {}
    for line in lp_path.read_text().splitlines():
        m = LINK_RE.match(line)
        if not m:
            continue
        for sign, coef, var, j in TERM_RE.findall(m.group(3)):
            signed = Decimal(coef) if sign == "+" else -Decimal(coef)
            cands[(var, int(j))] = -signed
    if not cands:
        sys.exit(f"no link rows parsed from {lp_path}")
    return cands


def load_values(paths, cands):
    """Resolve each T1/T2/T3 to its exact candidate via the binary set to 1."""
    vals = {}
    for path in paths:
        for line in path.read_text().splitlines():
            m = SOL_BIN_RE.match(line)
            if not m or int(m.group(3)) != 1:
                continue
            var, j = m.group(1), int(m.group(2))
            if (var, j) not in cands:
                sys.exit(f"{path.name}: no candidate {j} for {var} in the LP")
            exact = cands[(var, j)]
            if var in vals and vals[var] != exact:
                print(f"disagreement: {var} = {vals[var]} and {exact}")
            vals[var] = exact
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solutions", type=Path, nargs="+")
    ap.add_argument("--lp", type=Path, default=Path(__file__).parent / "milp-constraints.lp")
    ap.add_argument("--tol", type=Decimal, default=Decimal("1e-9"),
                    help="slack below -tol counts as a violation")
    args = ap.parse_args()

    cands = load_candidates(args.lp)
    vals = load_values(args.solutions, cands)
    n_t1 = sum(1 for k in vals if k.startswith("T1_"))
    n_t2 = sum(1 for k in vals if k.startswith("T2_"))
    n_t3 = sum(1 for k in vals if k.startswith("T3_"))
    print(f"loaded {len(vals)} values ({n_t1} T1, {n_t2} T2, {n_t3} T3)")

    checked = t3_checked = skipped = 0
    violations = []
    worst = Decimal(0)

    for line in args.lp.read_text().splitlines():
        m = COUPLE_RE.match(line)
        if m:
            name, op, bound = m.group(1), m.group(4), Decimal(m.group(5))
            a, b = m.group(2), m.group(3)
            if a not in vals or b not in vals:
                skipped += 1
                continue
            total = vals[a] + vals[b]
            checked += 1
        else:
            m = SUBNORM_RE.match(line)
            if not m:
                continue
            name, op, bound = m.group(1), m.group(3), Decimal(m.group(4))
            a, b = m.group(2), None
            if a not in vals:
                skipped += 1
                continue
            total = vals[a]
            t3_checked += 1

        slack = total - bound if op == ">=" else bound - total
        worst = min(worst, slack)
        if slack < -args.tol:
            violations.append((name, a, b, op, bound, total, slack))

    for v in violations[:10]:
        lhs = f"{v[1]} + {v[2]}" if v[2] else v[1]
        print(f"  VIOLATION {v[0]}: {lhs} {v[3]} {v[4]} -> {v[5]} (slack {v[6]:.3e})")
    if len(violations) > 10:
        print(f"  ... {len(violations) - 10} more")

    print(f"checked {checked} coupling rows, {t3_checked} subnormal rows "
          f"({skipped} skipped for missing vars)")
    print(f"violations: {len(violations)}, worst slack {worst:.3e}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
