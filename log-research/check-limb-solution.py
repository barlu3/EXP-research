#!/usr/bin/env python3
"""Exact re-validation of the limb tables against the per-limb MILP rows.

Why this exists. The coupling bounds sit ~1e-9 apart, tighter than glpsol's
default primal and integer tolerances, so `INTEGER OPTIMAL` is not evidence
that the source rows hold -- glpsol will report it on assignments that violate
them. This recomputes every row in exact decimal arithmetic from the generated
limb tables, which is the assignment the shipping code actually uses.

Usage (from the repo root):
  python3 log-research/check-limb-solution.py
  python3 log-research/check-limb-solution.py --lp OTHER.lp --header OTHER.h
Exit status is nonzero if any row is violated.
"""

import argparse
import re
import sys
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 80

TABLE_RE = r"static const __bf16 {name}\[\d+\]\[\d+\] = \{{(.*?)\n\}};"
ROW_RE = re.compile(r"\{([^}]*)\}")
COUPLE_RE = re.compile(r"^\s*c\d+_(lo|hi): T1_(\d+) \+ T2_(\d+) (>=|<=) (\S+)")
SUBNORM_RE = re.compile(r"^\s*s\d+_(lo|hi): T3_(\d+) (>=|<=) (\S+)")


def load_table(header_text, name):
    m = re.search(TABLE_RE.format(name=name), header_text, re.S)
    if not m:
        sys.exit(f"table {name} not found in header")
    rows = []
    for row in ROW_RE.findall(m.group(1)):
        # Exact: every literal is a hex float, so Decimal(float) loses nothing.
        rows.append([Decimal(float.fromhex(v.strip().rstrip("f")))
                     for v in row.split(",")])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", type=Path,
                    default=Path("log-research/milp-limb-constraints.lp"))
    ap.add_argument("--header", type=Path,
                    default=Path("implementations/log/logbf16-limb.h"))
    args = ap.parse_args()

    header = args.header.read_text()
    t1 = load_table(header, "T1L")
    t2 = load_table(header, "T2L")
    t3 = load_table(header, "T3L")

    sums1 = [sum(r) for r in t1]
    sums2 = [sum(r) for r in t2]
    sums3 = [sum(r) for r in t3]

    couple = subnorm = 0
    violations = []
    worst = Decimal(0)

    for line in args.lp.read_text().splitlines():
        m = COUPLE_RE.match(line)
        if m:
            kind, i1, i2, op, val = m.groups()
            got = sums1[int(i1)] + sums2[int(i2)]
            name = f"T1_{i1} + T2_{i2}"
            couple += 1
        else:
            m = SUBNORM_RE.match(line)
            if not m:
                continue
            kind, i3, op, val = m.groups()
            got = sums3[int(i3)]
            name = f"T3_{i3}"
            subnorm += 1

        bound = Decimal(val)
        slack = got - bound if op == ">=" else bound - got
        if slack < 0:
            violations.append((line.split(":")[0].strip(), name, op, bound, got, slack))
            if slack < worst:
                worst = slack

    print(f"checked {couple} coupling rows, {subnorm} subnormal rows")
    for v in violations[:10]:
        print(f"  VIOLATION {v[0]}: {v[1]} {v[2]} {v[3]} -> {v[4]} (slack {v[5]:.3e})")
    if len(violations) > 10:
        print(f"  ... {len(violations) - 10} more")
    print(f"violations: {len(violations)}, worst slack {worst:.3e}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
