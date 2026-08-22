#!/usr/bin/env python3
"""Split milp-constraints.lp into independently solvable fragments.

The MILP has a block structure: every coupling row is `T1_a + T2_b >=/<= bound`,
so a fragment that keeps a subset of T1 indices (plus all 128 T2 indices and
their sel/link/binary rows) is itself a valid, self-contained LP.

Decomposition modes:

  blocks   Slice the T1 index range into N chunks. Each fragment carries all of
           T2, so an INFEASIBLE fragment proves the full model INFEASIBLE. A
           fragment reporting OPTIMAL says nothing about the full model on its
           own -- feasibility does not compose, because fragments disagree about
           the shared T2 values.

  t2blocks Mirror image: slice T2 into N chunks, keep all T1. Same asymmetric
           logic. Use this to localize which T2 indices carry the conflict --
           the near-1 region lives in specific T2 entries, not specific T1 ones.

  pin-t2   Read a solved glpsol --output file, fix every zT2_k_j to the value it
           took there, and emit one fragment per T1 chunk. With T2 frozen the T1
           blocks are genuinely independent. Blocks that go INFEASIBLE name
           exactly which T1 entries the pinned T2 cannot serve.

           WARNING: all-OPTIMAL does NOT certify a full-model solution. The
           coupling bounds here are ~1e-9 apart, far tighter than glpsol's
           default primal/integer tolerances, so glpsol reports OPTIMAL on
           assignments that violate the original rows by up to ~1e-3. Always
           re-check a composed assignment against milp-constraints.lp in exact
           arithmetic before believing it -- check-solution.py does this.

  t2only   Just the T2 sel/link rows, no coupling. Sanity check that the T2
           candidate sets are non-empty. Should always be OPTIMAL.

T3 is not part of any decomposition. Its rows are single-variable
(`lb <= T3_k <= ub`), so the 127 T3 entries are independent of each other and
of T1/T2 -- there is nothing to decompose. They are emitted whole into one
`frag-t3-only.lp`, which is exactly the T3 subproblem. INFEASIBLE there names a
subnormal input no 8-bit grid value within +/-CAND_ULP can serve, and lifts to
the full model just as a T1 fragment's does.

Usage:
    python3 split-milp.py                       # 16 block fragments in fragments/
    python3 split-milp.py --blocks 32
    python3 split-milp.py --t1-range 1 8        # one fragment, T1_1..T1_8
    python3 split-milp.py --t2-blocks 16        # slice on T2 instead
    python3 split-milp.py --pin-t2 fragments/solutions/frag-t1-001-016.sol
"""

import argparse
import re
import sys
from pathlib import Path

HEADER_RE = re.compile(r"^(Minimize|Maximize|Subject To|Bounds|Binary|Binaries|General|Generals|End)\s*$")
SEL_RE = re.compile(r"^\s*sel(T[123])_(\d+):")
LINK_RE = re.compile(r"^\s*link(T[123])_(\d+):")
COUPLE_RE = re.compile(r"^\s*c\d+_(?:lo|hi): T1_(\d+) \+ T2_(\d+) ")
SUBNORM_RE = re.compile(r"^\s*s\d+_(?:lo|hi): T3_(\d+) ")
BOUND_RE = re.compile(r"^\s*(T[123])_(\d+)\s+free")
BINARY_RE = re.compile(r"^\s*z(T[123])_(\d+)_\d+\s*$")
SOL_COL_RE = re.compile(r"^\s*\d+\s+(zT2_\d+_\d+)\s+\*?\s+(\d+)")


class Model:
    """Parsed sections of the source LP, indexed for cheap filtering."""

    def __init__(self):
        self.comments = []
        self.sel = {}       # (table, idx) -> line
        self.link = {}      # (table, idx) -> line
        self.couple = {}    # t1_idx -> [line, ...]
        self.subnorm = {}   # t3_idx -> [line, ...]
        self.bounds = {}    # (table, idx) -> line
        self.binaries = {}  # (table, idx) -> [line, ...]


def parse(path):
    model = Model()
    section = None

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("\\"):
            model.comments.append(line)
            continue

        header = HEADER_RE.match(line)
        if header:
            section = header.group(1)
            continue

        if section in ("Minimize", "Maximize"):
            continue  # fragments emit their own zero objective
        elif section == "Subject To":
            m = SEL_RE.match(line)
            if m:
                model.sel[(m.group(1), int(m.group(2)))] = line
                continue
            m = LINK_RE.match(line)
            if m:
                model.link[(m.group(1), int(m.group(2)))] = line
                continue
            m = COUPLE_RE.match(line)
            if m:
                model.couple.setdefault(int(m.group(1)), []).append(line)
                continue
            m = SUBNORM_RE.match(line)
            if m:
                model.subnorm.setdefault(int(m.group(1)), []).append(line)
                continue
            sys.exit(f"unrecognized constraint row: {line[:80]}")
        elif section == "Bounds":
            m = BOUND_RE.match(line)
            if not m:
                sys.exit(f"unrecognized bound row: {line[:80]}")
            model.bounds[(m.group(1), int(m.group(2)))] = line
        elif section in ("Binary", "Binaries"):
            m = BINARY_RE.match(line)
            if not m:
                sys.exit(f"unrecognized binary row: {line[:80]}")
            model.binaries.setdefault((m.group(1), int(m.group(2))), []).append(line)
        elif section == "End":
            break

    return model


def parse_pins(path):
    """Extract zT2_k_j -> 0/1 from a glpsol --output solution file."""
    pins = {}
    for raw in path.read_text().splitlines():
        m = SOL_COL_RE.match(raw)
        if m:
            pins[m.group(1)] = int(m.group(2))
    if not pins:
        sys.exit(f"no zT2_* columns found in {path}; is it a glpsol --output file?")
    chosen = sum(pins.values())
    print(f"pins: {len(pins)} zT2 binaries, {chosen} set to 1")
    return pins


def require_full_t2_coverage(pins, t2_all):
    """A partially-pinned fragment leaves T2 free, which silently defeats --pin-t2."""
    pinned = {int(m.group(1)) for m in (re.match(r"zT2_(\d+)_", v) for v in pins) if m}
    missing = sorted(set(t2_all) - pinned)
    if missing:
        sys.exit(f"--pin-t2 file covers {len(pinned)}/{len(t2_all)} T2 indices; "
                 f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}")


def emit(model, out_path, t1_keep, t2_keep, note, pins=None):
    """Write a fragment holding only the named T1 and T2 indices.

    `pins` fixes zT2 binaries to a known assignment, which decouples the T1
    blocks from each other entirely.
    """
    keys = [("T1", i) for i in sorted(t1_keep)] + [("T2", i) for i in sorted(t2_keep)]
    t2_set = set(t2_keep)

    lines = list(model.comments)
    lines.append(f"\\ FRAGMENT: {note}")
    lines.append("Minimize")
    anchor = f"T1_{min(t1_keep)}" if t1_keep else f"T2_{min(t2_keep)}"
    lines.append(f" obj: 0 {anchor}")
    lines.append("Subject To")
    for key in keys:
        lines.append(model.sel[key])
        lines.append(model.link[key])
    for i in sorted(t1_keep):
        lines.extend(line for line in model.couple[i]
                     if int(COUPLE_RE.match(line).group(2)) in t2_set)
    lines.append("Bounds")
    lines.extend(model.bounds[key] for key in keys)
    if pins:
        for key in keys:
            for var in (BINARY_RE.match(b).group(0).strip() for b in model.binaries[key]):
                if var in pins:
                    lines.append(f" {var} = {pins[var]}")
    lines.append("Binary")
    for key in keys:
        lines.extend(model.binaries[key])
    lines.append("End")

    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def emit_t3(model, out_path, t3_keep):
    """Write the T3 subproblem: sel/link plus the single-variable bound rows.

    T3 has its own emitter rather than sharing `emit` because its rows name one
    variable, not a T1/T2 pair -- there is no coupling to filter and no shared
    table to duplicate across fragments.
    """
    keys = [("T3", i) for i in sorted(t3_keep)]

    lines = list(model.comments)
    lines.append(f"\\ FRAGMENT: T3_{min(t3_keep)}..T3_{max(t3_keep)}, subnormal lookups")
    lines.append("Minimize")
    lines.append(f" obj: 0 T3_{min(t3_keep)}")
    lines.append("Subject To")
    for key in keys:
        lines.append(model.sel[key])
        lines.append(model.link[key])
    for i in sorted(t3_keep):
        lines.extend(model.subnorm[i])
    lines.append("Bounds")
    lines.extend(model.bounds[key] for key in keys)
    lines.append("Binary")
    for key in keys:
        lines.extend(model.binaries[key])
    lines.append("End")

    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path(__file__).parent / "milp-constraints.lp")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "fragments")
    ap.add_argument("--blocks", type=int, default=16, help="number of T1 chunks")
    ap.add_argument("--t1-range", type=int, nargs=2, metavar=("LO", "HI"),
                    help="emit a single fragment covering T1_LO..T1_HI")
    ap.add_argument("--t2-blocks", type=int, metavar="N",
                    help="slice on T2 into N chunks (all T1 kept) instead of on T1")
    ap.add_argument("--pin-t2", type=Path, metavar="SOL",
                    help="fix zT2 binaries to their values in a glpsol --output file")
    args = ap.parse_args()

    model = parse(args.input)
    t1_all = sorted({i for (t, i) in model.sel if t == "T1"})
    t2_all = sorted({i for (t, i) in model.sel if t == "T2"})
    t3_all = sorted({i for (t, i) in model.sel if t == "T3"})
    print(f"parsed: {len(t1_all)} T1, {len(t2_all)} T2, {len(t3_all)} T3, "
          f"{sum(len(v) for v in model.couple.values())} coupling rows, "
          f"{sum(len(v) for v in model.subnorm.values())} subnormal rows")

    args.outdir.mkdir(exist_ok=True)

    # T3 always ships as one whole fragment: single-variable rows, nothing to
    # slice. Skipped for --t1-range, which deliberately emits a partial model.
    if t3_all and not args.t1_range:
        t3_out = args.outdir / "frag-t3-only.lp"
        n = emit_t3(model, t3_out, t3_all)
        print(f"{t3_out.name}: {n} lines, {len(t3_all)} T3 entries")

    if args.t1_range:
        lo, hi = args.t1_range
        keep = [i for i in t1_all if lo <= i <= hi]
        if not keep:
            sys.exit(f"no T1 indices in range {lo}..{hi}")
        out = args.outdir / f"frag-t1-{lo}-{hi}.lp"
        n = emit(model, out, keep, t2_all, f"T1_{lo}..T1_{hi}, all T2")
        print(f"{out.name}: {n} lines, {len(keep)} T1 blocks")
        return

    if args.t2_blocks:
        size = -(-len(t2_all) // args.t2_blocks)
        for b in range(0, len(t2_all), size):
            chunk = t2_all[b:b + size]
            out = args.outdir / f"frag-t2-{chunk[0]:03d}-{chunk[-1]:03d}.lp"
            n = emit(model, out, t1_all, chunk, f"T2_{chunk[0]}..T2_{chunk[-1]}, all T1")
            print(f"{out.name}: {n} lines, {len(chunk)} T2 blocks")
        return

    pins = None
    if args.pin_t2:
        pins = parse_pins(args.pin_t2)
        require_full_t2_coverage(pins, t2_all)
    tag = "pinned" if pins else "all T2"

    if not pins:
        t2_out = args.outdir / "frag-t2-only.lp"
        n = emit(model, t2_out, [], t2_all, "T2 sel/link only, no coupling")
        print(f"{t2_out.name}: {n} lines (sanity check, expect OPTIMAL)")

    size = -(-len(t1_all) // args.blocks)
    for b in range(0, len(t1_all), size):
        chunk = t1_all[b:b + size]
        prefix = "fragpin" if pins else "frag"
        out = args.outdir / f"{prefix}-t1-{chunk[0]:03d}-{chunk[-1]:03d}.lp"
        n = emit(model, out, chunk, t2_all,
                 f"T1_{chunk[0]}..T1_{chunk[-1]}, {tag}", pins=pins)
        print(f"{out.name}: {n} lines, {len(chunk)} T1 blocks")


if __name__ == "__main__":
    main()
