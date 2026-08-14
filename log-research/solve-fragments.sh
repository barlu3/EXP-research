#!/usr/bin/env bash
# Solve every fragment in fragments/ and summarize status.
#
# Each fragment holds all 128 T2 variables, so INFEASIBLE on any fragment is a
# proof that the full model is INFEASIBLE. OPTIMAL on every fragment does NOT
# prove the full model feasible -- fragments pick T2 values independently.
set -uo pipefail

cd "$(dirname "$0")"
tmlim=${TMLIM:-300}
outdir=fragments/solutions

# Clear prior results first. Logs from an earlier CAND_ULP linger otherwise, and
# a stale INTEGER OPTIMAL next to a current INFEASIBLE reads as a contradiction.
rm -rf "$outdir"
mkdir -p "$outdir"

grep -m1 'CAND_ULP' fragments/*.lp 2>/dev/null | head -1 | cut -d'\' -f2-

printf '%-26s %-14s %s\n' FRAGMENT STATUS TIME
fail=0
for lp in fragments/*.lp; do
    name=$(basename "$lp" .lp)
    log="$outdir/$name.log"
    start=$SECONDS
    glpsol --lp "$lp" --tmlim "$tmlim" --output "$outdir/$name.sol" >"$log" 2>&1
    elapsed=$((SECONDS - start))

    if   grep -q "INTEGER OPTIMAL SOLUTION FOUND"     "$log"; then status=OPTIMAL
    elif grep -q "HAS NO INTEGER FEASIBLE SOLUTION"   "$log"; then status=INFEASIBLE; fail=1
    elif grep -q "HAS NO PRIMAL FEASIBLE SOLUTION"    "$log"; then status=INFEASIBLE; fail=1
    elif grep -q "HAS NO FEASIBLE SOLUTION"           "$log"; then status=INFEASIBLE; fail=1
    elif grep -q "TIME LIMIT EXCEEDED"                "$log"; then status=TIMEOUT;    fail=1
    else                                                         status=UNKNOWN;    fail=1
    fi
    printf '%-26s %-14s %ss\n' "$name" "$status" "$elapsed"
done

exit $fail
