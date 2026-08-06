#!/usr/bin/env bash
# Regenerate milp-constraints.lp and split it into solvable fragments.
#
# Run this before solve-fragments.sh. The three generator steps are the CMake
# targets described in log-research/CMakeLists.txt; they must run from the repo
# root because bound-calc and milp-gen hardcode "log-research/"-prefixed paths.
#
# Only milp-constraints.lp is generated. core-near1.lp, core-i1_126.lp,
# core-i1_127.lp and milp-constraints-min.lp have no generator in this repo --
# they are kept artifacts and are never touched here.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
here=$root/log-research
build=${BUILD_DIR:-$root/build}
jobs=${JOBS:-$(nproc 2>/dev/null || echo 2)}

# Fragment layout: --blocks N slices T1, --t2-blocks N slices T2, and
# --pin-t2 FILE freezes T2 from a solved fragment. See split-milp.py.
split_args=("$@")
[ ${#split_args[@]} -eq 0 ] && split_args=(--blocks 16)

step() { printf '\n== %s\n' "$1"; }

step "configure ($build)"
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null

step "build generators"
cmake --build "$build" --target precompute bound-calc milp-gen -j "$jobs" >/dev/null

step "step 1/3: ln-precompute.txt"
cmake --build "$build" --target ln-precompute

step "step 2/3: ln-bounds.txt + lp-constraints.txt"
cmake --build "$build" --target ln-bounds

step "step 3/3: milp-constraints.lp"
cmake --build "$build" --target ln-milp

lp=$here/milp-constraints.lp
[ -s "$lp" ] || { echo "error: $lp missing or empty" >&2; exit 1; }

step "split into fragments"
rm -f "$here"/fragments/*.lp
python3 "$here/split-milp.py" "${split_args[@]}"

# Conservation check: a fragment set that drops coupling rows would solve fast
# and report OPTIMAL while proving nothing. --t1-range deliberately emits a
# partial model, so the check does not apply there.
if [[ " ${split_args[*]} " == *" --t1-range "* ]]; then
    echo "ok -- partial fragment (--t1-range); skipping conservation check"
    exit 0
fi

src=$(grep -cP '^ c[0-9]+_(lo|hi):' "$lp")
shopt -s nullglob
frags=("$here"/fragments/frag-t[12]-[0-9]*.lp "$here"/fragments/fragpin-t[12]-[0-9]*.lp)
shopt -u nullglob
frg=$(cat "${frags[@]}" | grep -cP '^ c[0-9]+_(lo|hi):') || frg=0
printf '\ncoupling rows: source=%s fragments=%s\n' "$src" "$frg"
if [ "$src" -ne "$frg" ]; then
    echo "error: fragments dropped $((src - frg)) coupling rows" >&2
    exit 1
fi

echo "ok -- next: ./solve-fragments.sh"
