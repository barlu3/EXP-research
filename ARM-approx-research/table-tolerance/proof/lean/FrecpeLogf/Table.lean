import FrecpeLogf.CellCheck

/-!
# FRECPE's 256 cells

`chat i` is FRECPE's output for significands whose top eight fraction bits are
`i`, from Arm's `RecipEstimate` pseudocode. `gen/check.py` checks these 256
values against the ones the hardware returned.

The kernel uses `base i`: FRECPE's value, except `c = 1` in cell 0, without which
results just above `x = 1` break one ulp.
-/

namespace FrecpeLogf

/-- Arm's `RecipEstimate` for cell `i`, as a value in `[1/2, 1)`. -/
def chat (i : ℕ) : ℚ := (((2 ^ 19 / (2 * (256 + i) + 1) + 1) / 2 : ℕ) : ℚ) / 512

def base (i : ℕ) : ℚ := if i = 0 then 1 else chat i

/-- Cell `i`'s significands run from `mloOf i` up to its last binary32, `mhiOf i`. -/
def mloOf (i : ℕ) : ℚ := 1 + (i : ℚ) / 256
def mhiOf (i : ℕ) : ℚ := 1 + ((i : ℚ) + 1) / 256 - 1 / 2 ^ 23

/-- At `k = 0`, cell 0 starts one binary32 above `1`: `logf(1)` returns before the
kernel runs. -/
def mlo0Of (i : ℕ) : ℚ := if i = 0 then 1 + 1 / 2 ^ 23 else mloOf i

/-- Drifts are whole multiples of `2^-22`. -/
def epsUnit : ℚ := 1 / 2 ^ 22

def downOK (i e : ℕ) : Bool :=
  sideOK (mloOf i) (mlo0Of i) (mhiOf i) (base i * (1 - (e : ℚ) * epsUnit)) (base i)

def upOK (i e : ℕ) : Bool :=
  sideOK (mloOf i) (mlo0Of i) (mhiOf i) (base i) (base i * (1 + (e : ℚ) * epsUnit))

end FrecpeLogf
