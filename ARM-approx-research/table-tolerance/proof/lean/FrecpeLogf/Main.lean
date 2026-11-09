import FrecpeLogf.Cells

/-!
# The bound-only theorem

For each of FRECPE's 256 cells, `frecpe_logf_drift_bound` gives a window of
reciprocals around the kernel's own within which every input stays within one
ulp of the correctly rounded `log x`. The window comes from the generated
`epsTable`, but nothing about the table is trusted: `cells_checked` runs
`downOK` and `upOK` on every entry in Lean's kernel.
-/

namespace FrecpeLogf

def epsDown (i : ℕ) : ℕ := (epsTable.getD i (0, 0)).1
def epsUp (i : ℕ) : ℕ := (epsTable.getD i (0, 0)).2

def cellsCheck : ℕ → Bool
  | 0 => true
  | n + 1 => cellsCheck n && downOK n (epsDown n) && upOK n (epsUp n)

theorem cells_checked : cellsCheck 256 = true := by decide +kernel

lemma cell_of_check {n i : ℕ} (hi : i < n) (h : cellsCheck n = true) :
    downOK i (epsDown i) = true ∧ upOK i (epsUp i) = true := by
  induction n with
  | zero => omega
  | succ n ih =>
    simp only [cellsCheck, Bool.and_eq_true] at h
    obtain ⟨⟨h1, h2⟩, h3⟩ := h
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with h' | rfl
    · exact ih h' h1
    · exact ⟨h2, h3⟩

/-- **The drift bound.** Take any significand `m` of cell `i` and any exponent `k`
of a binary32 (`x = 2^k·m`, and `x ≠ 1`), and any reciprocal `c` in the cell's
window, whether or not it is on any grid. Then `frecpe_logf`, with `c` in place of
its estimate and the log entry rounded for that `c`, returns the correctly rounded
`log x` or a neighbouring binary32.

`R` and `rn32` are the hypotheses on rounding, in `Model.lean`. -/
theorem frecpe_logf_drift_bound (R : RN64) (rn32 : ℝ → ℝ) (h32 : ∀ Y, Nearest32 Y (rn32 Y))
    {i : ℕ} (hi : i < 256) {k : ℤ} (hk1 : -149 ≤ k) (hk2 : k ≤ 127) {m c : ℝ}
    (hm1 : (mloOf i : ℝ) ≤ m) (hm2 : m ≤ mhiOf i) (h0 : k = 0 → (mlo0Of i : ℝ) ≤ m)
    (hc1 : ((base i * (1 - (epsDown i : ℚ) * epsUnit) : ℚ) : ℝ) ≤ c)
    (hc2 : c ≤ ((base i * (1 + (epsUp i : ℚ) * epsUnit) : ℚ) : ℝ)) :
    WithinOneUlp (rn32 (kernel R.rn k m c (R.rn (-Real.log c))))
      (rn32 (Real.log ((2 : ℝ) ^ k * m))) := by
  obtain ⟨hd, hu⟩ := cell_of_check hi cells_checked
  rcases le_total c ((base i : ℚ) : ℝ) with h | h
  · exact side_sound R rn32 h32 hd hk1 hk2 hm1 hm2 h0 hc1 h
  · exact side_sound R rn32 h32 hu hk1 hk2 hm1 hm2 h0 h hc2

/-- Cell 68, the median cell's upward bound in the pilot: at least `3.87%`. -/
theorem cell68_up : (387 : ℚ) / 10000 ≤ (epsUp 68 : ℚ) * epsUnit := by decide +kernel

end FrecpeLogf
