import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Data.Int.Log
import Mathlib.Tactic

/-!
# The floating-point model

binary32 values as a set of reals, binary32 round-to-nearest as "some nearest
value", and binary64 round-to-nearest through the error bound it satisfies.
Nothing here is an axiom: the theorems take these as hypotheses, so they hold
for any rounding that meets them, IEEE 754's included.
-/

namespace FrecpeLogf

/-- A binary32 value, subnormals included: an integer significand below `2^24`
times a power of two from `2^-149` to `2^104`. -/
def IsF32 (y : ℝ) : Prop :=
  ∃ M e : ℤ, |M| < 2 ^ 24 ∧ -149 ≤ e ∧ e ≤ 104 ∧ y = M * (2 : ℝ) ^ e

/-- `y` is a binary32 nearest to `Y`, under any tie rule. -/
def Nearest32 (Y y : ℝ) : Prop :=
  IsF32 y ∧ ∀ f, IsF32 f → |y - Y| ≤ |f - Y|

/-- No binary32 lies strictly between `a` and `b`: their distance in binary32
steps is at most one, the `ulps ≤ 1` that the benchmarks count. -/
def WithinOneUlp (a b : ℝ) : Prop :=
  ∀ f, IsF32 f → ¬ (min a b < f ∧ f < max a b)

/-- binary64's unit roundoff and its underflow floor. -/
noncomputable def u64 : ℝ := 1 / 2 ^ 53
noncomputable def eta64 : ℝ := 1 / 2 ^ 1075

/-- binary64 round-to-nearest, through the bound it meets for `|z| ≤ 1024`:
relative error `2^-53`, plus `2^-1075` for the subnormal range. A fused
multiply-add is `rn (a * b + c)`, one rounding of the exact value. -/
structure RN64 where
  rn : ℝ → ℝ
  err : ∀ z : ℝ, |z| ≤ 1024 → |rn z - z| ≤ u64 * |z| + eta64

lemma two_zpow_pos (e : ℤ) : (0 : ℝ) < 2 ^ e := zpow_pos (by norm_num) e

lemma IsF32.neg {y : ℝ} (h : IsF32 y) : IsF32 (-y) := by
  obtain ⟨M, e, hM, he1, he2, rfl⟩ := h
  exact ⟨-M, e, by simpa using hM, he1, he2, by push_cast; ring⟩

lemma IsF32.two_zpow {e : ℤ} (h1 : -149 ≤ e) (h2 : e ≤ 104) : IsF32 ((2 : ℝ) ^ e) :=
  ⟨1, e, by norm_num, h1, h2, by simp⟩

lemma Nearest32.le {Y y f : ℝ} (h : Nearest32 Y y) (hf : IsF32 f) : |y - Y| ≤ |f - Y| :=
  h.2 f hf

lemma Nearest32.neg {Y y : ℝ} (h : Nearest32 Y y) : Nearest32 (-Y) (-y) := by
  refine ⟨h.1.neg, fun f hf => ?_⟩
  have := h.2 (-f) hf.neg
  calc |(-y) - (-Y)| = |y - Y| := by rw [← abs_neg]; ring_nf
    _ ≤ |-f - Y| := this
    _ = |f - -Y| := by rw [← abs_neg]; ring_nf

lemma WithinOneUlp.of_neg {a b : ℝ} (h : WithinOneUlp (-a) (-b)) : WithinOneUlp a b := by
  intro f hf ⟨h1, h2⟩
  refine h (-f) hf.neg ⟨?_, ?_⟩
  · rw [min_neg_neg]; linarith
  · rw [max_neg_neg]; linarith

/-- A binary32 at least `2^E` in magnitude is a multiple of `2^(E-23)`, the
spacing of the binade `[2^E, 2^(E+1))`. -/
lemma IsF32.multiple {f : ℝ} {E : ℤ} (hf : IsF32 f) (hfE : (2 : ℝ) ^ E ≤ |f|) :
    ∃ n : ℤ, f = n * (2 : ℝ) ^ (E - 23) := by
  obtain ⟨M, e, hM, -, -, rfl⟩ := hf
  have h2 := two_zpow_pos e
  have hMr : |(M : ℝ)| < 2 ^ 24 := by exact_mod_cast hM
  have hlt : (2 : ℝ) ^ E < 2 ^ (e + 24) := by
    calc (2 : ℝ) ^ E ≤ |(M : ℝ) * 2 ^ e| := hfE
      _ = |(M : ℝ)| * 2 ^ e := by rw [abs_mul, abs_of_pos h2]
      _ < 2 ^ 24 * 2 ^ e := by gcongr
      _ = 2 ^ (e + 24) := by rw [zpow_add₀ two_ne_zero]; norm_num; ring
  have hE : E < e + 24 := (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)).1 hlt
  refine ⟨M * 2 ^ (e - (E - 23)).toNat, ?_⟩
  have hk : ((e - (E - 23)).toNat : ℤ) = e - (E - 23) := Int.toNat_of_nonneg (by omega)
  have : (2 : ℝ) ^ e = 2 ^ ((e - (E - 23)).toNat) * 2 ^ (E - 23) := by
    rw [← zpow_natCast, hk, ← zpow_add₀ two_ne_zero]; ring_nf
  rw [this]; push_cast; ring

/-- Multiples of a binade's spacing, up to its top, are binary32 values. -/
lemma isF32_of_mul {n E : ℤ} (hn : |n| ≤ 2 ^ 24) (hE1 : -126 ≤ E) (hE2 : E ≤ 103) :
    IsF32 (n * (2 : ℝ) ^ (E - 23)) := by
  rcases lt_or_eq_of_le hn with h | h
  · exact ⟨n, E - 23, h, by omega, by omega, rfl⟩
  · have h' : n = 2 ^ 24 ∨ n = -2 ^ 24 := by
      rcases abs_eq (by norm_num : (0 : ℤ) ≤ 2 ^ 24) |>.1 h with h' | h' <;> simp [h']
    have hz : (2 : ℝ) ^ (E - 23) = 2 ^ (E - 22) / 2 := by
      rw [show E - 22 = (E - 23) + 1 by ring, zpow_add₀ two_ne_zero]; ring
    rcases h' with rfl | rfl
    · exact ⟨2 ^ 23, E - 22, by norm_num, by omega, by omega, by rw [hz]; push_cast; ring⟩
    · exact ⟨-2 ^ 23, E - 22, by norm_num, by omega, by omega, by rw [hz]; push_cast; ring⟩

/-- Below a binade, the spacing halves: multiples of `2^(E-24)` under `2^E` are
binary32 values. -/
lemma isF32_of_mul_low {n E : ℤ} (hn : |n| < 2 ^ 24) (hE1 : -125 ≤ E) (hE2 : E ≤ 104) :
    IsF32 (n * (2 : ℝ) ^ (E - 24)) :=
  ⟨n, E - 24, hn, by omega, by omega, rfl⟩

end FrecpeLogf
