import FrecpeLogf.Accept
import FrecpeLogf.PolyErr

/-!
# One FRECPE cell, checked by computation

`sideOK` is an executable check over exact rationals; `side_sound` proves that
when it returns `true`, every significand of the cell, at every exponent, with
every reciprocal `c` in the window, gives a `logf` result within one ulp of the
correctly rounded one.

A window `[cl, ch]` passes if, for each exponent case, the polynomial's error
plus rounding stays under the acceptance threshold:

* `k = 0`: `t = log m`, the smallest results when `m` is low in `[1, 2)`;
* `k = -1`: `t = log(m/2)`, the smallest when `m` is high;
* any other `k`: `|t| ≥ log 2`, a generous threshold, whatever the exponent.

The first two split `m` adaptively, since a cell can straddle a binade of `t`.
-/

namespace FrecpeLogf

open Real

/-! ### Exact powers of two -/

def pow2 : ℤ → ℚ
  | .ofNat n => 2 ^ n
  | .negSucc n => 1 / 2 ^ (n + 1)

lemma pow2_cast (E : ℤ) : ((pow2 E : ℚ) : ℝ) = (2 : ℝ) ^ E := by
  cases E with
  | ofNat n => simp [pow2]
  | negSucc n => simp [pow2, zpow_negSucc]

/-! ### The acceptance threshold for a range of `|t|` -/

/-- The binade of `L > 0`; `tau` re-checks what it relies on, so this only has
to be right for the threshold to be useful, not for it to be sound. -/
def binade (L : ℚ) : ℤ :=
  let E0 : ℤ := (Nat.log2 L.num.natAbs : ℤ) - (Nat.log2 L.den : ℤ)
  if pow2 E0 ≤ L then (if pow2 (E0 + 1) ≤ L then E0 + 1 else E0) else E0 - 1

/-- How far `Y` may be from any `t` with `L ≤ |t|`: a full spacing of `L`'s binade
when `L` is at least a spacing above its bottom, three quarters of one otherwise,
and `0` if `L` is unusable. A larger `|t|` only has more room. -/
def tau (L : ℚ) : ℚ :=
  let E := binade L
  if -100 ≤ E ∧ E ≤ 100 ∧ pow2 E + pow2 (E - 23) ≤ L then pow2 (E - 23)
  else if -100 ≤ E ∧ pow2 E ≤ L then 3 / 2 * pow2 (E - 24)
  else 0

/-- `|t|` a spacing above `2^E`, and an error below that spacing: accepted in
`E`'s binade by `accept_inner`, and in any higher one by `accept`, whose allowance
there is already larger. -/
lemma accept_inner_ge {E : ℤ} (hE1 : -100 ≤ E) {t Y y a : ℝ}
    (ht1 : (2 : ℝ) ^ E + 2 ^ (E - 23) ≤ |t|) (ht2 : |t| < 2 ^ 100)
    (hY : |Y - t| < 2 ^ (E - 23)) (hy : Nearest32 Y y) (ha : Nearest32 t a) :
    WithinOneUlp y a := by
  have hs := two_zpow_pos (E - 23)
  have hP := two_zpow_pos E
  have htpos : 0 < |t| := by linarith
  set E' := Int.log 2 |t|
  have h1 : (2 : ℝ) ^ E' ≤ |t| := Int.zpow_log_le_self (by norm_num) htpos
  have h2 : |t| < 2 ^ (E' + 1) := Int.lt_zpow_succ_log_self (by norm_num) |t|
  have hEE : E ≤ E' := by
    by_contra! h
    have : (2 : ℝ) ^ (E' + 1) ≤ 2 ^ E := zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  have hE'2 : E' ≤ 100 := by
    by_contra! h
    have : (2 : ℝ) ^ (100 : ℤ) ≤ 2 ^ E' := zpow_le_zpow_right₀ (by norm_num) (by omega)
    norm_num at this; linarith
  rcases eq_or_lt_of_le hEE with rfl | hlt
  · rcases le_or_gt 0 t with ht | ht
    · rw [abs_of_nonneg ht] at ht1 h2
      exact accept_inner (by omega) (by omega) ht1 h2 hY hy ha
    · rw [abs_of_neg ht] at ht1 h2
      apply WithinOneUlp.of_neg
      refine accept_inner (by omega) (by omega) ht1 h2 ?_ hy.neg ha.neg
      rw [show -Y - -t = -(Y - t) by ring, abs_neg]; exact hY
  · have hq : (2 : ℝ) ^ (E - 23) ≤ 2 ^ (E' - 24) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have hq0 := two_zpow_pos (E' - 24)
    exact accept (E := E') (by omega) h1 ht2 (by linarith) hy ha

theorem tau_sound {L : ℚ} {t Y y a : ℝ} (hL : (L : ℝ) ≤ |t|)
    (ht : |t| < 2 ^ 100) (hY : |Y - t| < tau L) (hy : Nearest32 Y y) (ha : Nearest32 t a) :
    WithinOneUlp y a := by
  unfold tau at hY
  simp only at hY
  split_ifs at hY with h1 h2
  · obtain ⟨e1, -, e3⟩ := h1
    have e3' : ((pow2 (binade L) + pow2 (binade L - 23) : ℚ) : ℝ) ≤ L := by exact_mod_cast e3
    push_cast [pow2_cast] at e3' hY
    exact accept_inner_ge e1 (by linarith) ht hY hy ha
  · obtain ⟨e1, e2⟩ := h2
    have e2' : ((pow2 (binade L) : ℚ) : ℝ) ≤ L := by exact_mod_cast e2
    push_cast [pow2_cast] at e2' hY
    exact accept e1 (by linarith) ht hY hy ha
  · exact absurd hY (by push_cast; exact not_lt.mpr (abs_nonneg _))

/-! ### Exact bounds on `-log a` -/

def seriesQ (x : ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => seriesQ x n + x ^ (n + 1) / ((n : ℚ) + 1)

lemma seriesQ_cast (x : ℚ) (n : ℕ) :
    ((seriesQ x n : ℚ) : ℝ) = ∑ i ∈ Finset.range n, (x : ℝ) ^ (i + 1) / ((i : ℝ) + 1) := by
  induction n with
  | zero => simp [seriesQ]
  | succ n ih => rw [Finset.sum_range_succ, ← ih]; simp [seriesQ]

/-- Terms enough that the remainder is far below the threshold at that size. -/
def nTerms (x : ℚ) : ℕ := if x ≤ 1 / 256 then 8 else if x ≤ 1 / 16 then 12 else 32

def negLogLo (a : ℚ) : ℚ :=
  seriesQ (1 - a) (nTerms (1 - a)) - (1 - a) ^ (nTerms (1 - a) + 1) / (1 - (1 - a))

lemma negLog_lo {a : ℚ} (h0 : 0 < a) (h1 : a ≤ 1) :
    ((negLogLo a : ℚ) : ℝ) ≤ -log a := by
  have h0' : (0 : ℝ) < a := by exact_mod_cast h0
  have h1' : (a : ℝ) ≤ 1 := by exact_mod_cast h1
  set n := nTerms (1 - a)
  have hx0 : (0 : ℝ) ≤ 1 - a := by linarith
  have hx1 : |(1 - a : ℝ)| < 1 := by rw [abs_of_nonneg hx0]; linarith
  have h := Real.abs_log_sub_add_sum_range_le hx1 n
  rw [abs_of_nonneg hx0, show (1 - (1 - a) : ℝ) = a by ring] at h
  have hc := seriesQ_cast (1 - a) n
  push_cast at hc
  rw [← hc, abs_le] at h
  simp only [negLogLo]
  push_cast
  rw [show (1 - (1 - a) : ℝ) = a by ring]
  linarith [h.1, h.2]

/-- `log m` from below, for `m ≥ ma ≥ 1`. -/
lemma log_lo {ma : ℚ} {m : ℝ} (h1 : 1 ≤ ma) (h2 : (ma : ℝ) ≤ m) :
    ((negLogLo (1 / ma) : ℚ) : ℝ) ≤ log m := by
  have ha0 : 0 < ma := by linarith
  have ha := negLog_lo (a := 1 / ma) (by positivity) (by rw [div_le_iff₀ ha0]; linarith)
  have ca : (0 : ℝ) < ma := by exact_mod_cast ha0
  have e1 : -log (((1 / ma : ℚ)) : ℝ) = log ma := by push_cast; rw [one_div, log_inv]; ring
  rw [e1] at ha
  exact ha.trans (Real.log_le_log ca h2)

/-- `-log(m/2)` from below, for `0 < m ≤ mb ≤ 2`: the size of `t` at `k = -1`. -/
lemma log_half_lo {mb : ℚ} {m : ℝ} (hb0 : 0 < mb) (hb : mb ≤ 2) (hm : 0 < m) (h3 : m ≤ mb) :
    ((negLogLo (mb / 2) : ℚ) : ℝ) ≤ log 2 - log m := by
  have cb : (0 : ℝ) < mb := by exact_mod_cast hb0
  have bb := negLog_lo (a := mb / 2) (by positivity) (by linarith)
  have e1 : -log (((mb / 2 : ℚ)) : ℝ) = log 2 - log mb := by
    push_cast; rw [Real.log_div cb.ne' two_ne_zero]; ring
  rw [e1] at bb
  have l2 := Real.log_le_log hm h3
  linarith

/-! ### The checks -/

/-- `kernel_error`'s rounding term, rounded up to a rational and made strict. -/
def rhoQ (K : ℚ) : ℚ := (11 + 3 * K) / 2 ^ 53 + K / 2 ^ 54

/-- The box stays where `kernel_error` and `err_sound` apply. -/
def guard (ma mb cl ch : ℚ) : Bool :=
  decide (1 ≤ ma ∧ ma ≤ mb ∧ mb ≤ 2 ∧ 9 / 20 ≤ cl ∧ cl ≤ ch ∧ ch ≤ 11 / 10 ∧
    -1 / 16 ≤ ma * cl - 1 ∧ mb * ch - 1 ≤ 1 / 16)

/-- `k = 0`. -/
def check0 (cl ch ma mb : ℚ) : Bool :=
  guard ma mb cl ch && decide (errSup (ma * cl - 1) (mb * ch - 1) ≤
    tau (negLogLo (1 / ma)) - rhoQ 0)

/-- `k = -1`. -/
def checkM1 (cl ch ma mb : ℚ) : Bool :=
  guard ma mb cl ch && decide (errSup (ma * cl - 1) (mb * ch - 1) ≤
    tau (negLogLo (mb / 2)) - rhoQ 1)

/-- Any other `k`. -/
def checkO (cl ch ma mb : ℚ) : Bool :=
  guard ma mb cl ch && decide (errSup (ma * cl - 1) (mb * ch - 1) ≤
    3 / 2 * pow2 (-25) - rhoQ 150)

/-- Tries the whole range, and on failure each half. -/
def splitOK (f : ℚ → ℚ → Bool) : ℕ → ℚ → ℚ → Bool
  | 0, a, b => f a b
  | n + 1, a, b => f a b || (splitOK f n a ((a + b) / 2) && splitOK f n ((a + b) / 2) b)

lemma splitOK_sound {f : ℚ → ℚ → Bool} {Q : ℝ → Prop}
    (hf : ∀ a b : ℚ, f a b = true → ∀ m : ℝ, (a : ℝ) ≤ m → m ≤ b → Q m)
    (n : ℕ) (a b : ℚ) (h : splitOK f n a b = true) {m : ℝ} (h1 : (a : ℝ) ≤ m) (h2 : m ≤ b) :
    Q m := by
  induction n generalizing a b with
  | zero => exact hf a b h m h1 h2
  | succ n ih =>
    simp only [splitOK, Bool.or_eq_true, Bool.and_eq_true] at h
    rcases h with h | ⟨ha, hb⟩
    · exact hf a b h m h1 h2
    · rcases le_total m (((a + b) / 2 : ℚ) : ℝ) with hm | hm
      · exact ih _ _ ha h1 hm
      · exact ih _ _ hb hm h2

/-- One side of one cell: reciprocals in `[cl, ch]`, significands in
`[mlo, mhi]`, and in `[mlo0, mhi]` when `k = 0`. -/
def sideOK (mlo mlo0 mhi cl ch : ℚ) : Bool :=
  splitOK (check0 cl ch) 24 mlo0 mhi && splitOK (checkM1 cl ch) 24 mlo mhi &&
    checkO cl ch mlo mhi

/-! ### Soundness -/

/-- The common step: a passing box bounds the kernel's error by any threshold
`T` that the check compared against. -/
lemma box_error (R : RN64) {k : ℤ} {K ma mb cl ch T : ℚ} {m c : ℝ}
    (hk : |(k : ℝ)| ≤ K) (hK : K ≤ 150) (hg : guard ma mb cl ch = true)
    (he : errSup (ma * cl - 1) (mb * ch - 1) ≤ T - rhoQ K)
    (hm1 : (ma : ℝ) ≤ m) (hm2 : m ≤ mb) (hc1 : (cl : ℝ) ≤ c) (hc2 : c ≤ ch) :
    |kernel R.rn k m c (R.rn (-log c)) - (k * log 2 + log m)| < T := by
  simp only [guard, decide_eq_true_eq] at hg
  obtain ⟨g1, g2, g3, g4, g5, g6, g7, g8⟩ := hg
  have c1 : (1 : ℝ) ≤ ma := by exact_mod_cast g1
  have c3 : (mb : ℝ) ≤ 2 := by exact_mod_cast g3
  have c4 : (9 / 20 : ℝ) ≤ cl := by
    have h := (Rat.cast_le (K := ℝ)).2 g4; push_cast at h; linarith
  have c6 : (ch : ℝ) ≤ 11 / 10 := by
    have h := (Rat.cast_le (K := ℝ)).2 g6; push_cast at h; linarith
  have c7 : (-1 / 16 : ℝ) ≤ ((ma * cl - 1 : ℚ) : ℝ) := by
    have h := (Rat.cast_le (K := ℝ)).2 g7; push_cast at h ⊢; linarith
  have c8 : ((mb * ch - 1 : ℚ) : ℝ) ≤ 1 / 16 := by
    have h := (Rat.cast_le (K := ℝ)).2 g8; push_cast at h ⊢; linarith
  have cK : (K : ℝ) ≤ 150 := by exact_mod_cast hK
  have hmc1 : (ma : ℝ) * cl ≤ m * c := mul_le_mul hm1 hc1 (by linarith) (by linarith)
  have hmc2 : m * c ≤ (mb : ℝ) * ch := mul_le_mul hm2 hc2 (by linarith) (by linarith)
  have r1 : ((ma * cl - 1 : ℚ) : ℝ) ≤ m * c - 1 := by push_cast; linarith
  have r2 : m * c - 1 ≤ ((mb * ch - 1 : ℚ) : ℝ) := by push_cast; linarith
  have hr : |m * c - 1| ≤ 1 / 16 := by rw [abs_le]; constructor <;> linarith
  have hE := kernel_error R (K := (K : ℝ)) hk (by linarith) (by linarith) (by linarith)
    (by linarith) (by linarith) hr
  have hP := err_sound c7 c8 r1 r2
  rw [show (1 + (m * c - 1) : ℝ) = m * c by ring] at hP
  have he' : ((errSup (ma * cl - 1) (mb * ch - 1) : ℚ) : ℝ) ≤ ((T - rhoQ K : ℚ) : ℝ) := by
    exact_mod_cast he
  have hK0 : (0 : ℝ) ≤ K := (abs_nonneg _).trans hk
  have hd := ln2d_err
  have hKd : (K : ℝ) * |ln2d - log 2| ≤ K * (1 / 2 ^ 54) := mul_le_mul_of_nonneg_left hd hK0
  have hu0 := u64_pos; have he1 := eta64_le; have he0 := eta64_pos
  have hrho : ((rhoQ K : ℚ) : ℝ) = (11 + 3 * K) * u64 + K * (1 / 2 ^ 54) := by
    simp only [rhoQ, u64]; push_cast; ring
  rw [Rat.cast_sub, hrho] at he'
  have : (16 : ℝ) * eta64 < u64 := by linarith
  have hexp : ((10 : ℝ) + 3 * K) * u64 + 16 * eta64 < (11 + 3 * K) * u64 := by linarith
  linarith

/-- If `sideOK` passes, every input of the cell at every exponent, with every
reciprocal in the window, rounds to within one ulp of the correctly rounded
`log x`. -/
theorem side_sound (R : RN64) (rn32 : ℝ → ℝ) (h32 : ∀ Y, Nearest32 Y (rn32 Y))
    {mlo mlo0 mhi cl ch : ℚ} (h : sideOK mlo mlo0 mhi cl ch = true)
    {k : ℤ} (hk1 : -149 ≤ k) (hk2 : k ≤ 127) {m c : ℝ}
    (hm1 : (mlo : ℝ) ≤ m) (hm2 : m ≤ mhi) (h0 : k = 0 → (mlo0 : ℝ) ≤ m)
    (hc1 : (cl : ℝ) ≤ c) (hc2 : c ≤ ch) :
    WithinOneUlp (rn32 (kernel R.rn k m c (R.rn (-log c)))) (rn32 (log ((2 : ℝ) ^ k * m))) := by
  simp only [sideOK, Bool.and_eq_true] at h
  obtain ⟨⟨hA, hB⟩, hC⟩ := h
  -- The other case's guard pins down `m` for the size bounds.
  have hgO := hC
  simp only [checkO, Bool.and_eq_true, decide_eq_true_eq] at hgO
  obtain ⟨hg, heO⟩ := hgO
  have hg' := hg
  simp only [guard, decide_eq_true_eq] at hg'
  obtain ⟨g1, -, g3, -, -, -, -, -⟩ := hg'
  have c1 : (1 : ℝ) ≤ mlo := by exact_mod_cast g1
  have c3 : (mhi : ℝ) ≤ 2 := by exact_mod_cast g3
  have hm0 : 0 < m := by linarith
  rw [log_scaled k hm0]
  have l2 := ln2d_err
  unfold ln2d at l2
  rw [abs_le] at l2
  have l2a : (69 / 100 : ℝ) < log 2 := by linarith [l2.2]
  have l2b : log 2 < 7 / 10 := by linarith [l2.1]
  have lm0 : 0 ≤ log m := Real.log_nonneg (by linarith)
  have lm1 : log m ≤ log 2 := Real.log_le_log hm0 (by linarith)
  have hsize : |(k : ℝ) * log 2 + log m| < 2 ^ 100 := by
    have hk1' : (-149 : ℝ) ≤ k := by exact_mod_cast hk1
    have hk2' : (k : ℝ) ≤ 127 := by exact_mod_cast hk2
    have hl : |(k : ℝ) * log 2| ≤ 149 * 1 := by
      rw [abs_mul, abs_of_pos (by linarith : 0 < log 2)]
      exact mul_le_mul (by rw [abs_le]; constructor <;> linarith) (by linarith) (by linarith)
        (by norm_num)
    have hm' : |log m| ≤ 1 := by rw [abs_of_nonneg lm0]; linarith
    have := abs_add_le ((k : ℝ) * log 2) (log m)
    calc _ ≤ (150 : ℝ) := by linarith
      _ < 2 ^ 100 := by norm_num
  rcases eq_or_ne k 0 with rfl | hk0
  · -- k = 0: `t = log m`.
    have hm0' := h0 rfl
    refine splitOK_sound (Q := fun m => (0 : ℝ) < m → 0 ≤ log m →
        |((0 : ℤ) : ℝ) * log 2 + log m| < 2 ^ 100 →
        WithinOneUlp (rn32 (kernel R.rn 0 m c (R.rn (-log c))))
          (rn32 (((0 : ℤ) : ℝ) * log 2 + log m))) ?_ 24 mlo0 mhi hA hm0' hm2 hm0 lm0 hsize
    intro a b hab m' ha hb hpos hl0 hsz
    simp only [check0, Bool.and_eq_true, decide_eq_true_eq] at hab
    obtain ⟨hg0, he0⟩ := hab
    have hg0' := hg0
    simp only [guard, decide_eq_true_eq] at hg0'
    have hbd := log_lo hg0'.1 ha
    have hE := box_error R (k := 0) (K := 0) (by simp) (by norm_num) hg0 he0 ha hb hc1 hc2
    refine tau_sound (L := negLogLo (1 / a)) ?_ hsz hE (h32 _) (h32 _)
    simp only [Int.cast_zero, zero_mul, zero_add]; rw [abs_of_nonneg hl0]; exact hbd
  rcases eq_or_ne k (-1) with rfl | hk1'
  · -- k = -1: `t = log m - log 2`.
    refine splitOK_sound (Q := fun m => (0 : ℝ) < m → log m ≤ log 2 →
        |((-1 : ℤ) : ℝ) * log 2 + log m| < 2 ^ 100 →
        WithinOneUlp (rn32 (kernel R.rn (-1) m c (R.rn (-log c))))
          (rn32 (((-1 : ℤ) : ℝ) * log 2 + log m))) ?_ 24 mlo mhi hB hm1 hm2 hm0 lm1 hsize
    intro a b hab m' ha hb hpos hl1 hsz
    simp only [checkM1, Bool.and_eq_true, decide_eq_true_eq] at hab
    obtain ⟨hg0, he0⟩ := hab
    have hg0' := hg0
    simp only [guard, decide_eq_true_eq] at hg0'
    have hb0 : 0 < b := by
      have h1 := hg0'.1
      have h2 := hg0'.2.1
      linarith
    have hbd := log_half_lo hb0 hg0'.2.2.1 hpos hb
    have hE := box_error R (k := -1) (K := 1) (by simp) (by norm_num) hg0 he0 ha hb hc1 hc2
    have hneg : ((-1 : ℤ) : ℝ) * log 2 + log m' ≤ 0 := by push_cast; linarith
    refine tau_sound (L := negLogLo (b / 2)) ?_ hsz hE (h32 _) (h32 _)
    rw [abs_of_nonpos hneg]; push_cast; linarith
  · -- Any other `k`: `|t| ≥ log 2 ≥ 1/2`, with room to spare.
    have hK : |(k : ℝ)| ≤ ((150 : ℚ) : ℝ) := by
      have hk1' : (-149 : ℝ) ≤ k := by exact_mod_cast hk1
      have hk2' : (k : ℝ) ≤ 127 := by exact_mod_cast hk2
      push_cast
      rw [abs_le]; constructor <;> linarith
    have hE := box_error R (K := 150) hK (by norm_num) hg heO hm1 hm2 hc1 hc2
    rw [Rat.cast_mul, pow2_cast] at hE
    have ht : (2 : ℝ) ^ (-1 : ℤ) ≤ |(k : ℝ) * log 2 + log m| := by
      have h12 : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by norm_num
      rw [h12]
      rcases lt_or_gt_of_ne hk0 with hneg | hpos
      · -- `k ≤ -2`
        have hk2' : k ≤ -2 := by omega
        have hk3 : (k : ℝ) ≤ -2 := by exact_mod_cast hk2'
        have : (k : ℝ) * log 2 ≤ -2 * log 2 :=
          mul_le_mul_of_nonneg_right hk3 (by linarith)
        have hnp : (k : ℝ) * log 2 + log m ≤ 0 := by linarith
        rw [abs_of_nonpos hnp]; linarith
      · have hk3 : (1 : ℝ) ≤ k := by exact_mod_cast hpos
        have : 1 * log 2 ≤ (k : ℝ) * log 2 :=
          mul_le_mul_of_nonneg_right hk3 (by linarith)
        have hnn : 0 ≤ (k : ℝ) * log 2 + log m := by linarith
        rw [abs_of_nonneg hnn]; linarith
    refine accept (E := -1) (by norm_num) ht hsize ?_ (h32 _) (h32 _)
    rw [show (-1 : ℤ) - 24 = -25 by norm_num]
    simpa using hE

end FrecpeLogf
