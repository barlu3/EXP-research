import FrecpeLogf.Model

/-!
# Acceptance: from an error bound to "within one ulp"

If the kernel's double result `Y` is close enough to the true `t = log x`,
then the binary32 it rounds to is the correctly rounded result or a neighbour
of it. Write `q = 2^(E-24)` for a binade `[2^E, 2^(E+1))`: its spacing is
`2q`, and the spacing just below `2^E` is `q`.

* `accept_inner`: inside a binade, `|Y - t| < 2q` suffices.
* `accept_pos`, `accept`: anywhere, `|Y - t| < 3q/2` suffices. Less is
  allowed because when `t` rounds to `2^E` itself, `Y` can land on the finer
  grid below it.
-/

namespace FrecpeLogf

section Grid

variable {E : ℤ}

lemma spacing_eq (E : ℤ) : (2 : ℝ) ^ (E - 23) = 2 * 2 ^ (E - 24) := by
  rw [show E - 23 = (E - 24) + 1 by ring, zpow_add₀ two_ne_zero]; ring

lemma bot_eq (E : ℤ) : (2 : ℝ) ^ E = 2 ^ 24 * 2 ^ (E - 24) := by
  rw [zpow_sub₀ two_ne_zero]; norm_num; field_simp

lemma top_eq (E : ℤ) : (2 : ℝ) ^ (E + 1) = 2 ^ 25 * 2 ^ (E - 24) := by
  rw [zpow_add₀ two_ne_zero, bot_eq E]; norm_num; ring

lemma half_eq (E : ℤ) : (2 : ℝ) ^ (E - 1) = 2 ^ 23 * 2 ^ (E - 24) := by
  rw [zpow_sub₀ two_ne_zero, bot_eq E]; norm_num; ring

/-- A grid point of the binade within `q` of any `x` in it. -/
lemma near_grid (hE1 : -126 ≤ E) (hE2 : E ≤ 103) {x : ℝ}
    (h1 : (2 : ℝ) ^ E ≤ x) (h2 : x ≤ 2 ^ (E + 1)) :
    ∃ g, IsF32 g ∧ |g - x| ≤ 2 ^ (E - 24) := by
  have hq := two_zpow_pos (E - 24)
  have hs : (0 : ℝ) < 2 ^ (E - 23) := two_zpow_pos _
  have hsq := spacing_eq E
  set n := round (x / 2 ^ (E - 23))
  have hr := abs_sub_round (x / 2 ^ (E - 23))
  have hx1 : (2 : ℝ) ^ 23 ≤ x / 2 ^ (E - 23) := by
    rw [le_div_iff₀ hs]; rw [bot_eq] at h1; nlinarith
  have hx2 : x / 2 ^ (E - 23) ≤ 2 ^ 24 := by
    rw [div_le_iff₀ hs]; rw [top_eq] at h2; nlinarith
  rw [abs_le] at hr
  refine ⟨n * 2 ^ (E - 23), isF32_of_mul ?_ hE1 hE2, ?_⟩
  · have h3 : (n : ℝ) < 2 ^ 24 + 1 := by linarith [hr.1]
    have h4 : (0 : ℝ) < n := by linarith [hr.2]
    have h3' : n < 2 ^ 24 + 1 := by exact_mod_cast h3
    have h4' : 0 < n := by exact_mod_cast h4
    rw [abs_of_pos h4']; omega
  · have : n * (2 : ℝ) ^ (E - 23) - x = -((x / 2 ^ (E - 23) - n) * 2 ^ (E - 23)) := by
      field_simp; ring
    rw [this, abs_neg, abs_mul, abs_of_pos hs]
    calc |x / 2 ^ (E - 23) - n| * 2 ^ (E - 23) ≤ 1 / 2 * 2 ^ (E - 23) := by
          gcongr; exact abs_sub_round _
      _ = 2 ^ (E - 24) := by rw [hsq]; ring

/-- A nearest value is on the same side of any binary32 as its argument. -/
lemma Nearest32.ge_of_le {X z F : ℝ} (hz : Nearest32 X z) (hF : IsF32 F) (h : F ≤ X) :
    F ≤ z := by
  by_contra! hlt
  have := hz.le hF
  rw [abs_of_nonpos (by linarith : z - X ≤ 0), abs_of_nonpos (by linarith : F - X ≤ 0)] at this
  linarith

lemma Nearest32.le_of_ge {X z F : ℝ} (hz : Nearest32 X z) (hF : IsF32 F) (h : X ≤ F) :
    z ≤ F := by
  by_contra! hlt
  have := hz.le hF
  rw [abs_of_nonneg (by linarith : 0 ≤ z - X), abs_of_nonneg (by linarith : 0 ≤ F - X)] at this
  linarith

/-- Two multiples of the binade's spacing, less than two steps apart and both at
least `2^E`, have no binary32 between them. -/
lemma adjacent_of_close {y a : ℝ} (hy : IsF32 y) (ha : IsF32 a)
    (hyP : (2 : ℝ) ^ E ≤ y) (haP : (2 : ℝ) ^ E ≤ a) (hya : |y - a| < 2 * 2 ^ (E - 23)) :
    WithinOneUlp y a := by
  have hs : (0 : ℝ) < 2 ^ (E - 23) := two_zpow_pos _
  have hP := two_zpow_pos E
  obtain ⟨ny, rfl⟩ := hy.multiple (E := E) (by rw [abs_of_pos (by linarith)]; exact hyP)
  obtain ⟨na, rfl⟩ := ha.multiple (E := E) (by rw [abs_of_pos (by linarith)]; exact haP)
  have hd : |(ny : ℝ) - na| < 2 := by
    rw [← sub_mul, abs_mul, abs_of_pos hs] at hya
    exact lt_of_mul_lt_mul_right (by linarith) hs.le
  rw [abs_lt] at hd
  have hd1 : ny - na < 2 := by exact_mod_cast hd.2
  have hd2 : -2 < ny - na := by exact_mod_cast hd.1
  intro f hf ⟨h1, h2⟩
  have hmin := le_min hyP haP
  have hfE : (2 : ℝ) ^ E ≤ f := by linarith
  have hfP : (2 : ℝ) ^ E ≤ |f| := by rw [abs_of_pos (by linarith)]; exact hfE
  obtain ⟨nf, rfl⟩ := hf.multiple hfP
  have key : ∀ i j : ℤ, (i : ℝ) * 2 ^ (E - 23) < j * 2 ^ (E - 23) → i < j := fun i j h => by
    exact_mod_cast lt_of_mul_lt_mul_right h hs.le
  rcases le_total ny na with h | h
  · rw [min_eq_left (by gcongr)] at h1
    rw [max_eq_right (by gcongr)] at h2
    have := key _ _ h1; have := key _ _ h2; omega
  · rw [min_eq_right (by gcongr)] at h1
    rw [max_eq_left (by gcongr)] at h2
    have := key _ _ h1; have := key _ _ h2; omega

/-- A nearest value of `X` in the binade `[2^E, 2^(E+1)]` stays in it, within `q`. -/
lemma nearest_in_binade (hE1 : -126 ≤ E) (hE2 : E ≤ 103) {X z : ℝ}
    (h1 : (2 : ℝ) ^ E ≤ X) (h2 : X ≤ 2 ^ (E + 1)) (hz : Nearest32 X z) :
    2 ^ E ≤ z ∧ z ≤ 2 ^ (E + 1) ∧ |z - X| ≤ 2 ^ (E - 24) := by
  obtain ⟨g, hg, hgX⟩ := near_grid hE1 hE2 h1 h2
  exact ⟨hz.ge_of_le (IsF32.two_zpow (by omega) (by omega)) h1,
    hz.le_of_ge (IsF32.two_zpow (by omega) (by omega)) h2, (hz.le hg).trans hgX⟩

end Grid

/-- A spacing above the bottom of a binade, an error below one spacing is
accepted. Only the bottom needs that margin: below `2^E` the grid is finer, while
crossing the top lands on a coarser one, where `y` can only be `2^(E+1)`. -/
theorem accept_inner {E : ℤ} (hE1 : -126 ≤ E) (hE2 : E ≤ 102) {t Y y a : ℝ}
    (ht1 : (2 : ℝ) ^ E + 2 ^ (E - 23) ≤ t) (ht2 : t < 2 ^ (E + 1))
    (hY : |Y - t| < 2 ^ (E - 23)) (hy : Nearest32 Y y) (ha : Nearest32 t a) :
    WithinOneUlp y a := by
  have hsq := spacing_eq E
  have hP2 := top_eq E
  have hP := bot_eq E
  have hq := two_zpow_pos (E - 24)
  rw [abs_lt] at hY
  obtain ⟨ha1, -, ha3⟩ := nearest_in_binade hE1 (by omega) (by linarith) ht2.le ha
  rw [abs_le] at ha3
  rcases le_or_gt Y (2 ^ (E + 1)) with hY2 | hY2
  · obtain ⟨hy1, -, hy3⟩ := nearest_in_binade hE1 (by omega) (by linarith) hY2 hy
    refine adjacent_of_close hy.1 ha.1 hy1 ha1 ?_
    rw [abs_le] at hy3; rw [abs_lt]; constructor <;> linarith
  · -- `Y` above the top: the grid there is `4q`, so `y` is `2^(E+1)` exactly.
    have fP2 : IsF32 ((2 : ℝ) ^ (E + 1)) := IsF32.two_zpow (by omega) (by omega)
    have hy1 : (2 : ℝ) ^ (E + 1) ≤ y := hy.ge_of_le fP2 hY2.le
    have hyY := hy.le fP2
    rw [abs_of_neg (by linarith : (2 : ℝ) ^ (E + 1) - Y < 0)] at hyY
    have hyU : y < 2 ^ (E + 1) + 4 * 2 ^ (E - 24) := by
      rcases le_or_gt y Y with h | h
      · linarith
      · rw [abs_of_pos (by linarith)] at hyY; linarith
    obtain ⟨n, hn⟩ := hy.1.multiple (E := E + 1) (by rw [abs_of_pos (by linarith)]; exact hy1)
    have hz : (2 : ℝ) ^ (E + 1 - 23) = 4 * 2 ^ (E - 24) := by
      rw [show E + 1 - 23 = (E - 24) + 2 by ring, zpow_add₀ two_ne_zero]; norm_num; ring
    rw [hz] at hn
    have h1 : (2 : ℝ) ^ 23 ≤ n := by rw [hn, hP2] at hy1; nlinarith
    have h2 : (n : ℝ) < 2 ^ 23 + 1 := by rw [hn, hP2] at hyU; nlinarith
    have h1' : (2 : ℤ) ^ 23 ≤ n := by exact_mod_cast h1
    have h2' : n < (2 : ℤ) ^ 23 + 1 := by exact_mod_cast h2
    have hn' : n = 2 ^ 23 := by omega
    have hyP : y = 2 ^ (E + 1) := by rw [hn, hn', hP2]; push_cast; ring
    refine adjacent_of_close hy.1 ha.1 ?_ ha1 ?_
    · rw [hyP, hP2, hP]; nlinarith
    · rw [hyP, abs_lt]; constructor <;> linarith

/-- Anywhere in a binade, an error below `3q/2` is accepted. -/
theorem accept_pos {E : ℤ} (hE1 : -100 ≤ E) (hE2 : E ≤ 100) {t Y y a : ℝ}
    (ht1 : (2 : ℝ) ^ E ≤ t) (ht2 : t < 2 ^ (E + 1))
    (hY : |Y - t| < 3 / 2 * 2 ^ (E - 24)) (hy : Nearest32 Y y) (ha : Nearest32 t a) :
    WithinOneUlp y a := by
  have hsq := spacing_eq E
  have hP := bot_eq E
  have hP2 := top_eq E
  have hq := two_zpow_pos (E - 24)
  set q := (2 : ℝ) ^ (E - 24) with hqdef
  rw [abs_lt] at hY
  have fP : IsF32 ((2 : ℝ) ^ E) := IsF32.two_zpow (by omega) (by omega)
  have fP2 : IsF32 ((2 : ℝ) ^ (E + 1)) := IsF32.two_zpow (by omega) (by omega)
  obtain ⟨ha1, ha2, ha3⟩ := nearest_in_binade (by omega) (by omega) ht1 ht2.le ha
  rw [abs_le] at ha3
  rcases le_or_gt (2 ^ E) Y with hYP | hYP
  · -- `Y` at or above `2^E`: `y` is a multiple of the spacing, at most `3q/2` away.
    have hy1 : (2 : ℝ) ^ E ≤ y := hy.ge_of_le fP hYP
    have hy3 : |y - Y| ≤ 3 / 2 * q := by
      rcases le_or_gt Y (2 ^ (E + 1)) with hY2 | hY2
      · exact (nearest_in_binade (by omega) (by omega) hYP hY2 hy).2.2.trans (by linarith)
      · refine (hy.le fP2).trans ?_
        rw [abs_of_neg (by linarith)]; linarith
    refine adjacent_of_close hy.1 ha.1 hy1 ha1 ?_
    rw [abs_le] at hy3; rw [abs_lt]; constructor <;> linarith
  · -- `Y` below `2^E`: then `t` is within `3q/2` of `2^E`, and `a` is `2^E` or the next.
    have fPm : IsF32 ((2 : ℝ) ^ E - q) := by
      have := isF32_of_mul_low (n := 2 ^ 24 - 1) (E := E) (by norm_num) (by omega) (by omega)
      convert this using 1; rw [hP]; push_cast; ring
    -- `y` is within `3q/2` of `Y`, so in `(2^E - 3q, 2^E + 3q/2)`.
    have hyY : |y - Y| < 3 / 2 * q := by
      refine (hy.le fP).trans_lt ?_; rw [abs_of_pos (by linarith)]; linarith
    rw [abs_lt] at hyY
    -- `y` is not above `2^E`: the next binary32 up is `2^E + 2q`.
    have hyle : y ≤ 2 ^ E := by
      by_contra! hgt
      obtain ⟨n, hn⟩ := hy.1.multiple (E := E) (by rw [abs_of_pos (by linarith)]; exact hgt.le)
      have h1 : (2 : ℝ) ^ 23 < n := by
        rw [hn, hP, hsq] at hgt; nlinarith
      have h1' : (2 : ℤ) ^ 23 + 1 ≤ n := by
        have : (2 : ℤ) ^ 23 < n := by exact_mod_cast h1
        omega
      have h1'' : (2 : ℝ) ^ 23 + 1 ≤ n := by exact_mod_cast h1'
      have : (2 : ℝ) ^ E + 2 * q ≤ y := by rw [hn, hP, hsq]; nlinarith
      linarith
    -- `y` is a multiple of `q`, and not `2^E - 2q` or lower: `2^E - q` is nearer.
    have hyge : (2 : ℝ) ^ E - q ≤ y := by
      have hyhalf : (2 : ℝ) ^ (E - 1) ≤ |y| := by
        rw [abs_of_pos (by linarith), half_eq]; linarith
      obtain ⟨n, hn⟩ := hy.1.multiple hyhalf
      rw [show E - 1 - 23 = E - 24 by ring] at hn
      by_contra! hlt
      have h1 : (n : ℝ) < 2 ^ 24 - 1 := by
        rw [hn, hP] at hlt; nlinarith
      have h1' : n ≤ 2 ^ 24 - 2 := by
        have : n < (2 : ℤ) ^ 24 - 1 := by exact_mod_cast h1
        omega
      have h1'' : (n : ℝ) ≤ 2 ^ 24 - 2 := by exact_mod_cast h1'
      have hy2 : y ≤ 2 ^ E - 2 * q := by rw [hn, hP]; nlinarith
      have := hy.le fPm
      rw [abs_of_neg (by linarith)] at this
      rcases le_or_gt 0 (2 ^ E - q - Y) with hd | hd
      · rw [abs_of_nonneg hd] at this; linarith
      · rw [abs_of_neg hd] at this; linarith
    -- Every binary32 strictly between `2^E - q` and `2^E` is a multiple of `q`: none.
    have no_low : ∀ f, IsF32 f → (2 : ℝ) ^ E - q < f → f < 2 ^ E → False := by
      intro f hf h1 h2
      have hfhalf : (2 : ℝ) ^ (E - 1) ≤ |f| := by
        rw [abs_of_pos (by linarith), half_eq]; linarith
      obtain ⟨n, hn⟩ := hf.multiple hfhalf
      rw [show E - 1 - 23 = E - 24 by ring] at hn
      rw [hn, hP] at h1 h2
      have h3 : (2 : ℝ) ^ 24 - 1 < n := by nlinarith
      have h4 : (n : ℝ) < 2 ^ 24 := by nlinarith
      have h3' : (2 : ℤ) ^ 24 - 1 < n := by exact_mod_cast h3
      have h4' : n < (2 : ℤ) ^ 24 := by exact_mod_cast h4
      omega
    -- `a` is `2^E` or `2^E + 2q`.
    obtain ⟨na, hna⟩ := ha.1.multiple (E := E) (by rw [abs_of_pos (by linarith)]; exact ha1)
    have hna1 : (2 : ℤ) ^ 23 ≤ na := by
      have : (2 : ℝ) ^ 23 ≤ na := by rw [hna, hP, hsq] at ha1; nlinarith
      exact_mod_cast this
    have ht' : t < 2 ^ E + 3 / 2 * q := by linarith [hY.1]
    have hna2 : na ≤ 2 ^ 23 + 1 := by
      have h5 : (na : ℝ) * (2 * q) ≤ t + q := by rw [hna, hsq] at ha3; linarith [ha3.2, hqdef]
      have : (na : ℝ) < 2 ^ 23 + 2 := by rw [hP] at ht'; nlinarith
      have : na < (2 : ℤ) ^ 23 + 2 := by exact_mod_cast this
      omega
    rcases (show na = 2 ^ 23 ∨ na = 2 ^ 23 + 1 by omega) with h | h
    · -- `a = 2^E`: `y` is `2^E - q` or `2^E`.
      have haP : a = 2 ^ E := by rw [hna, h, hP, hsq]; push_cast; ring
      intro f hf ⟨h1, h2⟩
      rw [haP] at h1 h2
      rw [min_eq_left hyle] at h1; rw [max_eq_right hyle] at h2
      exact no_low f hf (by linarith) h2
    · -- `a = 2^E + 2q`: then `t ≥ 2^E + q`, so `y = 2^E`.
      have haP : a = 2 ^ E + 2 * q := by rw [hna, h, hP, hsq]; push_cast; ring
      have ht3 : (2 : ℝ) ^ E + q ≤ t := by
        have := ha.le fP
        rw [haP, abs_of_pos (by linarith : (0 : ℝ) < 2 ^ E + 2 * q - t),
          abs_of_nonpos (by linarith : (2 : ℝ) ^ E - t ≤ 0)] at this
        linarith
      have hyP : y = 2 ^ E := by
        rcases lt_or_eq_of_le hyle with hlt | heq
        · exfalso
          rcases lt_or_eq_of_le hyge with hlt' | heq'
          · exact no_low y hy.1 hlt' hlt
          · -- `y = 2^E - q` is farther from `Y` than `2^E` is.
            have hyY' := hy.le fP
            rw [← heq', abs_of_neg (by linarith), abs_of_pos (by linarith)] at hyY'
            linarith
        · exact heq
      intro f hf ⟨h1, h2⟩
      rw [hyP, haP] at h1 h2
      rw [min_eq_left (by linarith)] at h1; rw [max_eq_right (by linarith)] at h2
      obtain ⟨n, hn⟩ := hf.multiple (E := E) (by rw [abs_of_pos (by linarith)]; linarith)
      rw [hn, hP, hsq] at h1 h2
      have h3 : (2 : ℝ) ^ 23 < n := by nlinarith
      have h4 : (n : ℝ) < 2 ^ 23 + 1 := by nlinarith
      have h3' : (2 : ℤ) ^ 23 < n := by exact_mod_cast h3
      have h4' : n < (2 : ℤ) ^ 23 + 1 := by exact_mod_cast h4
      omega

/-- The signed form: `|t| ≥ 2^E` and `|Y - t| < 3q/2` give one ulp, in any
binade at or above `E`. -/
theorem accept {E : ℤ} (hE1 : -100 ≤ E) {t Y y a : ℝ}
    (ht1 : (2 : ℝ) ^ E ≤ |t|) (ht2 : |t| < 2 ^ 100)
    (hY : |Y - t| < 3 / 2 * 2 ^ (E - 24)) (hy : Nearest32 Y y) (ha : Nearest32 t a) :
    WithinOneUlp y a := by
  have htpos : 0 < |t| := lt_of_lt_of_le (two_zpow_pos E) ht1
  -- The binade of `|t|`, at or above `E`.
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
  have hq : (2 : ℝ) ^ (E - 24) ≤ 2 ^ (E' - 24) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have hY' : |Y - t| < 3 / 2 * 2 ^ (E' - 24) := by linarith
  rcases le_or_gt 0 t with ht | ht
  · rw [abs_of_nonneg ht] at h1 h2
    exact accept_pos (by omega) hE'2 h1 h2 hY' hy ha
  · rw [abs_of_neg ht] at h1 h2
    apply WithinOneUlp.of_neg
    refine accept_pos (by omega) hE'2 h1 h2 ?_ hy.neg ha.neg
    rw [show -Y - -t = -(Y - t) by ring, abs_neg]; exact hY'

end FrecpeLogf
