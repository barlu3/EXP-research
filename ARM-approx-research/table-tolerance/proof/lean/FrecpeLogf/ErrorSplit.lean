import FrecpeLogf.Model

/-!
# The kernel and its forward error

`kernel` is `frecpe_logf` (`replicas.c`) instruction for instruction, as Apple
clang 21 compiles it with `-O2 -ffp-contract=fast`: `r = (m·c) - 1` as a
multiply then a subtract, then five fused or plain operations. Every
instruction rounds once.

`kernel_error` splits its distance from `log x = k·log 2 + log m` into the
polynomial's own error at the exact `r* = m·c - 1`, `|P(r*) - log(m·c)|`, plus
a rounding term `ρ` that is a few units of `2^-53`: the polynomial's error is
the only part that depends on how far the reciprocal `c` has drifted.
-/

namespace FrecpeLogf

open Real

/-- glibc 2.44's `logf` constants (`e_logf_data.c`), exactly: `ln2d` is
`0x1.62e42fefa39efp-1`; `A0`, `A1`, `A2` are `poly[0..2]`. -/
noncomputable def ln2d : ℝ := 6243314768165359 / 2 ^ 53
noncomputable def A0 : ℝ := -4519694089880372 / 2 ^ 54
noncomputable def A1 : ℝ := 6007023063796586 / 2 ^ 54
noncomputable def A2 : ℝ := -9007194725564707 / 2 ^ 54

/-- The polynomial glibc's `logf` evaluates for `log(1 + r)`. -/
noncomputable def P (r : ℝ) : ℝ := r + A2 * r ^ 2 + A1 * r ^ 3 + A0 * r ^ 4

/-- `frecpe_logf`, one `rn` per instruction (`fmul`, `fsub`, `fmadd`, `fmul`,
`fmadd`, `fmadd`, `fadd`, `fmadd`). `m` is the significand, `k` the exponent,
`c` the reciprocal and `lg` its log entry. -/
noncomputable def kernel (rn : ℝ → ℝ) (k : ℤ) (m c lg : ℝ) : ℝ :=
  let r := rn (rn (m * c) - 1)
  let y0 := rn (k * ln2d + lg)
  let r2 := rn (r * r)
  let y1 := rn (A1 * r + A2)
  let y2 := rn (A0 * r2 + y1)
  let s := rn (y0 + r)
  rn (y2 * r2 + s)

lemma A0_abs : |A0| ≤ 251 / 1000 := by unfold A0; rw [abs_le]; constructor <;> norm_num
lemma A1_abs : |A1| ≤ 334 / 1000 := by unfold A1; rw [abs_le]; constructor <;> norm_num
lemma A2_abs : |A2| ≤ 1 / 2 := by unfold A2; rw [abs_le]; constructor <;> norm_num
lemma ln2d_pos : 0 < ln2d := by unfold ln2d; norm_num
lemma ln2d_le : ln2d ≤ 7 / 10 := by unfold ln2d; norm_num

lemma u64_pos : 0 < u64 := by unfold u64; norm_num
lemma u64_le : u64 ≤ 1 / 10 ^ 15 := by unfold u64; norm_num
lemma eta64_pos : 0 < eta64 := by unfold eta64; norm_num
/-- `2^-1075` is far below a hundredth of `2^-53`; `2^1075` is past Lean's
default `exponentiation.threshold`, so it is raised for this one lemma. -/
lemma eta64_le : eta64 ≤ u64 / 100 := by
  unfold eta64 u64
  set_option exponentiation.threshold 1100 in
  rw [div_div, one_div_le_one_div (by positivity) (by positivity)]; norm_num

/-- The rounding bound with the argument's size bounded by `B`. -/
lemma RN64.err_le (R : RN64) {z B : ℝ} (hz : |z| ≤ B) (hB : B ≤ 1024) :
    |R.rn z - z| ≤ u64 * B + eta64 := by
  have := u64_pos
  exact (R.err z (hz.trans hB)).trans (by gcongr)

/-- `|x| ≤ |y| + |x - y|`, the step every size bound below takes. -/
lemma abs_le_of_sub {x y B d : ℝ} (hy : |y| ≤ B) (hxy : |x - y| ≤ d) : |x| ≤ B + d := by
  calc |x| = |y + (x - y)| := by ring_nf
    _ ≤ |y| + |x - y| := abs_add_le _ _
    _ ≤ B + d := add_le_add hy hxy

lemma u64_mul_le {B : ℝ} (hB0 : 0 ≤ B) (hB : B ≤ 110) : u64 * B ≤ 1 / 100 := by
  have h1 := u64_pos; have h2 := u64_le
  calc u64 * B ≤ (1 / 10 ^ 15) * 110 := mul_le_mul h2 hB hB0 (by norm_num)
    _ ≤ 1 / 100 := by norm_num

lemma abs_mul_self_le {r : ℝ} (hr : |r| ≤ 1 / 15) : |r| * |r| ≤ 1 / 225 := by
  calc |r| * |r| ≤ (1 / 15) * (1 / 15) := mul_le_mul hr hr (abs_nonneg r) (by norm_num)
    _ = 1 / 225 := by norm_num

/-- `log x = k·log 2 + log m` for `x = 2^k·m`. -/
lemma log_scaled (k : ℤ) {m : ℝ} (hm : 0 < m) : log ((2 : ℝ) ^ k * m) = k * log 2 + log m := by
  rw [Real.log_mul (zpow_pos (by norm_num) k).ne' hm.ne', Real.log_zpow]

/-- `P(r) - P(r*)` is at most `11/10 · |r - r*|` for `|r|, |r*| ≤ 1/15`. -/
lemma P_lipschitz {r s : ℝ} (hr : |r| ≤ 1 / 15) (hs : |s| ≤ 1 / 15) :
    |P r - P s| ≤ 11 / 10 * |r - s| := by
  have hA0 := A0_abs; have hA1 := A1_abs; have hA2 := A2_abs
  have hfac : P r - P s = (r - s) *
      (1 + A2 * (r + s) + A1 * (r ^ 2 + r * s + s ^ 2) + A0 * ((r + s) * (r ^ 2 + s ^ 2))) := by
    unfold P; ring
  rw [hfac, abs_mul, mul_comm]
  gcongr
  -- the factor is within `1/10` of `1`
  have h1 : |r + s| ≤ 2 / 15 := (abs_add_le _ _).trans (by linarith)
  have h2 : |r ^ 2 + r * s + s ^ 2| ≤ 3 / 225 := by
    have : |r| * |r| ≤ 1 / 225 := by nlinarith [abs_nonneg r]
    have : |r| * |s| ≤ 1 / 225 := by nlinarith [abs_nonneg r, abs_nonneg s]
    have : |s| * |s| ≤ 1 / 225 := by nlinarith [abs_nonneg s]
    calc |r ^ 2 + r * s + s ^ 2| ≤ |r ^ 2| + |r * s| + |s ^ 2| :=
          (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ = |r| * |r| + |r| * |s| + |s| * |s| := by rw [abs_pow, abs_mul, abs_pow]; ring
      _ ≤ 3 / 225 := by linarith
  have h3 : |(r + s) * (r ^ 2 + s ^ 2)| ≤ 2 / 15 * (2 / 225) := by
    rw [abs_mul]
    gcongr
    calc |r ^ 2 + s ^ 2| ≤ |r ^ 2| + |s ^ 2| := abs_add_le _ _
      _ = |r| * |r| + |s| * |s| := by rw [abs_pow, abs_pow]; ring
      _ ≤ 2 / 225 := by nlinarith [abs_nonneg r, abs_nonneg s]
  have e2 : |A2 * (r + s)| ≤ 1 / 2 * (2 / 15) := by rw [abs_mul]; gcongr
  have e3 : |A1 * (r ^ 2 + r * s + s ^ 2)| ≤ 334 / 1000 * (3 / 225) := by rw [abs_mul]; gcongr
  have e4 : |A0 * ((r + s) * (r ^ 2 + s ^ 2))| ≤ 251 / 1000 * (2 / 15 * (2 / 225)) := by
    rw [abs_mul]; gcongr
  calc |1 + A2 * (r + s) + A1 * (r ^ 2 + r * s + s ^ 2) + A0 * ((r + s) * (r ^ 2 + s ^ 2))|
      ≤ |(1 : ℝ)| + |A2 * (r + s)| + |A1 * (r ^ 2 + r * s + s ^ 2)| +
          |A0 * ((r + s) * (r ^ 2 + s ^ 2))| := by
        refine (abs_add_le _ _).trans ?_
        gcongr
        refine (abs_add_le _ _).trans ?_
        gcongr
        exact abs_add_le _ _
    _ ≤ 11 / 10 := by rw [abs_one]; linarith

set_option maxHeartbeats 1000000 in
/-- The kernel's distance from `log x` is the polynomial's error at the exact
`r* = m·c - 1`, plus rounding: `(10 + 3K)` units of `2^-53`, the underflow
floor, and `K·|ln2d - log 2|` from the rounded `ln 2`. -/
theorem kernel_error (R : RN64) {k : ℤ} {K m c : ℝ} (hk : |(k : ℝ)| ≤ K) (hK : K ≤ 150)
    (hm1 : 1 ≤ m) (hm2 : m ≤ 2) (hc1 : 9 / 20 ≤ c) (hc2 : c ≤ 11 / 10)
    (hr : |m * c - 1| ≤ 1 / 16) :
    |kernel R.rn k m c (R.rn (-log c)) - (k * log 2 + log m)| ≤
      |P (m * c - 1) - log (m * c)| +
        ((10 + 3 * K) * u64 + 16 * eta64 + K * |ln2d - log 2|) := by
  have hu0 := u64_pos; have hu1 := u64_le; have he0 := eta64_pos; have he1 := eta64_le
  have hA0 := A0_abs; have hA1 := A1_abs; have hA2 := A2_abs
  have hl0 := ln2d_pos; have hl1 := ln2d_le
  have hK0 : 0 ≤ K := (abs_nonneg _).trans hk
  have hcpos : 0 < c := by linarith
  have hmpos : 0 < m := by linarith
  have hlogc : |log c| ≤ 5 / 4 := by
    rw [abs_le]; constructor
    · have h := Real.one_sub_inv_le_log_of_pos hcpos
      have : c⁻¹ ≤ 20 / 9 := by rw [inv_le_comm₀ hcpos (by norm_num)]; linarith
      linarith
    · have := Real.log_le_sub_one_of_pos hcpos; linarith
  have hlogmc : log (m * c) = log m + log c := Real.log_mul hmpos.ne' hcpos.ne'
  simp only [kernel]
  -- r = rn(rn(m·c) - 1), within `3u` of `r* = m·c - 1`
  have hz1 : |m * c| ≤ 11 / 5 := by
    have : m * c ≤ 2 * (11 / 10) := mul_le_mul hm2 hc2 hcpos.le (by norm_num)
    rw [abs_of_pos (mul_pos hmpos hcpos)]; linarith
  have t1 := R.err_le hz1 (by norm_num)
  set r1 := R.rn (m * c)
  have hz2 : |r1 - 1| ≤ 1 / 15 := by
    have := abs_le_of_sub hr (show |(r1 - 1) - (m * c - 1)| ≤ u64 * (11 / 5) + eta64 by
      ring_nf; ring_nf at t1; exact t1)
    linarith
  have t2 := R.err_le hz2 (by norm_num)
  set r := R.rn (r1 - 1)
  have hdr : |r - (m * c - 1)| ≤ 3 * u64 := by
    calc |r - (m * c - 1)| = |(r - (r1 - 1)) + (r1 - m * c)| := by ring_nf
      _ ≤ |r - (r1 - 1)| + |r1 - m * c| := abs_add_le _ _
      _ ≤ (u64 * (1 / 15) + eta64) + (u64 * (11 / 5) + eta64) := add_le_add t2 t1
      _ ≤ 3 * u64 := by linarith
  have hr' : |r| ≤ 1 / 15 := by have := abs_le_of_sub hr hdr; linarith
  have hrs : |m * c - 1| ≤ 1 / 15 := by linarith
  -- lg = rn(-log c)
  have t0 := R.err_le (show |-log c| ≤ 5 / 4 by rwa [abs_neg]) (by norm_num)
  set lg := R.rn (-log c)
  have hlg : |lg| ≤ 13 / 10 := by
    have := abs_le_of_sub (show |-log c| ≤ 5 / 4 by rwa [abs_neg]) t0; linarith
  -- y0 = rn(k·ln2d + lg)
  have hkl : |(k : ℝ) * ln2d| ≤ K * (7 / 10) := by
    rw [abs_mul, abs_of_pos hl0]; gcongr
  have hz6 : |(k : ℝ) * ln2d + lg| ≤ 7 / 10 * K + 13 / 10 := by
    have := abs_add_le ((k : ℝ) * ln2d) lg; linarith
  have t6 := R.err_le hz6 (by linarith)
  set y0 := R.rn ((k : ℝ) * ln2d + lg)
  have hy0 : |y0| ≤ 7 / 10 * K + 14 / 10 := by
    have h1 := abs_le_of_sub hz6 t6
    have h2 := u64_mul_le (B := 7 / 10 * K + 13 / 10) (by linarith) (by linarith)
    linarith only [h1, h2, he1, hu1, hu0]
  -- r2 = rn(r·r)
  have hz3 : |r * r| ≤ 1 / 225 := by
    rw [abs_mul]; exact abs_mul_self_le hr'
  have t3 := R.err_le hz3 (by norm_num)
  set r2 := R.rn (r * r)
  have hr2 : |r2| ≤ 1 / 200 := by have := abs_le_of_sub hz3 t3; linarith
  -- y1 = rn(A1·r + A2)
  have hz4 : |A1 * r + A2| ≤ 53 / 100 := by
    have : |A1 * r| ≤ 334 / 1000 * (1 / 15) := by rw [abs_mul]; gcongr
    have := abs_add_le (A1 * r) A2; linarith
  have t4 := R.err_le hz4 (by norm_num)
  set y1 := R.rn (A1 * r + A2)
  have hy1 : |y1| ≤ 54 / 100 := by have := abs_le_of_sub hz4 t4; linarith
  -- y2 = rn(A0·r2 + y1)
  have hz5 : |A0 * r2 + y1| ≤ 55 / 100 := by
    have : |A0 * r2| ≤ 251 / 1000 * (1 / 200) := by rw [abs_mul]; gcongr
    have := abs_add_le (A0 * r2) y1; linarith
  have t5 := R.err_le hz5 (by norm_num)
  set y2 := R.rn (A0 * r2 + y1)
  have hy2 : |y2| ≤ 56 / 100 := by have := abs_le_of_sub hz5 t5; linarith
  -- s = rn(y0 + r)
  have hz7 : |y0 + r| ≤ 7 / 10 * K + 15 / 10 := by
    have := abs_add_le y0 r; linarith
  have t7 := R.err_le hz7 (by linarith)
  set s := R.rn (y0 + r)
  have hs : |s| ≤ 7 / 10 * K + 16 / 10 := by
    have h1 := abs_le_of_sub hz7 t7
    have h2 := u64_mul_le (B := 7 / 10 * K + 15 / 10) (by linarith) (by linarith)
    linarith only [h1, h2, he1, hu1, hu0]
  -- Y = rn(y2·r2 + s)
  have hz8 : |y2 * r2 + s| ≤ 7 / 10 * K + 17 / 10 := by
    have : |y2 * r2| ≤ 56 / 100 * (1 / 200) := by rw [abs_mul]; gcongr
    have := abs_add_le (y2 * r2) s; linarith
  have t8 := R.err_le hz8 (by linarith)
  -- The split. `E1` collects the polynomial evaluation's roundings.
  have hsplit : R.rn (y2 * r2 + s) - (k * log 2 + log m) =
      (P (m * c - 1) - log (m * c)) + (P r - P (m * c - 1)) + k * (ln2d - log 2) +
        (lg - -log c) + (y0 - (k * ln2d + lg)) + (s - (y0 + r)) +
        (R.rn (y2 * r2 + s) - (y2 * r2 + s)) +
        ((y2 - (A0 * r ^ 2 + A1 * r + A2)) * r2 + (A0 * r ^ 2 + A1 * r + A2) * (r2 - r * r)) := by
    rw [hlogmc]; unfold P; ring
  have hE1 : |(y2 - (A0 * r ^ 2 + A1 * r + A2)) * r2 + (A0 * r ^ 2 + A1 * r + A2) * (r2 - r * r)|
      ≤ u64 / 50 + 2 * eta64 := by
    have hq : |A0 * r ^ 2 + A1 * r + A2| ≤ 54 / 100 := by
      have h1 : |A0 * r ^ 2| ≤ 251 / 1000 * (1 / 225) := by
        have h := abs_mul_self_le hr'
        rw [← sq] at h
        rw [abs_mul, abs_pow]
        exact mul_le_mul hA0 h (by positivity) (by norm_num)
      have h2 := abs_add_le (A0 * r ^ 2) (A1 * r + A2)
      have h3 : A0 * r ^ 2 + A1 * r + A2 = A0 * r ^ 2 + (A1 * r + A2) := by ring
      rw [h3]; linarith only [h1, h2, hz4]
    have hd : |y2 - (A0 * r ^ 2 + A1 * r + A2)| ≤ 2 * u64 + 3 * eta64 := by
      have : y2 - (A0 * r ^ 2 + A1 * r + A2) =
          (y2 - (A0 * r2 + y1)) + (y1 - (A1 * r + A2)) + A0 * (r2 - r * r) := by ring
      rw [this]
      have h1 : |A0 * (r2 - r * r)| ≤ 251 / 1000 * (u64 * (1 / 225) + eta64) := by
        rw [abs_mul]; exact mul_le_mul hA0 t3 (abs_nonneg _) (by norm_num)
      have h2 := abs_add_le (y2 - (A0 * r2 + y1) + (y1 - (A1 * r + A2))) (A0 * (r2 - r * r))
      have h3 := abs_add_le (y2 - (A0 * r2 + y1)) (y1 - (A1 * r + A2))
      linarith only [h1, h2, h3, t4, t5, hu0, he0]
    calc _ ≤ |(y2 - (A0 * r ^ 2 + A1 * r + A2)) * r2| + |(A0 * r ^ 2 + A1 * r + A2) * (r2 - r * r)| :=
          abs_add_le _ _
      _ = |y2 - (A0 * r ^ 2 + A1 * r + A2)| * |r2| + |A0 * r ^ 2 + A1 * r + A2| * |r2 - r * r| := by
          rw [abs_mul, abs_mul]
      _ ≤ (2 * u64 + 3 * eta64) * (1 / 200) + 54 / 100 * (u64 * (1 / 225) + eta64) :=
          add_le_add (mul_le_mul hd hr2 (abs_nonneg _) (by linarith only [hu0, he0]))
            (mul_le_mul hq t3 (abs_nonneg _) (by norm_num))
      _ ≤ u64 / 50 + 2 * eta64 := by linarith only [hu0, he0]
  have hP := P_lipschitz hr' hrs
  have hkd : |(k : ℝ) * (ln2d - log 2)| ≤ K * |ln2d - log 2| := by
    rw [abs_mul]; gcongr
  rw [hsplit]
  -- Nine terms, each bounded.
  have a1 := abs_add_le (P (m * c - 1) - log (m * c)) (P r - P (m * c - 1))
  have a2 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1))) (k * (ln2d - log 2))
  have a3 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1)) + k * (ln2d - log 2))
    (lg - -log c)
  have a4 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1)) + k * (ln2d - log 2) +
    (lg - -log c)) (y0 - (k * ln2d + lg))
  have a5 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1)) + k * (ln2d - log 2) +
    (lg - -log c) + (y0 - (k * ln2d + lg))) (s - (y0 + r))
  have a6 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1)) + k * (ln2d - log 2) +
    (lg - -log c) + (y0 - (k * ln2d + lg)) + (s - (y0 + r))) (R.rn (y2 * r2 + s) - (y2 * r2 + s))
  have a7 := abs_add_le (P (m * c - 1) - log (m * c) + (P r - P (m * c - 1)) + k * (ln2d - log 2) +
    (lg - -log c) + (y0 - (k * ln2d + lg)) + (s - (y0 + r)) + (R.rn (y2 * r2 + s) - (y2 * r2 + s)))
    ((y2 - (A0 * r ^ 2 + A1 * r + A2)) * r2 + (A0 * r ^ 2 + A1 * r + A2) * (r2 - r * r))
  have hPd : |P r - P (m * c - 1)| ≤ 33 / 10 * u64 := by
    have := mul_le_mul_of_nonneg_left hdr (by norm_num : (0 : ℝ) ≤ 11 / 10)
    linarith only [this, hP]
  have hKu := mul_nonneg hK0 hu0.le
  linarith only [a1, a2, a3, a4, a5, a6, a7, hPd, hkd, t0, t6, t7, t8, hE1, hKu, hu0, he0]

/-- `|ln2d - log 2| ≤ 2^-54`, from the series `log 2 = Σ 2^-(i+1)/(i+1)`. -/
lemma ln2d_err : |ln2d - log 2| ≤ 1 / 2 ^ 54 := by
  have h := Real.abs_log_sub_add_sum_range_le (x := 1 / 2) (by norm_num) 58
  have hl : log (1 - 1 / 2 : ℝ) = -log 2 := by
    rw [show (1 - 1 / 2 : ℝ) = 2⁻¹ by norm_num, Real.log_inv]
  rw [hl] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
  norm_num at h
  unfold ln2d
  rw [abs_le] at h ⊢
  constructor <;> linarith [h.1, h.2]

end FrecpeLogf
