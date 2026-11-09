import FrecpeLogf.ErrorSplit

/-!
# The polynomial's error, bounded by exact computation

`e(r) = P(r) - log(1 + r)` is the only part of the kernel's error that grows as
the reciprocal drifts. `errSup lo hi` is an exact rational bound on `|e|` over
`[lo, hi]`, and `err_sound` proves it.

Mathlib's series bound replaces `log(1 + r)` by a degree-9 polynomial, leaving
a polynomial `Q` plus a remainder of `|r|^10 / (1 - |r|)`. On `[-1/16, 1/16]`, `Q`
turns only three times away from `r = 0`, so it is monotone on five stretches,
each proved once by interval arithmetic on its derivative. On a monotone stretch
the worst `|Q|` over a range is at the range's ends, which `esup` evaluates
exactly; tiny boxes around the three turning points use interval arithmetic.

The bound involves no sampling and no search: every check here runs in Lean's
kernel by `decide +kernel`, on exact rationals.
-/

namespace FrecpeLogf

open Real

/-! ### Exact rational helpers, written to reduce well in the kernel -/

def qmin (a b : ℚ) : ℚ := if a ≤ b then a else b
def qmax (a b : ℚ) : ℚ := if a ≤ b then b else a
def qabs (a : ℚ) : ℚ := if 0 ≤ a then a else -a

lemma qmin_cast (a b : ℚ) : ((qmin a b : ℚ) : ℝ) = min (a : ℝ) b := by
  unfold qmin; split_ifs with h
  · exact (min_eq_left (by exact_mod_cast h)).symm
  · exact (min_eq_right (by exact_mod_cast (not_le.mp h).le)).symm

lemma qmax_cast (a b : ℚ) : ((qmax a b : ℚ) : ℝ) = max (a : ℝ) b := by
  unfold qmax; split_ifs with h
  · exact (max_eq_right (by exact_mod_cast h)).symm
  · exact (max_eq_left (by exact_mod_cast (not_le.mp h).le)).symm

lemma qabs_cast (a : ℚ) : ((qabs a : ℚ) : ℝ) = |(a : ℝ)| := by
  unfold qabs; split_ifs with h
  · exact (abs_of_nonneg (by exact_mod_cast h)).symm
  · push_cast
    exact (abs_of_neg (by exact_mod_cast not_le.mp h)).symm

/-! ### Polynomials as coefficient lists -/

/-- `c₀ + x (c₁ + x (c₂ + ...))`, over the reals. -/
noncomputable def peval : List ℚ → ℝ → ℝ
  | [], _ => 0
  | c :: t, x => (c : ℝ) + x * peval t x

/-- The same, over the rationals. -/
def pevalQ : List ℚ → ℚ → ℚ
  | [], _ => 0
  | c :: t, x => c + x * pevalQ t x

lemma pevalQ_cast (cs : List ℚ) (x : ℚ) : ((pevalQ cs x : ℚ) : ℝ) = peval cs x := by
  induction cs with
  | nil => simp [pevalQ, peval]
  | cons c t ih => simp [pevalQ, peval, ih]

/-- The derivative of `peval`, by the product rule at each step of the nesting. -/
noncomputable def pd : List ℚ → ℝ → ℝ
  | [], _ => 0
  | _ :: t, x => peval t x + x * pd t x

lemma peval_hasDerivAt (cs : List ℚ) (x : ℝ) : HasDerivAt (peval cs) (pd cs x) x := by
  induction cs with
  | nil =>
    have : peval [] = fun _ => (0 : ℝ) := by funext y; simp [peval]
    rw [this]; exact hasDerivAt_const x 0
  | cons c t ih =>
    have hfun : peval (c :: t) = fun y => (c : ℝ) + y * peval t y := by funext y; rfl
    rw [hfun]
    have h := ((hasDerivAt_id' x).mul ih).const_add (c : ℝ)
    convert h using 1
    simp [pd]

/-! ### Interval arithmetic with rational ends -/

structure Iv where
  lo : ℚ
  hi : ℚ

def Iv.Mem (I : Iv) (x : ℝ) : Prop := (I.lo : ℝ) ≤ x ∧ x ≤ (I.hi : ℝ)

def Iv.add (a b : Iv) : Iv := ⟨a.lo + b.lo, a.hi + b.hi⟩

def Iv.mul (a b : Iv) : Iv :=
  ⟨qmin (qmin (a.lo * b.lo) (a.lo * b.hi)) (qmin (a.hi * b.lo) (a.hi * b.hi)),
   qmax (qmax (a.lo * b.lo) (a.lo * b.hi)) (qmax (a.hi * b.lo) (a.hi * b.hi))⟩

lemma mul_corners {l h x m n y : ℝ} (hx1 : l ≤ x) (hx2 : x ≤ h) (hy1 : m ≤ y) (hy2 : y ≤ n) :
    min (min (l * m) (l * n)) (min (h * m) (h * n)) ≤ x * y ∧
      x * y ≤ max (max (l * m) (l * n)) (max (h * m) (h * n)) := by
  have A : min (l * y) (h * y) ≤ x * y ∧ x * y ≤ max (l * y) (h * y) := by
    rcases le_total 0 y with h0 | h0
    · exact ⟨(min_le_left _ _).trans (mul_le_mul_of_nonneg_right hx1 h0),
        (mul_le_mul_of_nonneg_right hx2 h0).trans (le_max_right _ _)⟩
    · exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_right hx2 h0),
        (mul_le_mul_of_nonpos_right hx1 h0).trans (le_max_left _ _)⟩
  have B : ∀ p : ℝ, min (p * m) (p * n) ≤ p * y ∧ p * y ≤ max (p * m) (p * n) := fun p => by
    rcases le_total 0 p with h0 | h0
    · exact ⟨(min_le_left _ _).trans (mul_le_mul_of_nonneg_left hy1 h0),
        (mul_le_mul_of_nonneg_left hy2 h0).trans (le_max_right _ _)⟩
    · exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_left hy2 h0),
        (mul_le_mul_of_nonpos_left hy1 h0).trans (le_max_left _ _)⟩
  exact ⟨le_trans (le_min ((min_le_left _ _).trans (B l).1) ((min_le_right _ _).trans (B h).1)) A.1,
    A.2.trans (max_le ((B l).2.trans (le_max_left _ _)) ((B h).2.trans (le_max_right _ _)))⟩

lemma Iv.mul_mem {a b : Iv} {x y : ℝ} (hx : a.Mem x) (hy : b.Mem y) : (a.mul b).Mem (x * y) := by
  have := mul_corners hx.1 hx.2 hy.1 hy.2
  simpa only [Iv.Mem, Iv.mul, qmin_cast, qmax_cast, Rat.cast_mul] using this

lemma Iv.add_mem {a b : Iv} {x y : ℝ} (hx : a.Mem x) (hy : b.Mem y) : (a.add b).Mem (x + y) := by
  simp only [Iv.Mem, Iv.add, Rat.cast_add]
  exact ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

/-- Horner's rule over intervals. -/
def hornerIv : List ℚ → Iv → Iv
  | [], _ => ⟨0, 0⟩
  | c :: t, I => Iv.add ⟨c, c⟩ (Iv.mul I (hornerIv t I))

lemma hornerIv_mem (cs : List ℚ) {I : Iv} {x : ℝ} (hx : I.Mem x) :
    (hornerIv cs I).Mem (peval cs x) := by
  induction cs with
  | nil => simp [hornerIv, peval, Iv.Mem]
  | cons c t ih => exact Iv.add_mem ⟨le_refl _, le_refl _⟩ (Iv.mul_mem hx ih)

/-! ### The polynomial `Q` -/

def qA0 : ℚ := -4519694089880372 / 2 ^ 54
def qA1 : ℚ := 6007023063796586 / 2 ^ 54
def qA2 : ℚ := -9007194725564707 / 2 ^ 54

lemma qA0_cast : (qA0 : ℝ) = A0 := by unfold qA0 A0; norm_num
lemma qA1_cast : (qA1 : ℝ) = A1 := by unfold qA1 A1; norm_num
lemma qA2_cast : (qA2 : ℝ) = A2 := by unfold qA2 A2; norm_num

/-- `Q = P` minus the degree-9 series for `log(1 + r)`. Its `r` term cancels. -/
def qs : List ℚ :=
  [0, 0, qA2 + 1 / 2, qA1 - 1 / 3, qA0 + 1 / 4, -1 / 5, 1 / 6, -1 / 7, 1 / 8, -1 / 9]

/-- `Q`'s derivative, as coefficients. -/
def dqs : List ℚ :=
  [0, 2 * (qA2 + 1 / 2), 3 * (qA1 - 1 / 3), 4 * (qA0 + 1 / 4), -1, 1, -1, 1, -1]

lemma pd_qs (r : ℝ) : pd qs r = peval dqs r := by
  simp only [qs, dqs, pd, peval]; push_cast; ring

/-- Mathlib's series bound splits `e(r)` into `Q(r)` and a remainder. -/
lemma e_split {r : ℝ} (hr : |r| < 1) :
    |P r - log (1 + r)| ≤ |peval qs r| + |r| ^ 10 / (1 - |r|) := by
  have h := Real.abs_log_sub_add_sum_range_le (x := -r) (by rwa [abs_neg]) 9
  rw [abs_neg, sub_neg_eq_add] at h
  have hQ : peval qs r = P r + ∑ i ∈ Finset.range 9, (-r) ^ (i + 1) / ((i : ℝ) + 1) := by
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, qs, peval, P]
    push_cast [qA0_cast, qA1_cast, qA2_cast]
    norm_num
    ring
  have : P r - log (1 + r) =
      peval qs r - (∑ i ∈ Finset.range 9, (-r) ^ (i + 1) / ((i : ℝ) + 1) + log (1 + r)) := by
    rw [hQ]; ring
  rw [this]
  exact (abs_sub _ _).trans (add_le_add le_rfl h)

/-! ### Monotone stretches, checked once -/

/-- Whether `Q' ≤ 0` on `[lo, hi]`, splitting while the interval is too wide. -/
def negOK : ℕ → ℚ → ℚ → Bool
  | 0, lo, hi => decide ((hornerIv dqs ⟨lo, hi⟩).hi ≤ 0)
  | n + 1, lo, hi => decide ((hornerIv dqs ⟨lo, hi⟩).hi ≤ 0) ||
      (negOK n lo ((lo + hi) / 2) && negOK n ((lo + hi) / 2) hi)

/-- Whether `Q' ≥ 0` on `[lo, hi]`. -/
def posOK : ℕ → ℚ → ℚ → Bool
  | 0, lo, hi => decide (0 ≤ (hornerIv dqs ⟨lo, hi⟩).lo)
  | n + 1, lo, hi => decide (0 ≤ (hornerIv dqs ⟨lo, hi⟩).lo) ||
      (posOK n lo ((lo + hi) / 2) && posOK n ((lo + hi) / 2) hi)

lemma negOK_sound (n : ℕ) (lo hi : ℚ) (h : negOK n lo hi = true) {r : ℝ}
    (h1 : (lo : ℝ) ≤ r) (h2 : r ≤ hi) : pd qs r ≤ 0 := by
  induction n generalizing lo hi with
  | zero =>
    simp only [negOK, decide_eq_true_eq] at h
    rw [pd_qs]
    exact (hornerIv_mem dqs (I := ⟨lo, hi⟩) ⟨h1, h2⟩).2.trans (by exact_mod_cast h)
  | succ n ih =>
    simp only [negOK, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases h with h | ⟨ha, hb⟩
    · rw [pd_qs]
      exact (hornerIv_mem dqs (I := ⟨lo, hi⟩) ⟨h1, h2⟩).2.trans (by exact_mod_cast h)
    · rcases le_total r (((lo + hi) / 2 : ℚ) : ℝ) with hm | hm
      · exact ih _ _ ha h1 hm
      · exact ih _ _ hb hm h2

lemma posOK_sound (n : ℕ) (lo hi : ℚ) (h : posOK n lo hi = true) {r : ℝ}
    (h1 : (lo : ℝ) ≤ r) (h2 : r ≤ hi) : 0 ≤ pd qs r := by
  induction n generalizing lo hi with
  | zero =>
    simp only [posOK, decide_eq_true_eq] at h
    rw [pd_qs]
    exact le_trans (by exact_mod_cast h) (hornerIv_mem dqs (I := ⟨lo, hi⟩) ⟨h1, h2⟩).1
  | succ n ih =>
    simp only [posOK, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases h with h | ⟨ha, hb⟩
    · rw [pd_qs]
      exact le_trans (by exact_mod_cast h) (hornerIv_mem dqs (I := ⟨lo, hi⟩) ⟨h1, h2⟩).1
    · rcases le_total r (((lo + hi) / 2 : ℚ) : ℝ) with hm | hm
      · exact ih _ _ ha h1 hm
      · exact ih _ _ hb hm h2

lemma antitone_of_negOK {n : ℕ} {a b : ℚ} (h : negOK n a b = true) :
    AntitoneOn (peval qs) (Set.Icc (a : ℝ) b) := by
  refine antitoneOn_of_deriv_nonpos (convex_Icc _ _)
    (fun x _ => (peval_hasDerivAt qs x).continuousAt.continuousWithinAt)
    (fun x _ => (peval_hasDerivAt qs x).differentiableAt.differentiableWithinAt)
    (fun x hx => ?_)
  rw [(peval_hasDerivAt qs x).deriv]
  rw [interior_Icc] at hx
  exact negOK_sound n a b h hx.1.le hx.2.le

lemma monotone_of_posOK {n : ℕ} {a b : ℚ} (h : posOK n a b = true) :
    MonotoneOn (peval qs) (Set.Icc (a : ℝ) b) := by
  refine monotoneOn_of_deriv_nonneg (convex_Icc _ _)
    (fun x _ => (peval_hasDerivAt qs x).continuousAt.continuousWithinAt)
    (fun x _ => (peval_hasDerivAt qs x).differentiableAt.differentiableWithinAt)
    (fun x hx => ?_)
  rw [(peval_hasDerivAt qs x).deriv]
  rw [interior_Icc] at hx
  exact posOK_sound n a b h hx.1.le hx.2.le

/-! ### The stretches of `[-1/16, 1/16]`, on a `2^-24` grid -/

def b0 : ℚ := -1 / 16
def b1 : ℚ := -339434 / 2 ^ 24
def b2 : ℚ := -339402 / 2 ^ 24
def b3 : ℚ := -22615 / 2 ^ 24
def b4 : ℚ := -22583 / 2 ^ 24
def b5 : ℚ := 0
def b6 : ℚ := 308449 / 2 ^ 24
def b7 : ℚ := 308481 / 2 ^ 24
def b8 : ℚ := 1 / 16

lemma stretch1 : AntitoneOn (peval qs) (Set.Icc (b0 : ℝ) b1) :=
  antitone_of_negOK (n := 30) (by decide +kernel)
lemma stretch2 : MonotoneOn (peval qs) (Set.Icc (b2 : ℝ) b3) :=
  monotone_of_posOK (n := 30) (by decide +kernel)
lemma stretch3 : AntitoneOn (peval qs) (Set.Icc (b4 : ℝ) b5) :=
  antitone_of_negOK (n := 30) (by decide +kernel)
lemma stretch4 : MonotoneOn (peval qs) (Set.Icc (b5 : ℝ) b6) :=
  monotone_of_posOK (n := 30) (by decide +kernel)
lemma stretch5 : AntitoneOn (peval qs) (Set.Icc (b7 : ℝ) b8) :=
  antitone_of_negOK (n := 30) (by decide +kernel)

/-! ### The bound over any range -/

/-- A stretch where `Q` is monotone, or a tiny box around a turning point. -/
inductive Piece
  | mono (a b : ℚ)
  | box (a b : ℚ)

def Piece.a : Piece → ℚ
  | .mono a _ => a
  | .box a _ => a

def Piece.b : Piece → ℚ
  | .mono _ b => b
  | .box _ b => b

/-- The largest `|Q|` the piece allows over `[lo, hi]`; `0` if they don't meet. -/
def Piece.sup (p : Piece) (lo hi : ℚ) : ℚ :=
  if hi < p.a ∨ p.b < lo then 0 else
    match p with
    | .mono a b => qmax (qabs (pevalQ qs (qmax lo a))) (qabs (pevalQ qs (qmin hi b)))
    | .box a b =>
      qmax (qabs (hornerIv qs ⟨a, b⟩).lo) (qabs (hornerIv qs ⟨a, b⟩).hi)

def pieces : List Piece :=
  [.mono b0 b1, .box b1 b2, .mono b2 b3, .box b3 b4, .mono b4 b5, .mono b5 b6, .box b6 b7,
   .mono b7 b8]

def esup (lo hi : ℚ) : ℚ := pieces.foldr (fun p acc => qmax (p.sup lo hi) acc) 0

/-- The remainder of the series, for `|r| ≤ ρ`. -/
def remB (ρ : ℚ) : ℚ := ρ ^ 10 / (1 - ρ)

/-- A bound on `|P(r) - log(1 + r)|` over `[lo, hi]`, within `[-1/16, 1/16]`. -/
def errSup (lo hi : ℚ) : ℚ := esup lo hi + remB (qmax (qabs lo) (qabs hi))

lemma Piece.sup_mem_le {p : Piece} {lo hi : ℚ} (hp : p ∈ pieces) :
    ((p.sup lo hi : ℚ) : ℝ) ≤ esup lo hi := by
  have key : ∀ l : List Piece, p ∈ l →
      ((p.sup lo hi : ℚ) : ℝ) ≤ ((l.foldr (fun p acc => qmax (p.sup lo hi) acc) 0 : ℚ) : ℝ) := by
    intro l hl
    induction l with
    | nil => cases hl
    | cons q t ih =>
      rw [List.foldr_cons, qmax_cast]
      rcases List.mem_cons.1 hl with rfl | h
      · exact le_max_left _ _
      · exact (ih h).trans (le_max_right _ _)
  exact key _ hp

/-- On a monotone stretch, `|f|` over a range is largest at the range's ends. -/
lemma abs_le_ends {f : ℝ → ℝ} {a b u v r : ℝ}
    (hf : MonotoneOn f (Set.Icc a b) ∨ AntitoneOn f (Set.Icc a b))
    (hu : u ∈ Set.Icc a b) (hv : v ∈ Set.Icc a b) (hr : r ∈ Set.Icc a b)
    (h1 : u ≤ r) (h2 : r ≤ v) : |f r| ≤ max |f u| |f v| := by
  rcases hf with hf | hf
  · have h3 := hf hu hr h1; have h4 := hf hr hv h2
    rw [abs_le]; constructor
    · have := neg_abs_le (f u); have := le_max_left |f u| |f v|; linarith
    · have := le_abs_self (f v); have := le_max_right |f u| |f v|; linarith
  · have h3 := hf hu hr h1; have h4 := hf hr hv h2
    rw [abs_le]; constructor
    · have := neg_abs_le (f v); have := le_max_right |f u| |f v|; linarith
    · have := le_abs_self (f u); have := le_max_left |f u| |f v|; linarith

lemma Q_le_esup {lo hi : ℚ} {r : ℝ} (h1 : (lo : ℝ) ≤ r) (h2 : r ≤ hi)
    (h3 : (b0 : ℝ) ≤ r) (h4 : r ≤ b8) : |peval qs r| ≤ esup lo hi := by
  -- Whichever piece holds `r`, its `sup` is at least `|Q r|` and at most `esup`.
  have mono_case : ∀ {a b : ℚ}, Piece.mono a b ∈ pieces →
      (MonotoneOn (peval qs) (Set.Icc (a : ℝ) b) ∨ AntitoneOn (peval qs) (Set.Icc (a : ℝ) b)) →
      (a : ℝ) ≤ r → r ≤ b → |peval qs r| ≤ esup lo hi := by
    intro a b hp hf ha hb
    refine le_trans ?_ (Piece.sup_mem_le (lo := lo) (hi := hi) hp)
    have hlohi : ¬ (hi < a ∨ b < lo) := by
      rintro (h | h)
      · have : (hi : ℝ) < a := by exact_mod_cast h
        linarith
      · have : (b : ℝ) < lo := by exact_mod_cast h
        linarith
    have e : (Piece.mono a b).sup lo hi =
        qmax (qabs (pevalQ qs (qmax lo a))) (qabs (pevalQ qs (qmin hi b))) := by
      simp only [Piece.sup, Piece.a, Piece.b]; exact if_neg hlohi
    rw [e]
    simp only [qmax_cast, qabs_cast, pevalQ_cast, qmin_cast]
    have hab : (a : ℝ) ≤ b := ha.trans hb
    have hu : max (lo : ℝ) a ∈ Set.Icc (a : ℝ) b :=
      ⟨le_max_right _ _, max_le (h1.trans hb) hab⟩
    have hv : min (hi : ℝ) b ∈ Set.Icc (a : ℝ) b :=
      ⟨le_min (ha.trans h2) hab, min_le_right _ _⟩
    exact abs_le_ends hf hu hv ⟨ha, hb⟩ (max_le h1 ha) (le_min h2 hb)
  have box_case : ∀ {a b : ℚ}, Piece.box a b ∈ pieces → (a : ℝ) ≤ r → r ≤ b →
      |peval qs r| ≤ esup lo hi := by
    intro a b hp ha hb
    refine le_trans ?_ (Piece.sup_mem_le (lo := lo) (hi := hi) hp)
    have hlohi : ¬ (hi < a ∨ b < lo) := by
      rintro (h | h)
      · have : (hi : ℝ) < a := by exact_mod_cast h
        linarith
      · have : (b : ℝ) < lo := by exact_mod_cast h
        linarith
    have e : (Piece.box a b).sup lo hi =
        qmax (qabs (hornerIv qs ⟨a, b⟩).lo) (qabs (hornerIv qs ⟨a, b⟩).hi) := by
      simp only [Piece.sup, Piece.a, Piece.b]; exact if_neg hlohi
    rw [e]
    simp only [qmax_cast, qabs_cast]
    have hm := hornerIv_mem qs (I := ⟨a, b⟩) (x := r) ⟨ha, hb⟩
    rw [abs_le]; constructor
    · have := neg_abs_le ((hornerIv qs ⟨a, b⟩).lo : ℝ)
      have := le_max_left |((hornerIv qs ⟨a, b⟩).lo : ℝ)| |((hornerIv qs ⟨a, b⟩).hi : ℝ)|
      linarith [hm.1]
    · have := le_abs_self ((hornerIv qs ⟨a, b⟩).hi : ℝ)
      have := le_max_right |((hornerIv qs ⟨a, b⟩).lo : ℝ)| |((hornerIv qs ⟨a, b⟩).hi : ℝ)|
      linarith [hm.2]
  rcases le_total r (b1 : ℝ) with c1 | c1
  · exact mono_case (by simp [pieces]) (Or.inr stretch1) h3 c1
  rcases le_total r (b2 : ℝ) with c2 | c2
  · exact box_case (by simp [pieces]) c1 c2
  rcases le_total r (b3 : ℝ) with c3 | c3
  · exact mono_case (by simp [pieces]) (Or.inl stretch2) c2 c3
  rcases le_total r (b4 : ℝ) with c4 | c4
  · exact box_case (by simp [pieces]) c3 c4
  rcases le_total r (b5 : ℝ) with c5 | c5
  · exact mono_case (by simp [pieces]) (Or.inr stretch3) c4 c5
  rcases le_total r (b6 : ℝ) with c6 | c6
  · exact mono_case (by simp [pieces]) (Or.inl stretch4) c5 c6
  rcases le_total r (b7 : ℝ) with c7 | c7
  · exact box_case (by simp [pieces]) c6 c7
  · exact mono_case (by simp [pieces]) (Or.inr stretch5) c7 h4

/-- `errSup` bounds the polynomial's error over any `[lo, hi]` within
`[-1/16, 1/16]`. -/
theorem err_sound {lo hi : ℚ} {r : ℝ} (hlo : (-1 / 16 : ℝ) ≤ lo) (hhi : (hi : ℝ) ≤ 1 / 16)
    (h1 : (lo : ℝ) ≤ r) (h2 : r ≤ hi) :
    |P r - log (1 + r)| ≤ errSup lo hi := by
  have hb0 : (b0 : ℝ) = -1 / 16 := by unfold b0; norm_num
  have hb8 : (b8 : ℝ) = 1 / 16 := by unfold b8; norm_num
  have hQ : |peval qs r| ≤ esup lo hi :=
    Q_le_esup h1 h2 (by rw [hb0]; linarith) (by rw [hb8]; linarith)
  have hlo' : |(lo : ℝ)| ≤ 1 / 16 := by rw [abs_le]; constructor <;> linarith
  have hhi' : |(hi : ℝ)| ≤ 1 / 16 := by rw [abs_le]; constructor <;> linarith
  have hmax : max |(lo : ℝ)| |(hi : ℝ)| ≤ 1 / 16 := max_le hlo' hhi'
  have hρ : |r| ≤ max |(lo : ℝ)| |(hi : ℝ)| := by
    rw [abs_le]; constructor
    · have := neg_abs_le (lo : ℝ); have := le_max_left |(lo : ℝ)| |(hi : ℝ)|; linarith
    · have := le_abs_self (hi : ℝ); have := le_max_right |(lo : ℝ)| |(hi : ℝ)|; linarith
  have hsplit := e_split (r := r) (by linarith)
  simp only [errSup, remB]
  push_cast [qmax_cast, qabs_cast]
  refine hsplit.trans (add_le_add hQ ?_)
  exact div_le_div₀ (by positivity) (pow_le_pow_left₀ (abs_nonneg r) hρ 10) (by linarith)
    (by linarith)

end FrecpeLogf
