// Hand-written AArch64 kernels for benchmark-acosf.cpp.
//
// float32 acos(x) three ways, each in the two loop shapes described in
// approx-kernels.inc:
//
//   call     BL to the system libm's acosf: the baseline.
//   fsqrt    The hand kernel below, with sqrt(z) from FSQRT.
//   frsqrte  The same kernel, with sqrt(z) = z * rsqrt(z): FRSQRTE's 8-bit
//            estimate and two Newton steps, then one multiply.
//
// Everything but the SQRT macro is shared, so the timing difference between the
// two hand kernels is the square root alone.
//
// The algorithm, all of it in double, with one rounding to float at the end:
//   |x| <= 1/2   z = x^2, asin(x) = x + x*z*P(z), acos(x) = pi/2 - asin(x).
//   |x| >  1/2   z = (1 - |x|)/2 (exact), s = sqrt(z), asin(s) = s + s*z*P(z),
//                acos(x) = 2*asin(s) for x > 0, pi - 2*asin(s) for x < 0.
// Both are acos(x) = C + s2 + s2*z*P(z), with (C, s2) = (pi/2, -x), (0, 2s) or
// (pi, -2s), so after picking z, s2 and C the paths share one polynomial. asin
// is pi/2 - acos and shares the whole reduction; only acos is timed.
//
// P is a degree-5 weighted minimax fit to (asin(sqrt z) - sqrt z) / z^1.5 on
// [0, 1/4]; its error reaches the result as at most 2^-31.4 relative. The
// FRSQRTE square root is good to about 2^-30. Any error below 2^-24 before the
// final rounding keeps the result within 1 ulp of the correctly rounded acos,
// and the driver's accuracy pass holds both kernels to that. The square root is
// taken in double because a float32 FSQRT would already spend that whole 2^-24.
//
// The two paths are both inline, one taken branch each: |x| > 1/2 branches over
// the small path's setup, which branches over the large path's. Fresh inputs
// that mix the two mispredict about half the time; the driver's repeating
// buffer hides that (docs/BENCHMARKING.md, section 10.8).
//
// Off the hot path, in a stub after each kernel's ret: |x| >= 1, inf and NaN.
// At |x| = 1, z = 0 and z * rsqrt(z) is 0 * inf, NaN for any grouping, so the
// stub returns +0 or pi itself; both kernels take it, to stay identical. The
// stubs are in the same section as the kernels because Mach-O cannot relocate a
// conditional branch into another one.
//
// Registers, on top of approx-kernels.inc's (v31 = 0, LINK's v2/v3):
//   w5 = 0x3f000000 (0.5f)   w6 = 0x3f800000 (1.0f)   x3 = Lacosf_consts
//   d21-d26 = c0-c5          d27 = 1/2   d28 = pi/2   d29 = pi
//   scratch: w9 (ix), w10 (|ix|), d16 (x, then P), d17 (z), d18 (s2),
//            d19 (C), d20
// Every element reuses the same scratch registers; renaming makes that free.

#include "approx-kernels.inc"

    .text

// ─── The acos sequence ──────────────────────────────────────────────────────

.macro ACOSF_SETUP
    mov     w5, #0x3f000000
    mov     w6, #0x3f800000
    adr     x3, Lacosf_consts
    ldp     d21, d22, [x3]
    ldp     d23, d24, [x3, #16]
    ldp     d25, d26, [x3, #32]
    ldp     d27, d28, [x3, #48]
    ldr     d29, [x3, #64]
    movi    v31.2d, #0
.endm

// One Newton step for 1/sqrt(z) on d\e, as RSQRT_STEP in
// ../frsqrte/rsqrt-kernels.S: z*e first, with FMULX. Here z is in
// [2^-25, 1/4], where no grouping can overflow; it keeps the instruction mix
// that benchmark measured. Clobbers d\t.
.macro RSQRT_STEP_D e, x, t
    fmulx   d\t, d\x, d\e
    frsqrts d\t, d\t, d\e                   // (3 - t*e)/2, fused
    fmul    d\e, d\e, d\t
.endm

// d18 = sqrt(d17): the one place the two kernels differ. Clobbers d20.
.macro SQRT kind
.ifc \kind, fsqrt
    fsqrt   d18, d17
.else
    frsqrte d18, d17                        // e ~ 1/sqrt(z), 8 bits
    RSQRT_STEP_D 18, 17, 20                 // ~15 bits
    RSQRT_STEP_D 18, 17, 20                 // ~30 bits
    fmul    d18, d17, d18                   // sqrt(z) = z*e
.endif
.endm

// s\x = acosf(s\x), in place. \id names this instance's labels, which its
// ACOSF_COLD stub shares.
.macro ACOSF kind, x, id
    fmov    w9, s\x
    fcvt    d16, s\x                        // x, exact
    and     w10, w9, #0x7fffffff
    cmp     w10, w5
    b.hi    Lbig_\id                        // |x| > 1/2, or NaN
    fmul    d17, d16, d16                   // z = x^2, exact
    fneg    d18, d16                        // s2 = -x
    fmov    d19, d28                        // C = pi/2
    b       Lpoly_\id
Lbig_\id:
    cmp     w10, w6
    b.hs    Lcold_\id                       // |x| >= 1, inf or NaN
    fabs    d17, d16
    fmsub   d17, d17, d27, d27              // z = 1/2 - |x|/2, exact
    SQRT    \kind
    fadd    d18, d18, d18                   // 2s
    fneg    d20, d18
    tst     w9, #0x80000000
    fcsel   d18, d20, d18, ne               // s2 = -2s for x < 0
    fcsel   d19, d29, d31, ne               // C = pi for x < 0, else 0
Lpoly_\id:
    fmadd   d16, d26, d17, d25
    fmadd   d16, d16, d17, d24
    fmadd   d16, d16, d17, d23
    fmadd   d16, d16, d17, d22
    fmadd   d16, d16, d17, d21              // P(z)
    fmul    d20, d18, d17                   // s2*z
    fadd    d19, d19, d18                   // C + s2
    fmadd   d16, d20, d16, d19              // C + s2 + s2*z*P(z)
    fcvt    s\x, d16
Ldone_\id:
.endm

// The out-of-line half of ACOSF \kind, \x, \id. Placed after the kernel's ret.
// Entered with the flags of cmp |ix|, 1.0f.
.macro ACOSF_COLD x, id
Lcold_\id:
    b.ne    Lnan_\id
    tst     w9, #0x80000000
    fcsel   d16, d29, d31, ne               // acos(1) = +0, acos(-1) = pi
    fcvt    s\x, d16
    b       Ldone_\id
Lnan_\id:
    mov     w10, #0x7fc00000
    fmov    s\x, w10                        // |x| > 1 or NaN: NaN
    b       Ldone_\id
.endm

// ─── Kernels ────────────────────────────────────────────────────────────────

// Throughput: in place in s0-s7, so v16-v20 are free for scratch.
.macro ACOSF_TPUT kind, p
    .globl  _acosf_tput_\kind
    .p2align 4
_acosf_tput_\kind:
    ACOSF_SETUP
    .p2align 6
1:  LOAD8
    ACOSF   \kind, 0, \p\()0
    ACOSF   \kind, 1, \p\()1
    ACOSF   \kind, 2, \p\()2
    ACOSF   \kind, 3, \p\()3
    ACOSF   \kind, 4, \p\()4
    ACOSF   \kind, 5, \p\()5
    ACOSF   \kind, 6, \p\()6
    ACOSF   \kind, 7, \p\()7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret
    ACOSF_COLD 0, \p\()0
    ACOSF_COLD 1, \p\()1
    ACOSF_COLD 2, \p\()2
    ACOSF_COLD 3, \p\()3
    ACOSF_COLD 4, \p\()4
    ACOSF_COLD 5, \p\()5
    ACOSF_COLD 6, \p\()6
    ACOSF_COLD 7, \p\()7
.endm

// Latency: the chain runs through s1, which LINK 1, 1 refills each link.
.macro ACOSF_LAT kind, p
    .globl  _acosf_lat_\kind
    .p2align 4
_acosf_lat_\kind:
    ACOSF_SETUP
    movi    v1.2d, #0
    .p2align 6
1:  LINK    1, 1
    ACOSF   \kind, 1, \p\()0
    LINK    1, 1
    ACOSF   \kind, 1, \p\()1
    LINK    1, 1
    ACOSF   \kind, 1, \p\()2
    LINK    1, 1
    ACOSF   \kind, 1, \p\()3
    subs    x1, x1, #4
    b.ne    1b
    fmov    s0, s1
    ret
    ACOSF_COLD 1, \p\()0
    ACOSF_COLD 1, \p\()1
    ACOSF_COLD 1, \p\()2
    ACOSF_COLD 1, \p\()3
.endm

    ACOSF_TPUT fsqrt,   ts
    ACOSF_TPUT frsqrte, te
    ACOSF_LAT  fsqrt,   ls
    ACOSF_LAT  frsqrte, le

// The libm baseline, shaped like logf-kernels.S's _call pair: loop state in
// callee-saved registers, because acosf may clobber every caller-saved one.
    .globl  _acosf_tput_call
    .p2align 4
_acosf_tput_call:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    .p2align 6
1:
    .rept 8
    ldr     s0, [x19], #4
    bl      _acosf
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _acosf_lat_call
    .p2align 4
_acosf_lat_call:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     d8, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    movi    v8.2d, #0                       // the zero, callee-saved across BL
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    0, 0, 8, x19                    // operand straight into s0, the argument
    bl      _acosf                          // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ACOSF_SETUP's constants, in this section so ADR reaches them without a
// relocation. c0-c5 come from a Lawson-weighted least-squares fit (weight
// max(z, 1/64), the factor z*P(z) carries into the result), rounded to double.
    .p2align 3
Lacosf_consts:
    .quad   0x3fc55555247f8893              // c0 =  0.16666664392602523
    .quad   0x3fb33366782ce258              // c1 =  0.07500305589645373
    .quad   0x3fa6cf0bae6220e6              // c2 =  0.044548382794404692
    .quad   0x3fa02b6af680891d              // c3 =  0.031581251707637238
    .quad   0x3f8f20bd9d1d9308              // c4 =  0.015199166638646414
    .quad   0x3fa2b699210fa052              // c5 =  0.036549363408824434
    .quad   0x3fe0000000000000              // 1/2
    .quad   0x3ff921fb54442d18              // pi/2, rounded to double
    .quad   0x400921fb54442d18              // pi, rounded to double
