// Hand-written AArch64 kernels for benchmark-powf.cpp.
//
// float32 pow(x, y) = exp2(y * log2(x)) three ways, each in the two loop shapes
// described in approx-kernels.inc. y is the same for a whole kernel call: it is
// read from powf_y once, and the per-element code treats it like any operand
// (nothing about y is decided outside the loop).
//
//   call    BL to the system libm's powf: the baseline.
//   frecpe  log2's range reduction as in ../logf/: r = x*FRECPE(x) - 1, exact
//           in float32 with |r| <= 2^-8.45, FRECPE(x) = 2^-k * ehat(i).
//   tab     The same r with FRECPE's values read from memory: z = x*FRECPX(x)
//           = 2m exactly, then r = z * (ehat(i)/2) - 1. Bit for bit the same r,
//           so both kernels return identical results and differ only in how
//           the reciprocal is fetched.
//
// The rest is shared, all of it in double:
//   log2(x) = k - log2(ehat(i)) + log2(1 + r), the last a degree-4 Taylor
//             polynomial in r (relative error <= 2^-34.3, worst near 1).
//   t       = 32y * log2(x). pow amplifies log2's error by |y log2 x|, which is
//             at most 150 for any result that is neither 0 nor inf after
//             rounding, so t's error costs at most 2^-27.6 of the result.
//   exp2(t/32) = 2^(n/32) * 2^(rr/32), n = round(t), |rr| <= 1/2: a 32-entry
//             table whose entries absorb n >> 5 into their exponent, and a
//             degree-3 polynomial (relative error <= 2^-30.7).
// One rounding to float at the end. The total, under 2^-27.4, keeps every
// result within 1 ulp of the correctly rounded one; FCVT's own rounding makes
// overflow, underflow and subnormal results come out right with no extra code.
//
// Off the hot path, in a stub after each kernel's ret:
//   - x not a positive normal: subnormals are scaled and rejoin; +-0, +-inf,
//     NaN and y = +-0 or NaN take C's special values; x < 0 with an even
//     integer y rejoins with |x|, with a non-integer y is NaN, and with an odd
//     integer y runs a second copy of the hot path on |x| that negates its
//     result. The copy costs the hot path nothing.
//   - x in [1, 1 + 2^-8): r = x - 1 and k - log2(ehat) = 0, as in ../logf/.
//   - |t| >= 2^14 (|y log2 x| >= 512) or t NaN: the result is +inf, +0, NaN, or
//     1 for x = 1.
// The hot path pays two not-taken branches per element. The stubs are in the
// same section as the kernels because Mach-O cannot relocate a conditional
// branch into another one.
//
// Registers, on top of approx-kernels.inc's (v29 = 1.0f, v31 = 0, LINK's v2/v3):
//   w5 = 0x7f000000   w6 = 0x3f800000 (1.0f)   w7 = 0x8000   w14 = y's bits
//   x3 = powf_log2c   x4 = powf_recip (tab only)   x8 = powf_exp2
//   x13 = 2^14 - bits(0x1.8p52), the |t| range check
//   d20 = 32y   d21-d24 = c1-c4 (log2(1+r))   d25 = 0x1.8p52
//   d26-d28 = e1-e3 (2^(rr/32) - 1)
//   scratch: w9 (ix), w10-w12, w15 (i), w16 (k), d16-d19, d30
// Every element reuses the same scratch registers; renaming makes that free.

#include "approx-kernels.inc"

    .text

// ─── The pow sequence ───────────────────────────────────────────────────────

// Constants, table addresses and y, once per kernel call.
.macro POWF_SETUP kind
    mov     w5, #0x7f000000
    mov     w6, #0x3f800000
    mov     w7, #0x8000
    adrp    x3, _powf_log2c@PAGE
    add     x3, x3, _powf_log2c@PAGEOFF
.ifc \kind, tab
    adrp    x4, _powf_recip@PAGE
    add     x4, x4, _powf_recip@PAGEOFF
.endif
    adrp    x8, _powf_exp2@PAGE
    add     x8, x8, _powf_exp2@PAGEOFF
    adrp    x10, _powf_poly@PAGE
    add     x10, x10, _powf_poly@PAGEOFF
    ldp     d21, d22, [x10]
    ldp     d23, d24, [x10, #16]
    ldp     d26, d27, [x10, #32]
    ldr     d28, [x10, #48]
    movz    x10, #0x4338, lsl #48
    fmov    d25, x10                        // 0x1.8p52
    movz    x13, #0x4000
    movk    x13, #0xbcc8, lsl #48           // 2^14 - 0x4338000000000000
    adrp    x10, _powf_y@PAGE
    ldr     w14, [x10, _powf_y@PAGEOFF]
    fmov    s20, w14
    fcvt    d20, s20
    movz    x10, #0x4040, lsl #48
    fmov    d16, x10                        // 32.0
    fmul    d20, d20, d16                   // 32y, exact
    fmov    s29, #1.0
.endm

// s\x = pow(s\x, y), in place. \id names this instance's labels, which its
// POWF_COLD stub shares.
.macro POWF kind, x, id
Lentry_\id:
    fmov    w9, s\x
    sub     w10, w9, #0x800, lsl #12        // ix - 0x00800000
    cmp     w10, w5                         // lo: x is a positive normal
    sub     w12, w9, w6
    ccmp    w12, w7, #0, lo                 // then lo: x in [1, 1 + 2^-8);
    b.lo    Lcold_\id                       // anything else forces lo too
Lmain_\id:
    ubfx    w15, w9, #15, #8                // i: the top eight fraction bits
    asr     w16, w9, #23
    sub     w16, w16, #127                  // k (asr: a scaled subnormal's is negative)
.ifc \kind, frecpe
    frecpe  s16, s\x                        // e = 2^-k * ehat(i)
    fnmsub  s16, s\x, s16, s29              // r = x*e - 1, exact
.else
    ldr     s17, [x4, w15, uxtw #2]         // ehat(i)/2
    frecpx  s16, s\x
    fmul    s16, s\x, s16                   // z = 2m, exact
    fnmsub  s16, s16, s17, s29              // r = z*ehat(i)/2 - 1, exact
.endif
    fcvt    d16, s16
    ldr     d18, [x3, w15, uxtw #3]         // -log2(ehat(i))
    scvtf   d19, w16
    fadd    d18, d19, d18                   // y0 = k - log2(ehat(i))
Lpoly_\id:
    fmul    d17, d16, d16                   // r^2
    fmadd   d19, d24, d16, d23              // c3 + c4 r
    fmadd   d30, d22, d16, d21              // c1 + c2 r
    fmadd   d19, d19, d17, d30              // c1 + c2 r + c3 r^2 + c4 r^3
    fmadd   d18, d19, d16, d18              // log2(x) = y0 + r*(...)
    fmul    d18, d18, d20                   // t = 32 y log2(x)
    fadd    d16, d18, d25                   // n = round(t), in the low bits
    fmov    x10, d16
    add     x12, x10, x13
    cmp     x12, #8, lsl #12                // hs: |n| >= 2^14, or t is NaN
    b.hs    Lbig_\id
    fsub    d16, d16, d25                   // n
    fsub    d16, d18, d16                   // rr = t - n, exact, |rr| <= 1/2
    and     w11, w10, #31
    ldr     x11, [x8, w11, uxtw #3]         // bits(2^(j/32)) - (j << 47)
    add     x11, x11, x10, lsl #47          // s = 2^(n/32)
    fmov    d17, x11
    fmul    d18, d16, d16                   // rr^2
    fmadd   d19, d28, d16, d27              // e2 + e3 rr
    fmul    d30, d16, d26                   // e1 rr
    fmadd   d19, d19, d18, d30              // 2^(rr/32) - 1
    fmadd   d17, d17, d19, d17              // s * 2^(rr/32)
    fcvt    s\x, d17
Ldone_\id:
.endm

// The out-of-line half of POWF \kind, \x, \id. Placed after the kernel's ret.
// \copy = 1 is the stub of the negating copy, which only ever sees x > 0.
.macro POWF_COLD kind, x, id, copy=0
Lcold_\id:
    cmp     w10, w5
    b.hs    Lspecial_\id
    fsub    s16, s\x, s29                   // x in [1, 1 + 2^-8): r = x - 1, exact
    fcvt    d16, s16
    movi    d18, #0                         // y0 = 0
    b       Lpoly_\id
Lspecial_\id:                               // x is not a positive normal
.if \copy == 0
    lsl     w10, w14, #1
    cbz     w10, Lone_\id                   // pow(x, +-0) = 1, even for x NaN
    fcmp    s\x, s\x
    b.vs    Lnan_\id
    fmov    s17, w14                        // y
    fcmp    s17, s17
    b.vs    Lnan_\id
    tbnz    w9, #31, Lneg_\id
.endif
    lsl     w10, w9, #1
    cbz     w10, Lxzero_\id
    lsr     w10, w9, #23
    cbnz    w10, Lxinf_\id
    ucvtf   s\x, w9                         // positive subnormal: x * 2^149, exact
    fmov    w9, s\x
    mov     w10, #0x4a800000                // 149 << 23, so k comes out 149 lower
    sub     w9, w9, w10
    b       Lmain_\id
Lxzero_\id:                                 // pow(+0, y): +inf for y < 0, else +0
    tbnz    w14, #31, Linf_\id
    b       Lzero_\id
Lxinf_\id:                                  // pow(+inf, y): +0 for y < 0, else +inf
    tbnz    w14, #31, Lzero_\id
    b       Linf_\id
Lbig_\id:                                   // |y log2 x| >= 512, or t NaN
    cmp     w9, w6                          // ix, not s\x: 2^-149 scales to exactly 1
    b.eq    Lone_\id                        // pow(1, y) = 1, even for y NaN or inf
    fcmp    d18, #0.0
    b.vs    Lnan_\id
    b.gt    Linf_\id
Lzero_\id:
    fmov    s\x, wzr
    b       Ldone_\id
Linf_\id:
    mov     w10, #0x7f800000
    fmov    s\x, w10
    b       Ldone_\id
Lone_\id:
    fmov    s\x, #1.0
    b       Ldone_\id
Lnan_\id:
    mov     w10, #0x7fc00000
    fmov    s\x, w10
    b       Ldone_\id
.if \copy == 0
Lneg_\id:                                   // x < 0; y (s17) neither 0 nor NaN
    fabs    s\x, s\x
    frintz  s18, s17
    fcmp    s18, s17
    b.ne    Lnotint_\id
    fmov    s19, #0.5
    fmul    s18, s17, s19
    frintz  s19, s18
    fcmp    s19, s18
    b.eq    Lentry_\id                      // y even, or +-inf: pow(|x|, y)
    b       Lentry_\id\()n                  // y odd: -pow(|x|, y)
Lnotint_\id:
    lsl     w10, w9, #1
    cbz     w10, Lentry_\id                 // pow(-0, y) = pow(+0, y)
    lsr     w10, w9, #23
    cmp     w10, #0x1ff
    b.eq    Lentry_\id                      // pow(-inf, y) = pow(+inf, y)
    b       Lnan_\id                        // finite x < 0: NaN
    POWF    \kind, \x, \id\()n
    fneg    s\x, s\x
    b       Ldone_\id
    POWF_COLD \kind, \x, \id\()n, 1
.endif
.endm

// ─── Kernels ────────────────────────────────────────────────────────────────

// Throughput: in place in s0-s7, so v16-v19 and v30 are free for scratch.
.macro POWF_TPUT kind, p
    .globl  _powf_tput_\kind
    .p2align 4
_powf_tput_\kind:
    POWF_SETUP \kind
    .p2align 6
1:  LOAD8
    POWF    \kind, 0, \p\()0
    POWF    \kind, 1, \p\()1
    POWF    \kind, 2, \p\()2
    POWF    \kind, 3, \p\()3
    POWF    \kind, 4, \p\()4
    POWF    \kind, 5, \p\()5
    POWF    \kind, 6, \p\()6
    POWF    \kind, 7, \p\()7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret
    POWF_COLD \kind, 0, \p\()0
    POWF_COLD \kind, 1, \p\()1
    POWF_COLD \kind, 2, \p\()2
    POWF_COLD \kind, 3, \p\()3
    POWF_COLD \kind, 4, \p\()4
    POWF_COLD \kind, 5, \p\()5
    POWF_COLD \kind, 6, \p\()6
    POWF_COLD \kind, 7, \p\()7
.endm

// Latency: the chain runs through x in s1, which LINK 1, 1 refills each link.
.macro POWF_LAT kind, p
    .globl  _powf_lat_\kind
    .p2align 4
_powf_lat_\kind:
    POWF_SETUP \kind
    movi    v31.2d, #0
    movi    v1.2d, #0
    .p2align 6
1:  LINK    1, 1
    POWF    \kind, 1, \p\()0
    LINK    1, 1
    POWF    \kind, 1, \p\()1
    LINK    1, 1
    POWF    \kind, 1, \p\()2
    LINK    1, 1
    POWF    \kind, 1, \p\()3
    subs    x1, x1, #4
    b.ne    1b
    fmov    s0, s1
    ret
    POWF_COLD \kind, 1, \p\()0
    POWF_COLD \kind, 1, \p\()1
    POWF_COLD \kind, 1, \p\()2
    POWF_COLD \kind, 1, \p\()3
.endm

    POWF_TPUT frecpe, tf
    POWF_TPUT tab,    tt
    POWF_LAT  frecpe, lf
    POWF_LAT  tab,    lt

// The libm baseline, shaped like ../logf/'s _call pair: loop state and y in
// callee-saved registers, because powf may clobber every caller-saved one.
    .globl  _powf_tput_call
    .p2align 4
_powf_tput_call:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    str     d8, [sp, #40]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    adrp    x10, _powf_y@PAGE
    ldr     s8, [x10, _powf_y@PAGEOFF]
    .p2align 6
1:
    .rept 8
    ldr     s0, [x19], #4
    fmov    s1, s8
    bl      _powf
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     d8, [sp, #40]
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _powf_lat_call
    .p2align 4
_powf_lat_call:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    adrp    x10, _powf_y@PAGE
    ldr     s9, [x10, _powf_y@PAGEOFF]
    movi    v8.2d, #0                       // the zero, callee-saved across BL
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    0, 0, 8, x19                    // x straight into s0, the first argument
    fmov    s1, s9
    bl      _powf                           // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
