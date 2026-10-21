// Hand-written AArch64 kernels for benchmark-asinhf-acoshf.cpp.
//
// float32 asinh(x) and acosh(x) three ways each, in the two loop shapes
// described in approx-kernels.inc, all in double:
//
//   asinh(x) = sign(x) * log(w),  w = |x| + sqrt(x^2 + 1)
//   acosh(x) = log(w),            w = x + sqrt(x^2 - 1)
//
//   call    BL to the system libm's asinhf or acoshf: the baseline.
//   frecpe  log(w) with ../logf/'s reduction, on a double: r = w*FRECPE(w) - 1.
//           Double FRECPE uses the same 256-entry estimate, indexed by w's top
//           eight fraction bits, and r is still exact: 53 significant bits
//           times FRECPE's 9, less the eight that cancel against 1, fit in 53.
//   tab     The same r with the estimate read from memory: z = w*FRECPX(w) =
//           2m, then r = z * (ehat(i)/2) - 1. Bit for bit the same r, so both
//           kernels return identical results and differ only in the fetch.
//
// Unlike logf, the log's argument is computed mid-function. Both kernels move w
// to an integer register for i and k (the -log(ehat(i)) load and k*ln2); only
// tab also waits on that move for r.
//
// The log is ../logf/'s: y0 = k*ln2 - log(ehat(i)), log1p(r) ~ r - r^2/2 + r^3/3,
// one rounding to float at the end, within 1 ulp; the driver's accuracy pass
// holds every variant to that. w itself carries one rounding of 2^-52 or so,
// which costs nothing once log(w) >= 2^-8. Below that, out of line:
//   asinh  |x| < 2^-8 (so w < 1 + 2^-8): x(1 - x^2/6), relative error < 2^-35.7.
//          At or above 2^-8, w >= 1 + 2^-8 and the near-1 cell never comes up.
//   acosh  w in [1, 1 + 2^-8), x - 1 below about 2^-17: r = (x - 1) +
//          sqrt(x^2 - 1), which is w - 1 without w's rounding, and y0 = 0.
// Specials too: asinh(+-inf) = +-inf, acosh(x < 1) = NaN, NaN stays NaN. The
// stubs sit after each kernel's ret, in the same section, because Mach-O cannot
// relocate a conditional branch into another one.
//
// Registers, on top of approx-kernels.inc's (v30 = sign mask, v31 = 0, LINK's v2/v3):
//   w5 = 0x7f800000   x3 = ahyp_logc   x4 = ahyp_recip (tab only)
//   asinh: w6 = 2^-8 (bits)   w7 = 0x7f800000 - w6   d28 = -1/6
//   acosh: w6 = 1.0f (bits)   w7 = 0x7f800000 - w6   x13 = 1.0 (bits)   x14 = 2^44
//   d24 = 1.0   d25 = ln2   d26 = -1/2   d27 = 1/3
//   scratch: x9, x10, x15 (i), x16 (k), d16-d20
// Every element reuses the same scratch registers; renaming makes that free.

#include "approx-kernels.inc"

    .text

// ─── The sequences ──────────────────────────────────────────────────────────

.macro AHYP_SETUP fn, kind
    adrp    x3, _ahyp_logc@PAGE
    add     x3, x3, _ahyp_logc@PAGEOFF
.ifc \kind, tab
    adrp    x4, _ahyp_recip@PAGE
    add     x4, x4, _ahyp_recip@PAGEOFF
.endif
    mov     w5, #0x7f800000
.ifc \fn, asinhf
    mov     w6, #0x3b800000                 // 2^-8
    mov     w7, #0x44000000
    movi    v30.4s, #0x80, lsl #24
    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0xbfc5, lsl #48
    fmov    d28, x10                        // -1/6
.else
    mov     w6, #0x3f800000                 // 1.0f
    mov     w7, #0x40000000
    mov     x13, #0x3ff0000000000000        // 1.0
    mov     x14, #0x100000000000            // 2^44: w in [1, 1 + 2^-8) below it
.endif
    fmov    d24, #1.0
    movz    x10, #0x39ef
    movk    x10, #0xfefa, lsl #16
    movk    x10, #0x2e42, lsl #32
    movk    x10, #0x3fe6, lsl #48
    fmov    d25, x10                        // ln2, rounded to double
    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fd5, lsl #48
    fmov    d27, x10                        // 1/3
    fmov    d26, #-0.5
.endm

// d20 = log(w) for w = d19 >= 1, its bits in x9. A near-1 path joins at
// Lpoly_\id with r in d16 and y0 = 0 in d18.
.macro LOGW kind, id
    ubfx    x15, x9, #44, #8                // i: the top eight fraction bits
    lsr     x16, x9, #52
    sub     w16, w16, #1023                 // k
.ifc \kind, frecpe
    frecpe  d16, d19                        // e = 2^-k * ehat(i)
    fnmsub  d16, d19, d16, d24              // r = w*e - 1, exact
.else
    ldr     d17, [x4, x15, lsl #3]          // ehat(i)/2
    frecpx  d16, d19
    fmul    d16, d19, d16                   // z = 2m, exact
    fnmsub  d16, d16, d17, d24              // r = z*ehat(i)/2 - 1, exact
.endif
    ldr     d18, [x3, x15, lsl #3]          // -log(ehat(i))
    scvtf   d17, w16
    fmadd   d18, d17, d25, d18              // y0 = k*ln2 - log(ehat(i))
Lpoly_\id:
    fmul    d20, d16, d16
    fmadd   d17, d27, d16, d26              // r/3 - 1/2
    fmadd   d20, d17, d20, d16              // log1p(r) ~ r + r^2*(r/3 - 1/2)
    fadd    d20, d20, d18
.endm

// s\x = asinh(s\x), in place. \id names this instance's labels.
.macro asinhf kind, x, id
    fmov    w9, s\x
    and     w10, w9, #0x7fffffff
    sub     w10, w10, w6
    cmp     w10, w7                         // lo: 2^-8 <= |x| < inf
    b.hs    Lcold_\id
    fabs    s16, s\x
    fcvt    d17, s16                        // |x|
    fmadd   d16, d17, d17, d24              // x^2 + 1
    fsqrt   d16, d16
    fadd    d19, d17, d16                   // w >= 1 + 2^-8
    fmov    x9, d19
    LOGW    \kind, \id
    fcvt    s16, d20
    bif     v\x\().8b, v16.8b, v30.8b       // asinh(|x|), with x's sign
Ldone_\id:
.endm

.macro asinhf_cold x, id
Lcold_\id:
    and     w10, w9, #0x7fffffff
    cmp     w10, w5
    b.hs    Linfnan_\id
    fcvt    d17, s\x                        // |x| < 2^-8
    fmul    d16, d17, d17
    fmadd   d16, d16, d28, d24              // 1 - x^2/6
    fmul    d16, d16, d17                   // x(1 - x^2/6), keeping -0
    fcvt    s\x, d16
    b       Ldone_\id
Linfnan_\id:
    fadd    s\x, s\x, s\x                   // asinh(+-inf) = +-inf; NaN stays NaN
    b       Ldone_\id
.endm

// s\x = acosh(s\x), in place.
.macro acoshf kind, x, id
    fmov    w9, s\x
    sub     w10, w9, w6
    cmp     w10, w7                         // lo: 1 <= x < inf
    b.hs    Lcold_\id
    fcvt    d17, s\x
    fnmsub  d16, d17, d17, d24              // x^2 - 1
    fsqrt   d16, d16
    fadd    d19, d17, d16                   // w = x + sqrt(x^2 - 1)
    fmov    x9, d19
    sub     x10, x9, x13
    cmp     x10, x14                        // lo: w in [1, 1 + 2^-8)
    b.lo    Lnear_\id
    LOGW    \kind, \id
    fcvt    s\x, d20
Ldone_\id:
.endm

.macro acoshf_cold x, id
Lcold_\id:
    cmp     w9, w5
    b.eq    Ldone_\id                       // acosh(+inf) = +inf
    mov     w10, #0x7fc00000
    fmov    s\x, w10                        // x < 1, or NaN
    b       Ldone_\id
Lnear_\id:
    fsub    d17, d17, d24                   // x - 1, exact
    fadd    d16, d17, d16                   // r = (x - 1) + sqrt(x^2 - 1)
    movi    d18, #0                         // y0 = 0
    b       Lpoly_\id
.endm

// ─── Kernels ────────────────────────────────────────────────────────────────

// Throughput: in place in s0-s7, so v16-v20 are free for scratch.
.macro AHYP_TPUT fn, kind, p
    .globl  _\fn\()_tput_\kind
    .p2align 4
_\fn\()_tput_\kind:
    AHYP_SETUP \fn, \kind
    .p2align 6
1:  LOAD8
    \fn     \kind, 0, \p\()0
    \fn     \kind, 1, \p\()1
    \fn     \kind, 2, \p\()2
    \fn     \kind, 3, \p\()3
    \fn     \kind, 4, \p\()4
    \fn     \kind, 5, \p\()5
    \fn     \kind, 6, \p\()6
    \fn     \kind, 7, \p\()7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret
    \fn\()_cold 0, \p\()0
    \fn\()_cold 1, \p\()1
    \fn\()_cold 2, \p\()2
    \fn\()_cold 3, \p\()3
    \fn\()_cold 4, \p\()4
    \fn\()_cold 5, \p\()5
    \fn\()_cold 6, \p\()6
    \fn\()_cold 7, \p\()7
.endm

// Latency: the chain runs through s1, which LINK 1, 1 refills each link.
.macro AHYP_LAT fn, kind, p
    .globl  _\fn\()_lat_\kind
    .p2align 4
_\fn\()_lat_\kind:
    AHYP_SETUP \fn, \kind
    movi    v31.2d, #0
    movi    v1.2d, #0
    .p2align 6
1:  LINK    1, 1
    \fn     \kind, 1, \p\()0
    LINK    1, 1
    \fn     \kind, 1, \p\()1
    LINK    1, 1
    \fn     \kind, 1, \p\()2
    LINK    1, 1
    \fn     \kind, 1, \p\()3
    subs    x1, x1, #4
    b.ne    1b
    fmov    s0, s1
    ret
    \fn\()_cold 1, \p\()0
    \fn\()_cold 1, \p\()1
    \fn\()_cold 1, \p\()2
    \fn\()_cold 1, \p\()3
.endm

// The libm baseline, shaped like ../logf/'s _call pair.
.macro AHYP_CALL fn
    .globl  _\fn\()_tput_call
    .p2align 4
_\fn\()_tput_call:
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
    bl      _\fn
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _\fn\()_lat_call
    .p2align 4
_\fn\()_lat_call:
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
    bl      _\fn                            // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.endm

    AHYP_TPUT asinhf, frecpe, stf
    AHYP_TPUT asinhf, tab,    stt
    AHYP_LAT  asinhf, frecpe, slf
    AHYP_LAT  asinhf, tab,    slt
    AHYP_CALL asinhf
    AHYP_TPUT acoshf, frecpe, ctf
    AHYP_TPUT acoshf, tab,    ctt
    AHYP_LAT  acoshf, frecpe, clf
    AHYP_LAT  acoshf, tab,    clt
    AHYP_CALL acoshf
