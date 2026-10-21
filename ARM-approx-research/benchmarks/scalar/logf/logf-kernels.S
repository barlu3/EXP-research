// Hand-written AArch64 kernels for benchmark-logf.cpp.
//
// float32 log(x) three ways, each in the two loop shapes described in
// approx-kernels.inc:
//
//   call    BL to the system libm's logf: the baseline.
//   frecpe  r = x*FRECPE(x) - 1. For every normal x, FRECPE's 8-bit estimate is
//           exactly 2^-k * ehat(i), where k is x's unbiased exponent and i its
//           top eight fraction bits: a 256-entry reciprocal table that applies
//           the 2^-k itself. The FMA that forms r is exact in float32 and
//           |r| <= 2^-8.45. Then log(x) = k*ln2 - log(ehat(i)) + log1p(r).
//   tab     The same reduction with FRECPE's values read from memory instead:
//           z = x*FRECPX(x) = 2m exactly, then r = z * (ehat(i)/2) - 1. That is
//           bit for bit the same r, so both kernels return identical results
//           and differ only in how the reciprocal is fetched.
//
// The rest is shared, all of it in double: -log(ehat(i)) from a 256-entry table
// built by the driver, y0 = k*ln2 - log(ehat(i)), and
// log1p(r) ~ r - r^2/2 + r^3/3, whose truncation error is below 2^-35.8. One
// rounding to float at the end, so both stay within 1 ulp of the correctly
// rounded result; the driver's accuracy pass holds them to that.
//
// Off the hot path, in a stub after each kernel's ret:
//   - inputs that are not positive normals. Subnormals are scaled to normals
//     and rejoin the main path; the rest are -inf, +inf or NaN.
//   - x in [1, 1 + 2^-8). ehat is 511/512 there rather than 1, so log(x) would
//     be the difference of two terms near 2^-9, with absolute error ~2^-36 on a
//     result that can be as small as 2^-23. The stub takes r = x - 1 (exact)
//     and y0 = 0 instead. The cell just below 1 needs nothing: there ehat = 1/2
//     exactly, so y0 = -ln2 + ln2 = 0 exactly and r = x - 1 already.
// The hot path pays one not-taken branch per element for both. The stubs are in
// the same section as the kernels because Mach-O cannot relocate a conditional
// branch into another one.
//
// Assembled once per base. With LOGF_BASE (2 or 10) defined the same kernels
// compute log2 or log10, under the same symbol names: the driver fills
// logf_logc with -log_b(ehat(i)), d25 is log_b(2), and the last step is
// y0 + log1p(r)/ln(b), one FMADD in place of the FADD. Every error bound above
// is relative, so it holds in any base.
//
// Registers, on top of approx-kernels.inc's (v29 = 1.0f, v31 = 0, LINK's v2/v3):
//   w5 = 0x7f000000   w6 = 0x3f800000 (1.0f)   w7 = 0x8000
//   x3 = logf_logc    x4 = logf_recip (tab only)
//   d25 = log_b(2)    d26 = -1/2               d27 = 1/3     d28 = 1/ln(b)
//   scratch: w9 (ix), w10, w12, w15 (i), w16 (k), d16-d20
// Every element reuses the same scratch registers; renaming makes that free.

#include "approx-kernels.inc"

#if !defined(LOGF_BASE)
#define LOGF_LIBM _logf
#elif LOGF_BASE == 2
#define LOGF_LIBM _log2f
#elif LOGF_BASE == 10
#define LOGF_LIBM _log10f
#else
#error "LOGF_BASE must be 2 or 10, or undefined for the natural log"
#endif

    .text

// ─── The log sequence ───────────────────────────────────────────────────────

// Constants and table addresses, once per kernel call.
.macro LOGF_SETUP kind
    mov     w5, #0x7f000000
    mov     w6, #0x3f800000
    mov     w7, #0x8000
    adrp    x3, _logf_logc@PAGE
    add     x3, x3, _logf_logc@PAGEOFF
.ifc \kind, tab
    adrp    x4, _logf_recip@PAGE
    add     x4, x4, _logf_recip@PAGEOFF
.endif
    adrp    x10, _logf_base@PAGE
    add     x10, x10, _logf_base@PAGEOFF
    ldp     d25, d28, [x10]                 // log_b(2), 1/ln(b); rounded to double
    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fd5, lsl #48
    fmov    d27, x10                        // 1/3
    fmov    d26, #-0.5
    fmov    s29, #1.0
.endm

// s\x = logf(s\x), in place. \id names this instance's labels, which its
// LOGF_COLD stub shares.
.macro LOGF kind, x, id
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
    ldr     d18, [x3, w15, uxtw #3]         // -log(ehat(i))
    scvtf   d19, w16
    fmadd   d18, d19, d25, d18              // y0 = k*ln2 - log(ehat(i))
Lpoly_\id:
    fmul    d20, d16, d16
    fmadd   d19, d27, d16, d26              // r/3 - 1/2
    fmadd   d20, d19, d20, d16              // log1p(r) ~ r + r^2*(r/3 - 1/2)
#if defined(LOGF_BASE)
    fmadd   d20, d20, d28, d18              // y0 + log1p(r)/ln(b)
#else
    fadd    d20, d20, d18
#endif
    fcvt    s\x, d20
Ldone_\id:
.endm

// The out-of-line half of LOGF \kind, \x, \id. Placed after the kernel's ret.
.macro LOGF_COLD x, id
Lcold_\id:
    cmp     w10, w5
    b.hs    Lspecial_\id
    fsub    s16, s\x, s29                   // x in [1, 1 + 2^-8): r = x - 1, exact
    fcvt    d16, s16
    movi    d18, #0                         // y0 = 0
    b       Lpoly_\id
Lspecial_\id:
    lsl     w10, w9, #1
    cbz     w10, Lzero_\id
    tbnz    w9, #31, Lnan_\id
    lsr     w10, w9, #23
    cbnz    w10, Linfnan_\id
    ucvtf   s\x, w9                         // positive subnormal: x * 2^149, exact
    fmov    w9, s\x
    mov     w10, #0x4a800000                // 149 << 23, so k comes out 149 lower
    sub     w9, w9, w10
    b       Lmain_\id
Lzero_\id:
    mov     w10, #0xff800000
    fmov    s\x, w10                        // log(+-0) = -inf
    b       Ldone_\id
Lnan_\id:
    mov     w10, #0x7fc00000
    fmov    s\x, w10                        // log(x < 0) = NaN
    b       Ldone_\id
Linfnan_\id:
    fadd    s\x, s\x, s\x                   // log(+inf) = +inf; NaN stays NaN
    b       Ldone_\id
.endm

// ─── Kernels ────────────────────────────────────────────────────────────────

// Throughput: in place in s0-s7, so v16-v20 are free for scratch.
.macro LOGF_TPUT kind, p
    .globl  _logf_tput_\kind
    .p2align 4
_logf_tput_\kind:
    LOGF_SETUP \kind
    .p2align 6
1:  LOAD8
    LOGF    \kind, 0, \p\()0
    LOGF    \kind, 1, \p\()1
    LOGF    \kind, 2, \p\()2
    LOGF    \kind, 3, \p\()3
    LOGF    \kind, 4, \p\()4
    LOGF    \kind, 5, \p\()5
    LOGF    \kind, 6, \p\()6
    LOGF    \kind, 7, \p\()7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret
    LOGF_COLD 0, \p\()0
    LOGF_COLD 1, \p\()1
    LOGF_COLD 2, \p\()2
    LOGF_COLD 3, \p\()3
    LOGF_COLD 4, \p\()4
    LOGF_COLD 5, \p\()5
    LOGF_COLD 6, \p\()6
    LOGF_COLD 7, \p\()7
.endm

// Latency: the chain runs through s1, which LINK 1, 1 refills each link.
.macro LOGF_LAT kind, p
    .globl  _logf_lat_\kind
    .p2align 4
_logf_lat_\kind:
    LOGF_SETUP \kind
    movi    v31.2d, #0
    movi    v1.2d, #0
    .p2align 6
1:  LINK    1, 1
    LOGF    \kind, 1, \p\()0
    LINK    1, 1
    LOGF    \kind, 1, \p\()1
    LINK    1, 1
    LOGF    \kind, 1, \p\()2
    LINK    1, 1
    LOGF    \kind, 1, \p\()3
    subs    x1, x1, #4
    b.ne    1b
    fmov    s0, s1
    ret
    LOGF_COLD 1, \p\()0
    LOGF_COLD 1, \p\()1
    LOGF_COLD 1, \p\()2
    LOGF_COLD 1, \p\()3
.endm

    LOGF_TPUT frecpe, tf
    LOGF_TPUT tab,    tt
    LOGF_LAT  frecpe, lf
    LOGF_LAT  tab,    lt

// The libm baseline (logf, log2f or log10f), shaped like recip-kernels.S's
// _call pair: loop state in callee-saved registers, because the call may
// clobber every caller-saved one.
    .globl  _logf_tput_call
    .p2align 4
_logf_tput_call:
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
    bl      LOGF_LIBM
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _logf_lat_call
    .p2align 4
_logf_lat_call:
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
    bl      LOGF_LIBM                       // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
