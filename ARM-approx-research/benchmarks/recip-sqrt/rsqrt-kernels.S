// Hand-written AArch64 kernels for benchmark-rsqrt.cpp.
//
// Ways to get 1/sqrt(x) in float32, each in the two loop shapes described in
// approx-kernels.inc. The loops are written here rather than in C so that no
// compiler sits between the instructions and the cycle count; the control
// loops live in approx-common.S.
//
//   sqdiv   FSQRT, then FDIV against 1.0: what clang -O3 emits for
//           1.0f / sqrtf(x). AArch64 has no reciprocal-square-root instruction
//           that is not an estimate, so this is the exact route -- but it rounds
//           twice, so it is faithful (within 1 ulp), not correctly rounded: 26%
//           of positive inputs come out 1 ulp off.
//   call    The same behind a BL to rsqrt_fn, which is byte-for-byte what clang
//           -O3 emits for `float rsqrt(float x) { return 1.0f / sqrtf(x); }`.
//   est0    FRSQRTE alone: the bare estimate, no Newton steps.
//   est1    FRSQRTE, then one Newton step.
//   est2    FRSQRTE, then two Newton steps: within 2 ulp of the correctly
//           rounded result on every finite input, NaN for x < 0, +-inf at +-0.
//
// FPCR.AH works as it does for FRECPE: with FEAT_AFP and FEAT_RPRES, setting it
// makes single-precision FRSQRTE a 12-bit estimate, and the driver sets it
// around whole runs, so est0 and est1 run in both modes. Under AH, FRSQRTE also
// flushes subnormal inputs to zero: every positive subnormal comes back +inf,
// and every negative one -inf where NaN is due.

#include "approx-kernels.inc"

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// One Newton step for 1/sqrt(x) on s\e: e *= (3 - (x*e)*e) / 2. Clobbers s\t.
//
// The grouping matters. The textbook order, e*e first, overflows for
// x < 2^-128, where e >= 2^64, and goes subnormal for x > 2^126; x*e first
// stays in range for every finite x. But at x = +-0 it meets e = +-inf, and
// FMUL makes 0*inf a NaN. FMULX exists for exactly this -- it is FMUL except
// 0*inf = +-2 -- and with it two steps land on +-inf at +-0 by themselves. One
// step alone leaves +0 at -inf.
.macro RSQRT_STEP e, x, t
    fmulx   s\t, s\x, s\e                   // x*e, with 0*inf = +-2
    frsqrts s\t, s\t, s\e                   // (3 - t*e)/2, fused
    fmul    s\e, s\e, s\t
.endm

.macro RSQRT_EST1 e, x, t
    frsqrte s\e, s\x                        // ~8-bit estimate (12-bit under AH)
    RSQRT_STEP \e, \x, \t                   // ~16 bits
.endm

.macro RSQRT_EST2 e, x, t
    frsqrte s\e, s\x
    RSQRT_STEP \e, \x, \t
    RSQRT_STEP \e, \x, \t                   // within 2 ulp
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  _rsqrt_tput_sqdiv
    .p2align 4
_rsqrt_tput_sqdiv:
    fmov    s29, #1.0
    .p2align 6
1:  LOAD8
    fsqrt   s0, s0
    fsqrt   s1, s1
    fsqrt   s2, s2
    fsqrt   s3, s3
    fsqrt   s4, s4
    fsqrt   s5, s5
    fsqrt   s6, s6
    fsqrt   s7, s7
    fdiv    s0, s29, s0
    fdiv    s1, s29, s1
    fdiv    s2, s29, s2
    fdiv    s3, s29, s3
    fdiv    s4, s29, s4
    fdiv    s5, s29, s5
    fdiv    s6, s29, s6
    fdiv    s7, s29, s7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _rsqrt_tput_call
    .p2align 4
_rsqrt_tput_call:
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
    bl      _rsqrt_fn
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _rsqrt_tput_est0
    .p2align 6
_rsqrt_tput_est0:
1:  LOAD8
    frsqrte s0, s0
    frsqrte s1, s1
    frsqrte s2, s2
    frsqrte s3, s3
    frsqrte s4, s4
    frsqrte s5, s5
    frsqrte s6, s6
    frsqrte s7, s7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _rsqrt_tput_est1
    .p2align 6
_rsqrt_tput_est1:
1:  LOAD8
    // Four temporaries shared round-robin: renaming makes the reuse free.
    RSQRT_EST1 16, 0, 24
    RSQRT_EST1 17, 1, 25
    RSQRT_EST1 18, 2, 26
    RSQRT_EST1 19, 3, 27
    RSQRT_EST1 20, 4, 24
    RSQRT_EST1 21, 5, 25
    RSQRT_EST1 22, 6, 26
    RSQRT_EST1 23, 7, 27
    STORE8  16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _rsqrt_tput_est2
    .p2align 6
_rsqrt_tput_est2:
1:  LOAD8
    RSQRT_EST2 16, 0, 24
    RSQRT_EST2 17, 1, 25
    RSQRT_EST2 18, 2, 26
    RSQRT_EST2 19, 3, 27
    RSQRT_EST2 20, 4, 24
    RSQRT_EST2 21, 5, 25
    RSQRT_EST2 22, 6, 26
    RSQRT_EST2 23, 7, 27
    STORE8  16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #8
    b.ne    1b
    ret

// The called function for the _call kernels. Kept in asm so it cannot change
// with compiler version, but it is exactly what clang emits for the C one-liner.
    .globl  _rsqrt_fn
    .p2align 4
_rsqrt_fn:
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  _rsqrt_lat_sqdiv
    .p2align 4
_rsqrt_lat_sqdiv:
    fmov    s29, #1.0
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    fsqrt   s1, s1
    fdiv    s0, s29, s1
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _rsqrt_lat_call
    .p2align 4
_rsqrt_lat_call:
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
    bl      _rsqrt_fn                       // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _rsqrt_lat_est0
    .p2align 4
_rsqrt_lat_est0:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    frsqrte s0, s1                          // the chain is AND -> ORR -> FRSQRTE
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _rsqrt_lat_est1
    .p2align 4
_rsqrt_lat_est1:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    RSQRT_EST1 0, 1, 4
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _rsqrt_lat_est2
    .p2align 4
_rsqrt_lat_est2:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    RSQRT_EST2 0, 1, 4
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret
