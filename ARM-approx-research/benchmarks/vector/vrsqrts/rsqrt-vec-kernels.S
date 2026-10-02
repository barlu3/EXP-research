// Hand-written AArch64 kernels for benchmark-rsqrt-vec.cpp.
//
// The variants of ../../scalar/frsqrte/rsqrt-kernels.S, instruction for
// instruction, on 4-lane float32 vectors (.4s) instead of scalars, in the two
// loop shapes described in ../vec-kernels.inc. Each lane computes exactly what
// the scalar kernel computes for that input, so the accuracy pass must
// reproduce the scalar report's table row for row; only the timings can differ.
//
//   sqdiv   FSQRT v.4s, then FDIV v.4s against 1.0: what clang -O3 emits for
//           1.0f / sqrt over a float32x4_t. Faithful, not correctly rounded.
//   call    The same behind a BL to rsqrt4_fn, byte-for-byte what clang -O3
//           emits for the one-line float32x4_t function.
//   est0    FRSQRTE alone.
//   est1    FRSQRTE, then one Newton step.
//   est2    FRSQRTE, then two Newton steps.
//
// FPCR.AH applies to the vector forms as it does to the scalar ones, so est0
// and est1 run in both modes, as in the scalar benchmark.

#include "vec-kernels.inc"

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// One Newton step for 1/sqrt(x) on v\e, per lane: e *= (3 - (x*e)*e) / 2.
// Clobbers v\t. Grouped as RSQRT_STEP in ../../scalar/frsqrte/rsqrt-kernels.S,
// and for the same reasons: x*e first stays in range for every finite x, and
// FMULX makes 0*inf = +-2 so +-0 lands on +-inf after two steps.
.macro RSQRT_STEPV e, x, t
    fmulx   v\t\().4s, v\x\().4s, v\e\().4s         // x*e, with 0*inf = +-2
    frsqrts v\t\().4s, v\t\().4s, v\e\().4s         // (3 - t*e)/2, fused
    fmul    v\e\().4s, v\e\().4s, v\t\().4s
.endm

.macro RSQRT_EST1V e, x, t
    frsqrte v\e\().4s, v\x\().4s                    // ~8-bit estimate (12-bit under AH)
    RSQRT_STEPV \e, \x, \t                          // ~16 bits
.endm

.macro RSQRT_EST2V e, x, t
    frsqrte v\e\().4s, v\x\().4s
    RSQRT_STEPV \e, \x, \t
    RSQRT_STEPV \e, \x, \t                          // within 2 ulp
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  _rsqrtv_tput_sqdiv
    .p2align 4
_rsqrtv_tput_sqdiv:
    fmov    v29.4s, #1.0
    .p2align 6
1:  LOADV8
    fsqrt   v0.4s, v0.4s
    fsqrt   v1.4s, v1.4s
    fsqrt   v2.4s, v2.4s
    fsqrt   v3.4s, v3.4s
    fsqrt   v4.4s, v4.4s
    fsqrt   v5.4s, v5.4s
    fsqrt   v6.4s, v6.4s
    fsqrt   v7.4s, v7.4s
    fdiv    v0.4s, v29.4s, v0.4s
    fdiv    v1.4s, v29.4s, v1.4s
    fdiv    v2.4s, v29.4s, v2.4s
    fdiv    v3.4s, v29.4s, v3.4s
    fdiv    v4.4s, v29.4s, v4.4s
    fdiv    v5.4s, v29.4s, v5.4s
    fdiv    v6.4s, v29.4s, v6.4s
    fdiv    v7.4s, v29.4s, v7.4s
    STOREV8 0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _rsqrtv_tput_call
    .p2align 4
_rsqrtv_tput_call:
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
    ldr     q0, [x19], #16
    bl      _rsqrt4_fn
    str     q0, [x20], #16
    .endr
    subs    x21, x21, #32
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _rsqrtv_tput_est0
    .p2align 6
_rsqrtv_tput_est0:
1:  LOADV8
    frsqrte v0.4s, v0.4s
    frsqrte v1.4s, v1.4s
    frsqrte v2.4s, v2.4s
    frsqrte v3.4s, v3.4s
    frsqrte v4.4s, v4.4s
    frsqrte v5.4s, v5.4s
    frsqrte v6.4s, v6.4s
    frsqrte v7.4s, v7.4s
    STOREV8 0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _rsqrtv_tput_est1
    .p2align 6
_rsqrtv_tput_est1:
1:  LOADV8
    // Four temporaries shared round-robin: renaming makes the reuse free.
    RSQRT_EST1V 16, 0, 24
    RSQRT_EST1V 17, 1, 25
    RSQRT_EST1V 18, 2, 26
    RSQRT_EST1V 19, 3, 27
    RSQRT_EST1V 20, 4, 24
    RSQRT_EST1V 21, 5, 25
    RSQRT_EST1V 22, 6, 26
    RSQRT_EST1V 23, 7, 27
    STOREV8 16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _rsqrtv_tput_est2
    .p2align 6
_rsqrtv_tput_est2:
1:  LOADV8
    RSQRT_EST2V 16, 0, 24
    RSQRT_EST2V 17, 1, 25
    RSQRT_EST2V 18, 2, 26
    RSQRT_EST2V 19, 3, 27
    RSQRT_EST2V 20, 4, 24
    RSQRT_EST2V 21, 5, 25
    RSQRT_EST2V 22, 6, 26
    RSQRT_EST2V 23, 7, 27
    STOREV8 16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #32
    b.ne    1b
    ret

// The called function for the _call kernels. Kept in asm so it cannot change
// with compiler version, but it is exactly what clang emits for the C one-liner.
    .globl  _rsqrt4_fn
    .p2align 4
_rsqrt4_fn:
    fsqrt   v0.4s, v0.4s
    fmov    v1.4s, #1.0
    fdiv    v0.4s, v1.4s, v0.4s
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  _rsqrtv_lat_sqdiv
    .p2align 4
_rsqrtv_lat_sqdiv:
    fmov    v29.4s, #1.0
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    fsqrt   v1.4s, v1.4s
    fdiv    v0.4s, v29.4s, v1.4s
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _rsqrtv_lat_call
    .p2align 4
_rsqrtv_lat_call:
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
    LINKV   0, 0, 8, x19                    // operand straight into v0, the argument
    bl      _rsqrt4_fn                      // result back in v0, feeding the next LINKV
    .endr
    subs    x20, x20, #16
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _rsqrtv_lat_est0
    .p2align 4
_rsqrtv_lat_est0:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    frsqrte v0.4s, v1.4s                    // the chain is AND -> ORR -> FRSQRTE
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _rsqrtv_lat_est1
    .p2align 4
_rsqrtv_lat_est1:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    RSQRT_EST1V 0, 1, 4
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _rsqrtv_lat_est2
    .p2align 4
_rsqrtv_lat_est2:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    RSQRT_EST2V 0, 1, 4
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret
