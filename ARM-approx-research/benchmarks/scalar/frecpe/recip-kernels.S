// Hand-written AArch64 kernels for benchmark-recip.cpp.
//
// Three ways to get 1/x in float32, each in two loop shapes. The loops are
// written here rather than in C so that no compiler sits between the
// instructions and the cycle count: what is listed below is exactly what
// retires, and the only C involvement is one call per 4096-element pass.
//
//   fdiv    FDIV against 1.0. Correctly rounded; the reference.
//   call    The same FDIV behind a BL to recip_fn, which is byte-for-byte what
//           clang -O3 emits for `float recip(float x) { return 1.0f / x; }`.
//   est2    FRECPE, then two FRECPS/FMUL Newton steps, then a sign fix.
//           Within 1 ulp of 1/x for every finite float, verified exhaustively
//           by the benchmark's accuracy pass. One Newton step (the textbook
//           three-instruction sequence) is up to 129 ulp off, so it does not
//           reach float32.
//   est0    FRECPE alone: the bare estimate, no Newton steps. Not a float32
//           reciprocal -- it is timed to isolate the estimate instruction's
//           own latency and throughput from the refinement est2 adds.
//   est1    FRECPE, then one FRECPS/FMUL Newton step: the textbook sequence.
//           Run in both modes: on the 8-bit estimate (~2^-16) and, with
//           FPCR.AH set, on the 12-bit one (below).
//
// The control loops, the loop-shape macros and the register conventions are
// shared with the other kernel files: see approx-kernels.inc and
// approx-common.S.
//
// FPCR.AH and the 12-bit estimate. With FEAT_AFP and FEAT_RPRES (both present
// on Apple M5), setting FPCR.AH makes single-precision FRECPE return a 12-bit
// estimate instead of an 8-bit one. The driver, not these kernels, sets it:
// toggling AH costs ~75 cycles, about 2% of a 4096-element pass, so it is set
// once around a whole timed run instead. est0 and est1 therefore run in both
// modes.
// AH also forces flush-to-zero inside FRECPE and FRECPS, so under AH every
// subnormal input becomes infinity and every result below 2^-126 becomes zero;
// benchmark-recip.cpp has the measured consequences.

#include "approx-kernels.inc"

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// s\e = 1/s\x within 1 ulp; clobbers s\t. Needs the sign mask in v30.
//
// The BIT is not optional. For |x| < 2^-128 the true reciprocal overflows, and
// FRECPE returns an infinity of the right sign. FRECPS then computes
// 2 - x*inf = -inf, and multiplying the estimate by it flips the sign; the
// second step flips nothing back. Every such input (4,194,302 of them) came out
// as the wrong infinity until the sign was copied over from x.
.macro RECIP_EST2 e, x, t
    frecpe  s\e, s\x                        // ~8-bit estimate
    frecps  s\t, s\x, s\e                   // t = 2 - x*e, fused
    fmul    s\e, s\e, s\t                   // Newton step 1: ~16 bits
    frecps  s\t, s\x, s\e
    fmul    s\e, s\e, s\t                   // Newton step 2: within 1 ulp
    bit     v\e\().8b, v\x\().8b, v30.8b    // take x's sign bit
.endm

// s\e = FRECPE refined by one Newton step; clobbers s\t. With FPCR.AH set the
// estimate is 12-bit and this lands within 3 ulp on normal inputs and results;
// with AH clear it is the 8-bit sequence, up to 129 ulp off. No sign fix: under
// AH the inputs that needed one are flushed to zero, and FRECPS(0, inf) = 2.
.macro RECIP_EST1 e, x, t
    frecpe  s\e, s\x
    frecps  s\t, s\x, s\e
    fmul    s\e, s\e, s\t
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  _recip_tput_fdiv
    .p2align 4
_recip_tput_fdiv:
    fmov    s29, #1.0
    .p2align 6
1:  LOAD8
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

    .globl  _recip_tput_call
    .p2align 4
_recip_tput_call:
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
    bl      _recip_fn
    str     s0, [x20], #4
    .endr
    subs    x21, x21, #8
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _recip_tput_est2
    .p2align 4
_recip_tput_est2:
    movi    v30.2s, #0x80, lsl #24
    .p2align 6
1:  LOAD8
    // Four temporaries shared round-robin: renaming makes the reuse free.
    RECIP_EST2 16, 0, 24
    RECIP_EST2 17, 1, 25
    RECIP_EST2 18, 2, 26
    RECIP_EST2 19, 3, 27
    RECIP_EST2 20, 4, 24
    RECIP_EST2 21, 5, 25
    RECIP_EST2 22, 6, 26
    RECIP_EST2 23, 7, 27
    STORE8  16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _recip_tput_est0
    .p2align 6
_recip_tput_est0:
1:  LOAD8
    frecpe  s0, s0
    frecpe  s1, s1
    frecpe  s2, s2
    frecpe  s3, s3
    frecpe  s4, s4
    frecpe  s5, s5
    frecpe  s6, s6
    frecpe  s7, s7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _recip_tput_est1
    .p2align 6
_recip_tput_est1:
1:  LOAD8
    RECIP_EST1 16, 0, 24
    RECIP_EST1 17, 1, 25
    RECIP_EST1 18, 2, 26
    RECIP_EST1 19, 3, 27
    RECIP_EST1 20, 4, 24
    RECIP_EST1 21, 5, 25
    RECIP_EST1 22, 6, 26
    RECIP_EST1 23, 7, 27
    STORE8  16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #8
    b.ne    1b
    ret

// The called function for the _call kernels. Kept in asm so it cannot change
// with compiler version, but it is exactly what clang emits for the C one-liner.
    .globl  _recip_fn
    .p2align 4
_recip_fn:
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  _recip_lat_fdiv
    .p2align 4
_recip_lat_fdiv:
    fmov    s29, #1.0
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    fdiv    s0, s29, s1
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _recip_lat_call
    .p2align 4
_recip_lat_call:
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
    bl      _recip_fn                       // result back in s0, feeding the next LINK
    .endr
    subs    x20, x20, #4
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _recip_lat_est2
    .p2align 4
_recip_lat_est2:
    movi    v30.2s, #0x80, lsl #24
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    RECIP_EST2 0, 1, 4
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _recip_lat_est0
    .p2align 4
_recip_lat_est0:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    frecpe  s0, s1                          // the chain is AND -> ORR -> FRECPE
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

    .globl  _recip_lat_est1
    .p2align 4
_recip_lat_est1:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    1, 0
    RECIP_EST1 0, 1, 4
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret
