// Hand-written AArch64 kernels for benchmark-recip-vec.cpp.
//
// The variants of ../../scalar/frecpe/recip-kernels.S, instruction for
// instruction, on 4-lane float32 vectors (.4s) instead of scalars, in the two
// loop shapes described in ../vec-kernels.inc. Each lane computes exactly what
// the scalar kernel computes for that input, so the accuracy pass must
// reproduce the scalar report's table row for row; only the timings can differ.
//
//   fdiv    FDIV v.4s against 1.0 in every lane. Correctly rounded.
//   call    The same FDIV behind a BL to recip4_fn, which is byte-for-byte what
//           clang -O3 emits for
//           `float32x4_t recip4(float32x4_t x) { return 1.0f / x; }`.
//   est2    FRECPE, then two FRECPS/FMUL Newton steps, then the sign fix.
//   est0    FRECPE alone.
//   est1    FRECPE, then one FRECPS/FMUL Newton step.
//
// FPCR.AH applies to the vector forms as it does to the scalar ones, so est0
// and est1 run in both modes, as in the scalar benchmark.

#include "vec-kernels.inc"

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// v\e = 1/v\x within 1 ulp per lane; clobbers v\t. Needs the sign mask in v30.
// The BIT copies each lane's sign for the same reason as RECIP_EST2 in
// ../../scalar/frecpe/recip-kernels.S: below 2^-128 the two steps otherwise
// land on the wrong-signed infinity.
.macro RECIP_EST2V e, x, t
    frecpe  v\e\().4s, v\x\().4s                    // ~8-bit estimate
    frecps  v\t\().4s, v\x\().4s, v\e\().4s         // t = 2 - x*e, fused
    fmul    v\e\().4s, v\e\().4s, v\t\().4s         // Newton step 1: ~16 bits
    frecps  v\t\().4s, v\x\().4s, v\e\().4s
    fmul    v\e\().4s, v\e\().4s, v\t\().4s         // Newton step 2: within 1 ulp
    bit     v\e\().16b, v\x\().16b, v30.16b         // take x's sign bit
.endm

// v\e = FRECPE refined by one Newton step; clobbers v\t.
.macro RECIP_EST1V e, x, t
    frecpe  v\e\().4s, v\x\().4s
    frecps  v\t\().4s, v\x\().4s, v\e\().4s
    fmul    v\e\().4s, v\e\().4s, v\t\().4s
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  _recipv_tput_fdiv
    .p2align 4
_recipv_tput_fdiv:
    fmov    v29.4s, #1.0
    .p2align 6
1:  LOADV8
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

    .globl  _recipv_tput_call
    .p2align 4
_recipv_tput_call:
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
    bl      _recip4_fn
    str     q0, [x20], #16
    .endr
    subs    x21, x21, #32
    b.ne    1b
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _recipv_tput_est2
    .p2align 4
_recipv_tput_est2:
    movi    v30.4s, #0x80, lsl #24
    .p2align 6
1:  LOADV8
    // Four temporaries shared round-robin: renaming makes the reuse free.
    RECIP_EST2V 16, 0, 24
    RECIP_EST2V 17, 1, 25
    RECIP_EST2V 18, 2, 26
    RECIP_EST2V 19, 3, 27
    RECIP_EST2V 20, 4, 24
    RECIP_EST2V 21, 5, 25
    RECIP_EST2V 22, 6, 26
    RECIP_EST2V 23, 7, 27
    STOREV8 16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _recipv_tput_est0
    .p2align 6
_recipv_tput_est0:
1:  LOADV8
    frecpe  v0.4s, v0.4s
    frecpe  v1.4s, v1.4s
    frecpe  v2.4s, v2.4s
    frecpe  v3.4s, v3.4s
    frecpe  v4.4s, v4.4s
    frecpe  v5.4s, v5.4s
    frecpe  v6.4s, v6.4s
    frecpe  v7.4s, v7.4s
    STOREV8 0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _recipv_tput_est1
    .p2align 6
_recipv_tput_est1:
1:  LOADV8
    RECIP_EST1V 16, 0, 24
    RECIP_EST1V 17, 1, 25
    RECIP_EST1V 18, 2, 26
    RECIP_EST1V 19, 3, 27
    RECIP_EST1V 20, 4, 24
    RECIP_EST1V 21, 5, 25
    RECIP_EST1V 22, 6, 26
    RECIP_EST1V 23, 7, 27
    STOREV8 16, 17, 18, 19, 20, 21, 22, 23
    subs    x2, x2, #32
    b.ne    1b
    ret

// The called function for the _call kernels. Kept in asm so it cannot change
// with compiler version, but it is exactly what clang emits for the C one-liner.
    .globl  _recip4_fn
    .p2align 4
_recip4_fn:
    fmov    v1.4s, #1.0
    fdiv    v0.4s, v1.4s, v0.4s
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  _recipv_lat_fdiv
    .p2align 4
_recipv_lat_fdiv:
    fmov    v29.4s, #1.0
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    fdiv    v0.4s, v29.4s, v1.4s
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _recipv_lat_call
    .p2align 4
_recipv_lat_call:
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
    bl      _recip4_fn                      // result back in v0, feeding the next LINKV
    .endr
    subs    x20, x20, #16
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

    .globl  _recipv_lat_est2
    .p2align 4
_recipv_lat_est2:
    movi    v30.4s, #0x80, lsl #24
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    RECIP_EST2V 0, 1, 4
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _recipv_lat_est0
    .p2align 4
_recipv_lat_est0:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    frecpe  v0.4s, v1.4s                    // the chain is AND -> ORR -> FRECPE
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret

    .globl  _recipv_lat_est1
    .p2align 4
_recipv_lat_est1:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   1, 0
    RECIP_EST1V 0, 1, 4
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret
