// Hand-written x86-64 kernels for benchmark-recip.cpp: the counterpart of
// ARM-approx-research/benchmarks/scalar/frecpe/recip-kernels.S.
//
// Three ways to get 1/x in float32, each in two loop shapes:
//
//   fdiv    VDIVSS into 1.0. Correctly rounded; the reference.
//   call    The same VDIVSS behind a CALL to recip_fn, which is byte-for-byte
//           what gcc 13 -O3 -march=native emits on Ubuntu 24.04 for
//           `float recip(float x) { return 1.0f / x; }`.
//   est0    VRCPSS alone: the bare 12-bit estimate (|rel err| <= 1.5 * 2^-12
//           per the vendors' manuals; Intel's and AMD's tables differ, so the
//           bits do too).
//   est1    VRCPSS, then one FMA Newton step: e += e * (1 - x*e).
//   est2    VRCPSS, then two FMA Newton steps, with range fixups off the hot
//           path. Within 1 ulp of 1/x for every float, held to that by the
//           driver's accuracy pass.
//
// x86 has no FRECPS, so a Newton step is two FMAs instead of FRECPS + FMUL.
// The residual 1 - x*e comes out of one fused operation, exact to within one
// rounding, which is what lets the step square the error.
//
// Why est2 needs fixups. VRCPSS treats subnormal inputs as zero (returns
// +-inf) and flushes tiny results to zero, and 0 * inf = NaN inside the
// residual, so the bare pipeline is NaN at +-0 and |x| < 2^-126 and wrong for
// |x| >= ~2^126. On exponent fields 1..252 it is within 1 ulp as it stands
// (measured exhaustively), so that is the hot range. Everything else goes to
// recip_est2_slow:
//   +-0, +-inf, NaN   VRCPSS alone is already exact: +-inf, +-0, NaN.
//   subnormal x       x * 2^64 is exact and lands in the hot range; the
//                     result times 2^64 is exact unless it overflows, and at
//                     the overflow edge inf and FLT_MAX are 1 ulp apart.
//   |x| >= 2^126      x * 2^-64 is exact and in range; scaling the result back
//                     rounds once into the subnormals. Error < 0.5 + 1 ulp of
//                     the subnormal grid, so at most 1 ulp from the correctly
//                     rounded result.
// The hot path pays an integer range check and a not-taken branch per element.
// est1 has no fixups: like the AArch64 est1 it is timed, not a contender.

#include "approx-kernels.inc"

    .section .rodata
    .p2align 4
recip_zero:
    .long   0, 0, 0, 0
recip_one:
    .long   0x3f800000
recip_two64:
    .long   0x5f800000                      // 2^64
recip_twom64:
    .long   0x1f800000                      // 2^-64

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// xmm\e = VRCPSS(xmm\x) refined by one Newton step; clobbers xmm\t. Needs 1.0
// in xmm13.
.macro RECIP_EST1 e, x, t
    vrcpss       xmm\e, xmm\x, xmm\x       // 12-bit estimate
    vmovaps      xmm\t, xmm13
    vfnmadd231ss xmm\t, xmm\x, xmm\e       // t = 1 - x*e, fused
    vfmadd132ss  xmm\e, xmm\e, xmm\t       // e = e*t + e: ~23 bits
.endm

.macro RECIP_EST2 e, x, t
    RECIP_EST1 \e, \x, \t
    vmovaps      xmm\t, xmm13
    vfnmadd231ss xmm\t, xmm\x, xmm\e
    vfmadd132ss  xmm\e, xmm\e, xmm\t       // second step
.endm

// RECIP_EST2 for every float: inputs outside exponent fields 1..252 branch to
// a stub in .text.unlikely that runs recip_est2_slow and jumps back. The check
// is integer work on a copy of x, so it never sits on the FP result's chain.
// Clobbers eax; the stub also clobbers xmm10-12.
.macro RECIP_EST2_FULL e, x, t
    vmovd   eax, xmm\x
    and     eax, 0x7fffffff
    sub     eax, 0x00800000                 // exponent field 1 -> 0; 0 wraps high
    cmp     eax, 0x7e000000                 // field 253 and up, and the wrapped 0
    jae     .Lcold\@
    RECIP_EST2 \e, \x, \t
.Lback\@:
    .pushsection .text.unlikely, "ax", @progbits
.Lcold\@:
    vmovaps xmm10, xmm\x
    call    recip_est2_slow
    vmovaps xmm\e, xmm10
    jmp     .Lback\@
    .popsection
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  recip_tput_fdiv
    .p2align 4
recip_tput_fdiv:
    vmovss  xmm13, dword ptr [rip + recip_one]
    .p2align 6
1:  LOAD8
    vdivss  xmm0, xmm13, xmm0
    vdivss  xmm1, xmm13, xmm1
    vdivss  xmm2, xmm13, xmm2
    vdivss  xmm3, xmm13, xmm3
    vdivss  xmm4, xmm13, xmm4
    vdivss  xmm5, xmm13, xmm5
    vdivss  xmm6, xmm13, xmm6
    vdivss  xmm7, xmm13, xmm7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    sub     rdx, 8
    jnz     1b
    ret

    .globl  recip_tput_call
    .p2align 4
recip_tput_call:
    push    rbx
    push    r12
    push    r13                             // rsp now 16-byte aligned for the calls
    mov     rbx, rdi
    mov     r12, rsi
    mov     r13, rdx
    .p2align 6
1:
    .irp off, 0, 4, 8, 12, 16, 20, 24, 28
    vmovss  xmm0, dword ptr [rbx + \off]
    call    recip_fn
    vmovss  dword ptr [r12 + \off], xmm0
    .endr
    add     rbx, 32
    add     r12, 32
    sub     r13, 8
    jnz     1b
    pop     r13
    pop     r12
    pop     rbx
    ret

    .globl  recip_tput_est0
    .p2align 6
recip_tput_est0:
1:  LOAD8
    vrcpss  xmm0, xmm0, xmm0
    vrcpss  xmm1, xmm1, xmm1
    vrcpss  xmm2, xmm2, xmm2
    vrcpss  xmm3, xmm3, xmm3
    vrcpss  xmm4, xmm4, xmm4
    vrcpss  xmm5, xmm5, xmm5
    vrcpss  xmm6, xmm6, xmm6
    vrcpss  xmm7, xmm7, xmm7
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    sub     rdx, 8
    jnz     1b
    ret

    .globl  recip_tput_est1
    .p2align 4
recip_tput_est1:
    vmovss  xmm13, dword ptr [rip + recip_one]
    .p2align 6
1:  LOAD8
    // xmm13-15 are reserved, so eight results cannot each get a register as
    // on AArch64. Each is stored at once and xmm8/xmm9 reused; renaming makes
    // the reuse free.
    RECIP_EST1 8, 0, 9
    vmovss  dword ptr [rsi], xmm8
    RECIP_EST1 8, 1, 9
    vmovss  dword ptr [rsi + 4], xmm8
    RECIP_EST1 8, 2, 9
    vmovss  dword ptr [rsi + 8], xmm8
    RECIP_EST1 8, 3, 9
    vmovss  dword ptr [rsi + 12], xmm8
    RECIP_EST1 8, 4, 9
    vmovss  dword ptr [rsi + 16], xmm8
    RECIP_EST1 8, 5, 9
    vmovss  dword ptr [rsi + 20], xmm8
    RECIP_EST1 8, 6, 9
    vmovss  dword ptr [rsi + 24], xmm8
    RECIP_EST1 8, 7, 9
    vmovss  dword ptr [rsi + 28], xmm8
    add     rsi, 32
    sub     rdx, 8
    jnz     1b
    ret

    .globl  recip_tput_est2
    .p2align 4
recip_tput_est2:
    vmovss  xmm13, dword ptr [rip + recip_one]
    .p2align 6
1:  LOAD8
    RECIP_EST2_FULL 8, 0, 9
    vmovss  dword ptr [rsi], xmm8
    RECIP_EST2_FULL 8, 1, 9
    vmovss  dword ptr [rsi + 4], xmm8
    RECIP_EST2_FULL 8, 2, 9
    vmovss  dword ptr [rsi + 8], xmm8
    RECIP_EST2_FULL 8, 3, 9
    vmovss  dword ptr [rsi + 12], xmm8
    RECIP_EST2_FULL 8, 4, 9
    vmovss  dword ptr [rsi + 16], xmm8
    RECIP_EST2_FULL 8, 5, 9
    vmovss  dword ptr [rsi + 20], xmm8
    RECIP_EST2_FULL 8, 6, 9
    vmovss  dword ptr [rsi + 24], xmm8
    RECIP_EST2_FULL 8, 7, 9
    vmovss  dword ptr [rsi + 28], xmm8
    add     rsi, 32
    sub     rdx, 8
    jnz     1b
    ret

// The called function for the _call kernels. Kept in asm so it cannot change
// with compiler version; endbr64 included because Ubuntu's gcc emits it.
    .globl  recip_fn
    .p2align 4
recip_fn:
    endbr64
    vmovss  xmm1, dword ptr [rip + recip_one]
    vdivss  xmm0, xmm1, xmm0
    ret

// xmm10 = 1/xmm10 within 1 ulp, for the inputs RECIP_EST2_FULL sends here
// (cases at the top of this file). Needs 1.0 in xmm13; clobbers xmm11, xmm12
// and eax, which no est2 kernel keeps anything in.
    .p2align 4
recip_est2_slow:
    vmovd   eax, xmm10
    and     eax, 0x7fffffff
    jz      3f                              // +-0
    cmp     eax, 0x00800000
    jb      1f                              // subnormal
    cmp     eax, 0x7f800000
    jb      2f                              // |x| >= 2^126, finite
3:  vrcpss  xmm10, xmm10, xmm10             // +-0, +-inf, NaN: already exact
    ret
1:  vmulss  xmm10, xmm10, dword ptr [rip + recip_two64]
    RECIP_EST2 11, 10, 12
    vmulss  xmm10, xmm11, dword ptr [rip + recip_two64]
    ret
2:  vmulss  xmm10, xmm10, dword ptr [rip + recip_twom64]
    RECIP_EST2 11, 10, 12
    vmulss  xmm10, xmm11, dword ptr [rip + recip_twom64]
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  recip_lat_fdiv
    .p2align 4
recip_lat_fdiv:
    vmovss  xmm13, dword ptr [rip + recip_one]
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    vdivss  xmm0, xmm13, xmm1
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

    .globl  recip_lat_call
    .p2align 4
recip_lat_call:
    push    rbx
    push    r12
    sub     rsp, 8                          // align rsp to 16 for the calls
    mov     rbx, rdi
    mov     r12, rsi
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    // No xmm register survives a SysV call, so the zero comes from memory.
    // NOTE: that adds a load to this link that the control's link lacks; the
    // load is off the chain, so it costs issue slots, not latency.
    .irp off, 0, 4, 8, 12
    LINK    0, 0, \off, [rip + recip_zero], rbx
    call    recip_fn
    .endr
    add     rbx, 16
    sub     r12, 4
    jnz     1b
    add     rsp, 8
    pop     r12
    pop     rbx
    ret

    .globl  recip_lat_est0
    .p2align 4
recip_lat_est0:
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    vrcpss  xmm0, xmm1, xmm1                // the chain is AND -> OR -> RCPSS
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

    .globl  recip_lat_est1
    .p2align 4
recip_lat_est1:
    vmovss  xmm13, dword ptr [rip + recip_one]
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    RECIP_EST1 0, 1, 4
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

    .globl  recip_lat_est2
    .p2align 4
recip_lat_est2:
    vmovss  xmm13, dword ptr [rip + recip_one]
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    RECIP_EST2_FULL 0, 1, 4
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret
