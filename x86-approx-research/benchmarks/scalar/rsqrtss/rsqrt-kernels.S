// Hand-written x86-64 kernels for benchmark-rsqrt.cpp: the counterpart of
// ARM-approx-research/benchmarks/scalar/frsqrte/rsqrt-kernels.S.
//
// Ways to get 1/sqrt(x) in float32, each in the two loop shapes described in
// ../../approx-kernels.inc:
//
//   sqdiv   VSQRTSS, then VDIVSS into 1.0. x86 has no exact reciprocal square
//           root either, so this is the exact route; it rounds twice, so it is
//           faithful, not correctly rounded.
//   call    rsqrt_fn: byte-for-byte what gcc 13 -O3 -march=native emits on
//           Ubuntu 24.04 for `float rsqrt(float x) { return 1.0f / sqrtf(x); }`.
//           Unlike clang on macOS, gcc keeps errno: x < 0 goes to libm's sqrtf
//           through the PLT, so negative inputs pay a real call.
//   est0    VRSQRTSS alone: the 12-bit estimate (|rel err| <= 1.5 * 2^-12).
//   est1    VRSQRTSS, then one Newton step.
//   est2    VRSQRTSS, then two Newton steps, with range fixups off the hot path:
//           held to 2 ulp of the correctly rounded result, like the AArch64 est2.
//   est2u   est2 without the range check or fixups, timed beside it so the
//           check's cost reads off one report. Reported, never checked.
//
// The Newton step is e += (e/2) * (1 - (x*e)*e). x*e first, as on AArch64: e*e
// first overflows for small x and goes subnormal for huge x. x86 has no FMULX,
// so 0*inf stays NaN and the step is wrong wherever VRSQRTSS returns an
// infinity or a zero.
//
// Why est2 needs fixups. VRSQRTSS treats subnormal inputs as zero (returns
// +-inf), so the steps meet 0*inf or inf-inf: NaN at +-0, +inf and every
// subnormal. Exponent fields 1..254 are fine as they stand, either sign (a
// negative normal is NaN from VRSQRTSS on, as it should be), so that is the hot
// range. The rest goes to rsqrt_est2_slow:
//   +-0, +-inf, NaN   VRSQRTSS alone is already right: +-inf, +0 or NaN, NaN.
//   subnormal x       x * 2^64 is exact and in range, and 1/sqrt(2^64) = 2^-32
//                     exactly, so the result times 2^32 is exact too: est2's
//                     own error, nothing added. Negative subnormals come out
//                     NaN through the same path.
// The throughput kernel checks each 8-element block with one 256-bit compare;
// the latency kernel checks per element, off the FP chain.

#include "approx-kernels.inc"

    .section .rodata
    .p2align 4
rsqrt_zero:
    .long   0, 0, 0, 0
rsqrt_one:
    .long   0x3f800000
rsqrt_half:
    .long   0x3f000000
rsqrt_two64:
    .long   0x5f800000                      // 2^64
rsqrt_two32:
    .long   0x4f800000                      // 2^32
rsqrt_abs_mask:
    .long   0x7fffffff
rsqrt_range_bias:
    .long   0x7f800000                      // 2^31 - 0x00800000: hot range -> [INT_MIN, -2^24)
    .p2align 5
rsqrt_range_last:
    .rept 8
    .long   0xfeffffff                      // last biased value in the hot range
    .endr

    .text

// ─── The estimate sequences ─────────────────────────────────────────────────

// One Newton step on xmm\e; clobbers xmm\t, xmm\u. Needs 1.0 in xmm13 and 0.5
// in xmm12.
.macro RSQRT_STEP e, x, t, u
    vmulss       xmm\t, xmm\x, xmm\e       // x*e
    vmovaps      xmm\u, xmm13
    vfnmadd231ss xmm\u, xmm\t, xmm\e       // u = 1 - (x*e)*e, fused
    vmulss       xmm\u, xmm\u, xmm12       // u/2, exact
    vfmadd231ss  xmm\e, xmm\e, xmm\u       // e += e * u/2
.endm

.macro RSQRT_EST1 e, x, t, u
    vrsqrtss     xmm\e, xmm\x, xmm\x       // 12-bit estimate
    RSQRT_STEP \e, \x, \t, \u
.endm

.macro RSQRT_EST2 e, x, t, u
    RSQRT_EST1 \e, \x, \t, \u
    RSQRT_STEP \e, \x, \t, \u
.endm

// RSQRT_EST2 for every float: inputs outside exponent fields 1..254 branch to
// a stub in .text.unlikely that runs rsqrt_est2_slow and jumps back. Clobbers
// eax; the stub also clobbers xmm8-11.
.macro RSQRT_EST2_FULL e, x, t, u
    vmovd   eax, xmm\x
    and     eax, 0x7fffffff
    sub     eax, 0x00800000                 // exponent field 1 -> 0; 0 wraps high
    cmp     eax, 0x7f000000                 // field 255, and the wrapped 0
    jae     .Lcold\@
    RSQRT_EST2 \e, \x, \t, \u
.Lback\@:
    .pushsection .text.unlikely, "ax", @progbits
.Lcold\@:
    vmovaps xmm9, xmm\x
    call    rsqrt_est2_slow
    vmovaps xmm\e, xmm9
    jmp     .Lback\@
    .popsection
.endm

// Throughput: 8 inputs from [rdi] through \body (EST2 or EST2_FULL), each
// stored as soon as it is done; xmm13-15 are reserved, so eight results cannot
// each keep a register.
.macro RSQRT_BLOCK body
    LOAD8
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7
    \body   8, \i, 10, 11
    vmovss  dword ptr [rsi + 4*\i], xmm8
    .endr
.endm

// ─── Throughput kernels ─────────────────────────────────────────────────────

    .globl  rsqrt_tput_sqdiv
    .p2align 4
rsqrt_tput_sqdiv:
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    .p2align 6
1:  LOAD8
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7
    vsqrtss xmm\i, xmm\i, xmm\i
    .endr
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7
    vdivss  xmm\i, xmm13, xmm\i
    .endr
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    sub     rdx, 8
    jnz     1b
    ret

    .globl  rsqrt_tput_call
    .p2align 4
rsqrt_tput_call:
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
    call    rsqrt_fn
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

    .globl  rsqrt_tput_est0
    .p2align 6
rsqrt_tput_est0:
1:  LOAD8
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7
    vrsqrtss xmm\i, xmm\i, xmm\i
    .endr
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    sub     rdx, 8
    jnz     1b
    ret

    .globl  rsqrt_tput_est1
    .p2align 4
rsqrt_tput_est1:
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    vmovss  xmm12, dword ptr [rip + rsqrt_half]
    .p2align 6
1:  RSQRT_BLOCK RSQRT_EST1
    add     rsi, 32
    sub     rdx, 8
    jnz     1b
    ret

    .globl  rsqrt_tput_est2
    .p2align 4
rsqrt_tput_est2:
    // One range check per 8-element block: the biased |x| bits are
    // signed-below rsqrt_range_last exactly on exponent fields 1..254, so a
    // block with any lane above it takes the per-element path at 3:.
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    vmovss  xmm12, dword ptr [rip + rsqrt_half]
    vpbroadcastd ymm14, dword ptr [rip + rsqrt_abs_mask]
    vpbroadcastd ymm15, dword ptr [rip + rsqrt_range_bias]
    .p2align 6
1:  vpand   ymm8, ymm14, ymmword ptr [rdi]
    vpaddd  ymm8, ymm8, ymm15
    vpcmpgtd ymm8, ymm8, ymmword ptr [rip + rsqrt_range_last]
    vptest  ymm8, ymm8
    jnz     3f
    RSQRT_BLOCK RSQRT_EST2
2:  add     rsi, 32
    sub     rdx, 8
    jnz     1b
    vzeroupper                              // ymm state dirty; callers may run SSE
    ret
3:  RSQRT_BLOCK RSQRT_EST2_FULL
    jmp     2b

    .globl  rsqrt_tput_est2u
    .p2align 4
rsqrt_tput_est2u:
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    vmovss  xmm12, dword ptr [rip + rsqrt_half]
    .p2align 6
1:  RSQRT_BLOCK RSQRT_EST2
    add     rsi, 32
    sub     rdx, 8
    jnz     1b
    ret

// The called function for the _call kernels, as gcc emits it (see the top of
// this file). Kept in asm so it cannot change with compiler version.
    .globl  rsqrt_fn
    .p2align 4
rsqrt_fn:
    endbr64
    vxorps  xmm1, xmm1, xmm1
    vucomiss xmm1, xmm0
    ja      1f
    vmovss  xmm1, dword ptr [rip + rsqrt_one]
    vsqrtss xmm0, xmm0, xmm0
    vdivss  xmm0, xmm1, xmm0
    ret
1:  push    rax
    call    sqrtf@PLT
    vmovss  xmm1, dword ptr [rip + rsqrt_one]
    pop     rdx
    vdivss  xmm0, xmm1, xmm0
    ret

// xmm9 = 1/sqrt(xmm9) within est2's bound, for the inputs RSQRT_EST2_FULL sends
// here (cases at the top of this file). Needs 1.0 in xmm13 and 0.5 in xmm12;
// clobbers xmm8, xmm10, xmm11 and eax, which no est2 kernel keeps anything in
// at the call (the throughput kernels' xmm8 is the result the stub overwrites).
    .p2align 4
rsqrt_est2_slow:
    vmovd   eax, xmm9
    test    eax, 0x7f800000                 // only fields 0 and 255 come here
    jnz     2f                              // field 255: +-inf, NaN
    test    eax, 0x7fffffff
    jz      2f                              // +-0
    vmulss  xmm9, xmm9, dword ptr [rip + rsqrt_two64]
    RSQRT_EST2 8, 9, 10, 11
    vmulss  xmm9, xmm8, dword ptr [rip + rsqrt_two32]
    ret
2:  vrsqrtss xmm9, xmm9, xmm9               // already right for these
    ret

// ─── Latency kernels ────────────────────────────────────────────────────────

    .globl  rsqrt_lat_sqdiv
    .p2align 4
rsqrt_lat_sqdiv:
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    vsqrtss xmm1, xmm1, xmm1
    vdivss  xmm0, xmm13, xmm1
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

    .globl  rsqrt_lat_call
    .p2align 4
rsqrt_lat_call:
    push    rbx
    push    r12
    sub     rsp, 8                          // align rsp to 16 for the calls
    mov     rbx, rdi
    mov     r12, rsi
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    // No xmm register survives a SysV call, so the zero comes from memory.
    .irp off, 0, 4, 8, 12
    LINK    0, 0, \off, [rip + rsqrt_zero], rbx
    call    rsqrt_fn
    .endr
    add     rbx, 16
    sub     r12, 4
    jnz     1b
    add     rsp, 8
    pop     r12
    pop     rbx
    ret

    .globl  rsqrt_lat_est0
    .p2align 4
rsqrt_lat_est0:
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    vrsqrtss xmm0, xmm1, xmm1               // the chain is AND -> OR -> RSQRTSS
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

// The latency kernels below differ only in the sequence they chain through.
.macro RSQRT_LAT body
    vmovss  xmm13, dword ptr [rip + rsqrt_one]
    vmovss  xmm12, dword ptr [rip + rsqrt_half]
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    1, 0, \off
    \body   0, 1, 4, 5
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret
.endm

    .globl  rsqrt_lat_est1
    .p2align 4
rsqrt_lat_est1:
    RSQRT_LAT RSQRT_EST1

    .globl  rsqrt_lat_est2
    .p2align 4
rsqrt_lat_est2:
    RSQRT_LAT RSQRT_EST2_FULL

    .globl  rsqrt_lat_est2u
    .p2align 4
rsqrt_lat_est2u:
    RSQRT_LAT RSQRT_EST2
