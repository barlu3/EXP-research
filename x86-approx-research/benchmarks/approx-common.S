// Kernels every benchmark in x86-approx-research/benchmarks links, under the
// symbol names ARM-approx-research/benchmarks/approx-bench.hpp declares: the
// control loops (each shape with the arithmetic removed) and the calibration
// chain bench-cycles.hpp converts nanoseconds to cycles with. Shapes and
// register conventions are described in approx-kernels.inc.

#include "approx-kernels.inc"

    .text

    .globl  approx_tput_ctrl
    .p2align 6
approx_tput_ctrl:
1:  LOAD8
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    sub     rdx, 8
    jnz     1b
    ret

    .globl  approx_lat_ctrl
    .p2align 4
approx_lat_ctrl:
    vxorps  xmm15, xmm15, xmm15
    vxorps  xmm0, xmm0, xmm0
    .p2align 6
1:
    .irp off, 0, 4, 8, 12
    LINK    0, 0, \off
    .endr
    add     rdi, 16
    sub     rsi, 4
    jnz     1b
    ret

// uint64_t approx_calib_chain(uint64_t iters)
//
// The contract bench-cycles.hpp checks: an 8-cycle dependent chain and 14
// instructions per iteration. Eight dependent ADDs set the chain on any core
// with single-cycle ADD (all of Zen and Intel's big cores); four independent
// ADDs and the loop control ride in its shadow.
    .globl  approx_calib_chain
    .p2align 4
approx_calib_chain:
    xor     eax, eax
    .p2align 6
1:  add     rax, 1
    add     rdx, 1
    add     rax, 1
    add     rcx, 1
    add     rax, 1
    add     r8, 1
    add     rax, 1
    add     r9, 1
    add     rax, 1
    add     rax, 1
    add     rax, 1
    add     rax, 1
    sub     rdi, 1
    jnz     1b
    ret
