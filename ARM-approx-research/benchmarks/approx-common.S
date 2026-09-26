// Kernels every benchmark in ARM-approx-research/benchmarks links: the control
// loops, which are each shape with the arithmetic removed so the scaffolding
// can be told apart from what is being measured, and the cycle-counter
// calibration chain bench-cycles.hpp relies on. Shapes and register
// conventions are described in approx-kernels.inc.

#include "approx-kernels.inc"

    .text

    .globl  _approx_tput_ctrl
    .p2align 6
_approx_tput_ctrl:
1:  LOAD8
    STORE8  0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #8
    b.ne    1b
    ret

    .globl  _approx_lat_ctrl
    .p2align 4
_approx_lat_ctrl:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINK    0, 0
    .endr
    subs    x1, x1, #4
    b.ne    1b
    ret

// uint64_t approx_calib_chain(uint64_t iters)
//
// Eight dependent ADDs per iteration set an 8-cycle critical path on any core
// with single-cycle ADD. Four independent ADDs and the loop control ride in its
// shadow, so a working pair of counters reads 8 cycles and 14 instructions per
// iteration. That gap is what lets bench-cycles.hpp tell the two fixed counters
// apart and check both, and in timer mode the same kernel converts ns to cycles.
// The shadow work is kept to four ADDs so a narrower efficiency core still fits
// it inside 8 cycles.
    .globl  _approx_calib_chain
    .p2align 4
_approx_calib_chain:
    mov     x1, #0
    .p2align 6
1:  add     x1, x1, #1
    add     x2, x2, #1
    add     x1, x1, #1
    add     x3, x3, #1
    add     x1, x1, #1
    add     x4, x4, #1
    add     x1, x1, #1
    add     x5, x5, #1
    add     x1, x1, #1
    add     x1, x1, #1
    add     x1, x1, #1
    add     x1, x1, #1
    subs    x0, x0, #1
    b.ne    1b
    mov     x0, x1
    ret
