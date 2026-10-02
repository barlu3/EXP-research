// The control loops for the vector benchmarks: each vector loop shape with the
// arithmetic removed, as ../approx-common.S has for the scalar shapes. The
// cycle-counter calibration chain is shared with the scalar benchmarks and
// comes from approx-common.S. Shapes and register conventions are described in
// vec-kernels.inc.

#include "vec-kernels.inc"

    .text

    .globl  _vec_tput_ctrl
    .p2align 6
_vec_tput_ctrl:
1:  LOADV8
    STOREV8 0, 1, 2, 3, 4, 5, 6, 7
    subs    x2, x2, #32
    b.ne    1b
    ret

    .globl  _vec_lat_ctrl
    .p2align 4
_vec_lat_ctrl:
    movi    v31.2d, #0
    movi    v0.2d, #0
    .p2align 6
1:
    .rept 4
    LINKV   0, 0
    .endr
    subs    x1, x1, #16
    b.ne    1b
    ret
