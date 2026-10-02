// Assembler macros for the 4-lane vector kernels in vrecpe/ and vrsqrts/: the
// same two loop shapes as ../approx-kernels.inc, with each scalar s register
// replaced by a full 128-bit q register holding four float32 lanes. Included
// (through the C preprocessor) by vec-common.S, vrecpe/recip-vec-kernels.S and
// vrsqrts/rsqrt-vec-kernels.S.
//
// Throughput -- void <fn>_tput_<v>(const float* in, float* out, uint64_t n)
//   out[i] = f(in[i]) for n independent inputs, eight vectors (32 elements) per
//   iteration. n: positive multiple of 32 (approx::TPUT_STEP).
//
// Latency -- float <fn>_lat_<v>(const float* in, uint64_t n)
//   One dependent chain through all n inputs, four at a time: link i's operand
//   is in[4i..4i+3] bit-for-bit, but cannot issue until link i-1's whole vector
//   result exists. Returns lane 0 of the last result. n: positive multiple of
//   16 (four links per iteration). The driver reports cycles per link
//   (Spec::lat_lanes = 4), so a row is the latency of one vector call.
//
// Register conventions are approx-kernels.inc's, applied to all four lanes:
//
//   v29 = 1.0f x4   v30 = 0x80000000 x4 (sign mask)   v31 = 0
//
// Apple arm64 only: symbols carry the Mach-O leading underscore.

// ─── Throughput shape ───────────────────────────────────────────────────────

.macro LOADV8
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0, #64]
    ldp     q6, q7, [x0, #96]
    add     x0, x0, #128
.endm

.macro STOREV8 a, b, c, d, e, f, g, h
    stp     q\a, q\b, [x1]
    stp     q\c, q\d, [x1, #32]
    stp     q\e, q\f, [x1, #64]
    stp     q\g, q\h, [x1, #96]
    add     x1, x1, #128
.endm

// ─── Latency shape ──────────────────────────────────────────────────────────

// v\dst = the next four inputs bit-for-bit, made dependent on v\prev.
//
// LINK's trick at full width. The AND stays 64-bit: writing v2.8b zeroes v2's
// upper half too, so v2 is all zero either way, and the core tracks the
// dependency per register, not per lane, so all four lanes of the ORR still wait
// for the previous result. It also means the AND reads only the low half of
// v\zero, which is all AAPCS64 preserves of v8 across the _call kernels' BL.
.macro LINKV dst, prev, zero=31, ptr=x0
    and     v2.8b, v\prev\().8b, v\zero\().8b
    ldr     q3, [\ptr], #16
    orr     v\dst\().16b, v3.16b, v2.16b
.endm
