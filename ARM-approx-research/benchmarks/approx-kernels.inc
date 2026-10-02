// Assembler macros shared by every kernel file in ARM-approx-research/benchmarks.
// Included (through the C preprocessor) by approx-common.S,
// scalar/frecpe/recip-kernels.S and scalar/frsqrte/rsqrt-kernels.S.
//
// Every kernel file uses the same two loop shapes:
//
// Throughput -- void <fn>_tput_<v>(const float* in, float* out, uint64_t n)
//   out[i] = f(in[i]) for n independent inputs. n: positive multiple of 8.
//
// Latency -- float <fn>_lat_<v>(const float* in, uint64_t n)
//   One dependent chain through all n inputs: element i's operand is in[i]
//   bit-for-bit, but cannot issue until element i-1's result exists. Returns the
//   last result. n: positive multiple of 4.
//
// Registers follow AAPCS64. Kernels live in caller-saved registers, except the
// _call pairs, which make real calls and so keep their loop state in
// callee-saved x19/x20 and d8 and save them. By convention:
//
//   v29 = 1.0f   v30 = 0x80000000 per lane (sign mask)   v31 = 0
//
// Apple arm64 only: symbols carry the Mach-O leading underscore.

// ─── Throughput shape ───────────────────────────────────────────────────────

.macro LOAD8
    ldp     s0, s1, [x0]
    ldp     s2, s3, [x0, #8]
    ldp     s4, s5, [x0, #16]
    ldp     s6, s7, [x0, #24]
    add     x0, x0, #32
.endm

.macro STORE8 a, b, c, d, e, f, g, h
    stp     s\a, s\b, [x1]
    stp     s\c, s\d, [x1, #8]
    stp     s\e, s\f, [x1, #16]
    stp     s\g, s\h, [x1, #24]
    add     x1, x1, #32
.endm

// ─── Latency shape ──────────────────────────────────────────────────────────

// v\dst = the next input bit-for-bit, made dependent on v\prev.
//
// AND with a register that holds zero yields zero, but the core cannot know
// that -- zero idioms are recognised by encoding, not by value -- so the ORR
// waits for the previous result. The load itself is independent and issues
// early. The chain costs one AND plus one ORR per element on top of the
// function; approx_lat_ctrl measures exactly that and nothing else.
.macro LINK dst, prev, zero=31, ptr=x0
    and     v2.8b, v\prev\().8b, v\zero\().8b
    ldr     s3, [\ptr], #4
    orr     v\dst\().8b, v3.8b, v2.8b
.endm
