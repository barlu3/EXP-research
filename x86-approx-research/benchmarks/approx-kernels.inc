// Assembler macros shared by every kernel file in x86-approx-research/benchmarks:
// the x86-64 counterpart of ARM-approx-research/benchmarks/approx-kernels.inc,
// with the same two loop shapes so the reports read against each other.
//
// Throughput -- void <fn>_tput_<v>(const float* in, float* out, uint64_t n)
//   out[i] = f(in[i]) for n independent inputs. n: positive multiple of 8.
//
// Latency -- float <fn>_lat_<v>(const float* in, uint64_t n)
//   One dependent chain through all n inputs: element i's operand is in[i]
//   bit-for-bit, but cannot issue until element i-1's result exists. Returns the
//   last result. n: positive multiple of 4.
//
// Registers follow the System V AMD64 ABI: in = rdi, out/n = rsi, n = rdx.
// VEX encodings throughout, so no SSE/AVX transition can land inside a timed
// loop. SysV has no callee-saved xmm registers, so the _call kernels keep their
// loop state in rbx/r12/r13. By convention:
//
//   xmm13 = 1.0f   xmm14 = 0x80000000 per lane (sign mask)   xmm15 = 0
//
// Linux x86-64 only: ELF symbols, no leading underscore.

    .intel_syntax noprefix
    .section .note.GNU-stack, "", @progbits

// ─── Throughput shape ───────────────────────────────────────────────────────

.macro LOAD8
    vmovss  xmm0, dword ptr [rdi]
    vmovss  xmm1, dword ptr [rdi + 4]
    vmovss  xmm2, dword ptr [rdi + 8]
    vmovss  xmm3, dword ptr [rdi + 12]
    vmovss  xmm4, dword ptr [rdi + 16]
    vmovss  xmm5, dword ptr [rdi + 20]
    vmovss  xmm6, dword ptr [rdi + 24]
    vmovss  xmm7, dword ptr [rdi + 28]
    add     rdi, 32
.endm

.macro STORE8 a, b, c, d, e, f, g, h
    vmovss  dword ptr [rsi], xmm\a
    vmovss  dword ptr [rsi + 4], xmm\b
    vmovss  dword ptr [rsi + 8], xmm\c
    vmovss  dword ptr [rsi + 12], xmm\d
    vmovss  dword ptr [rsi + 16], xmm\e
    vmovss  dword ptr [rsi + 20], xmm\f
    vmovss  dword ptr [rsi + 24], xmm\g
    vmovss  dword ptr [rsi + 28], xmm\h
    add     rsi, 32
.endm

// ─── Latency shape ──────────────────────────────────────────────────────────

// xmm\dst = the input at [\ptr + \off] bit-for-bit, made dependent on xmm\prev.
//
// Same trick as the AArch64 LINK: AND with a register holding zero yields zero,
// but x86 recognises zero idioms by encoding (xor/sub of a register with
// itself), not by value, so the OR waits for the previous result. The load is
// independent and issues early. Callers unroll four links with offsets 0..12
// and step \ptr once, so the chain pays no pointer arithmetic per element.
.macro LINK dst, prev, off, zero=xmm15, ptr=rdi
    vandps  xmm2, xmm\prev, \zero
    vmovss  xmm3, dword ptr [\ptr + \off]
    vorps   xmm\dst, xmm3, xmm2
.endm
