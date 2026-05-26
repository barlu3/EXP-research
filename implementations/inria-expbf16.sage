# Table generator for the correctly-rounded bfloat16 exponential function.
#
# This file is part of the CORE-MATH project (https://core-math.gitlabpages.inria.fr/).
# It is a SageMath script — run it with "sage inria-expbf16.sage" to produce the
# C source that declares the T1[] and T2[] lookup tables used by cr_exp_bf16() in
# inria-expbf16.c.
#
# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND: the bfloat16 (brain float 16) format
# ──────────────────────────────────────────────────────────────────────────────
# A bfloat16 value occupies 16 bits laid out as:
#
#   bit 15    : sign s  (0 = positive, 1 = negative)
#   bits 14–7 : biased exponent e  (8 bits, bias = 127)  ← same as binary32
#   bits  6–0 : mantissa m  (7 bits)
#
# For a normal value (e ≠ 0 and e ≠ 255):
#   value = (-1)^s × 2^(e−127) × (1 + m / 2^7)
#
# Compared to float16 (5-bit exponent, 10-bit mantissa), bfloat16 keeps the
# full float32 exponent range but uses only 7 mantissa bits instead of 10.
# This means bfloat16 can represent the same magnitudes as float32, but with
# much lower precision (~3 decimal digits vs ~7).
#
# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM: split-table multiplication
# ──────────────────────────────────────────────────────────────────────────────
# The identity  exp(x) = exp(x1) × exp(x2)  is used, where x = x1 + x2.
#
# The 7 mantissa bits of x are split:
#   "upper 4"  captured by i1 (coarse table)
#   "lower 3"  captured by i2 (fine table)
#
# Formally, for a normal bfloat16 with exponent field e and mantissa m:
#   m = m_hi × 8 + m_lo    (m_hi = bits 6–3, m_lo = bits 2–0)
#   x1 = (-1)^s × 2^(e−127) × (1 + m_hi / 16)   [coarse: lower 3 bits = 0]
#   x2 = (-1)^s × 2^(e−134) × m_lo               [fine: residual from low 3 bits]
#
# Both T1[i1] and T2[i2] are stored as binary32 (float) values.
# The product  T1[i1] × T2[i2]  is computed in binary32 and then rounded to
# bfloat16 in the calling C function.
#
# The index encoding used by cr_exp_bf16() in inria-expbf16.c is:
#
#   i1 = ((u >> 15) << 8)  +  (au >> 3)  − 0x760
#         ↑ sign bit         ↑ top 12 bits of au   ↑ subtract base 0x3b00>>3
#
#   i2 = ((u >> 15) << 7)  +  ((au >> 7) << 3  |  au & 0x7)  − 0x3b0
#         ↑ sign bit         ↑ exponent field + low 3 mantissa bits  ↑ base
#
#   where u  = raw 16-bit encoding of x,
#         au = u & 0x7fff  (absolute-value bits, i.e. sign stripped).
#
# The base offsets 0x760 = 0x3b00 >> 3 and 0x3b0 = 118 × 8 correspond to the
# minimum biased exponent of inputs in the valid range (e = 118, i.e. unbiased
# exponent = −9), which corresponds to |x| ≥ 2^−9 ≈ 0.002.  Inputs smaller
# than that round to 1.0 and are handled separately without a table lookup.
#
# ──────────────────────────────────────────────────────────────────────────────
# HELPER CONVENTIONS (defined in the calling SageMath environment)
# ──────────────────────────────────────────────────────────────────────────────
#   R24(v)          — round v to 24-bit (binary32) precision (nearest-even)
#   asbfloat16(u)   — reinterpret 16-bit integer u as a bfloat16 and return its
#                     mathematical value as an exact rational/real
#   get_hex(v)      — format v as a C hexadecimal float literal, e.g. "0x1.8p+3"
#
# ──────────────────────────────────────────────────────────────────────────────

def table1(out):
    # Generate T1[], the "coarse" table: 2 × 2^8 = 512 entries.
    #
    # T1[i1] stores RN_32(exp(±x1)), the binary32 value nearest to the
    # exponential of the "coarse" part x1 of the bfloat16 input.
    #
    # Index layout (i1 ∈ [0, 512)):
    #   i1 = sgn × 2^8  +  j1
    #   sgn ∈ {0, 1}   : whether x was negative
    #   j1  ∈ [0, 256) : encodes the upper 4 mantissa bits alongside the exponent
    #
    # The bfloat16 bit pattern for x1 is  u1 = 118 × 2^7 + j1 × 2^3:
    #   • 118 × 2^7 = 0x3b00:  the lowest biased exponent (118) in the valid
    #     input range, with all mantissa bits zero.
    #   • j1 × 2^3 steps through every combination of (exponent, upper 4
    #     mantissa bits) because multiplying j1 by 8 = 2^3 advances by one
    #     step through bits 6–3 of the mantissa field (and carries into the
    #     exponent when those bits overflow).
    #   For j1 = 0…255:  exponent field e = 118 + j1 // 16,  mantissa m_hi = j1 & 0xf
    #   → exponent field spans 118…133  (unbiased: −9…+6)
    #
    # The sign is applied after computing exp(x1):  exp(−x1) = 1/exp(x1),
    # so negative-x entries are in the upper half of the table.
    #
    # Values larger than binary32 MAX_FLT (≥ 2^128) are capped to MAX_FLT.
    # No underflow capping is needed here because the coarse part alone cannot
    # push exp(x) below the representable range.

    f = open(out,"w")
    f.write ("static const float T1[] = {\n")
    s = ""
    for i1 in range(2^9):
        sgn, j1 = divmod(i1, 2^8)  # sgn = 0 (positive) or 1 (negative)

        # Construct the bfloat16 encoding for the coarse argument x1.
        # 118 × 128 = 0x3b00 is the bfloat16 with exponent 118 and mantissa 0.
        # Adding j1 × 8 walks through all (exponent, upper-mantissa) combinations.
        u1 = 118*2^7+j1*2^3

        # Decode u1 as a bfloat16 real value (positive; sign applied below).
        # NOTE: "if u1 = h*2^4+m, then ulp(x1) = 2^(h-16)" (comment from original).
        x1 = R24(asbfloat16(u1))

        # Apply sign and compute exp.
        y1 = exp((-1)^sgn*x1)

        # Cap overflow: values larger than binary32 MAX_FLT become MAX_FLT.
        if y1>=2^128: # values larger than MAX_FLT are capped to MAX_FLT
            y1 = R24(2^128).nextbelow()   # = 0x1.fffffep+127

        v = get_hex(y1)
        t = " " + v + ","
        if len(s)+len(t)>=79:
            f.write (s + "\n")
            s = t
        else:
            s += t
    if s!="":
        f.write (s + "\n")
    f.write ("};\n")
    f.close()


def table2(out):
    # Generate T2[], the "fine" table: 2 × 2^7 = 256 entries.
    #
    # T2[i2] stores RN_32(exp(±x2)), the binary32 value nearest to the
    # exponential of the "fine" residual x2 of the bfloat16 input.
    #
    # Index layout (i2 ∈ [0, 256)):
    #   i2 = sgn × 2^7  +  j2
    #   sgn ∈ {0, 1}  : whether x was negative
    #   j2  ∈ [0, 128): encodes (exponent field, lower 3 mantissa bits)
    #
    #   j2 = h × 2^3 + l
    #   h ∈ [0, 16)   : exponent field offset above 118 (i.e. e − 118)
    #   l ∈ [0,  8)   : lower 3 mantissa bits of the bfloat16 input
    #
    # The residual x2 captures the contribution of the low 3 mantissa bits:
    #   For a bfloat16 with exponent field e and low mantissa l:
    #     x2 = 2^(e−134) × l    (= 2^(h−16) × l  since  h = e − 118)
    #
    # The range of x2 is tiny (at most a few ULPs of the bfloat16 format),
    # which is why T2 entries are all very close to 1.0.
    #
    # One manual adjustment:
    #   i2 == 120 is nudged one ULP upward (nextabove) to avoid a "double
    #   rounding" exception: without it, the product T1[i1] × T2[i2] for the
    #   corresponding bfloat16 input would not round correctly to bfloat16.
    #   This index was identified by exhaustive verification against an oracle.

    f = open(out,"w")
    f.write ("static const float T2[] = {\n")
    s = ""
    for i2 in range(2^8):
        sgn, j2 = divmod(i2, 2^7)   # sgn = 0 (positive) or 1 (negative)
        h,l = divmod(j2,2^3)         # h = exponent offset (0..15), l = low mantissa bits (0..7)

        # Compute the fine residual:  x2 = 2^(h−16) × l
        #   h = e − 118,  and  x2 = 2^(e−134) × l  = 2^((e−118)−16) × l
        # For h = 0, l > 0: x2 = 2^−16 × l  (very small)
        # For h = 15, l = 7: x2 = 2^−1 × 7 = 3.5  (the largest fine residual)
        x2 = R24(2^(h-16)*l)

        # Apply sign.
        y2 = exp((-1)^sgn*x2)

        # Manual 1-ULP adjustment for one "hard" input index.
        # Without this, the product T1[i1] × T2[i2] for the matching bfloat16
        # input would land on exactly the wrong side of a rounding boundary.
        if i2==120: # adjust to avoid exception
            y2 = y2.nextabove()

        v = get_hex(y2)
        t = " " + v + ","
        if len(s)+len(t)>=79:
            f.write (s + "\n")
            s = t
        else:
            s += t
    if s!="":
        f.write (s + "\n")
    f.write ("};\n")
    f.close()
