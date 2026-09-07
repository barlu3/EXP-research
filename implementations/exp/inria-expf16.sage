# Table generator for the correctly-rounded float16 exponential function.
#
# This file is part of the CORE-MATH project (https://core-math.gitlabpages.inria.fr/).
# It is a SageMath script — run it with "sage inria-expf16.sage" to produce the
# C source that declares the T1[] and T2[] lookup tables used by cr_expf16() in
# inria-expf16.c.
#
# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND: the float16 (binary16) format
# ──────────────────────────────────────────────────────────────────────────────
# A float16 value is stored in 16 bits laid out as:
#
#   bit 15    : sign s  (0 = positive, 1 = negative)
#   bits 14–10: biased exponent e  (5 bits, bias = 15)
#   bits  9–0 : mantissa m  (10 bits)
#
# For a normal value (e ≠ 0 and e ≠ 31):
#   value = (-1)^s × 2^(e−15) × (1 + m / 2^10)
#
# Special encodings:
#   e == 31, m == 0  →  ±Infinity
#   e == 31, m != 0  →  NaN
#   e ==  0          →  subnormal  (implicit leading bit is 0, not 1)
#
# Range of representable normal values: ~6.1×10⁻⁵  to  ~65504
#
# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM: split-table multiplication
# ──────────────────────────────────────────────────────────────────────────────
# The identity  exp(x) = exp(x1) × exp(x2)  is used, where x = x1 + x2.
#
# Given the 16-bit encoding u of x, define:
#   i1 = u >> 5          (drop the 5 lowest mantissa bits  → 11-bit index)
#   i2 = (u >> 10) << 5  |  (u & 0x1f)
#           (keep the sign+exponent field and the 5 lowest mantissa bits)
#
# Geometrically, the 10 mantissa bits of x are split into:
#   "upper 5"  captured by i1 (along with sign+exponent)
#   "lower 5"  captured by i2 (along with sign+exponent)
#
# x1  = the float16 whose bit pattern is  i1 × 2^5  (lower 5 mantissa bits = 0)
# x2  = the residual part: x − x1
#      For a normal float16 with exponent field e and mantissa m_hi‖m_lo:
#        x1 = (-1)^s × 2^(e−15) × (1 + m_hi / 2^5)          where m_hi = bits 9–5
#        x2 = (-1)^s × 2^(e−25) × m_lo                        where m_lo = bits 4–0
#      (For subnormals, e = 0 and there is no implicit 1, so x2 = (-1)^s × 2^(e−24) × m_lo)
#
# Both T1[i1] and T2[i2] are stored as binary32 (float) values.
# The final result  T1[i1].f × T2[i2].f  is computed in binary32 and then
# rounded to float16.
#
# Using binary32 intermediates gives enough extra precision (8 extra mantissa
# bits) that the product is correctly rounded — i.e., it equals the float16
# value nearest to the true mathematical exp(x).
#
# ──────────────────────────────────────────────────────────────────────────────
# HELPER CONVENTIONS (defined in the calling SageMath environment)
# ──────────────────────────────────────────────────────────────────────────────
#   R24(v)         — round v to 24-bit (binary32) precision (nearest-even)
#   asfloat16(u)   — reinterpret 16-bit integer u as a float16 and return its
#                    mathematical value as an exact rational/real
#   get_hex(v)     — format v as a C hexadecimal floating-point literal, e.g.
#                    "0x1.8p+3"  (suitable for pasting into a C initialiser)
#
# ──────────────────────────────────────────────────────────────────────────────

def table1(out):
    # Generate T1[], the "coarse" table: 2^11 = 2048 entries.
    #
    # T1[i1] stores RN_32(exp(x1)), the binary32 value nearest to exp(x1),
    # where x1 is the float16 whose bit pattern is  i1 × 2^5  (i.e., the
    # float16 formed by treating i1 as the top 11 bits and filling the bottom
    # 5 bits with zeros).
    #
    # Index range: i1 ∈ [0, 2^11)  →  2048 entries in T1.
    #
    # Special cases handled inline:
    #   • e == 31 (NaN or ±Infinity in float16):
    #       −Infinity (i1 = 2^10 + e<<5 with e = 31, i.e. i1 = 0x7c0):
    #         exp(−∞) = 0, so T1[i1] = 0.
    #       All other NaN/Inf encodings:
    #         The binary32 payload is reconstructed to propagate the NaN/Inf
    #         through the multiplication in inria-expf16.c.
    #   • Overflow: if exp(x1) ≥ 2^128 (beyond binary32 MAX_FLT), cap to
    #     the largest representable binary32 value  (nextbelow(2^128)).
    #   • Underflow: if exp(x1) < 2^−149 (below binary32 MIN_SUBNORMAL), floor
    #     to the smallest positive binary32 subnormal  2^−149.

    f = open(out,"w")
    f.write ("static const b32u32_u T1[] = {\n")
    s = ""
    for i1 in range(2^11):
        # Reconstruct the float16 bit pattern that i1 represents.
        # Shifting left by 5 restores the sign+exponent+upper-mantissa field;
        # the lower 5 mantissa bits are implicitly zero.
        u1 = i1*2^5

        # e = the 5-bit exponent field of u1
        e = (u1>>10) % (2^5)

        if e==31: # NaN or Inf
            if i1==2^10+(e<<5): # −Inf: bit pattern 0xFC00 → i1=0x7C0
                v1 = 0          # exp(−∞) = 0
            else:
                # Propagate the special value into binary32 so the downstream
                # multiplication in the C code sees a proper NaN/Inf.
                # The mapping widens the 5-bit float16 exponent to the 8-bit
                # binary32 exponent by adding 224 (= 255 − 31), preserving
                # the sign and the 10-bit mantissa payload (shifted to fit the
                # 23-bit binary32 mantissa field by shifting left 13).
                v1 = ((u1>>31)<<31) | ((224+e)<<23) | ((u1 % (2^10)) << 13)
            v = "{.u = " + hex(v1) + "}"
        else:
            # Normal / subnormal float16: compute exp(x1) in exact arithmetic,
            # then round to binary32.
            x1 = R24(asfloat16(u1))
            y1 = exp(x1)

            # Cap to the binary32 overflow boundary.
            if y1>=2^128.:
                y1 = R24(2^128).nextbelow()   # = 0x1.fffffep+127

            # Floor to the binary32 underflow boundary (smallest positive subnormal).
            if y1<2^-149.:
                y1 = R24(2^-149)              # = 0x1p-149

            v = "{" + get_hex(y1) + "}"

        # Line-wrap the output at 79 characters so the generated C file is readable.
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
    # Generate T2[], the "fine" table: 2^11 = 2048 entries.
    #
    # T2[i2] stores RN_32(exp(x2)), the binary32 value nearest to exp(x2),
    # where x2 is the small residual formed by the sign, exponent field, and
    # the lower 5 mantissa bits of the original float16 input:
    #
    #   Given i2 ∈ [0, 2^11):
    #     h = i2 // 2^5  →  the sign+exponent "header" (6 bits: bit 5 is sign)
    #     l = i2  % 2^5  →  the 5 low mantissa bits
    #     u2 = h × 2^10 + l   (reassembled bit pattern for table addressing)
    #
    #   For a normal float16 (e > 0), x2 is the residual value:
    #     x2 = 2^(e−25) × l
    #   For a subnormal float16 (e = 0), the implicit leading bit is absent:
    #     x2 = 2^(e−24) × l  = 2^−24 × l
    #
    #   If h ≥ 2^5 (i.e., the sign bit in h is set), x2 is negated.
    #
    # Special cases:
    #   • e == 31 (NaN or ±Infinity), handled identically to table1.
    #   • Five indices [159, 190, 249, 303, 493] are manually adjusted by one
    #     ULP downward (nextbelow).  These are "exceptional" cases where the
    #     true exp(x) for the full float16 input lands so close to the midpoint
    #     between two float16 values that a 1-ULP adjustment in T2 is needed to
    #     guarantee that  T1[i1] × T2[i2]  rounds to the correct float16.  These
    #     five were identified by exhaustive verification against an oracle.
    #   • Overflow / underflow: same caps as table1.

    f = open(out,"w")
    f.write ("static const b32u32_u T2[] = {\n")
    s = ""
    for i2 in range(2^11):
        # Decode i2 into the "header" h (sign + exponent) and the low mantissa l.
        h = i2 // (2^5)   # 6-bit header:  h[5] = sign bit, h[4:0] = exponent field
        l = i2 % (2^5)    # 5 low mantissa bits

        # Reconstruct the bit pattern: put the header back in bits 15..10,
        # put the low mantissa in bits 4..0 (bits 9..5 are zero).
        u2 = h*2^10+l

        # e = the 5-bit exponent field
        e = h % (2^5)

        if e==31: # NaN or Inf
            if i2==2^10+(e<<5): # −Inf
                v2 = 0
            else:
                v2 = ((u2>>31)<<31) | ((224+e)<<23) | ((u2 % (2^10)) << 13)
            v = "{.u = " + hex(v2) + "}"
        else:
            # Compute x2 = the residual contribution of the lower 5 mantissa bits.
            #   For normal floats (e > 0):  ulp of x1 = 2^(e−15−10) = 2^(e−25),
            #     so the residual is  x2 = 2^(e−25) × l.
            #   For subnormals (e = 0):     ulp of x1 = 2^(0−24)   = 2^−24,
            #     so the residual is  x2 = 2^(e−24) × l = 2^−24 × l.
            if e>0:
                x2 = R24(2^(e-25)*l)
            else:
                x2 = R24(2^(e-24)*l)

            # Apply sign: if h ≥ 2^5 the original float16 was negative.
            if h>=2^5:
                x2 = -x2

            y2 = exp(x2)

            # Manual 1-ULP adjustment for five "hard" inputs.
            # These five i2 values correspond to full float16 inputs x where
            # the exact result exp(x) falls within one binary32 ULP of the
            # exact midpoint between two consecutive float16 values.  In these
            # cases, the table-lookup algorithm would produce an incorrectly
            # rounded result without this fix.
            if i2 in [159,190,249,303,493]:
                y2 = y2.nextbelow()

            if y2>=2^128.:
                y2 = R24(2^128).nextbelow()
            if y2<2^-149.:
                y2 = R24(2^-149)

            v = "{" + get_hex(y2) + "}"

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
