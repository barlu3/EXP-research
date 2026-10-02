/* The input clusters and correctly rounded reference for float32 1/x, shared
   by the scalar benchmark (benchmark-recip.cpp) and its vector counterpart
   (../../vector/vrecpe/benchmark-recip-vec.cpp), so the two time the same inputs and
   are checked against the same answer. */

#ifndef RECIP_REF_HPP
#define RECIP_REF_HPP

#include "approx-bench.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace recip_ref {

// Input regions whose handling differs inside the kernels: FRECPE saturates to
// infinity below 2^-128 and produces subnormal estimates above 2^126 (under
// FPCR.AH it flushes both regions instead), and FDIV latency may depend on its
// operands. "any finite" draws every finite bit pattern with equal
// probability -- the literal "all real inputs" -- and so spends ~99% of its
// samples on normals.
inline std::uint32_t any_finite(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r(); if (((b >> 23) & 0xff) != 0xff) return b; }
}
inline std::uint32_t unit_binade(std::mt19937& r) {
    return ((std::uint32_t)r() & 0x807fffffu) | 0x3f800000u;
}
inline std::uint32_t subnormal(std::mt19937& r) {
    for (;;) { const std::uint32_t b = (std::uint32_t)r() & 0x807fffffu; if (b & 0x7fffffu) return b; }
}
inline std::uint32_t huge(std::mt19937& r) {   // exponent 253 or 254: 1/x is subnormal
    return ((std::uint32_t)r() & 0x807fffffu) | ((253u + ((std::uint32_t)r() & 1u)) << 23);
}

inline const std::vector<approx::Cluster> CLUSTERS = {
    { "any finite",     any_finite  },
    { "|x| in [1,2)",   unit_binade },
    { "subnormal",      subnormal   },
    { "|x| >= 2^126",   huge        },
};

// IEEE division is correctly rounded; no fast-math in the build.
inline constexpr const char* REFERENCE_NOTE = "1.0f/x (IEEE division)";
inline void reference(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = 1.0f / in[i];
}

// x and 1/x both normal: exponent field 1..252, either sign.
inline constexpr const char* NORMAL_NOTE = "x and 1/x both normal";
inline bool normal_block(std::uint32_t hi) {
    const std::uint32_t e = (hi >> 7) & 0xff;
    return e >= 1 && e <= 252;
}

// y*x is exact in double (24 x 24 bits), so this is the true relative error.
inline double rel_error(float x, float y) { return std::fabs((double)y * (double)x - 1.0); }

}  // namespace recip_ref

#endif  // RECIP_REF_HPP
