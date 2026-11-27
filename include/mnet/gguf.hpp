// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_GGUF_HPP
#define MNET_GGUF_HPP

#include <cmath>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstdint>
#include <cstring>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mnet/detail/iq1s_grid.hpp>
#include <mnet/detail/iq_grids.hpp>

namespace mnet::gguf {

// GGUF metadata value types.
inline constexpr std::uint32_t kTypeUint8 = 0;
inline constexpr std::uint32_t kTypeInt8 = 1;
inline constexpr std::uint32_t kTypeUint16 = 2;
inline constexpr std::uint32_t kTypeInt16 = 3;
inline constexpr std::uint32_t kTypeUint32 = 4;
inline constexpr std::uint32_t kTypeInt32 = 5;
inline constexpr std::uint32_t kTypeFloat32 = 6;
inline constexpr std::uint32_t kTypeBool = 7;
inline constexpr std::uint32_t kTypeString = 8;
inline constexpr std::uint32_t kTypeArray = 9;
inline constexpr std::uint32_t kTypeUint64 = 10;
inline constexpr std::uint32_t kTypeInt64 = 11;
inline constexpr std::uint32_t kTypeFloat64 = 12;

// GGML tensor types used by the supported models.
inline constexpr std::uint32_t kGgmlF32 = 0;
inline constexpr std::uint32_t kGgmlF16 = 1;
inline constexpr std::uint32_t kGgmlQ8_0 = 8;
inline constexpr std::uint32_t kGgmlQ4_K = 12;
inline constexpr std::uint32_t kGgmlQ5_K = 13;
inline constexpr std::uint32_t kGgmlQ6_K = 14;
inline constexpr std::uint32_t kGgmlIq2Xxs = 16;
inline constexpr std::uint32_t kGgmlIq2Xs = 17;
inline constexpr std::uint32_t kGgmlIq3Xxs = 18;
inline constexpr std::uint32_t kGgmlIq1S = 19;
inline constexpr std::uint32_t kGgmlIq2S = 22;
inline constexpr std::uint32_t kGgmlQ8_K = 15;
inline constexpr std::uint32_t kGgmlBf16 = 30;
inline constexpr std::uint32_t kGgmlNvfp4 = 40;
inline constexpr std::uint32_t kGgmlIq4Xs = 23;
inline constexpr std::uint32_t kGgmlQ2_0 = 42;

inline constexpr std::uint32_t kGgufMagic = 0x46554747U;

inline constexpr float kIq1sDelta = 0.125F;

inline constexpr std::int8_t kMxfp4Values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline constexpr std::int8_t kIq4NlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                 1,    13,   25,  38,  53,  69,  89,  113};

struct TypeInfo {
    std::string_view name;
    std::uint32_t block_elements;
    std::uint32_t block_bytes;
    bool supported;
};

[[nodiscard]] inline TypeInfo type_info(std::uint32_t type) {
    switch (type) {
        case kGgmlF32:
            return {"F32", 1, 4, true};
        case kGgmlF16:
            return {"F16", 1, 2, true};
        case kGgmlBf16:
            return {"BF16", 1, 2, true};
        case kGgmlQ8_0:
            return {"Q8_0", 32, 34, true};
        case kGgmlQ4_K:
            return {"Q4_K", 256, 144, true};
        case kGgmlQ5_K:
            return {"Q5_K", 256, 176, true};
        case kGgmlIq2Xxs:
            return {"IQ2_XXS", 256, 66, true};
        case kGgmlIq2Xs:
            return {"IQ2_XS", 256, 74, true};
        case kGgmlIq3Xxs:
            return {"IQ3_XXS", 256, 98, true};
        case kGgmlIq2S:
            return {"IQ2_S", 256, 82, true};
        case kGgmlQ6_K:
            return {"Q6_K", 256, 210, true};
        case kGgmlIq4Xs:
            return {"IQ4_XS", 256, 136, true};
        case kGgmlIq1S:
            return {"IQ1_S", 256, 50, true};
        case kGgmlNvfp4:
            return {"NVFP4", 64, 36, true};
        case kGgmlQ2_0:
            return {"Q2_0", 64, 18, true};
        default:
            return {"UNKNOWN", 1, 1, false};
    }
}

struct TensorInfo {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> dims;  // fastest-first, as stored by GGUF
    std::uint64_t offset = 0;
    std::uint64_t n_elements = 0;

    [[nodiscard]] std::uint64_t n_bytes() const {
        const auto info = type_info(this->type);
        if (!info.supported || this->n_elements % info.block_elements != 0) {
            return 0;
        }
        return this->n_elements / info.block_elements * info.block_bytes;
    }
};

[[nodiscard]] inline float half_to_float(std::uint16_t bits);
[[nodiscard]] inline float ue4m3_to_float(std::uint8_t bits);

[[nodiscard]] inline int nearest_int(float value) {
    // The barrier keeps the compiler from contracting the caller's `x * id` into
    // an fma with the +12582912 below: that would round the product+offset once
    // instead of rounding the product first, which flips exact .5 ties.
    volatile float rounded_input = value;
    const float val = rounded_input + 12582912.0F;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(bits));
    return static_cast<int>(bits & 0x007FFFFFU) - 0x00400000;
}

[[nodiscard]] inline std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t abs = bits & 0x7FFFFFFFU;
    if (abs >= 0x7F800000U) {
        return static_cast<std::uint16_t>(sign | 0x7C00U | (abs > 0x7F800000U ? 0x0200U : 0x0000U));
    }
    if (abs >= 0x477FF000U) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (abs < 0x38800000U) {
        const float scaled = std::ldexp(std::fabs(value), 24);
        const auto q = static_cast<std::uint32_t>(std::lround(scaled));
        if (q >= 0x0400U) {
            return static_cast<std::uint16_t>(sign | 0x0400U);
        }
        return static_cast<std::uint16_t>(sign | q);
    }
    std::uint32_t exponent = (abs >> 23U) - 112U;
    std::uint32_t mantissa = abs & 0x7FFFFFU;
    mantissa += 0x0FFFU + ((mantissa >> 13U) & 1U);
    if ((mantissa & 0x800000U) != 0) {
        mantissa = 0;
        ++exponent;
    }
    return static_cast<std::uint16_t>(sign | (exponent << 10U) | (mantissa >> 13U));
}

// llama.cpp quantizes the matmul input to vec_dot_type first. Reconstructing
// that quantized vector reproduces the same lossy input before the dot.
[[nodiscard]] inline std::vector<float> quantize_q8_k(const float* input, std::size_t columns) {
    std::vector<float> out(columns);
    for (std::size_t base = 0; base < columns; base += 256) {
        float max = 0.0F;
        float amax = 0.0F;
        for (std::size_t j = 0; j < 256; ++j) {
            const float value = input[base + j];
            const float ax = std::fabs(value);
            if (ax > amax) {
                amax = ax;
                max = value;
            }
        }
        if (amax == 0.0F) {
            continue;
        }
        const float iscale = -127.0F / max;
        const float d = 1.0F / iscale;
        for (std::size_t j = 0; j < 256; ++j) {
            int q = nearest_int(iscale * input[base + j]);
            q = std::min(127, q);
            out[base + j] = static_cast<float>(q) * d;
        }
    }
    return out;
}

[[nodiscard]] inline std::vector<float> quantize_q8_0(const float* input, std::size_t columns) {
    std::vector<float> out(columns);
    for (std::size_t base = 0; base < columns; base += 32) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(input[base + j]));
        }
        const float d = amax / 127.0F;
        // llama's quantize_row_q8_0 uses 127/amax directly (one rounding), not
        // 1/(amax/127); the difference is 1 ulp and flips exact .5 ties.
        const float id = amax != 0.0F ? 127.0F / amax : 0.0F;
        const float stored = half_to_float(float_to_half(d));
        for (std::size_t j = 0; j < 32; ++j) {
            out[base + j] = static_cast<float>(nearest_int(input[base + j] * id)) * stored;
        }
    }
    return out;
}

[[nodiscard]] inline std::vector<float> round_bf16_input(const float* input, std::size_t columns) {
    std::vector<float> out(columns);
    for (std::size_t i = 0; i < columns; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &input[i], sizeof(bits));
        if ((bits & 0x7F800000U) != 0x7F800000U) {
            bits += 0x7FFFU + ((bits >> 16U) & 1U);
        }
        bits &= 0xFFFF0000U;
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return out;
}

[[nodiscard]] inline bool uses_q8_k(std::uint32_t type) {
    switch (type) {
        case kGgmlQ4_K:
        case kGgmlQ5_K:
        case kGgmlQ6_K:
        case kGgmlIq1S:
        case kGgmlIq2Xxs:
        case kGgmlIq2Xs:
        case kGgmlIq2S:
        case kGgmlIq3Xxs:
        case kGgmlIq4Xs:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] inline bool uses_q8_0(std::uint32_t type) {
    return type == kGgmlQ8_0 || type == kGgmlNvfp4 || type == kGgmlQ2_0;
}

struct Q8KBlocks {
    std::vector<std::int8_t> qs;
    std::vector<float> d;
    std::vector<std::int16_t> bsums;
};

struct Q8_0Blocks {
    std::vector<std::int8_t> qs;
    std::vector<float> d;
};

[[nodiscard]] inline Q8KBlocks make_q8_k(const float* input, std::size_t columns) {
    Q8KBlocks out;
    out.qs.resize(columns);
    out.d.resize(columns / 256);
    out.bsums.resize(columns / 16);
    for (std::size_t b = 0; b < columns / 256; ++b) {
        float max = 0.0F;
        float amax = 0.0F;
        for (std::size_t j = 0; j < 256; ++j) {
            const float value = input[b * 256 + j];
            const float ax = std::fabs(value);
            if (ax > amax) {
                amax = ax;
                max = value;
            }
        }
        if (amax == 0.0F) {
            out.d[b] = 0.0F;
            continue;
        }
        const float iscale = -127.0F / max;
        for (std::size_t j = 0; j < 256; ++j) {
            int q = nearest_int(iscale * input[b * 256 + j]);
            q = std::min(127, q);
            out.qs[b * 256 + j] = static_cast<std::int8_t>(q);
        }
        for (std::size_t g = 0; g < 16; ++g) {
            int sum = 0;
            for (std::size_t j = 0; j < 16; ++j) {
                sum += out.qs[b * 256 + g * 16 + j];
            }
            out.bsums[b * 16 + g] = static_cast<std::int16_t>(sum);
        }
        out.d[b] = 1.0F / iscale;
    }
    return out;
}

[[nodiscard]] inline Q8_0Blocks make_q8_0_blocks(const float* input, std::size_t columns) {
    Q8_0Blocks out;
    out.qs.resize(columns);
    out.d.resize(columns / 32);
    for (std::size_t b = 0; b < columns / 32; ++b) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(input[b * 32 + j]));
        }
        const float d = amax / 127.0F;
        // 127/amax, not 1/(amax/127): matches quantize_row_q8_0 bit for bit.
        const float id = amax != 0.0F ? 127.0F / amax : 0.0F;
        out.d[b] = half_to_float(float_to_half(d));
        for (std::size_t j = 0; j < 32; ++j) {
            out.qs[b * 32 + j] = static_cast<std::int8_t>(nearest_int(input[b * 32 + j] * id));
        }
    }
    return out;
}

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] inline float dot_q6_k_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] inline float dot_iq1_s_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] inline float dot_iq2_xxs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] inline float dot_iq2_xs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] inline float dot_iq3_xxs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] inline float dot_iq2_s_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
#endif

[[nodiscard]] inline float dot_q6_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q6_k_q8_k_avx2(data, q8, blocks);
#else
    float sums[8] = {};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* ql = data + b * 210;
        const std::uint8_t* qh = ql + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(ql + 192);
        const float d =
            half_to_float(static_cast<std::uint16_t>(ql[208] | (static_cast<std::uint16_t>(ql[209]) << 8U))) * q8.d[b];
        std::int8_t a[256];
        for (int half_index = 0; half_index < 2; ++half_index) {
            const std::uint8_t* qlh = ql + half_index * 64;
            const std::uint8_t* qhh = qh + half_index * 32;
            for (int l = 0; l < 32; ++l) {
                a[half_index * 128 + l] = static_cast<std::int8_t>(((qlh[l] & 0x0F) | (((qhh[l] >> 0) & 3) << 4))) - 32;
                a[half_index * 128 + l + 32] =
                    static_cast<std::int8_t>(((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4))) - 32;
                a[half_index * 128 + l + 64] =
                    static_cast<std::int8_t>(((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4))) - 32;
                a[half_index * 128 + l + 96] =
                    static_cast<std::int8_t>(((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4))) - 32;
            }
        }
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::int32_t aux32[8] = {};
        int is = 0;
        for (int j = 0; j < 16; ++j) {
            const int scale = scales[is++];
            for (int l = 0; l < 8; ++l) {
                aux32[l] += scale * q[l] * a[j * 16 + l];
            }
            for (int l = 0; l < 8; ++l) {
                aux32[l] += scale * q[8 + l] * a[j * 16 + 8 + l];
            }
            q += 16;
        }
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    float total = 0.0F;
    for (const float value : sums) {
        total += value;
    }
    return total;
#endif
}

[[nodiscard]] inline float dot_iq1_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq1_s_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 50;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::uint8_t* qs = block + 2;
        const auto* qh = reinterpret_cast<const std::uint16_t*>(block + 34);
        const std::int8_t* q = q8.qs.data() + b * 256;
        int sumi = 0;
        int sumi1 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
            const int delta = (qh[ib] & 0x8000U) != 0 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint16_t index =
                    static_cast<std::uint16_t>(qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8));
                const std::uint64_t grid = detail::kIq1sGrid[index];
                for (int j = 0; j < 8; ++j) {
                    const auto value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                    lsum += q[j] * static_cast<int>(value);
                }
                q += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta *
                     (q8.bsums[b * 16U + static_cast<std::uint64_t>(2 * ib)] +
                      q8.bsums[b * 16U + static_cast<std::uint64_t>(2 * ib + 1)]);
        }
        sumf += d * (static_cast<float>(sumi) + kIq1sDelta * static_cast<float>(sumi1));
    }
    return sumf;
#endif
}

[[nodiscard]] inline std::uint16_t float_to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 64U);
    }
    return static_cast<std::uint16_t>((bits + (0x7FFFU + ((bits >> 16U) & 1U))) >> 16U);
}

[[nodiscard]] inline float bf16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] inline float hsum_float_8_avx2(__m256 value) {
    __m128 res = _mm256_extractf128_ps(value, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(value));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

// The tinyBLAS kernels below are plain float code (no intrinsics): they describe
// llamafile's eight-lane accumulation order and must stay available on every
// architecture so the runners reproduce the same products off x86.

// llama.cpp enables llamafile sgemm (tinyBLAS) only for batches with two or
// more tokens (`llamafile_sgemm`: if (n < 2) return false). tinyBLAS computes
// every output with an eight-lane FMA accumulator over K and one fixed hsum:
//   ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))
// which differs in the last bits from ggml_vec_dot_f32. Set from the runner
// when the model is evaluated with more than one token.
inline bool& ggml_tinyblas_batch() {
    static bool enabled = false;
    return enabled;
}

[[nodiscard]] inline float dot_f32_tinyblas(const float* a, const float* b, std::uint64_t size) {
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::uint64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        for (std::uint64_t j = 0; j < 8; ++j) {
            acc[j] = std::fma(a[i + j], b[i + j], acc[j]);
        }
    }
    const std::uint64_t rem = size - i;
    for (std::uint64_t j = 0; j < rem; ++j) {
        acc[j] = std::fma(a[i + j], b[i + j], acc[j]);
    }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

// The batched (n >= 2) llamafile_sgemm BF16 kernel: weights and activations
// are bf16, promoted to f32 and fused into an eight-lane accumulator over K,
// then reduced as ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)) (sgemm.cpp hsum).
[[nodiscard]] inline float dot_bf16_tinyblas(const std::uint8_t* data, const std::uint16_t* input,
                                             std::uint64_t size) {
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::uint64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        for (std::uint64_t j = 0; j < 8; ++j) {
            const std::uint16_t bits =
                static_cast<std::uint16_t>(data[(i + j) * 2] | (data[(i + j) * 2 + 1] << 8U));
            acc[j] = std::fma(bf16_bits_to_float(bits), bf16_bits_to_float(input[i + j]), acc[j]);
        }
    }
    for (std::uint64_t j = 0; i + j < size; ++j) {
        const std::uint16_t bits = static_cast<std::uint16_t>(data[(i + j) * 2] | (data[(i + j) * 2 + 1] << 8U));
        acc[j] = std::fma(bf16_bits_to_float(bits), bf16_bits_to_float(input[i + j]), acc[j]);
    }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] inline float dot_f32_reference_avx2(const float* x, const float* y, std::uint64_t size) {
    const std::uint64_t aligned = size & ~static_cast<std::uint64_t>(31);
    __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (std::uint64_t i = 0; i < aligned; i += 32) {
        for (std::uint64_t j = 0; j < 4; ++j) {
            sums[j] = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + j * 8), _mm256_loadu_ps(y + i + j * 8), sums[j]);
        }
    }
    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sums[0]), _mm256_extractf128_ps(sums[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    float total = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    // ggml_vec_dot_f32 finishes with its leftover path: full groups of four are
    // rounded products added in order, the final size % 4 values are fused.
    std::uint64_t i = aligned;
    const std::uint64_t tail = size - aligned;
    const std::uint64_t rounded = tail - tail % 4;
    for (std::uint64_t k = 0; k < rounded; ++k, ++i) {
        volatile float product = x[i] * y[i];
        total += product;
    }
    for (; i < size; ++i) {
        total = std::fma(x[i], y[i], total);
    }
    return total;
}

[[nodiscard]] inline __m256 mul_sum_i8_pairs_float_avx2(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
}

[[nodiscard]] inline __m256i q2_0_unpack_32_avx2(const std::uint8_t* qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs)));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

[[nodiscard]] inline __m256 q2_0_dot_32_avx2(const std::uint8_t* qs, const std::int8_t* qy) {
    const __m256i v = q2_0_unpack_32_avx2(qs);
    const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qy));
    const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(v, y), _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

[[nodiscard]] inline float dot_q8_0_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
    __m256 acc = _mm256_setzero_ps();
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const __m256 d = _mm256_set1_ps(half_to_float(static_cast<std::uint16_t>(data[b * 34]) |
                                                      (static_cast<std::uint16_t>(data[b * 34 + 1]) << 8U)) *
                                        q8.d[b]);
        const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + b * 34 + 2));
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8.qs.data() + b * 32));
        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float_avx2(qx, qy), acc);
    }
    return hsum_float_8_avx2(acc);
}

[[nodiscard]] inline __m256 q2_0_acc_block_avx2(const std::uint8_t* block, const float* dy, const std::int8_t* qy,
                                                __m256 acc) {
    const float dx = half_to_float(static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8U));
    const __m256 d0 = _mm256_set1_ps(dx * dy[0]);
    const __m256 d1 = _mm256_set1_ps(dx * dy[1]);
    acc = _mm256_fmadd_ps(d0, q2_0_dot_32_avx2(block + 2, qy), acc);
    return _mm256_fmadd_ps(d1, q2_0_dot_32_avx2(block + 2 + 8, qy + 32), acc);
}

[[nodiscard]] inline __m256 load_bf16_8_avx2(const std::uint8_t* data) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

// ggml_vec_dot_bf16: bf16 weight x bf16 activation, four 8-lane mul/add
// accumulators and one double accumulation per 32 elements.
[[nodiscard]] inline float dot_bf16_bf16_avx2(const std::uint8_t* data, const std::uint16_t* input,
                                              std::uint64_t columns) {
    std::uint64_t i = 0;
    double sumf = 0.0;
    const auto* input_bytes = reinterpret_cast<const std::uint8_t*>(input);
    // The four vector accumulators live across the whole loop (reduced once), so
    // the summation order matches ggml_vec_dot_bf16 exactly.
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= columns; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(load_bf16_8_avx2(data + i * 2), load_bf16_8_avx2(input_bytes + i * 2)), c1);
        c2 = _mm256_add_ps(
            _mm256_mul_ps(load_bf16_8_avx2(data + (i + 8) * 2), load_bf16_8_avx2(input_bytes + (i + 8) * 2)), c2);
        c3 = _mm256_add_ps(
            _mm256_mul_ps(load_bf16_8_avx2(data + (i + 16) * 2), load_bf16_8_avx2(input_bytes + (i + 16) * 2)), c3);
        c4 = _mm256_add_ps(
            _mm256_mul_ps(load_bf16_8_avx2(data + (i + 24) * 2), load_bf16_8_avx2(input_bytes + (i + 24) * 2)), c4);
    }
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4));
    {
        __m128 g = _mm_add_ps(_mm256_extractf128_ps(c1, 1), _mm256_castps256_ps128(c1));
        g = _mm_add_ps(g, _mm_movehl_ps(g, g));
        g = _mm_add_ss(g, _mm_movehdup_ps(g));
        sumf += static_cast<double>(_mm_cvtss_f32(g));
    }
    for (; i < columns; ++i) {
        const float weight = bf16_bits_to_float(static_cast<std::uint16_t>(data[i * 2] | (data[i * 2 + 1] << 8U)));
        sumf += static_cast<double>(weight * bf16_bits_to_float(input[i]));
    }
    return static_cast<float>(sumf);
}

[[nodiscard]] inline float dot_q2_0_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::uint64_t i = 0;
    for (; i + 1 < blocks; i += 2) {
        acc0 = q2_0_acc_block_avx2(data + i * 18, q8.d.data() + 2 * i, q8.qs.data() + 2 * i * 32, acc0);
        acc1 = q2_0_acc_block_avx2(data + (i + 1) * 18, q8.d.data() + 2 * i + 2, q8.qs.data() + (2 * i + 2) * 32, acc1);
    }
    if (i < blocks) {
        acc0 = q2_0_acc_block_avx2(data + i * 18, q8.d.data() + 2 * i, q8.qs.data() + 2 * i * 32, acc0);
    }
    return hsum_float_8_avx2(_mm256_add_ps(acc0, acc1));
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] inline __m256i mul_add_epi8_avx2(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

[[nodiscard]] inline float dot_iq4_xs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kIq4NlValues));
    const __m128i m4b = _mm_set1_epi8(0x0F);
    __m256 accum = _mm256_setzero_ps();
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 136;
        const std::uint8_t* qs = block + 8;
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::uint16_t sh = static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U));
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
            qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
            qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8b_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q4b_1 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(q4bits_1, m4b))),
                                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)), 1);
            const __m256i q4b_2 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(q4bits_2, m4b))),
                                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)), 1);
            const __m256i p16_1 = mul_add_epi8_avx2(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8_avx2(q4b_2, q8b_2);
            const int16_t ls1 = static_cast<int16_t>(((block[4 + ib / 2] & 0x0F) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = static_cast<int16_t>(((block[4 + ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
            sh = static_cast<std::uint16_t>(sh >> 4);
            sumi1 = _mm256_add_epi32(_mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1)), sumi1);
            sumi2 = _mm256_add_epi32(_mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2)), sumi2);
        }
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }
    return hsum_float_8_avx2(accum);
}

[[nodiscard]] inline float dot_nvfp4_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kMxfp4Values));
    const __m128i m4b = _mm_set1_epi8(0x0F);
    const __m256i one16 = _mm256_set1_epi16(1);
    __m256 accum = _mm256_setzero_ps();
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 36;
        const __m128i q4bits_01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 4));
        const __m128i q4bits_23 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 4 + 16));
        const __m256i q8_01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8.qs.data() + (2 * b) * 32));
        const __m256i q8_23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8.qs.data() + (2 * b + 1) * 32));
        const __m128i lo01 = _mm_shuffle_epi8(values, _mm_and_si128(q4bits_01, m4b));
        const __m128i hi01 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i lo23 = _mm_shuffle_epi8(values, _mm_and_si128(q4bits_23, m4b));
        const __m128i hi23 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));
        const __m256i q4_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(lo01, hi01)),
                                                      _mm_unpackhi_epi64(lo01, hi01), 1);
        const __m256i q4_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(lo23, hi23)),
                                                      _mm_unpackhi_epi64(lo23, hi23), 1);
        const __m256i p_1 = _mm256_madd_epi16(mul_add_epi8_avx2(q4_01, q8_01), one16);
        const __m256i p_2 = _mm256_madd_epi16(mul_add_epi8_avx2(q4_23, q8_23), one16);
        const float dy0 = q8.d[2 * b];
        const float dy1 = q8.d[2 * b + 1];
        const float s0 = 0.5F * ue4m3_to_float(block[0]) * dy0;
        const float s1 = 0.5F * ue4m3_to_float(block[1]) * dy0;
        const float s2 = 0.5F * ue4m3_to_float(block[2]) * dy1;
        const float s3 = 0.5F * ue4m3_to_float(block[3]) * dy1;
        const __m256 scales01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(s0)), _mm_set1_ps(s1), 1);
        const __m256 scales23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(s2)), _mm_set1_ps(s3), 1);
        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    return hsum_float_8_avx2(accum);
}

[[nodiscard]] inline __m128i iq_scale_shuffle_avx2(int index) {
    alignas(16) static constexpr std::uint8_t kShuffle[128] = {
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
        3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
        6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(kShuffle) + index);
}

[[nodiscard]] inline const std::uint64_t* even_signs_avx2() {
    static const std::vector<std::uint64_t> table = [] {
        std::vector<std::uint64_t> out(128, 0);
        for (std::size_t code = 0; code < 128; ++code) {
            std::uint64_t bits = 0;
            for (std::size_t j = 0; j < 8; ++j) {
                const std::uint64_t byte = (detail::kSignsIq2xs[code] & (1U << j)) != 0 ? 0xFFU : 0x01U;
                bits |= byte << (8 * j);
            }
            out[code] = bits;
        }
        return out;
    }();
    return table.data();
}

[[nodiscard]] inline float dot_q6_k_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 210;
        const std::uint8_t* ql = block;
        const std::uint8_t* qh = block + 128;
        const std::int8_t* q = q8.qs.data() + i * 256;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[208] | (static_cast<std::uint16_t>(block[209]) << 8U)));
        const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8.bsums.data() + i * 16));
        const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 192));
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);
        __m256i sumi = _mm256_setzero_si256();
        int is = 0;
        for (int j = 0; j < 2; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            ql += 32;
            const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            ql += 32;
            const __m256i q4bits_h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            qh += 32;
            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);
            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
            const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
            const __m128i scale_0 = _mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(is + 3));
            is += 4;
            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8_avx2(acc);
}

[[nodiscard]] inline float dot_iq1_s_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0.0F;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 50;
        const std::int8_t* q = q8.qs.data() + i * 256;
        const std::uint8_t* qs = block + 2;
        const auto* qh = reinterpret_cast<const std::uint16_t*>(block + 34);
        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i q1b_1 =
                _mm256_set_epi64x(static_cast<long long>(detail::kIq1sGrid[qs[3] | ((qh[ib] >> 1) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[2] | ((qh[ib] << 2) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[1] | ((qh[ib] << 5) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[0] | ((qh[ib] << 8) & 0x700)]));
            const __m256i q1b_2 =
                _mm256_set_epi64x(static_cast<long long>(detail::kIq1sGrid[qs[7] | ((qh[ib + 1] >> 1) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[6] | ((qh[ib + 1] << 2) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[5] | ((qh[ib + 1] << 5) & 0x700)]),
                                  static_cast<long long>(detail::kIq1sGrid[qs[4] | ((qh[ib + 1] << 8) & 0x700)]));
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8b_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i dot1 = mul_add_epi8_avx2(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8_avx2(q1b_2, q8b_2);
            const auto ls1 = static_cast<std::int16_t>(2 * ((qh[ib] >> 12) & 7) + 1);
            const auto ls2 = static_cast<std::int16_t>(2 * ((qh[ib + 1] >> 12) & 7) + 1);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            const std::size_t bi = static_cast<std::size_t>(ib);
            sumi1 += (q8.bsums[i * 16 + 2 * bi + 0] + q8.bsums[i * 16 + 2 * bi + 1]) *
                         ((qh[ib] & 0x8000U) != 0 ? -1 : 1) * ls1 +
                     (q8.bsums[i * 16 + 2 * bi + 2] + q8.bsums[i * 16 + 2 * bi + 3]) *
                         ((qh[ib + 1] & 0x8000U) != 0 ? -1 : 1) * ls2;
        }
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * static_cast<float>(sumi1);
    }
    return hsum_float_8_avx2(accum) + 0.125F * accum1;
}

[[nodiscard]] inline float dot_iq2_xxs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    const std::uint64_t* signs64 = even_signs_avx2();
    __m256 accumf = _mm256_setzero_ps();
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 66;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const auto* q2 = reinterpret_cast<const std::uint16_t*>(block + 2);
        const std::int8_t* q = q8.qs.data() + i * 256;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            std::uint32_t aux32[4] = {};
            std::memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
            const __m256i grid_1 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xxsGrid[aux8[3]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[2]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[1]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[0]]));
            const __m256i grid_2 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xxsGrid[aux8[11]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[10]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[9]]),
                                                     static_cast<long long>(detail::kIq2xxsGrid[aux8[8]]));
            const __m256i s2_1 = _mm256_set_epi64x(static_cast<long long>(signs64[(aux32[1] >> 21) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[1] >> 14) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[1] >> 7) & 127U]),
                                                   static_cast<long long>(signs64[aux32[1] & 127U]));
            const __m256i s2_2 = _mm256_set_epi64x(static_cast<long long>(signs64[(aux32[3] >> 21) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[3] >> 14) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[3] >> 7) & 127U]),
                                                   static_cast<long long>(signs64[aux32[3] & 127U]));
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1 = _mm256_maddubs_epi16(grid_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(grid_2, q8s_2);
            const auto ls1 = static_cast<std::uint16_t>(aux32[1] >> 28);
            const auto ls2 = static_cast<std::uint16_t>(aux32[3] >> 28);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(static_cast<std::int16_t>(2 * ls1 + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(static_cast<std::int16_t>(2 * ls2 + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125F * hsum_float_8_avx2(accumf);
}

[[nodiscard]] inline float dot_iq2_xs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    const __m256i mone = _mm256_set1_epi8(1);
    alignas(32) static constexpr std::uint8_t kSignShuffle1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
                                                                   4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6};
    alignas(32) static constexpr std::uint8_t kSignShuffle2[32] = {8,  8,  8,  8,  8,  8,  8,  8,  10, 10, 10,
                                                                   10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12,
                                                                   12, 12, 14, 14, 14, 14, 14, 14, 14, 14};
    alignas(32) static constexpr std::uint8_t kBitSelector[32] = {1,  2,   4,  8,  16,  32, 64, 128, 1,  2,  4,
                                                                  8,  16,  32, 64, 128, 1,  2,  4,   8,  16, 32,
                                                                  64, 128, 1,  2,  4,   8,  16, 32,  64, 128};
    alignas(32) static constexpr std::uint8_t kBitHelper[32] = {0,   128, 128, 0,   128, 0,   0,   128, 128, 0,   0,
                                                                128, 0,   128, 128, 0,   0,   128, 128, 0,   128, 0,
                                                                0,   128, 128, 0,   0,   128, 0,   128, 128, 0};
    const __m256i bit_selector = _mm256_load_si256(reinterpret_cast<const __m256i*>(kBitSelector));
    const __m256i sign_shuffle1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(kSignShuffle1));
    const __m256i sign_shuffle2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(kSignShuffle2));
    const __m256i bit_helper = _mm256_load_si256(reinterpret_cast<const __m256i*>(kBitHelper));
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m1 = _mm_set1_epi8(1);
    __m256 accumf = _mm256_setzero_ps();
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 74;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const auto* q2 = reinterpret_cast<const std::uint16_t*>(block + 2);
        const std::int8_t* q = q8.qs.data() + i * 256;
        std::uint64_t aux64 = 0;
        std::memcpy(&aux64, block + 66, 8);
        __m128i stmp = _mm_set1_epi64x(static_cast<long long>(aux64));
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 4) {
            const __m256i q2_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2));
            q2 += 16;
            const __m256i aux_gindex = _mm256_and_si256(q2_data, m511);
            const auto* gindex = reinterpret_cast<const std::uint16_t*>(&aux_gindex);
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i grid_1 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xsGrid[gindex[3]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[2]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[1]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[0]]));
            const __m256i grid_2 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xsGrid[gindex[7]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[6]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[5]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[4]]));
            const __m256i grid_3 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xsGrid[gindex[11]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[10]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[9]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[8]]));
            const __m256i grid_4 = _mm256_set_epi64x(static_cast<long long>(detail::kIq2xsGrid[gindex[15]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[14]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[13]]),
                                                     static_cast<long long>(detail::kIq2xsGrid[gindex[12]]));
            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(full_signs_l), full_signs_l, 1);
            const __m256i full_signs_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(full_signs_h), full_signs_h, 1);
            __m256i signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector), bit_selector);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));
            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector), bit_selector);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));
            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector), bit_selector);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));
            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector), bit_selector);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));
            const __m256i dot1 = _mm256_maddubs_epi16(grid_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(grid_2, q8s_2);
            const __m256i dot3 = _mm256_maddubs_epi16(grid_3, q8s_3);
            const __m256i dot4 = _mm256_maddubs_epi16(grid_4, q8s_4);
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(ib + 0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(ib + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(ib + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, iq_scale_shuffle_avx2(ib + 3)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125F * hsum_float_8_avx2(accumf);
}

[[nodiscard]] inline float dot_iq3_xxs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    const std::uint64_t* signs64 = even_signs_avx2();
    __m256 accumf = _mm256_setzero_ps();
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 98;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* q3 = block + 2;
        const std::uint8_t* gas = block + 2 + 64;
        const std::int8_t* q = q8.qs.data() + i * 256;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i grid_1 = _mm256_set_epi32(
                static_cast<int>(detail::kIq3xxsGrid[q3[7]]), static_cast<int>(detail::kIq3xxsGrid[q3[6]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[5]]), static_cast<int>(detail::kIq3xxsGrid[q3[4]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[3]]), static_cast<int>(detail::kIq3xxsGrid[q3[2]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[1]]), static_cast<int>(detail::kIq3xxsGrid[q3[0]]));
            q3 += 8;
            const __m256i grid_2 = _mm256_set_epi32(
                static_cast<int>(detail::kIq3xxsGrid[q3[7]]), static_cast<int>(detail::kIq3xxsGrid[q3[6]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[5]]), static_cast<int>(detail::kIq3xxsGrid[q3[4]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[3]]), static_cast<int>(detail::kIq3xxsGrid[q3[2]]),
                static_cast<int>(detail::kIq3xxsGrid[q3[1]]), static_cast<int>(detail::kIq3xxsGrid[q3[0]]));
            q3 += 8;
            std::uint32_t aux32[2] = {};
            std::memcpy(aux32, gas, sizeof(aux32));
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(static_cast<long long>(signs64[(aux32[0] >> 21) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[0] >> 14) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[0] >> 7) & 127U]),
                                                   static_cast<long long>(signs64[aux32[0] & 127U]));
            const __m256i s2_2 = _mm256_set_epi64x(static_cast<long long>(signs64[(aux32[1] >> 21) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[1] >> 14) & 127U]),
                                                   static_cast<long long>(signs64[(aux32[1] >> 7) & 127U]),
                                                   static_cast<long long>(signs64[aux32[1] & 127U]));
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1 = _mm256_maddubs_epi16(grid_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(grid_2, q8s_2);
            const auto ls1 = static_cast<std::uint16_t>(aux32[0] >> 28);
            const auto ls2 = static_cast<std::uint16_t>(aux32[1] >> 28);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(static_cast<std::int16_t>(2 * ls1 + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(static_cast<std::int16_t>(2 * ls2 + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.25F * hsum_float_8_avx2(accumf);
}

[[nodiscard]] inline const std::uint8_t* iq_scale_shuffle_k4_avx2() {
    static const std::vector<std::uint8_t> table = [] {
        std::vector<std::uint8_t> out(256, 0);
        for (std::size_t group = 0; group < 8; ++group) {
            for (std::size_t j = 0; j < 16; ++j) {
                out[group * 32 + 2 * j] = static_cast<std::uint8_t>(2 * group);
                out[group * 32 + 2 * j + 1] = static_cast<std::uint8_t>(2 * group + 1);
            }
        }
        return out;
    }();
    return table.data();
}

[[nodiscard]] inline float dot_iq2_s_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    alignas(32) static constexpr std::uint8_t kMask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    alignas(32) static constexpr std::uint8_t kMask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                                            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(kMask1));
    const __m256i mask2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(kMask2));
    const std::uint8_t* k4 = iq_scale_shuffle_k4_avx2();
    __m256 accumf = _mm256_setzero_ps();
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 82;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* qs = block + 2;
        const std::uint8_t* qh = block + 66;
        const auto* signs = reinterpret_cast<const std::uint16_t*>(block + 34);
        const std::int8_t* q = q8.qs.data() + i * 256;
        std::uint64_t aux64 = 0;
        std::memcpy(&aux64, block + 74, 8);
        const __m128i scales8 = _mm_add_epi8(
            _mm_slli_epi16(
                _mm_and_si128(_mm_set_epi64x(static_cast<long long>(aux64 >> 4), static_cast<long long>(aux64)), m4),
                1),
            m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i grid_1 =
                _mm256_set_epi64x(static_cast<long long>(detail::kIq2sGrid[qs[3] | ((qh[ib] << 2) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[2] | ((qh[ib] << 4) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[1] | ((qh[ib] << 6) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[0] | ((qh[ib] << 8) & 0x300)]));
            const __m256i grid_2 =
                _mm256_set_epi64x(static_cast<long long>(detail::kIq2sGrid[qs[7] | ((qh[ib + 1] << 2) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[6] | ((qh[ib + 1] << 4) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[5] | ((qh[ib + 1] << 6) & 0x300)]),
                                  static_cast<long long>(detail::kIq2sGrid[qs[4] | ((qh[ib + 1] << 8) & 0x300)]));
            qs += 8;
            __m256i aux256 = _mm256_set1_epi32(signs[0] | (static_cast<std::uint32_t>(signs[1]) << 16U));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | (static_cast<std::uint32_t>(signs[3]) << 16U));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            const __m256i dot1 = _mm256_maddubs_epi16(grid_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(grid_2, q8s_2);
            const __m256i sh1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k4) + (ib + 0) * 1);
            const __m256i sh2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k4) + (ib + 1) * 1);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, sh1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, sh2));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125F * hsum_float_8_avx2(accumf);
}
#endif

[[nodiscard]] inline float dot_q2_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q2_0_q8_0_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 18U;
        const float d0 =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* qs = block + 2;
        float sumi = 0.0F;
        for (int k = 0; k < 2; ++k) {
            const float d1 = q8.d[b * 2U + static_cast<std::uint64_t>(k)];
            const std::int8_t* qy = q8.qs.data() + (b * 2U + static_cast<std::uint64_t>(k)) * 32U;
            const std::uint8_t* q = qs + static_cast<std::uint64_t>(k) * 8U;
            int sumi_block = 0;
            for (int bb = 0; bb < 8; ++bb) {
                const std::uint8_t byte = q[bb];
                sumi_block += (((byte >> 0) & 3) - 1) * qy[static_cast<std::size_t>(bb) * 4U + 0U];
                sumi_block += (((byte >> 2) & 3) - 1) * qy[static_cast<std::size_t>(bb) * 4U + 1U];
                sumi_block += (((byte >> 4) & 3) - 1) * qy[static_cast<std::size_t>(bb) * 4U + 2U];
                sumi_block += (((byte >> 6) & 3) - 1) * qy[static_cast<std::size_t>(bb) * 4U + 3U];
            }
            sumi += d1 * static_cast<float>(sumi_block);
        }
        sumf += d0 * sumi;
    }
    return sumf;
#endif
}

[[nodiscard]] inline float dot_q8_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q8_0_q8_0_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 34;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        int sumi = 0;
        for (int j = 0; j < 32; ++j) {
            sumi += static_cast<int>(static_cast<std::int8_t>(block[2 + j])) *
                    q8.qs[b * 32U + static_cast<std::uint64_t>(j)];
        }
        sumf += d * static_cast<float>(sumi);
    }
    return sumf;
#endif
}

[[nodiscard]] inline float dot_iq2_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_xxs_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 66;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* q2 = block + 2 + 8 * ib;
            const std::uint32_t lo = static_cast<std::uint32_t>(q2[0]) | (static_cast<std::uint32_t>(q2[1]) << 8U) |
                                     (static_cast<std::uint32_t>(q2[2]) << 16U) |
                                     (static_cast<std::uint32_t>(q2[3]) << 24U);
            const std::uint32_t hi = static_cast<std::uint32_t>(q2[4]) | (static_cast<std::uint32_t>(q2[5]) << 8U) |
                                     (static_cast<std::uint32_t>(q2[6]) << 16U) |
                                     (static_cast<std::uint32_t>(q2[7]) << 24U);
            const int ls = 2 * static_cast<int>(hi >> 28U) + 1;
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = detail::kIq2xxsGrid[(lo >> (8 * l)) & 0xFFU];
                const std::uint8_t signs = detail::kSignsIq2xs[(hi >> (7 * l)) & 127U];
                for (int j = 0; j < 8; ++j) {
                    const int value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                    sumi += value * q[j] * ((signs & (1U << j)) != 0 ? -1 : 1);
                }
                q += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125F * sumf;
#endif
}

[[nodiscard]] inline float dot_iq2_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_xs_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 74;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::uint8_t* q2 = block + 2;
        const std::uint8_t* sc = block + 66;
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls1 = 2 * (sc[ib] & 0x0F) + 1;
            const int ls2 = 2 * (sc[ib] >> 4) + 1;
            for (int half = 0; half < 2; ++half) {
                int sumi = 0;
                for (int l = 2 * half; l < 2 * half + 2; ++l) {
                    const std::uint16_t qv = static_cast<std::uint16_t>(
                        q2[2 * (4 * ib + l)] | (static_cast<std::uint16_t>(q2[2 * (4 * ib + l) + 1]) << 8U));
                    const std::uint64_t grid = detail::kIq2xsGrid[qv & 511U];
                    const std::uint8_t signs = detail::kSignsIq2xs[qv >> 9U];
                    for (int j = 0; j < 8; ++j) {
                        const int value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                        sumi += value * q[j] * ((signs & (1U << j)) != 0 ? -1 : 1);
                    }
                    q += 8;
                }
                bsum += sumi * (half == 0 ? ls1 : ls2);
            }
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125F * sumf;
#endif
}

[[nodiscard]] inline float dot_iq2_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_s_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 82;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::uint8_t* qs = block + 2;
        const std::uint8_t* qh = block + 66;
        const std::uint8_t* signs = qs + 32;
        const std::uint8_t* scales = block + 74;
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls1 = 1 + 2 * (scales[ib] & 0x0F);
            const int ls2 = 1 + 2 * (scales[ib] >> 4);
            for (int half = 0; half < 2; ++half) {
                int sumi = 0;
                for (int l = 2 * half; l < 2 * half + 2; ++l) {
                    const std::uint32_t index = static_cast<std::uint32_t>(qs[4 * ib + l]) |
                                                ((static_cast<std::uint32_t>(qh[ib]) << (8 - 2 * l)) & 0x300U);
                    const std::uint64_t grid = detail::kIq2sGrid[index];
                    const std::uint8_t sign = signs[4 * ib + l];
                    for (int j = 0; j < 8; ++j) {
                        const int value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                        sumi += value * q[j] * ((sign & (1U << j)) != 0 ? -1 : 1);
                    }
                    q += 8;
                }
                bsum += sumi * (half == 0 ? ls1 : ls2);
            }
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125F * sumf;
#endif
}

[[nodiscard]] inline float dot_iq3_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq3_xxs_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 98;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::uint8_t* qs = block + 2;
        const std::uint8_t* scales_and_signs = qs + 64;
        const std::int8_t* q = q8.qs.data() + b * 256;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* aux = scales_and_signs + 4 * ib;
            const std::uint32_t packed =
                static_cast<std::uint32_t>(aux[0]) | (static_cast<std::uint32_t>(aux[1]) << 8U) |
                (static_cast<std::uint32_t>(aux[2]) << 16U) | (static_cast<std::uint32_t>(aux[3]) << 24U);
            const int ls = 2 * static_cast<int>(packed >> 28U) + 1;
            int sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = detail::kSignsIq2xs[(packed >> (7 * l)) & 127U];
                const std::uint32_t grid1 = detail::kIq3xxsGrid[qs[8 * ib + 2 * l]];
                const std::uint32_t grid2 = detail::kIq3xxsGrid[qs[8 * ib + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    sumi += static_cast<int>((grid1 >> (8 * j)) & 0xFFU) * q[j] * ((signs & (1U << j)) != 0 ? -1 : 1);
                }
                for (int j = 0; j < 4; ++j) {
                    sumi += static_cast<int>((grid2 >> (8 * j)) & 0xFFU) * q[4 + j] *
                            ((signs & (1U << (j + 4))) != 0 ? -1 : 1);
                }
                q += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.25F * sumf;
#endif
}

[[nodiscard]] inline float dot_iq4_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq4_xs_q8_k_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 136;
        const float d4d8 =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        std::uint16_t h = static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U));
        const std::uint8_t* scales_l = block + 4;
        const std::uint8_t* qs = block + 8;
        const std::int8_t* q = q8.qs.data() + b * 256;
        for (int ib = 0; ib < 8; ib += 2) {
            const int ls1 = (scales_l[ib / 2] & 0x0F) | ((h << 4) & 0x30);
            const int ls2 = (scales_l[ib / 2] >> 4) | ((h << 2) & 0x30);
            h = static_cast<std::uint16_t>(h >> 4);
            const float d1 = d4d8 * static_cast<float>(ls1 - 32);
            const float d2 = d4d8 * static_cast<float>(ls2 - 32);
            for (int half = 0; half < 2; ++half) {
                int sumi1 = 0;
                int sumi2 = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi1 += q[j] * kIq4NlValues[qs[j] & 0x0F];
                    sumi2 += q[j + 16] * kIq4NlValues[qs[j] >> 4];
                }
                sumf += (half == 0 ? d1 : d2) * static_cast<float>(sumi1 + sumi2);
                qs += 16;
                q += 32;
            }
        }
    }
    return sumf;
#endif
}

[[nodiscard]] inline float dot_nvfp4_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_nvfp4_q8_0_avx2(data, q8, blocks);
#else
    float sumf = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 36;
        for (int s = 0; s < 4; ++s) {
            const float d = 0.5F * ue4m3_to_float(block[s]);
            const std::uint64_t q8_block = 2 * b + static_cast<std::uint64_t>(s / 2);
            const std::uint64_t q8_off = static_cast<std::uint64_t>((s % 2) * 16);
            const float dy = q8.d[q8_block];
            const std::int8_t* qy = q8.qs.data() + q8_block * 32 + q8_off;
            int sumi_lo = 0;
            int sumi_hi = 0;
            for (int j = 0; j < 8; ++j) {
                const std::uint8_t qv = block[4 + s * 8 + j];
                sumi_lo += qy[j] * kMxfp4Values[qv & 0x0F];
                sumi_hi += qy[j + 8] * kMxfp4Values[qv >> 4];
            }
            sumf += dy * d * static_cast<float>(sumi_lo + sumi_hi);
        }
    }
    return sumf;
#endif
}

inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m);

// ggml_vec_dot_q4_K_q8_K (AVX2): each 256-value block keeps its integer partial
// sums in eight int32 lanes (four values per lane, one scale per 32-value
// group), folds the block into eight float accumulators with a fused
// multiply-add, and rides the quantised minima on a separate four-lane
// accumulator that is reduced first. Both reductions and the final add follow
// hsum_float_8 exactly.
[[nodiscard]] inline float dot_q4_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    float acc[8] = {};
    float acc_m[4] = {};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 144;
        const std::uint8_t* qs = block + 16;
        const std::int8_t* y = q8.qs.data() + b * 256;
        const std::int16_t* bsums = q8.bsums.data() + b * 16;
        const float dy = q8.d[b];
        const float d =
            dy * half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const float dmin =
            -dy * half_to_float(static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U)));
        std::int32_t lane[8] = {};
        std::int32_t sums[8] = {};
        std::int32_t mins[8] = {};
        for (int g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            mins[g] = mn;
            sums[g] = static_cast<std::int32_t>(bsums[2 * g]) + static_cast<std::int32_t>(bsums[2 * g + 1]);
            const std::uint8_t* bytes = qs + 32 * (g / 2);
            for (int l = 0; l < 8; ++l) {
                std::int32_t partial = 0;
                for (int k = 0; k < 4; ++k) {
                    const std::uint8_t byte = bytes[4 * l + k];
                    const int nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
                    partial += nibble * y[g * 32 + 4 * l + k];
                }
                lane[l] += static_cast<std::int32_t>(scale) * partial;
            }
        }
        for (int l = 0; l < 8; ++l) {
            acc[l] = std::fma(d, static_cast<float>(lane[l]), acc[l]);
        }
        for (int i = 0; i < 4; ++i) {
            const std::int32_t prod = mins[2 * i] * sums[2 * i] + mins[2 * i + 1] * sums[2 * i + 1];
            acc_m[i] = std::fma(dmin, static_cast<float>(prod), acc_m[i]);
        }
    }
    const float t0[4] = {acc[0] + acc[4], acc[1] + acc[5], acc[2] + acc[6], acc[3] + acc[7]};
    const float quantised = (t0[0] + t0[1]) + (t0[2] + t0[3]);
    const float minima = (acc_m[0] + acc_m[2]) + (acc_m[1] + acc_m[3]);
    return quantised + minima;
}

// ggml_vec_dot_q5_K_q8_K (AVX2): same eight-lane int32 layout as Q4_K with the
// fifth bit taken from the high-bit plane, but the minima are folded into a
// single float scalar per block instead of a four-lane accumulator.
[[nodiscard]] inline float dot_q5_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    float acc[8] = {};
    float summs = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 176;
        const std::uint8_t* qh = block + 16;
        const std::uint8_t* qs = block + 48;
        const std::int8_t* y = q8.qs.data() + b * 256;
        const std::int16_t* bsums = q8.bsums.data() + b * 16;
        const float dy = q8.d[b];
        const float d =
            dy * half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const float dmin =
            -dy * half_to_float(static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U)));
        std::int32_t lane[8] = {};
        std::int32_t minima[4] = {};
        for (int g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            const std::uint8_t* bytes = qs + 32 * (g / 2);
            for (int l = 0; l < 8; ++l) {
                std::int32_t partial = 0;
                for (int k = 0; k < 4; ++k) {
                    const std::uint8_t byte = bytes[4 * l + k];
                    const int nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
                    const int bit = (qh[4 * l + k] >> g) & 1;
                    partial += (nibble + 16 * bit) * y[g * 32 + 4 * l + k];
                }
                lane[l] += static_cast<std::int32_t>(scale) * partial;
            }
            minima[g / 2] += static_cast<std::int32_t>(mn) *
                             (static_cast<std::int32_t>(bsums[2 * g]) + static_cast<std::int32_t>(bsums[2 * g + 1]));
        }
        for (int l = 0; l < 8; ++l) {
            acc[l] = std::fma(d, static_cast<float>(lane[l]), acc[l]);
        }
        const std::int32_t prod = (minima[0] + minima[1]) + (minima[2] + minima[3]);
        summs = std::fma(dmin, static_cast<float>(prod), summs);
    }
    // the reduced order of this build's hsum_float_8 lowers the pairwise halves
    // to (a0 + a2) + (a1 + a3); Q4_K reduces through GGML_F32_VEC_REDUCE instead
    const float t0[4] = {acc[0] + acc[4], acc[1] + acc[5], acc[2] + acc[6], acc[3] + acc[7]};
    return ((t0[0] + t0[2]) + (t0[1] + t0[3])) + summs;
}

// llama.cpp repacks Q4_K weights (CPU_REPACK, rows divisible by 8) on AVX2 and
// runs ggml_gemv_q4_K_8x8_q8_K: each row keeps one float accumulator for the
// scale part and one for the minima, both fed once per 256-value block, and the
// minima are subtracted after the block loop instead of being reduced
// separately. The integer sums are the full block sums either way.
[[nodiscard]] inline float dot_q4_k_q8_k_8x8(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    float acc = 0.0F;
    float acc_min = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 144;
        const std::int8_t* y = q8.qs.data() + b * 256;
        const std::int16_t* bsums = q8.bsums.data() + b * 16;
        const float dy = q8.d[b];
        const float d =
            dy * half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const float dmin =
            dy * half_to_float(static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U)));
        std::int32_t isum = 0;
        std::int32_t imin = 0;
        for (int g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            const std::uint8_t* bytes = block + 16 + 32 * (g / 2);
            std::int32_t partial = 0;
            for (int l = 0; l < 32; ++l) {
                const std::uint8_t byte = bytes[l];
                const int nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
                partial += nibble * y[g * 32 + l];
            }
            isum += static_cast<std::int32_t>(scale) * partial;
            imin += static_cast<std::int32_t>(mn) *
                    (static_cast<std::int32_t>(bsums[2 * g]) + static_cast<std::int32_t>(bsums[2 * g + 1]));
        }
        acc = std::fma(static_cast<float>(isum), d, acc);
        acc_min = std::fma(static_cast<float>(imin), dmin, acc_min);
    }
    return acc - acc_min;
}

[[nodiscard]] inline bool has_int_q8k_dot(std::uint32_t type) {
    return type == kGgmlQ4_K || type == kGgmlQ5_K || type == kGgmlQ6_K || type == kGgmlIq1S || type == kGgmlIq2Xxs ||
           type == kGgmlIq2Xs || type == kGgmlIq2S || type == kGgmlIq3Xxs || type == kGgmlIq4Xs;
}

[[nodiscard]] inline bool has_int_q80_dot(std::uint32_t type) {
    return type == kGgmlQ2_0 || type == kGgmlQ8_0 || type == kGgmlNvfp4;
}

class Reader {
public:
    explicit Reader(std::string path) : path_(std::move(path)) {
        this->file_.open(this->path_, std::ios::binary);
        if (!this->file_) {
            throw std::runtime_error("cannot open " + this->path_);
        }
        this->parse_header();
    }

    [[nodiscard]] const std::string& path() const { return this->path_; }
    [[nodiscard]] std::uint32_t version() const { return this->version_; }
    [[nodiscard]] std::uint64_t data_start() const { return this->data_start_; }
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return this->tensors_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const { return this->strings_; }
    [[nodiscard]] const std::unordered_map<std::string, std::uint64_t>& integers() const { return this->integers_; }
    [[nodiscard]] const std::unordered_map<std::string, double>& floats() const { return this->floats_; }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::int64_t>>& int_arrays() const {
        return this->int_arrays_;
    }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::string>>& string_arrays() const {
        return this->string_arrays_;
    }

    // Sign vector for a Hadamard-folded weight width, concatenated as stored in
    // `prism.hadamard.sign_widths` / `sign_values`.
    [[nodiscard]] std::vector<std::int32_t> signs_for_width(std::uint32_t width) const {
        const auto widths = this->int_arrays_.find("prism.hadamard.sign_widths");
        const auto values = this->int_arrays_.find("prism.hadamard.sign_values");
        if (widths == this->int_arrays_.end() || values == this->int_arrays_.end()) {
            return {};
        }
        std::size_t offset = 0;
        for (const std::int64_t candidate : widths->second) {
            if (candidate < 0) {
                return {};
            }
            const auto size = static_cast<std::size_t>(candidate);
            if (candidate == static_cast<std::int64_t>(width)) {
                if (offset + size > values->second.size()) {
                    return {};
                }
                std::vector<std::int32_t> signs(size);
                for (std::size_t i = 0; i < size; ++i) {
                    signs[i] = static_cast<std::int32_t>(values->second[offset + i]);
                }
                return signs;
            }
            offset += size;
        }
        return {};
    }

    [[nodiscard]] const TensorInfo& tensor(std::string_view name) const {
        for (const auto& item : this->tensors_) {
            if (item.name == name) {
                return item;
            }
        }
        throw std::runtime_error("tensor not found: " + std::string(name));
    }

    [[nodiscard]] std::vector<std::uint8_t> read_raw(const TensorInfo& info, std::uint64_t byte_offset = 0,
                                                     std::uint64_t byte_count = 0) const {
        const std::uint64_t size = byte_count == 0 ? info.n_bytes() : byte_count;
        if (size == 0) {
            throw std::runtime_error("unsupported or empty tensor: " + info.name);
        }
        std::ifstream file(this->path_, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open " + this->path_);
        }
        file.seekg(static_cast<std::streamoff>(this->data_start_ + info.offset + byte_offset));
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        if (!file) {
            throw std::runtime_error("short read for tensor: " + info.name);
        }
        return bytes;
    }

    // Dequantize one tensor to fp32. `expert` slices a 3-D MoE weight and
    // applies the per-expert `<name>.scale` when present.
    [[nodiscard]] std::vector<float> dequantize(const TensorInfo& info, std::int32_t expert = -1) const {
        const auto type = type_info(info.type);
        if (!type.supported) {
            throw std::runtime_error("unsupported GGML type for tensor: " + info.name);
        }
        std::uint64_t count = info.n_elements;
        std::uint64_t byte_offset = 0;
        std::uint64_t byte_count = 0;
        if (info.dims.size() == 3) {
            if (expert < 0 || static_cast<std::uint64_t>(expert) >= info.dims[2]) {
                throw std::runtime_error("expert index required for 3-D tensor: " + info.name);
            }
            const std::uint64_t per_expert = info.dims[0] * info.dims[1];
            if (per_expert % type.block_elements != 0) {
                throw std::runtime_error("expert slice is not block aligned: " + info.name);
            }
            const std::uint64_t per_expert_bytes = per_expert / type.block_elements * type.block_bytes;
            byte_offset = static_cast<std::uint64_t>(expert) * per_expert_bytes;
            byte_count = per_expert_bytes;
            count = per_expert;
        }
        const std::vector<std::uint8_t> raw = this->read_raw(info, byte_offset, byte_count);
        std::vector<float> values = dequantize_block(info.type, raw, count);
        if (info.dims.size() == 3 && expert >= 0) {
            const std::string base =
                info.name.ends_with(".weight") ? info.name.substr(0, info.name.size() - 7) : info.name;
            const std::string scale_name = base + ".scale";
            for (const auto& candidate : this->tensors_) {
                if (candidate.name == scale_name && candidate.n_elements > static_cast<std::uint64_t>(expert)) {
                    const auto scale_bytes = this->read_raw(candidate);
                    const auto* scales = reinterpret_cast<const float*>(scale_bytes.data());
                    const float factor = scales[expert];
                    for (float& value : values) {
                        value *= factor;
                    }
                    break;
                }
            }
        }
        return values;
    }

    [[nodiscard]] static std::vector<float> dequantize_block(std::uint32_t type, const std::vector<std::uint8_t>& raw,
                                                             std::uint64_t count);
    [[nodiscard]] static float dot_row(std::uint32_t type, const std::uint8_t* data, const float* input,
                                       std::uint64_t columns);

    // A tensor kept in its GGUF block-quantized form. `matvec` dequantizes each
    // block directly into the dot product, so the fp32 copy of the weight is
    // never materialized (a 16x saving for Q2_0 versus fp32).
    struct Quantized {
        std::uint32_t type = kGgmlF32;
        std::uint64_t rows = 0;     // output features
        std::uint64_t columns = 0;  // input features
        float scale = 1.0F;         // per-expert MoE scale, folded at the output
        std::vector<std::uint8_t> blocks;

        [[nodiscard]] bool empty() const { return this->blocks.empty(); }
    };

    [[nodiscard]] Quantized quantized(const TensorInfo& info, std::int32_t expert = -1) const {
        const auto type = type_info(info.type);
        if (!type.supported) {
            throw std::runtime_error("unsupported GGML type for tensor: " + info.name);
        }
        Quantized result;
        result.type = info.type;
        result.rows = info.dims.size() >= 2 ? info.dims[1] : 1;
        result.columns = info.dims[0];
        std::uint64_t byte_offset = 0;
        std::uint64_t byte_count = 0;
        if (info.dims.size() == 3) {
            if (expert < 0 || static_cast<std::uint64_t>(expert) >= info.dims[2]) {
                throw std::runtime_error("expert index required for 3-D tensor: " + info.name);
            }
            const std::uint64_t per_expert = info.dims[0] * info.dims[1];
            if (per_expert % type.block_elements != 0) {
                throw std::runtime_error("expert slice is not block aligned: " + info.name);
            }
            const std::uint64_t per_expert_bytes = per_expert / type.block_elements * type.block_bytes;
            byte_offset = static_cast<std::uint64_t>(expert) * per_expert_bytes;
            byte_count = per_expert_bytes;
            result.rows = info.dims[1];
            const std::string base =
                info.name.ends_with(".weight") ? info.name.substr(0, info.name.size() - 7) : info.name;
            const std::string scale_name = base + ".scale";
            for (const auto& candidate : this->tensors_) {
                if (candidate.name == scale_name && candidate.n_elements > static_cast<std::uint64_t>(expert)) {
                    const auto scale_bytes = this->read_raw(candidate);
                    result.scale = reinterpret_cast<const float*>(scale_bytes.data())[expert];
                    break;
                }
            }
        }
        result.blocks = this->read_raw(info, byte_offset, byte_count);
        return result;
    }

    // y = W x with W kept quantized. Deterministic: each output element is
    // accumulated left to right in double, independent of thread count.
    static void matvec(const Quantized& matrix, const float* input, float* output) {
        const auto info = type_info(matrix.type);
        if (matrix.columns % info.block_elements != 0) {
            throw std::runtime_error("quantized row is not block aligned");
        }
        const std::size_t row_bytes = matrix.columns / info.block_elements * info.block_bytes;
        // Mirror llama.cpp: quantize the matmul input to the weight's
        // vec_dot_type before the dot product.
        Q8KBlocks q8k;
        Q8_0Blocks q80;
        std::vector<float> converted;
        const float* source = input;
        const bool int_q8k = uses_q8_k(matrix.type) && has_int_q8k_dot(matrix.type);
        const bool int_q80 = uses_q8_0(matrix.type) && has_int_q80_dot(matrix.type);
        if (int_q8k) {
            q8k = make_q8_k(input, matrix.columns);
        } else if (int_q80) {
            q80 = make_q8_0_blocks(input, matrix.columns);
        } else if (uses_q8_k(matrix.type)) {
            converted = quantize_q8_k(input, matrix.columns);
            source = converted.data();
        } else if (uses_q8_0(matrix.type)) {
            converted = quantize_q8_0(input, matrix.columns);
            source = converted.data();
        } else if (matrix.type == kGgmlBf16) {
            converted = round_bf16_input(input, matrix.columns);
            source = converted.data();
        }
        const std::uint64_t blocks = matrix.columns / info.block_elements;
        // Rows are independent and keep a fixed accumulation order, so the
        // result is identical for any thread count.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (matrix.rows >= 128)
#endif
        for (std::uint64_t row = 0; row < matrix.rows; ++row) {
            const std::uint8_t* data = matrix.blocks.data() + row * row_bytes;
            float value = 0.0F;
            if (int_q8k) {
                switch (matrix.type) {
                    case kGgmlQ4_K:
                        value = matrix.rows % 8 == 0 ? dot_q4_k_q8_k_8x8(data, q8k, blocks)
                                                     : dot_q4_k_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlQ5_K:
                        value = dot_q5_k_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlQ6_K:
                        value = dot_q6_k_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlIq1S:
                        value = dot_iq1_s_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlIq2Xxs:
                        value = dot_iq2_xxs_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlIq2Xs:
                        value = dot_iq2_xs_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlIq2S:
                        value = dot_iq2_s_q8_k(data, q8k, blocks);
                        break;
                    case kGgmlIq3Xxs:
                        value = dot_iq3_xxs_q8_k(data, q8k, blocks);
                        break;
                    default:
                        value = dot_iq4_xs_q8_k(data, q8k, blocks);
                        break;
                }
            } else if (int_q80) {
                value = matrix.type == kGgmlNvfp4 ? dot_nvfp4_q8_0(data, q80, blocks)
                                                  : (matrix.type == kGgmlQ2_0 ? dot_q2_0_q8_0(data, q80, blocks)
                                                                              : dot_q8_0_q8_0(data, q80, blocks));
            } else {
                value = dot_row(matrix.type, data, source, matrix.columns);
            }
            output[row] = matrix.scale * value;
        }
    }

    static void matmul(const Quantized& matrix, const float* input, float* output, std::size_t tokens,
                       std::size_t input_stride, std::size_t output_stride) {
        for (std::size_t t = 0; t < tokens; ++t) {
            matvec(matrix, input + t * input_stride, output + t * output_stride);
        }
    }

private:
    static std::uint16_t read_u16(std::istream& in) {
        std::uint8_t b[2] = {};
        in.read(reinterpret_cast<char*>(b), 2);
        return static_cast<std::uint16_t>(b[0] | (static_cast<std::uint16_t>(b[1]) << 8));
    }
    static std::uint32_t read_u32(std::istream& in) {
        std::uint8_t b[4] = {};
        in.read(reinterpret_cast<char*>(b), 4);
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
               (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    }
    static std::uint64_t read_u64(std::istream& in) {
        std::uint64_t lo = read_u32(in);
        std::uint64_t hi = read_u32(in);
        return lo | (hi << 32);
    }
    static std::string read_string(std::istream& in) {
        const std::uint64_t length = read_u64(in);
        std::string value(static_cast<std::size_t>(length), '\0');
        in.read(value.data(), static_cast<std::streamsize>(length));
        return value;
    }
    static std::int64_t read_scalar_int(std::istream& in, std::uint32_t type) {
        switch (type) {
            case kTypeUint8:
                return static_cast<std::int64_t>(in.get());
            case kTypeInt8:
                return static_cast<std::int64_t>(static_cast<std::int8_t>(in.get()));
            case kTypeUint16:
                return static_cast<std::int64_t>(read_u16(in));
            case kTypeInt16:
                return static_cast<std::int64_t>(static_cast<std::int16_t>(read_u16(in)));
            case kTypeUint32:
                return static_cast<std::int64_t>(read_u32(in));
            case kTypeInt32:
                return static_cast<std::int64_t>(static_cast<std::int32_t>(read_u32(in)));
            case kTypeUint64:
                return static_cast<std::int64_t>(read_u64(in));
            case kTypeInt64:
                return static_cast<std::int64_t>(read_u64(in));
            case kTypeBool:
                return static_cast<std::int64_t>(in.get());
            default:
                throw std::runtime_error("not an integer GGUF metadata type");
        }
    }

    static void skip_value(std::istream& in, std::uint32_t type) {
        switch (type) {
            case kTypeString:
                static_cast<void>(read_string(in));
                return;
            case kTypeUint8:
            case kTypeInt8:
            case kTypeBool:
                in.seekg(1, std::ios::cur);
                return;
            case kTypeUint16:
            case kTypeInt16:
                in.seekg(2, std::ios::cur);
                return;
            case kTypeUint32:
            case kTypeInt32:
            case kTypeFloat32:
                in.seekg(4, std::ios::cur);
                return;
            case kTypeUint64:
            case kTypeInt64:
            case kTypeFloat64:
                in.seekg(8, std::ios::cur);
                return;
            case kTypeArray: {
                const std::uint32_t element_type = read_u32(in);
                const std::uint64_t count = read_u64(in);
                for (std::uint64_t i = 0; i < count; ++i) {
                    skip_value(in, element_type);
                }
                return;
            }
            default:
                throw std::runtime_error("unknown GGUF metadata type");
        }
    }

    void parse_header() {
        const std::uint32_t magic = read_u32(this->file_);
        if (magic != kGgufMagic) {
            throw std::runtime_error("not a GGUF file");
        }
        this->version_ = read_u32(this->file_);
        const std::uint64_t tensor_count = read_u64(this->file_);
        const std::uint64_t kv_count = read_u64(this->file_);
        if (tensor_count > 1'000'000ULL || kv_count > 1'000'000ULL) {
            throw std::runtime_error("implausible GGUF header counts");
        }

        for (std::uint64_t i = 0; i < kv_count; ++i) {
            const std::string key = read_string(this->file_);
            const std::uint32_t type = read_u32(this->file_);
            if (type == kTypeString) {
                this->strings_[key] = read_string(this->file_);
            } else if (type == kTypeUint32 || type == kTypeInt32) {
                this->integers_[key] = read_u32(this->file_);
            } else if (type == kTypeUint64 || type == kTypeInt64) {
                this->integers_[key] = read_u64(this->file_);
            } else if (type == kTypeBool) {
                this->integers_[key] = static_cast<std::uint64_t>(this->file_.get());
            } else if (type == kTypeFloat32) {
                const std::uint32_t bits = read_u32(this->file_);
                float value = 0.0F;
                std::memcpy(&value, &bits, sizeof(value));
                this->floats_[key] = static_cast<double>(value);
            } else if (type == kTypeArray) {
                const std::uint32_t element_type = read_u32(this->file_);
                const std::uint64_t count = read_u64(this->file_);
                if (element_type == kTypeString) {
                    auto& values = this->string_arrays_[key];
                    values.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t index = 0; index < count; ++index) {
                        values.push_back(read_string(this->file_));
                    }
                } else if (element_type == kTypeUint8 || element_type == kTypeInt8 || element_type == kTypeBool ||
                           element_type == kTypeUint16 || element_type == kTypeInt16 || element_type == kTypeUint32 ||
                           element_type == kTypeInt32 || element_type == kTypeUint64 || element_type == kTypeInt64) {
                    auto& values = this->int_arrays_[key];
                    values.reserve(static_cast<std::size_t>(count));
                    for (std::uint64_t index = 0; index < count; ++index) {
                        values.push_back(read_scalar_int(this->file_, element_type));
                    }
                } else {
                    for (std::uint64_t index = 0; index < count; ++index) {
                        skip_value(this->file_, element_type);
                    }
                }
            } else {
                skip_value(this->file_, type);
            }
        }

        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            TensorInfo info;
            info.name = read_string(this->file_);
            const std::uint32_t n_dims = read_u32(this->file_);
            if (n_dims == 0 || n_dims > 4) {
                throw std::runtime_error("invalid tensor rank: " + info.name);
            }
            info.dims.resize(n_dims);
            info.n_elements = 1;
            for (std::uint32_t d = 0; d < n_dims; ++d) {
                info.dims[d] = read_u64(this->file_);
                info.n_elements *= info.dims[d];
            }
            info.type = read_u32(this->file_);
            info.offset = read_u64(this->file_);
            this->tensors_.push_back(std::move(info));
        }

        const std::streampos header_end = this->file_.tellg();
        const std::uint64_t alignment =
            this->integers_.contains("general.alignment") ? this->integers_.at("general.alignment") : 32ULL;
        const std::uint64_t position = static_cast<std::uint64_t>(header_end);
        this->data_start_ = (position + alignment - 1) / alignment * alignment;
    }

    std::string path_;
    mutable std::ifstream file_;
    std::uint32_t version_ = 0;
    std::uint64_t data_start_ = 0;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, std::uint64_t> integers_;
    std::unordered_map<std::string, double> floats_;
    std::unordered_map<std::string, std::vector<std::int64_t>> int_arrays_;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays_;
};

[[nodiscard]] inline float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x03FFU;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t m = mantissa;
            std::int32_t e = -1;
            while ((m & 0x0400U) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFU;
            result = sign | (static_cast<std::uint32_t>(e + 127 + 2 - 15) << 23) | (m << 13);
        }
    } else if (exponent == 0x1FU) {
        result = sign | 0x7F800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0F;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

inline void dequantize_q6_k(const std::uint8_t* data, float* out) {
    const auto* ql = data;
    const auto* qh = data + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(data + 192);
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[208] | (static_cast<std::uint16_t>(data[209]) << 8U));
    const float d = half_to_float(dBits);
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * static_cast<float>(scales[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        scales += 8;
    }
}

inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    const auto byte = [q](int index) { return static_cast<std::uint32_t>(q[index]); };
    if (j < 4) {
        d = static_cast<std::uint8_t>(byte(j) & 63U);
        m = static_cast<std::uint8_t>(byte(j + 4) & 63U);
    } else {
        d = static_cast<std::uint8_t>((byte(j + 4) & 0x0FU) | ((byte(j - 4) >> 6U) << 4U));
        m = static_cast<std::uint8_t>((byte(j + 4) >> 4U) | ((byte(j) >> 6U) << 4U));
    }
}

inline void dequantize_q4_k(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const std::uint16_t minBits = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const float d = half_to_float(dBits);
    const float min = half_to_float(minBits);
    const std::uint8_t* scales = data + 4;
    const std::uint8_t* q = data + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (int l = 0; l < 32; ++l) {
            *out++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *out++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

inline void dequantize_q5_k(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const std::uint16_t minBits = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const float d = half_to_float(dBits);
    const float min = half_to_float(minBits);
    const std::uint8_t* scales = data + 4;
    const std::uint8_t* qh = data + 16;
    const std::uint8_t* ql = data + 48;
    int is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (int l = 0; l < 32; ++l) {
            *out++ = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *out++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

inline void dequantize_iq2_xxs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* q = data + 2;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t lo =
            static_cast<std::uint32_t>(q[8 * ib32]) | (static_cast<std::uint32_t>(q[8 * ib32 + 1]) << 8U) |
            (static_cast<std::uint32_t>(q[8 * ib32 + 2]) << 16U) | (static_cast<std::uint32_t>(q[8 * ib32 + 3]) << 24U);
        const std::uint32_t hi =
            static_cast<std::uint32_t>(q[8 * ib32 + 4]) | (static_cast<std::uint32_t>(q[8 * ib32 + 5]) << 8U) |
            (static_cast<std::uint32_t>(q[8 * ib32 + 6]) << 16U) | (static_cast<std::uint32_t>(q[8 * ib32 + 7]) << 24U);
        const float db = d * (0.5F + static_cast<float>(hi >> 28U)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t grid = detail::kIq2xxsGrid[(lo >> (8 * l)) & 0xFFU];
            const std::uint8_t signs = detail::kSignsIq2xs[(hi >> (7 * l)) & 127U];
            for (int j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

inline void dequantize_iq2_xs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* scales = data + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F;
        const float db1 = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t qv = static_cast<std::uint16_t>(
                qs[2 * (4 * ib32 + l)] | (static_cast<std::uint16_t>(qs[2 * (4 * ib32 + l) + 1]) << 8U));
            const std::uint64_t grid = detail::kIq2xsGrid[qv & 511U];
            const std::uint8_t signs = detail::kSignsIq2xs[qv >> 9U];
            const float db = l < 2 ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

inline void dequantize_iq3_xxs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* scales_and_signs = qs + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t aux = static_cast<std::uint32_t>(scales_and_signs[4 * ib32]) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 1]) << 8U) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 2]) << 16U) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 3]) << 24U);
        const float db = d * (0.5F + static_cast<float>(aux >> 28U)) * 0.5F;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = detail::kSignsIq2xs[(aux >> (7 * l)) & 127U];
            const std::uint32_t grid1 = detail::kIq3xxsGrid[qs[8 * ib32 + 2 * l]];
            const std::uint32_t grid2 = detail::kIq3xxsGrid[qs[8 * ib32 + 2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
                *out++ =
                    db * static_cast<float>((grid1 >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
            for (int j = 0; j < 4; ++j) {
                *out++ = db * static_cast<float>((grid2 >> (8 * j)) & 0xFFU) *
                         ((signs & (1U << (j + 4))) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

inline void dequantize_iq2_s(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* qh = data + 66;
    const std::uint8_t* scales = data + 74;
    const std::uint8_t* signs = qs + 32;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F;
        const float db1 = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t index = static_cast<std::uint32_t>(qs[4 * ib32 + l]) |
                                        ((static_cast<std::uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300U);
            const std::uint64_t grid = detail::kIq2sGrid[index];
            const std::uint8_t sign = signs[4 * ib32 + l];
            const float db = l < 2 ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((sign & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

inline void dequantize_iq1_s(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const auto* qh = reinterpret_cast<const std::uint16_t*>(data + 34);
    for (int ib = 0; ib < 8; ++ib) {
        const std::uint16_t high = qh[ib];
        const float dl = d * static_cast<float>(2 * ((high >> 12) & 7) + 1);
        const float delta = (high & 0x8000U) != 0 ? -kIq1sDelta : kIq1sDelta;
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t index = static_cast<std::uint16_t>(qs[ib * 4 + l] | (((high >> (3 * l)) & 7) << 8));
            const std::uint64_t grid = detail::kIq1sGrid[index];
            for (int j = 0; j < 8; ++j) {
                const auto value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                out[ib * 32 + l * 8 + j] = dl * (static_cast<float>(value) + delta);
            }
        }
    }
}

inline void dequantize_iq4_xs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint16_t scales_h = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const std::uint8_t* scales_l = data + 4;
    const std::uint8_t* qs = data + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            out[j] = dl * static_cast<float>(kIq4NlValues[qs[j] & 0x0F]);
            out[j + 16] = dl * static_cast<float>(kIq4NlValues[qs[j] >> 4]);
        }
        out += 32;
        qs += 16;
    }
}

inline constexpr float kE2m1Magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

[[nodiscard]] inline float e2m1_to_float(std::uint8_t nibble) {
    const float magnitude = kE2m1Magnitudes[nibble & 0x7U];
    return (nibble & 0x8U) != 0 ? -magnitude : magnitude;
}

[[nodiscard]] inline float ue4m3_to_float(std::uint8_t bits) {
    if (bits == 0 || bits == 0x7F) {
        return 0.0F;
    }
    const std::uint32_t exponent = (bits >> 3) & 0xFU;
    const std::uint32_t mantissa = bits & 0x7U;
    if (exponent == 0) {
        return static_cast<float>(mantissa) * (1.0F / 512.0F);
    }
    return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, static_cast<int>(exponent) - 7);
}

inline float Reader::dot_row(std::uint32_t type, const std::uint8_t* data, const float* input, std::uint64_t columns) {
    switch (type) {
        case kGgmlF32: {
            const auto* values = reinterpret_cast<const float*>(data);
            if (ggml_tinyblas_batch()) {
                return dot_f32_tinyblas(values, input, columns);
            }
#if defined(__AVX2__) && defined(__FMA__)
            return dot_f32_reference_avx2(values, input, columns);
#else
            float sum = 0.0F;
            for (std::uint64_t i = 0; i < columns; ++i) {
                sum += values[i] * input[i];
            }
            return sum;
#endif
        }
        case kGgmlF16: {
            const auto* values = reinterpret_cast<const std::uint16_t*>(data);
            double sum = 0.0;
            for (std::uint64_t i = 0; i < columns; ++i) {
                sum += static_cast<double>(half_to_float(values[i])) * static_cast<double>(input[i]);
            }
            return static_cast<float>(sum);
        }
        case kGgmlBf16: {
            // llama.cpp: the activation is converted to bf16 (vec_dot_type), then
            // ggml_vec_dot_bf16 runs on bf16 x bf16.
            static thread_local std::vector<std::uint16_t> converted;
            converted.resize(columns);
            for (std::uint64_t i = 0; i < columns; ++i) {
                converted[i] = float_to_bf16_bits(input[i]);
            }
            if (ggml_tinyblas_batch()) {
                return dot_bf16_tinyblas(data, converted.data(), columns);
            }
#if defined(__AVX2__) && defined(__FMA__)
            return dot_bf16_bf16_avx2(data, converted.data(), columns);
#else
            double sumf = 0.0;
            for (std::uint64_t i = 0; i < columns; ++i) {
                const float value =
                    bf16_bits_to_float(static_cast<std::uint16_t>(data[i * 2] | (data[i * 2 + 1] << 8U)));
                sumf += static_cast<double>(value * bf16_bits_to_float(converted[i]));
            }
            return static_cast<float>(sumf);
#endif
        }
        case kGgmlQ8_0: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 32;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 34;
                const float scale =
                    half_to_float(static_cast<std::uint16_t>(base[0] | (static_cast<std::uint16_t>(base[1]) << 8U)));
                double block_sum = 0.0;
                for (std::uint64_t j = 0; j < 32; ++j) {
                    block_sum += static_cast<double>(static_cast<std::int8_t>(base[2 + j])) *
                                 static_cast<double>(input[b * 32 + j]);
                }
                sum += block_sum * static_cast<double>(scale);
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ6_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q6_k(data + b * 210, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq4Xs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq4_xs(data + b * 136, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ4_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q4_k(data + b * 144, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ5_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q5_k(data + b * 176, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2Xxs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_xxs(data + b * 66, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2Xs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_xs(data + b * 74, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq3Xxs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq3_xxs(data + b * 98, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2S: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_s(data + b * 82, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq1S: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq1_s(data + b * 50, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ2_0: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 64;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 18;
                const float scale =
                    half_to_float(static_cast<std::uint16_t>(base[0] | (static_cast<std::uint16_t>(base[1]) << 8U)));
                double block_sum = 0.0;
                for (std::uint64_t j = 0; j < 64; ++j) {
                    const std::uint8_t byte = base[2 + j / 4];
                    const auto q = static_cast<int>((byte >> ((j % 4) * 2U)) & 0x03U);
                    block_sum += static_cast<double>(q - 1) * static_cast<double>(input[b * 64 + j]);
                }
                sum += block_sum * static_cast<double>(scale);
            }
            return static_cast<float>(sum);
        }
        case kGgmlNvfp4: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 64;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 36;
                for (std::uint64_t s = 0; s < 4; ++s) {
                    const float scale = ue4m3_to_float(base[s]);
                    if (scale == 0.0F) {
                        continue;
                    }
                    double block_sum = 0.0;
                    const std::uint64_t offset = b * 64 + s * 16;
                    for (std::uint64_t j = 0; j < 8; ++j) {
                        const std::uint8_t byte = base[4 + s * 8 + j];
                        block_sum += static_cast<double>(e2m1_to_float(static_cast<std::uint8_t>(byte & 0x0FU))) *
                                     static_cast<double>(input[offset + j]);
                        block_sum += static_cast<double>(e2m1_to_float(static_cast<std::uint8_t>(byte >> 4U))) *
                                     static_cast<double>(input[offset + j + 8]);
                    }
                    sum += block_sum * static_cast<double>(scale);
                }
            }
            return static_cast<float>(sum);
        }
        default:
            throw std::runtime_error("unsupported GGML type " + std::to_string(type));
    }
}

inline std::vector<float> Reader::dequantize_block(std::uint32_t type, const std::vector<std::uint8_t>& raw,
                                                   std::uint64_t count) {
    std::vector<float> out(static_cast<std::size_t>(count), 0.0F);
    switch (type) {
        case kGgmlF32:
            std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
            return out;
        case kGgmlF16: {
            const auto* data = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = half_to_float(data[i]);
            }
            return out;
        }
        case kGgmlBf16: {
            const auto* data = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < out.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(data[i]) << 16U;
                std::memcpy(&out[i], &bits, sizeof(float));
            }
            return out;
        }
        case kGgmlQ8_0: {
            if (count % 32 != 0) {
                throw std::runtime_error("Q8_0 element count not a multiple of 32");
            }
            const std::size_t blocks = out.size() / 32;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 34;
                const float scale = half_to_float(
                    static_cast<std::uint16_t>(raw[base] | (static_cast<std::uint16_t>(raw[base + 1]) << 8U)));
                for (std::size_t j = 0; j < 32; ++j) {
                    out[b * 32 + j] = static_cast<float>(static_cast<std::int8_t>(raw[base + 2 + j])) * scale;
                }
            }
            return out;
        }
        case kGgmlQ6_K: {
            if (count % 256 != 0) {
                throw std::runtime_error("Q6_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q6_k(raw.data() + b * 210, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq4Xs: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ4_XS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq4_xs(raw.data() + b * 136, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ4_K: {
            if (count % 256 != 0) {
                throw std::runtime_error("Q4_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q4_k(raw.data() + b * 144, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ5_K: {
            if (count % 256 != 0) {
                throw std::runtime_error("Q5_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q5_k(raw.data() + b * 176, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2Xxs: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ2_XXS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_xxs(raw.data() + b * 66, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2Xs: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ2_XS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_xs(raw.data() + b * 74, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq3Xxs: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ3_XXS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq3_xxs(raw.data() + b * 98, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2S: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ2_S element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_s(raw.data() + b * 82, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq1S: {
            if (count % 256 != 0) {
                throw std::runtime_error("IQ1_S element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq1_s(raw.data() + b * 50, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ2_0: {
            if (count % 64 != 0) {
                throw std::runtime_error("Q2_0 element count not a multiple of 64");
            }
            const std::size_t blocks = out.size() / 64;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 18;
                const float scale = half_to_float(
                    static_cast<std::uint16_t>(raw[base] | (static_cast<std::uint16_t>(raw[base + 1]) << 8U)));
                for (std::size_t j = 0; j < 64; ++j) {
                    const std::uint8_t byte = raw[base + 2 + j / 4];
                    const auto q = static_cast<int>((byte >> ((j % 4) * 2U)) & 0x03U);
                    out[b * 64 + j] = static_cast<float>(q - 1) * scale;
                }
            }
            return out;
        }
        case kGgmlNvfp4: {
            if (count % 64 != 0) {
                throw std::runtime_error("NVFP4 element count not a multiple of 64");
            }
            const std::size_t blocks = out.size() / 64;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 36;
                for (std::size_t s = 0; s < 4; ++s) {
                    const float scale = ue4m3_to_float(raw[base + s]);
                    if (scale == 0.0F) {
                        continue;
                    }
                    const std::size_t sub = b * 64 + s * 16;
                    for (std::size_t j = 0; j < 8; ++j) {
                        const std::uint8_t byte = raw[base + 4 + s * 8 + j];
                        out[sub + j] = e2m1_to_float(static_cast<std::uint8_t>(byte & 0x0FU)) * scale;
                        out[sub + j + 8] = e2m1_to_float(static_cast<std::uint8_t>(byte >> 4U)) * scale;
                    }
                }
            }
            return out;
        }
        default:
            throw std::runtime_error("unsupported GGML type " + std::to_string(type));
    }
}

}  // namespace mnet::gguf

#endif  // MNET_GGUF_HPP
