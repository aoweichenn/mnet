// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstring>

#include <vector>

#include <gguf/internal.hpp>
#include <mnet/detail/iq1s_grid.hpp>
#include <mnet/detail/iq_grids.hpp>
#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>

namespace mnet::gguf {

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] static float dot_q6_k_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] static float dot_iq1_s_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] static float dot_iq2_xxs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] static float dot_iq2_xs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] static float dot_iq3_xxs_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
[[nodiscard]] static float dot_iq2_s_q8_k_avx2(const std::uint8_t*, const Q8KBlocks&, std::uint64_t);
#endif

[[nodiscard]] static const std::uint64_t* even_signs_avx2() {
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

// hsum of an 8-lane accumulator: ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)),
// the same reduction the AVX2 kernels use, available on every architecture.
[[nodiscard]] static float hsum_float_8(const float* lanes) {
    return ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
}

[[nodiscard]] float dot_q6_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q6_k_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_q6_k_q8_k_avx2: the same integer pipeline
    // (nibble split, maddubs pair sums, madd_epi16 with the per-16-value scales,
    // the 32 x bsums correction) accumulated into eight int32 lanes and folded
    // with one fma per block into eight float lanes.
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 210U;
        const std::uint8_t* ql = block;
        const std::uint8_t* qh = block + 128U;
        const std::int8_t* q = q8.qs.data() + i * 256U;
        const float d =
            q8.d[i] *
            half_to_float(static_cast<std::uint16_t>(block[208] | (static_cast<std::uint16_t>(block[209]) << 8U)));
        std::int32_t scales[16];
        for (std::size_t k = 0; k < 16; ++k) {
            scales[k] = static_cast<std::int8_t>(block[192U + k]);
        }
        // q8sclsub = 32 * sum over the 16-value groups of (bsum * scale).
        std::uint32_t q8sclsub[8];
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const std::int32_t pair =
                static_cast<std::int32_t>(q8.bsums[i * 16U + lane * 2U]) * scales[lane * 2U] +
                static_cast<std::int32_t>(q8.bsums[i * 16U + lane * 2U + 1U]) * scales[lane * 2U + 1U];
            q8sclsub[lane] = static_cast<std::uint32_t>(pair) << 5U;
        }
        std::uint32_t sumi[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        std::size_t is = 0;
        for (std::int32_t j = 0; j < 2; ++j) {
            const std::uint8_t* ql1 = ql + static_cast<std::size_t>(j) * 64U;
            const std::uint8_t* ql2 = ql1 + 32U;
            const std::uint8_t* qhb = qh + static_cast<std::size_t>(j) * 32U;
            const std::int8_t* q8b = q + static_cast<std::size_t>(j) * 128U;
            for (std::int32_t group = 0; group < 4; ++group) {
                std::int32_t p16[16];
                for (std::size_t idx = 0; idx < 32; ++idx) {
                    std::uint8_t low4;
                    std::uint8_t high2;
                    if (group == 0) {
                        low4 = static_cast<std::uint8_t>(ql1[idx] & 0x0FU);
                        high2 = static_cast<std::uint8_t>(qhb[idx] & 0x03U);
                    } else if (group == 1) {
                        low4 = static_cast<std::uint8_t>(ql2[idx] & 0x0FU);
                        high2 = static_cast<std::uint8_t>((qhb[idx] & 0x0CU) >> 2U);
                    } else if (group == 2) {
                        low4 = static_cast<std::uint8_t>(ql1[idx] >> 4U);
                        high2 = static_cast<std::uint8_t>((qhb[idx] & 0x30U) >> 4U);
                    } else {
                        low4 = static_cast<std::uint8_t>(ql2[idx] >> 4U);
                        high2 = static_cast<std::uint8_t>((qhb[idx] & 0xC0U) >> 6U);
                    }
                    const std::int32_t value =
                        static_cast<std::int32_t>(low4) | (static_cast<std::int32_t>(high2) << 4);
                    const std::int32_t product =
                        value * static_cast<std::int32_t>(q8b[static_cast<std::size_t>(group) * 32U + idx]);
                    if ((idx & 1U) == 0U) {
                        p16[idx / 2U] = product;
                    } else {
                        p16[idx / 2U] += product;
                    }
                }
                const std::int32_t scale0 = scales[2U * (is + static_cast<std::size_t>(group))];
                const std::int32_t scale1 = scales[2U * (is + static_cast<std::size_t>(group)) + 1U];
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const std::int32_t scale = lane < 4 ? scale0 : scale1;
                    const std::int32_t value = scale * (p16[lane * 2U] + p16[lane * 2U + 1U]);
                    sumi[lane] += static_cast<std::uint32_t>(value);
                }
            }
            is += 4;
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const std::int32_t value = static_cast<std::int32_t>(sumi[lane] - q8sclsub[lane]);
            acc[lane] = std::fma(d, static_cast<float>(value), acc[lane]);
        }
    }
    return hsum_float_8(acc);
#endif
}

[[nodiscard]] float dot_iq1_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq1_s_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq1_s_q8_k_avx2: four 11-bit grid rows per
    // 32-value group (8-bit signed codebook, mul_add_epi8 style products), one
    // int32 lane per four values accumulated across the group, the (2*ls+1)
    // scale per group, one fma per block into eight float lanes, and the
    // separate bsums correction accumulated in float and scaled by 1/8.
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    float accum1 = 0.0F;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 50U;
        const float d =
            q8.d[b] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* qs = block + 2U;
        const auto qh = reinterpret_cast<const std::uint16_t*>(block + 34);
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::int32_t lane_total[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        std::int32_t bsum = 0;
        for (std::size_t g = 0; g < 8; ++g) {
            const std::int32_t ls = 2 * static_cast<std::int32_t>((qh[g] >> 12U) & 7U) + 1;
            const std::int32_t delta = (qh[g] & 0x8000U) != 0U ? -1 : 1;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint32_t index = static_cast<std::uint32_t>(qs[4U * g + l]) |
                                            (((static_cast<std::uint32_t>(qh[g]) >> (3U * l)) & 7U) << 8U);
                const std::uint64_t grid = detail::kIq1sGrid[index];
                for (std::size_t j = 0; j < 8; ++j) {
                    const auto code = static_cast<std::int8_t>((grid >> (8U * j)) & 0xFFU);
                    lane_total[2U * l + j / 4U] +=
                        ls * static_cast<std::int32_t>(code) * static_cast<std::int32_t>(q[32U * g + 8U * l + j]);
                }
            }
            bsum += ls * delta *
                    (static_cast<std::int32_t>(q8.bsums[b * 16U + 2U * g]) +
                     static_cast<std::int32_t>(q8.bsums[b * 16U + 2U * g + 1U]));
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            accum[lane] = std::fma(d, static_cast<float>(lane_total[lane]), accum[lane]);
        }
        // The AVX2 kernel lets the compiler contract d*bsum + accum1 and
        // hsum + 0.125*accum1 into fmadd; spell both out so every target
        // rounds the same way.
        accum1 = std::fma(d, static_cast<float>(bsum), accum1);
    }
    return std::fma(kIq1sDelta, accum1, hsum_float_8(accum));
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] static float hsum_float_8_avx2(__m256 value) {
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

[[nodiscard]] float dot_f32_tinyblas(const float* a, const float* b, std::uint64_t size) {
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
[[nodiscard]] float dot_bf16_tinyblas(const std::uint8_t* data, const std::uint16_t* input, std::uint64_t size) {
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::uint64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        for (std::uint64_t j = 0; j < 8; ++j) {
            const std::uint16_t bits = static_cast<std::uint16_t>(data[(i + j) * 2] | (data[(i + j) * 2 + 1] << 8U));
            acc[j] = std::fma(bf16_bits_to_float(bits), bf16_bits_to_float(input[i + j]), acc[j]);
        }
    }
    for (std::uint64_t j = 0; i + j < size; ++j) {
        const std::uint16_t bits = static_cast<std::uint16_t>(data[(i + j) * 2] | (data[(i + j) * 2 + 1] << 8U));
        acc[j] = std::fma(bf16_bits_to_float(bits), bf16_bits_to_float(input[i + j]), acc[j]);
    }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

// Portable lane-faithful replicas of the AVX2 row dots used by dot_row, so the
// non-AVX2 builds reproduce ggml_vec_dot_f32 / ggml_vec_dot_bf16 exactly
// (four 8-lane accumulators, (s0+s2)+(s1+s3), hsum, and the same tails).
[[nodiscard]] float dot_f32_reference(const float* x, const float* y, std::uint64_t size) {
    const std::uint64_t aligned = size & ~static_cast<std::uint64_t>(31);
    float sums[4][8] = {};
    for (std::uint64_t i = 0; i < aligned; i += 32) {
        for (std::uint64_t j = 0; j < 4; ++j) {
            for (std::uint64_t l = 0; l < 8; ++l) {
                sums[j][l] = std::fma(x[i + j * 8 + l], y[i + j * 8 + l], sums[j][l]);
            }
        }
    }
    // _mm_hadd_ps twice: ((l0+l4)+(l1+l5)) + ((l2+l6)+(l3+l7)), which is a
    // different tree from hsum_float_8_avx2 used by the other quantised dots.
    float lanes[8];
    for (std::uint64_t l = 0; l < 8; ++l) {
        lanes[l] = (sums[0][l] + sums[2][l]) + (sums[1][l] + sums[3][l]);
    }
    float total = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    const std::uint64_t tail = size - aligned;
    const std::uint64_t rounded = tail - tail % 4;
    std::uint64_t i = aligned;
    for (std::uint64_t k = 0; k < rounded; ++k, ++i) {
        volatile float product = x[i] * y[i];
        total += product;
    }
    for (; i < size; ++i) {
        total = std::fma(x[i], y[i], total);
    }
    return total;
}

// ggml_vec_dot_bf16: four 8-lane mul/add accumulators (no fma), combined
// (c1+c3)+(c2+c4), hsum, then a double accumulator for the leftovers.
[[nodiscard]] float dot_bf16_bf16_reference(const std::uint8_t* data, const std::uint16_t* input,
                                            std::uint64_t columns) {
    const auto load8 = [](const std::uint8_t* bytes, float* out) {
        for (std::size_t l = 0; l < 8; ++l) {
            const std::uint16_t bits = static_cast<std::uint16_t>(bytes[l * 2] | (bytes[l * 2 + 1] << 8U));
            out[l] = bf16_bits_to_float(bits);
        }
    };
    float acc[4][8] = {};
    std::uint64_t i = 0;
    const auto* input_bytes = reinterpret_cast<const std::uint8_t*>(input);
    for (; i + 32 <= columns; i += 32) {
        for (std::uint64_t j = 0; j < 4; ++j) {
            float w[8];
            float a[8];
            load8(data + (i + j * 8) * 2, w);
            load8(input_bytes + (i + j * 8) * 2, a);
            for (std::uint64_t l = 0; l < 8; ++l) {
                acc[j][l] = acc[j][l] + w[l] * a[l];
            }
        }
    }
    float lanes[8];
    for (std::uint64_t l = 0; l < 8; ++l) {
        lanes[l] = (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]);
    }
    double sumf = static_cast<double>(hsum_float_8(lanes));
    for (; i < columns; ++i) {
        const float weight = bf16_bits_to_float(static_cast<std::uint16_t>(data[i * 2] | (data[i * 2 + 1] << 8U)));
        sumf += static_cast<double>(weight * bf16_bits_to_float(input[i]));
    }
    return static_cast<float>(sumf);
}

#if defined(__AVX2__) && defined(__FMA__)
[[nodiscard]] float dot_f32_reference_avx2(const float* x, const float* y, std::uint64_t size) {
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

[[nodiscard]] static __m256 mul_sum_i8_pairs_float_avx2(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
}

[[nodiscard]] static __m256i q2_0_unpack_32_avx2(const std::uint8_t* qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs)));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

[[nodiscard]] static __m256 q2_0_dot_32_avx2(const std::uint8_t* qs, const std::int8_t* qy) {
    const __m256i v = q2_0_unpack_32_avx2(qs);
    const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qy));
    const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(v, y), _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

[[nodiscard]] static float dot_q8_0_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
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

[[nodiscard]] static __m256 q2_0_acc_block_avx2(const std::uint8_t* block, const float* dy, const std::int8_t* qy,
                                                __m256 acc) {
    const float dx = half_to_float(static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8U));
    const __m256 d0 = _mm256_set1_ps(dx * dy[0]);
    const __m256 d1 = _mm256_set1_ps(dx * dy[1]);
    acc = _mm256_fmadd_ps(d0, q2_0_dot_32_avx2(block + 2, qy), acc);
    return _mm256_fmadd_ps(d1, q2_0_dot_32_avx2(block + 2 + 8, qy + 32), acc);
}

[[nodiscard]] static __m256 load_bf16_8_avx2(const std::uint8_t* data) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

// ggml_vec_dot_bf16: bf16 weight x bf16 activation, four 8-lane mul/add
// accumulators and one double accumulation per 32 elements.
[[nodiscard]] float dot_bf16_bf16_avx2(const std::uint8_t* data, const std::uint16_t* input, std::uint64_t columns) {
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

[[nodiscard]] static float dot_q2_0_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
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
[[nodiscard]] static __m256i mul_add_epi8_avx2(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

[[nodiscard]] static float dot_iq4_xs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
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
            const std::int16_t ls1 = static_cast<std::int16_t>(((block[4 + ib / 2] & 0x0F) | ((sh << 4) & 0x30)) - 32);
            const std::int16_t ls2 = static_cast<std::int16_t>(((block[4 + ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
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

[[nodiscard]] static float dot_nvfp4_q8_0_avx2(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
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

[[nodiscard]] static __m128i iq_scale_shuffle_avx2(std::int32_t index) {
    alignas(16) static constexpr std::uint8_t kShuffle[128] = {
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
        3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
        6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(kShuffle) + index);
}

[[nodiscard]] static float dot_q6_k_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        std::int32_t is = 0;
        for (std::int32_t j = 0; j < 2; ++j) {
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

[[nodiscard]] static float dot_iq1_s_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0.0F;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data + i * 50;
        const std::int8_t* q = q8.qs.data() + i * 256;
        const std::uint8_t* qs = block + 2;
        const auto* qh = reinterpret_cast<const std::uint16_t*>(block + 34);
        __m256i sumi = _mm256_setzero_si256();
        std::int32_t sumi1 = 0;
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const __m256i q1b_1 =
                _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq1sGrid[qs[3] | ((qh[ib] >> 1) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[2] | ((qh[ib] << 2) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[1] | ((qh[ib] << 5) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[0] | ((qh[ib] << 8) & 0x700)]));
            const __m256i q1b_2 =
                _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq1sGrid[qs[7] | ((qh[ib + 1] >> 1) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[6] | ((qh[ib + 1] << 2) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[5] | ((qh[ib + 1] << 5) & 0x700)]),
                                  static_cast<std::int64_t>(detail::kIq1sGrid[qs[4] | ((qh[ib + 1] << 8) & 0x700)]));
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
        // Fused explicitly so the result does not depend on whether the
        // translation unit is compiled with -ffp-contract=fast; the scalar
        // body spells out the same two fmadds.
        accum1 = std::fma(d, static_cast<float>(sumi1), accum1);
    }
    return std::fma(0.125F, accum1, hsum_float_8_avx2(accum));
}

[[nodiscard]] static float dot_iq2_xxs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            std::uint32_t aux32[4] = {};
            std::memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
            const __m256i grid_1 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[3]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[2]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[1]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[0]]));
            const __m256i grid_2 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[11]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[10]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[9]]),
                                                     static_cast<std::int64_t>(detail::kIq2xxsGrid[aux8[8]]));
            const __m256i s2_1 = _mm256_set_epi64x(static_cast<std::int64_t>(signs64[(aux32[1] >> 21) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[1] >> 14) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[1] >> 7) & 127U]),
                                                   static_cast<std::int64_t>(signs64[aux32[1] & 127U]));
            const __m256i s2_2 = _mm256_set_epi64x(static_cast<std::int64_t>(signs64[(aux32[3] >> 21) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[3] >> 14) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[3] >> 7) & 127U]),
                                                   static_cast<std::int64_t>(signs64[aux32[3] & 127U]));
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

[[nodiscard]] static float dot_iq2_xs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        __m128i stmp = _mm_set1_epi64x(static_cast<std::int64_t>(aux64));
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (std::int32_t ib = 0; ib < 8; ib += 4) {
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
            const __m256i grid_1 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[3]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[2]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[1]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[0]]));
            const __m256i grid_2 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[7]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[6]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[5]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[4]]));
            const __m256i grid_3 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[11]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[10]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[9]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[8]]));
            const __m256i grid_4 = _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[15]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[14]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[13]]),
                                                     static_cast<std::int64_t>(detail::kIq2xsGrid[gindex[12]]));
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

[[nodiscard]] static float dot_iq3_xxs_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i grid_1 = _mm256_set_epi32(static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[7]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[6]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[5]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[4]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[3]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[2]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[1]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[0]]));
            q3 += 8;
            const __m256i grid_2 = _mm256_set_epi32(static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[7]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[6]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[5]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[4]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[3]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[2]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[1]]),
                                                    static_cast<std::int32_t>(detail::kIq3xxsGrid[q3[0]]));
            q3 += 8;
            std::uint32_t aux32[2] = {};
            std::memcpy(aux32, gas, sizeof(aux32));
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(static_cast<std::int64_t>(signs64[(aux32[0] >> 21) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[0] >> 14) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[0] >> 7) & 127U]),
                                                   static_cast<std::int64_t>(signs64[aux32[0] & 127U]));
            const __m256i s2_2 = _mm256_set_epi64x(static_cast<std::int64_t>(signs64[(aux32[1] >> 21) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[1] >> 14) & 127U]),
                                                   static_cast<std::int64_t>(signs64[(aux32[1] >> 7) & 127U]),
                                                   static_cast<std::int64_t>(signs64[aux32[1] & 127U]));
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

[[nodiscard]] static const std::uint8_t* iq_scale_shuffle_k4_avx2() {
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

[[nodiscard]] static float dot_iq2_s_q8_k_avx2(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
                _mm_and_si128(_mm_set_epi64x(static_cast<std::int64_t>(aux64 >> 4), static_cast<std::int64_t>(aux64)),
                              m4),
                1),
            m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            q += 32;
            const __m256i grid_1 =
                _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2sGrid[qs[3] | ((qh[ib] << 2) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[2] | ((qh[ib] << 4) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[1] | ((qh[ib] << 6) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[0] | ((qh[ib] << 8) & 0x300)]));
            const __m256i grid_2 =
                _mm256_set_epi64x(static_cast<std::int64_t>(detail::kIq2sGrid[qs[7] | ((qh[ib + 1] << 2) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[6] | ((qh[ib + 1] << 4) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[5] | ((qh[ib + 1] << 6) & 0x300)]),
                                  static_cast<std::int64_t>(detail::kIq2sGrid[qs[4] | ((qh[ib + 1] << 8) & 0x300)]));
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

[[nodiscard]] float dot_q2_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q2_0_q8_0_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of q2_0_acc_block_avx2 / dot_q2_0_q8_0_avx2: eight
    // byte-group lanes, one fma per Q8_0 sub-block, two accumulators alternating
    // blocks, then the AVX2 hsum. Integer products are exact, so the only
    // rounding left is the same float sequence the vector kernel uses.
    const auto accumulate = [&](float (&acc)[8], const std::uint8_t* block, std::uint64_t sub) {
        const float dx =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        for (std::uint64_t k = 0; k < 2; ++k) {
            const float d = dx * q8.d[sub + k];
            const std::int8_t* qy = q8.qs.data() + (sub + k) * 32U;
            const std::uint8_t* q = block + 2 + k * 8U;
            for (std::size_t j = 0; j < 8; ++j) {
                const std::uint8_t byte = q[j];
                const std::int32_t lane =
                    (((byte >> 0) & 3) - 1) * qy[j * 4U + 0U] + (((byte >> 2) & 3) - 1) * qy[j * 4U + 1U] +
                    (((byte >> 4) & 3) - 1) * qy[j * 4U + 2U] + (((byte >> 6) & 3) - 1) * qy[j * 4U + 3U];
                acc[j] = std::fma(d, static_cast<float>(lane), acc[j]);
            }
        }
    };
    float acc0[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    float acc1[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::uint64_t i = 0;
    for (; i + 1 < blocks; i += 2) {
        accumulate(acc0, data + i * 18U, 2U * i);
        accumulate(acc1, data + (i + 1U) * 18U, 2U * i + 2U);
    }
    if (i < blocks) {
        accumulate(acc0, data + i * 18U, 2U * i);
    }
    for (std::size_t j = 0; j < 8; ++j) {
        acc0[j] += acc1[j];
    }
    return hsum_float_8(acc0);
#endif
}

[[nodiscard]] float dot_q8_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_q8_0_q8_0_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_q8_0_q8_0_avx2: one eight-lane accumulator,
    // lane j = sum of the four int8 products at 4j..4j+3, one fma per block.
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 34U;
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        const std::int8_t* qy = q8.qs.data() + b * 32U;
        for (std::size_t j = 0; j < 8; ++j) {
            // maddubs saturates each int16 pair before madd_epi16 adds the two
            // pairs; real Q8_0 codes stay in [-127, 127] so this only matters
            // for synthetic data containing -128.
            const auto pair = [&](std::size_t k0) {
                const std::int32_t sum =
                    static_cast<std::int32_t>(static_cast<std::int8_t>(block[2U + j * 4U + k0])) * qy[j * 4U + k0] +
                    static_cast<std::int32_t>(static_cast<std::int8_t>(block[2U + j * 4U + k0 + 1U])) *
                        qy[j * 4U + k0 + 1U];
                return std::min(32767, std::max(-32768, sum));
            };
            const std::int32_t lane = pair(0) + pair(2);
            acc[j] = std::fma(d, static_cast<float>(lane), acc[j]);
        }
    }
    return hsum_float_8(acc);
#endif
}

[[nodiscard]] float dot_iq2_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_xxs_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq2_xxs_q8_k_avx2: grid rows and sign masks
    // are expanded into 32 bytes, the Q8_K values take the sign mask, maddubs
    // pairs them against the unsigned grid, the per-group scale (2 * ls + 1)
    // folds each 4-value group into one int32 lane, and the blocks accumulate
    // with one fma into eight float lanes (0.125 scaling at the end).
    const std::uint64_t* signs64 = even_signs_avx2();
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 66U;
        const float d =
            q8.d[b] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const auto* q2 = reinterpret_cast<const std::uint16_t*>(block + 2);
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::uint32_t sumi1[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        std::uint32_t sumi2[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const std::int8_t* q8_1 = q;
            q += 32;
            const std::int8_t* q8_2 = q;
            q += 32;
            std::uint32_t aux32[4] = {};
            std::memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
            std::uint8_t grid1[32];
            std::uint8_t grid2[32];
            for (std::size_t k = 0; k < 4; ++k) {
                std::memcpy(grid1 + k * 8U, &detail::kIq2xxsGrid[aux8[k]], 8);
                std::memcpy(grid2 + k * 8U, &detail::kIq2xxsGrid[aux8[8U + k]], 8);
            }
            const std::uint64_t s1[4] = {signs64[aux32[1] & 127U], signs64[(aux32[1] >> 7U) & 127U],
                                         signs64[(aux32[1] >> 14U) & 127U], signs64[(aux32[1] >> 21U) & 127U]};
            const std::uint64_t s2[4] = {signs64[aux32[3] & 127U], signs64[(aux32[3] >> 7U) & 127U],
                                         signs64[(aux32[3] >> 14U) & 127U], signs64[(aux32[3] >> 21U) & 127U]};
            const std::int32_t fs1 = static_cast<std::int32_t>(2U * (aux32[1] >> 28U) + 1U);
            const std::int32_t fs2 = static_cast<std::int32_t>(2U * (aux32[3] >> 28U) + 1U);
            for (std::size_t lane = 0; lane < 8; ++lane) {
                std::int32_t p1 = 0;
                std::int32_t p2 = 0;
                for (std::size_t j = 0; j < 4; ++j) {
                    const std::size_t at = lane * 4U + j;
                    const auto sign1 = static_cast<std::int8_t>((s1[at / 8U] >> (8U * (at % 8U))) & 0xFFU);
                    const auto sign2 = static_cast<std::int8_t>((s2[at / 8U] >> (8U * (at % 8U))) & 0xFFU);
                    const std::int32_t v1 =
                        sign1 < 0 ? -static_cast<std::int32_t>(q8_1[at]) : static_cast<std::int32_t>(q8_1[at]);
                    const std::int32_t v2 =
                        sign2 < 0 ? -static_cast<std::int32_t>(q8_2[at]) : static_cast<std::int32_t>(q8_2[at]);
                    p1 += static_cast<std::int32_t>(grid1[at]) * v1;
                    p2 += static_cast<std::int32_t>(grid2[at]) * v2;
                }
                sumi1[lane] += static_cast<std::uint32_t>(fs1 * p1);
                sumi2[lane] += static_cast<std::uint32_t>(fs2 * p2);
            }
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const auto total = static_cast<std::int32_t>(sumi1[lane] + sumi2[lane]);
            accum[lane] = std::fma(d, static_cast<float>(total), accum[lane]);
        }
    }
    return 0.125F * hsum_float_8(accum);
#endif
}

[[nodiscard]] float dot_iq2_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_xs_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq2_xs_q8_k_avx2: two batches of sixteen
    // 10-bit grid codes, sign bits assembled from the high bits (with the pshufb
    // odd-bit expansion), four 32-value groups with their own grid rows and
    // 8-byte sign selectors, maddubs pair sums, per-group (2*ls+1) scales and one
    // fma per block into eight float lanes (0.125 scaling at the end).
    constexpr std::uint8_t kBitHelper[16] = {0, 128, 128, 0, 128, 0, 0, 128, 128, 0, 0, 128, 0, 128, 128, 0};
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 74U;
        const float d =
            q8.d[b] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const auto* q2 = reinterpret_cast<const std::uint16_t*>(block + 2);
        const std::uint8_t* sc_bytes = block + 66U;
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::int32_t scales[16];
        for (std::size_t j = 0; j < 8; ++j) {
            scales[2U * j] = 2 * static_cast<std::int32_t>(sc_bytes[j] & 0x0FU) + 1;
            scales[2U * j + 1U] = 2 * (sc_bytes[j] >> 4U) + 1;
        }
        std::uint32_t sumi1[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        std::uint32_t sumi2[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        for (std::int32_t ib = 0; ib < 8; ib += 4) {
            std::uint16_t codes[16];
            std::memcpy(codes, q2, sizeof(codes));
            q2 += 16;
            std::uint8_t full_signs[32];
            for (std::size_t i = 0; i < 16; ++i) {
                const std::uint16_t partial = static_cast<std::uint16_t>(codes[i] >> 9U);
                const std::uint16_t upper = static_cast<std::uint16_t>(codes[i] >> 13U);
                const std::uint16_t counting = static_cast<std::uint16_t>(partial ^ upper);
                const std::uint8_t count_lo = static_cast<std::uint8_t>(counting & 0xFFU);
                const std::uint8_t count_hi = static_cast<std::uint8_t>(counting >> 8U);
                const std::uint8_t odd_lo = (count_lo & 0x80U) != 0 ? 0U : kBitHelper[count_lo & 0x0FU];
                const std::uint8_t odd_hi = (count_hi & 0x80U) != 0 ? 0U : kBitHelper[count_hi & 0x0FU];
                // OR the odd bits into the partial sign bytes (not into counting).
                full_signs[2U * i] = static_cast<std::uint8_t>(static_cast<std::uint8_t>(partial & 0xFFU) | odd_lo);
                full_signs[2U * i + 1U] = static_cast<std::uint8_t>(static_cast<std::uint8_t>(partial >> 8U) | odd_hi);
            }
            for (std::int32_t group = 0; group < 4; ++group) {
                std::uint8_t grid[32];
                for (std::size_t k = 0; k < 4; ++k) {
                    const std::uint16_t index =
                        static_cast<std::uint16_t>(codes[4U * static_cast<std::size_t>(group) + k] & 511U);
                    std::memcpy(grid + k * 8U, &detail::kIq2xsGrid[index], 8);
                }
                const std::int8_t* q8g = q;
                q += 32;
                std::int32_t dot[16];
                for (std::size_t pair = 0; pair < 16; ++pair) {
                    std::int32_t sum = 0;
                    for (std::size_t e = 0; e < 2; ++e) {
                        const std::size_t at = pair * 2U + e;
                        const std::size_t sign_index = static_cast<std::size_t>(group) * 8U + 2U * ((at >> 3U) & 3U);
                        const bool negative = (full_signs[sign_index] & (1U << (at & 7U))) != 0;
                        const std::int32_t value =
                            negative ? -static_cast<std::int32_t>(q8g[at]) : static_cast<std::int32_t>(q8g[at]);
                        sum += static_cast<std::int32_t>(grid[at]) * value;
                    }
                    dot[pair] = static_cast<std::int32_t>(std::int16_t(std::min(32767, std::max(-32768, sum))));
                }
                const std::int32_t scale_lo =
                    scales[2U * (static_cast<std::size_t>(ib) + static_cast<std::size_t>(group))];
                const std::int32_t scale_hi =
                    scales[2U * (static_cast<std::size_t>(ib) + static_cast<std::size_t>(group)) + 1U];
                std::uint32_t* dst = (group % 2 == 0) ? sumi1 : sumi2;
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const std::int32_t scale = lane < 4 ? scale_lo : scale_hi;
                    const std::int32_t value = scale * (dot[lane * 2U] + dot[lane * 2U + 1U]);
                    dst[lane] += static_cast<std::uint32_t>(value);
                }
            }
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const auto total = static_cast<std::int32_t>(sumi1[lane] + sumi2[lane]);
            accum[lane] = std::fma(d, static_cast<float>(total), accum[lane]);
        }
    }
    return 0.125F * hsum_float_8(accum);
#endif
}

[[nodiscard]] float dot_iq2_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq2_s_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq2_s_q8_k_avx2: each 32-value group takes
    // four 10-bit grid rows, one sign byte per row, maddubs pair sums and one
    // int32 lane per four values. pshufb works per 128-bit lane, so the low
    // nibble of the scale byte covers values 0..15 (rows 0,1) and the high
    // nibble covers values 16..31 (rows 2,3). Blocks fold into eight float
    // lanes with one fma and the 0.125 factor.
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 82U;
        const float d =
            q8.d[b] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* qs = block + 2U;
        const std::uint8_t* signs = block + 34U;
        const std::uint8_t* qh = block + 66U;
        const std::uint8_t* scales = block + 74U;
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::int32_t lane_total[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (std::size_t g = 0; g < 8; ++g) {
            const std::int32_t scale_low = 1 + 2 * static_cast<std::int32_t>(scales[g] & 0x0FU);
            const std::int32_t scale_high = 1 + 2 * (scales[g] >> 4U);
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint32_t index = static_cast<std::uint32_t>(qs[4U * g + l]) |
                                            ((static_cast<std::uint32_t>(qh[g]) << (8U - 2U * l)) & 0x300U);
                const std::uint64_t grid = detail::kIq2sGrid[index];
                const std::uint8_t sign = signs[4U * g + l];
                const std::int32_t scale = l < 2 ? scale_low : scale_high;
                for (std::size_t j = 0; j < 8; ++j) {
                    const std::int32_t value = static_cast<std::int32_t>((grid >> (8U * j)) & 0xFFU);
                    const std::int32_t qv = (sign & (1U << j)) != 0U
                                                ? -static_cast<std::int32_t>(q[32U * g + 8U * l + j])
                                                : static_cast<std::int32_t>(q[32U * g + 8U * l + j]);
                    lane_total[2U * l + j / 4U] += scale * value * qv;
                }
            }
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            accum[lane] = std::fma(d, static_cast<float>(lane_total[lane]), accum[lane]);
        }
    }
    return 0.125F * hsum_float_8(accum);
#endif
}

[[nodiscard]] float dot_iq3_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq3_xxs_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq3_xxs_q8_k_avx2: four passes of two
    // 32-value groups, grid rows expanded to four bytes per code, Q8_K values
    // signed with the even_signs table, maddubs pair sums, one int32 lane per
    // grid row (four values), the (2*ls+1) scale per group, one fma per block
    // into eight float lanes and the 0.25 factor.
    const std::uint64_t* signs64 = even_signs_avx2();
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 98U;
        const float d =
            q8.d[b] *
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U)));
        const std::uint8_t* q3 = block + 2U;
        const std::uint8_t* gas = block + 66U;
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::int32_t sumi1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        std::int32_t sumi2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (std::int32_t it = 0; it < 4; ++it) {
            const std::int8_t* q8_1 = q;
            q += 32;
            const std::int8_t* q8_2 = q;
            q += 32;
            std::uint32_t aux32[2] = {};
            std::memcpy(aux32, gas, sizeof(aux32));
            gas += 8;
            const std::int32_t ls1 = 2 * static_cast<std::int32_t>(aux32[0] >> 28U) + 1;
            const std::int32_t ls2 = 2 * static_cast<std::int32_t>(aux32[1] >> 28U) + 1;
            for (std::size_t k = 0; k < 8; ++k) {
                const std::uint32_t g1 = detail::kIq3xxsGrid[q3[k]];
                const std::uint32_t g2 = detail::kIq3xxsGrid[q3[8U + k]];
                // Eight sign bytes cover the 32 values: grid row k (four
                // values) uses bytes 4*(k%2)..4*(k%2)+3 of the k/2-th chunk of
                // the sign word (0xFF negates, 0x01 keeps).
                const std::size_t code_index = k / 2U;
                const std::size_t byte_base = 4U * (k % 2U);
                const std::uint64_t w1 = signs64[(aux32[0] >> (7U * code_index)) & 127U];
                const std::uint64_t w2 = signs64[(aux32[1] >> (7U * code_index)) & 127U];
                std::int32_t p1 = 0;
                std::int32_t p2 = 0;
                for (std::size_t j = 0; j < 4; ++j) {
                    const std::int32_t v1 = static_cast<std::int32_t>((g1 >> (8U * j)) & 0xFFU);
                    const std::int32_t v2 = static_cast<std::int32_t>((g2 >> (8U * j)) & 0xFFU);
                    const std::int32_t value = static_cast<std::int32_t>(4U * k + j);
                    const auto m1 = static_cast<std::uint8_t>((w1 >> (8U * (byte_base + j))) & 0xFFU);
                    const auto m2 = static_cast<std::uint8_t>((w2 >> (8U * (byte_base + j))) & 0xFFU);
                    const std::int32_t q1 = (m1 & 0x80U) != 0U ? -static_cast<std::int32_t>(q8_1[value])
                                                               : static_cast<std::int32_t>(q8_1[value]);
                    const std::int32_t q2 = (m2 & 0x80U) != 0U ? -static_cast<std::int32_t>(q8_2[value])
                                                               : static_cast<std::int32_t>(q8_2[value]);
                    p1 += v1 * q1;
                    p2 += v2 * q2;
                }
                sumi1[k] += ls1 * p1;
                sumi2[k] += ls2 * p2;
            }
            q3 += 16;
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            accum[lane] = std::fma(d, static_cast<float>(sumi1[lane] + sumi2[lane]), accum[lane]);
        }
    }
    return 0.25F * hsum_float_8(accum);
#endif
}

[[nodiscard]] float dot_iq4_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_iq4_xs_q8_k_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_iq4_xs_q8_k_avx2: per block, four passes of
    // two 32-value groups, 4-bit codebook lookups (low nibbles then high),
    // pairwise products with the Q8_K values scaled by ls1/ls2, accumulated into
    // eight int32 lanes and folded with one fma into eight float lanes.
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 136U;
        const std::uint8_t* qs = block + 8U;
        const std::int8_t* q = q8.qs.data() + b * 256U;
        std::uint16_t sh = static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8U));
        std::uint32_t sumi[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        for (std::int32_t ib = 0; ib < 8; ib += 2) {
            const std::uint8_t* q4bits_1 = qs;
            qs += 16U;
            const std::uint8_t* q4bits_2 = qs;
            qs += 16U;
            const std::int8_t* q8b_1 = q;
            q += 32;
            const std::int8_t* q8b_2 = q;
            q += 32;
            const std::uint32_t high_bits = static_cast<std::uint32_t>(sh);
            const std::int32_t ls1 = static_cast<std::int32_t>(
                ((block[4U + static_cast<std::size_t>(ib) / 2U] & 0x0FU) | ((high_bits << 4U) & 0x30U)) - 32U);
            const std::int32_t ls2 = static_cast<std::int32_t>(
                ((block[4U + static_cast<std::size_t>(ib) / 2U] >> 4U) | ((high_bits << 2U) & 0x30U)) - 32U);
            sh = static_cast<std::uint16_t>(sh >> 4U);
            std::int32_t v1[32];
            std::int32_t v2[32];
            for (std::size_t idx = 0; idx < 16; ++idx) {
                v1[idx] = kIq4NlValues[q4bits_1[idx] & 0x0FU];
                v1[16U + idx] = kIq4NlValues[q4bits_1[idx] >> 4U];
                v2[idx] = kIq4NlValues[q4bits_2[idx] & 0x0FU];
                v2[16U + idx] = kIq4NlValues[q4bits_2[idx] >> 4U];
            }
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const std::size_t at = lane * 4U;
                const std::int32_t p1 = v1[at] * q8b_1[at] + v1[at + 1U] * q8b_1[at + 1U] +
                                        v1[at + 2U] * q8b_1[at + 2U] + v1[at + 3U] * q8b_1[at + 3U];
                const std::int32_t p2 = v2[at] * q8b_2[at] + v2[at + 1U] * q8b_2[at + 1U] +
                                        v2[at + 2U] * q8b_2[at + 2U] + v2[at + 3U] * q8b_2[at + 3U];
                sumi[lane] += static_cast<std::uint32_t>(ls1 * p1 + ls2 * p2);
            }
        }
        const float d =
            half_to_float(static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8U))) *
            q8.d[b];
        for (std::size_t lane = 0; lane < 8; ++lane) {
            accum[lane] = std::fma(d, static_cast<float>(static_cast<std::int32_t>(sumi[lane])), accum[lane]);
        }
    }
    return hsum_float_8(accum);
#endif
}

[[nodiscard]] float dot_nvfp4_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks) {
#if defined(__AVX2__) && defined(__FMA__)
    return dot_nvfp4_q8_0_avx2(data, q8, blocks);
#else
    // Lane-faithful replica of dot_nvfp4_q8_0_avx2: each 64-value block is two
    // 32-value groups, every group is two 16-value halves with their own
    // 0.5 * ue4m3 scale, maddubs pairs the codebook values against Q8_0 codes,
    // and each four-value lane folds with one fma per half into eight float
    // lanes reduced by the movehl tree.
    float accum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data + b * 36U;
        const float dy0 = q8.d[2U * b];
        const float dy1 = q8.d[2U * b + 1U];
        const float scale[4] = {0.5F * ue4m3_to_float(block[0]) * dy0, 0.5F * ue4m3_to_float(block[1]) * dy0,
                                0.5F * ue4m3_to_float(block[2]) * dy1, 0.5F * ue4m3_to_float(block[3]) * dy1};
        for (std::size_t g = 0; g < 2; ++g) {
            const std::int8_t* qy = q8.qs.data() + (2U * b + g) * 32U;
            for (std::size_t sh = 0; sh < 2; ++sh) {
                for (std::size_t ln = 0; ln < 4; ++ln) {
                    std::int32_t isum = 0;
                    for (std::size_t t = 0; t < 4; ++t) {
                        const std::size_t m = 4U * ln + t;
                        const std::uint8_t qv = block[4U + 16U * g + 8U * sh + (m % 8U)];
                        const std::uint8_t code =
                            m < 8U ? static_cast<std::uint8_t>(qv & 0x0FU) : static_cast<std::uint8_t>(qv >> 4U);
                        isum +=
                            static_cast<std::int32_t>(qy[16U * sh + m]) * static_cast<std::int32_t>(kMxfp4Values[code]);
                    }
                    const std::size_t lane = 4U * sh + ln;
                    accum[lane] = std::fma(scale[2U * g + sh], static_cast<float>(isum), accum[lane]);
                }
            }
        }
    }
    return hsum_float_8(accum);
#endif
}

// ggml_vec_dot_q4_K_q8_K (AVX2): each 256-value block keeps its integer partial
// sums in eight int32 lanes (four values per lane, one scale per 32-value
// group), folds the block into eight float accumulators with a fused
// multiply-add, and rides the quantised minima on a separate four-lane
// accumulator that is reduced first. Both reductions and the final add follow
// hsum_float_8 exactly.
[[nodiscard]] float dot_q4_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            mins[g] = mn;
            sums[g] = static_cast<std::int32_t>(bsums[2 * g]) + static_cast<std::int32_t>(bsums[2 * g + 1]);
            const std::uint8_t* bytes = qs + 32 * (g / 2);
            for (std::int32_t l = 0; l < 8; ++l) {
                std::int32_t partial = 0;
                for (std::int32_t k = 0; k < 4; ++k) {
                    const std::uint8_t byte = bytes[4 * l + k];
                    const std::int32_t nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
                    partial += nibble * y[g * 32 + 4 * l + k];
                }
                lane[l] += static_cast<std::int32_t>(scale) * partial;
            }
        }
        for (std::int32_t l = 0; l < 8; ++l) {
            acc[l] = std::fma(d, static_cast<float>(lane[l]), acc[l]);
        }
        for (std::int32_t i = 0; i < 4; ++i) {
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
[[nodiscard]] float dot_q5_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            const std::uint8_t* bytes = qs + 32 * (g / 2);
            for (std::int32_t l = 0; l < 8; ++l) {
                std::int32_t partial = 0;
                for (std::int32_t k = 0; k < 4; ++k) {
                    const std::uint8_t byte = bytes[4 * l + k];
                    const std::int32_t nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
                    const std::int32_t bit = (qh[4 * l + k] >> g) & 1;
                    partial += (nibble + 16 * bit) * y[g * 32 + 4 * l + k];
                }
                lane[l] += static_cast<std::int32_t>(scale) * partial;
            }
            minima[g / 2] += static_cast<std::int32_t>(mn) *
                             (static_cast<std::int32_t>(bsums[2 * g]) + static_cast<std::int32_t>(bsums[2 * g + 1]));
        }
        for (std::int32_t l = 0; l < 8; ++l) {
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
[[nodiscard]] float dot_q4_k_q8_k_8x8(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks) {
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
        for (std::int32_t g = 0; g < 8; ++g) {
            std::uint8_t scale = 0;
            std::uint8_t mn = 0;
            get_scale_min_k4(g, block + 4, scale, mn);
            const std::uint8_t* bytes = block + 16 + 32 * (g / 2);
            std::int32_t partial = 0;
            for (std::int32_t l = 0; l < 32; ++l) {
                const std::uint8_t byte = bytes[l];
                const std::int32_t nibble = (g & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
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

}  // namespace mnet::gguf
