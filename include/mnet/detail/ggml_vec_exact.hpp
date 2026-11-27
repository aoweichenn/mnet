#pragma once

// Portable reproductions of llama.cpp's 8-lane AVX2 kernels: the same lane
// layout, accumulation order and reduction tree, evaluated with scalar
// arithmetic. The arm64 builds of the bit-exactness runners use these so the
// numbers match the x86 reference build instead of a different summation order.

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mnet::detail {

// hsum of one 8-lane vector: ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)), the
// _mm_movehl_ps / _mm_movehdup_ps sequence.
[[nodiscard]] inline float ggml_hsum8(const float* lanes) {
    return ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
}

[[nodiscard]] inline bool ggml_abs_gt(float value, float limit) {
    // _mm256_cmp_ps(andnot(-0.0f, value), limit, _CMP_GT_OQ): ordered compare,
    // so NaN compares false.
    return (std::fabs(value) > limit) && !std::isnan(value);
}

// ggml_v_expf, the polynomial exp used by llama.cpp's SiLU / softmax.
[[nodiscard]] inline float ggml_expf(float x) {
    constexpr float kRounding = 0x1.8p23F;
    constexpr float kLog2e = 0x1.715476p+0F;
    const float z = std::fma(x, kLog2e, kRounding);
    const float n = z - kRounding;
    const float b = std::fma(-n, 0x1.7f7d1cp-20F, std::fma(-n, 0x1.62e4p-1F, x));
    const std::uint32_t e = std::bit_cast<std::uint32_t>(z) << 23U;
    const float k = std::bit_cast<float>(e + std::bit_cast<std::uint32_t>(1.0F));
    const bool c = ggml_abs_gt(n, 126.0F);
    const float u = b * b;
    const float j = std::fma(
        std::fma(std::fma(0x1.0e4020p-7F, b, 0x1.573e2ep-5F), u, std::fma(0x1.555e66p-3F, b, 0x1.fffdb6p-2F)), u,
        std::fma(0x1.ffffecp-1F, b, 0.0F));
    if (!c) {
        return std::fma(j, k, k);
    }
    const std::uint32_t g_bits = (n <= 0.0F ? 0xFFFFFFFFU : 0U) & 0x82000000U;
    const float s1 = std::bit_cast<float>(g_bits + 0x7f000000U);
    const float s2 = std::bit_cast<float>(e - g_bits);
    const bool d = ggml_abs_gt(n, 192.0F);
    if (d) {
        return s1 * s1;
    }
    const std::uint32_t low = c ? std::bit_cast<std::uint32_t>(std::fma(s2, j, s2) * s1)
                                : std::bit_cast<std::uint32_t>(std::fma(k, j, k));
    return std::bit_cast<float>(low);
}

[[nodiscard]] inline float ggml_silu(float x) {
    return x / (1.0F + ggml_expf(-x));
}

// ggml_vec_dot_f32: four 8-lane FMA accumulators over 32-element groups,
// (s0+s2)+(s1+s3) then hsum, and llama's leftover path (the final size % 4
// values fused into the running total).
[[nodiscard]] inline float ggml_dot_f32(const float* x, const float* y, std::size_t size) {
    const std::size_t aligned = size & ~static_cast<std::size_t>(31);
    float sums[4][8] = {};
    for (std::size_t i = 0; i < aligned; i += 32) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t l = 0; l < 8; ++l) {
                sums[j][l] = std::fma(x[i + j * 8 + l], y[i + j * 8 + l], sums[j][l]);
            }
        }
    }
    float lanes[8];
    for (std::size_t l = 0; l < 8; ++l) {
        lanes[l] = (sums[0][l] + sums[2][l]) + (sums[1][l] + sums[3][l]);
    }
    float total = ggml_hsum8(lanes);
    const std::size_t tail = size - aligned;
    const std::size_t rounded = tail - tail % 4;
    std::size_t i = aligned;
    for (std::size_t k = 0; k < rounded; ++k, ++i) {
        volatile float product = x[i] * y[i];
        total += product;
    }
    for (; i < size; ++i) {
        total = std::fma(x[i], y[i], total);
    }
    return total;
}

// ggml_vec_dot_f16_f32 with the f16 values already widened to f32.
[[nodiscard]] inline float ggml_dot_f16_f32(const float* x, const float* y, std::size_t size) {
    float acc[4][8] = {};
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        for (std::size_t a = 0; a < 4; ++a) {
            for (std::size_t l = 0; l < 8; ++l) {
                acc[a][l] = std::fma(x[i + a * 8 + l], y[i + a * 8 + l], acc[a][l]);
            }
        }
    }
    for (; i + 8 <= size; i += 8) {
        for (std::size_t l = 0; l < 8; ++l) {
            acc[0][l] = std::fma(x[i + l], y[i + l], acc[0][l]);
        }
    }
    float lanes[8];
    for (std::size_t l = 0; l < 8; ++l) {
        lanes[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }
    double total = static_cast<double>(ggml_hsum8(lanes));
    for (; i < size; ++i) {
        total += static_cast<double>(x[i] * y[i]);
    }
    return static_cast<float>(total);
}

// One row of ggml_vec_dot_f32 over a 4 x 8-lane accumulator set, matching the
// normalized-Hadamard rotation's dense matmul.
[[nodiscard]] inline float ggml_dot_f32_row(const float* weights, const float* input, std::size_t size) {
    return ggml_dot_f32(weights, input, size);
}

}  // namespace mnet::detail
