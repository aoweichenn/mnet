#pragma once

// llama.cpp's hot kernels behind one entry point per kernel. The accumulation
// order is spelled out in portable scalar form (and proven bit-identical to the
// AVX2/FMA originals), so model code calls these functions and never branches on
// the target: adding an architecture only changes the bodies here.
//
// Reduction trees matter: ggml_vec_dot_f32 reduces with _mm_hadd_ps
// (((l0+l4)+(l1+l5)) + ((l2+l6)+(l3+l7))), while the quantised dots use the
// movehl/movehdup tree (((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))). Both live in
// <mnet/detail/ggml_vec_exact.hpp>.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <limits>

#include <mnet/detail/ggml_vec_exact.hpp>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mnet::detail::ggml {

// ggml_vec_dot_f32: four 8-lane FMA accumulators over 32-element groups,
// (s0+s2)+(s1+s3), the hadd tree, then llama's rounded-product leftover path.
[[nodiscard]] inline float dot_f32(const float* x, const float* y, std::size_t size) {
#if defined(__AVX2__) && defined(__FMA__)
    const std::size_t aligned = size & ~static_cast<std::size_t>(31);
    __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (std::size_t i = 0; i < aligned; i += 32) {
        for (std::size_t j = 0; j < 4; ++j) {
            sums[j] = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + j * 8), _mm256_loadu_ps(y + i + j * 8), sums[j]);
        }
    }
    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sums[0]), _mm256_extractf128_ps(sums[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    float total = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    for (std::size_t i = aligned; i < size; ++i) {
        volatile float product = x[i] * y[i];
        total += product;
    }
    return total;
#else
    return ggml_dot_f32(x, y, size);
#endif
}

// ggml_vec_dot_f16_f32 with the f16 values already widened to f32: four
// accumulators over 32-element groups, one extra 8-lane group, then
// (a0+a1)+(a2+a3) and the movehl/movehdup hsum into a double accumulator.
[[nodiscard]] inline float dot_f16_f32(const float* x, const float* y, std::size_t size) {
#if defined(__AVX2__) && defined(__FMA__)
    std::size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= size; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }
    for (; i + 8 <= size; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
    }
    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    double total = static_cast<double>(_mm_cvtss_f32(r));
    for (; i < size; ++i) {
        total += static_cast<double>(x[i] * y[i]);
    }
    return static_cast<float>(total);
#else
    return ggml_dot_f16_f32(x, y, size);
#endif
}

// ggml_v_expf: llama.cpp's polynomial exponential (used by SiLU and softmax).
[[nodiscard]] inline float expf(float value) {
    return ggml_expf(value);
}

[[nodiscard]] inline float silu(float value) {
    return ggml_silu(value);
}

// ggml_vec_silu_f32: 8-wide polynomial body, then the scalar libm tail.
inline void silu_inplace(float* data, std::size_t size) {
    std::size_t i = 0;
    for (; i + 7 < size; i += 8) {
        for (std::size_t l = 0; l < 8; ++l) {
            data[i + l] = ggml_silu(data[i + l]);
        }
    }
    for (; i < size; ++i) {
        data[i] = data[i] / (1.0F + ggml_expf(-data[i]));
    }
}

// ggml_vec_swiglu_f32: silu(gate) * up, same body split.
inline void swiglu(float* out, const float* gate, const float* up, std::size_t size) {
    std::size_t i = 0;
    for (; i + 7 < size; i += 8) {
        for (std::size_t l = 0; l < 8; ++l) {
            out[i + l] = ggml_silu(gate[i + l]) * up[i + l];
        }
    }
    for (; i < size; ++i) {
        out[i] = (gate[i] / (1.0F + ggml_expf(-gate[i]))) * up[i];
    }
}

// Deterministic float transcendentals. libm's exp/tanh/log differ by an ulp
// between glibc, Apple libm and other C libraries, which is enough to make a
// 48-layer model route differently; these versions use only IEEE arithmetic
// (the same polynomial exp as ggml, plus fixed series) so every target rounds
// identically.
[[nodiscard]] inline float tanhf(float value) {
    const float ax = std::fabs(value);
    if (ax < 0.25F) {
        // tanh(x) = x - x^3/3 + 2x^5/15 - 17x^7/315 + 62x^9/2835 - 1382x^11/155925
        const float x2 = value * value;
        return value *
               (1.0F + x2 * (-1.0F / 3.0F +
                             x2 * (2.0F / 15.0F +
                                   x2 * (-17.0F / 315.0F + x2 * (62.0F / 2835.0F + x2 * (-1382.0F / 155925.0F))))));
    }
    // 1 - 2/(e^(2|x|)+1); the cancellation factor is below 4 for |x| >= 0.25.
    const float r = 1.0F - 2.0F / (expf(2.0F * ax) + 1.0F);
    return value < 0.0F ? -r : r;
}

[[nodiscard]] inline float logf(float value) {
    if (value <= 0.0F) {
        if (value == 0.0F) {
            return -std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::int32_t exponent = 0;                              // check-types: allow std::frexp takes int*
    const float mantissa = std::frexp(value, &exponent);    // [0.5, 1)
    const float t = (mantissa - 1.0F) / (mantissa + 1.0F);  // |t| <= 1/3
    const float t2 = t * t;
    const float series =
        1.0F +
        t2 * (1.0F / 3.0F +
              t2 * (1.0F / 5.0F + t2 * (1.0F / 7.0F + t2 * (1.0F / 9.0F + t2 * (1.0F / 11.0F + t2 * (1.0F / 13.0F))))));
    const double ln_mantissa = 2.0 * static_cast<double>(t) * static_cast<double>(series);
    return static_cast<float>(static_cast<double>(exponent) * 0.693147180559945309417232121458 + ln_mantissa);
}

[[nodiscard]] inline float log1pf(float value) {
    if (value > -0.5F && value < 0.5F) {
        // log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + ...
        float term = value;
        float sum = 0.0F;
        for (std::size_t k = 1; k <= 12; ++k) {
            const float sign = (k % 2U) == 1U ? 1.0F : -1.0F;
            sum += sign * term / static_cast<float>(k);
            term *= value;
        }
        return sum;
    }
    return logf(1.0F + value);
}

[[nodiscard]] inline float powf(float base, float exponent) {
    return expf(exponent * logf(base));
}

[[nodiscard]] inline double dlog(double value) {
    if (value <= 0.0) {
        return value == 0.0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    }
    std::int32_t exponent = 0;  // check-types: allow std::frexp takes int*
    const double mantissa = std::frexp(value, &exponent);
    const double t = (mantissa - 1.0) / (mantissa + 1.0);
    const double t2 = t * t;
    double series = 1.0;
    double power = t2;
    for (std::size_t k = 1; k <= 10; ++k) {
        series += power / static_cast<double>(2U * k + 1U);
        power *= t2;
    }
    return 2.0 * t * series + static_cast<double>(exponent) * 0.693147180559945309417232121458;
}

// Softmax row: exponentiate 8 lanes at a time with the polynomial exp, fold each
// group through the 8-lane hsum into a double accumulator, then the libm tail.
[[nodiscard]] inline double softmax_exp(const float* input, float* output, std::size_t size, float max) {
    std::size_t i = 0;
    double sum = 0.0;
    for (; i + 7 < size; i += 8) {
        float lanes[8];
        for (std::size_t l = 0; l < 8; ++l) {
            lanes[l] = ggml_expf(input[i + l] - max);
            output[i + l] = lanes[l];
        }
        sum += static_cast<double>(ggml_hsum8(lanes));
    }
    for (; i < size; ++i) {
        const float value = ggml_expf(input[i] - max);
        output[i] = value;
        sum += static_cast<double>(value);
    }
    return sum;
}

// One block of the normalized-Hadamard rotation: ggml_vec_dot_f32 per row.
inline void rotate_block(const float* matrix, const float* input, float* output, std::uint32_t block_size) {
    const std::uint32_t aligned = block_size & ~31U;
    const std::uint32_t rounded = aligned;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(block_size); ++row) {
        const float* weights = matrix + static_cast<std::size_t>(row) * block_size;
        float lanes[8];
        float sums[4][8] = {};
        for (std::uint32_t i = 0; i < rounded; i += 32) {
            for (std::uint32_t j = 0; j < 4; ++j) {
                for (std::uint32_t l = 0; l < 8; ++l) {
                    sums[j][l] = std::fma(weights[i + j * 8 + l], input[i + j * 8 + l], sums[j][l]);
                }
            }
        }
        for (std::uint32_t l = 0; l < 8; ++l) {
            lanes[l] = (sums[0][l] + sums[2][l]) + (sums[1][l] + sums[3][l]);
        }
        float total = ggml_hsum8_hadd(lanes);
        for (std::uint32_t i = aligned; i < block_size; ++i) {
            volatile float product = weights[i] * input[i];
            total += product;
        }
        output[row] = total;
    }
}

}  // namespace mnet::detail::ggml
