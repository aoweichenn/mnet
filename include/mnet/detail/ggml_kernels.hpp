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

#include <cstddef>
#include <cstdint>

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
        data[i] = data[i] / (1.0F + std::exp(-data[i]));
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
        out[i] = (gate[i] / (1.0F + std::exp(-gate[i]))) * up[i];
    }
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
        const float value = std::exp(input[i] - max);
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
