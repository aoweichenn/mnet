#pragma once

// Bit-exact reimplementations of the element-wise kernels ONNX Runtime and MLAS
// use for the Piper/Kokoro graphs, isolated from the model code so the
// arithmetic can be reused and audited on its own.
//
// Every routine was matched bit-for-bit against the pinned reference dumps; the
// comments record which MLAS/Eigen source and SIMD shape the arithmetic
// follows. In particular:
//   - MLAS kernels (MLAS_TARGET_AMD64) use FMA3 assembly, emulated here with
//     explicit std::fma;
//   - Eigen's vectorised helpers in the ONNX Runtime CPU EP are compiled for
//     4-lane SSE without FMA contraction, and their n % 4 tail runs the scalar
//     std::sin/std::cos/std::log1p - hence the *_element helpers;
//   - the Softmax kernel, the Gelu/Erf kernel and the convolutional GEMM
//     blocking rules come from mlas/lib and mlas/lib/x86_64.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <bit>
#include <thread>
#include <utility>
#include <vector>

namespace mnet::tts {

inline constexpr float kLogisticLowerRange = -18.0F;

inline constexpr float kLogisticUpperRange = 18.0F;

inline constexpr float kLogisticAlpha9 = 4.37031012579801e-11F;

inline constexpr float kLogisticAlpha7 = 1.15627324459942e-07F;

inline constexpr float kLogisticAlpha5 = 6.08574864600143e-05F;

inline constexpr float kLogisticAlpha3 = 8.51377133304701e-03F;

inline constexpr float kLogisticAlpha1 = 2.48287947061529e-01F;

inline constexpr float kLogisticBeta10 = 6.10247389755681e-13F;

inline constexpr float kLogisticBeta8 = 5.76102136993427e-09F;

inline constexpr float kLogisticBeta6 = 6.29106785017040e-06F;

inline constexpr float kLogisticBeta4 = 1.70198817374094e-03F;

inline constexpr float kLogisticBeta2 = 1.16817656904453e-01F;

inline constexpr float kLogisticBeta0 = 9.93151921023180e-01F;

inline constexpr float kLogisticHalf = 0.5F;

inline constexpr float kTanhLowerRange = -9.0F;

inline constexpr float kTanhUpperRange = 9.0F;

inline constexpr float kTanhAlpha13 = -2.76076847742355e-16F;

inline constexpr float kTanhAlpha11 = 2.00018790482477e-13F;

inline constexpr float kTanhAlpha9 = -8.60467152213735e-11F;

inline constexpr float kTanhAlpha7 = 5.12229709037114e-08F;

inline constexpr float kTanhAlpha5 = 1.48572235717979e-05F;

inline constexpr float kTanhAlpha3 = 6.37261928875436e-04F;

inline constexpr float kTanhAlpha1 = 4.89352455891786e-03F;

inline constexpr float kTanhBeta6 = 1.19825839466702e-06F;

inline constexpr float kTanhBeta4 = 1.18534705686654e-04F;

inline constexpr float kTanhBeta2 = 2.26843463243900e-03F;

inline constexpr float kTanhBeta0 = 4.89352518554385e-03F;

struct EigenFrexp {
    float mantissa = 0.0F;
    float exponent = 0.0F;
};

[[nodiscard]] EigenFrexp eigen_frexp(float value);

[[nodiscard]] float eigen_log_positive(float value);

[[nodiscard]] float eigen_exp(float value);

[[nodiscard]] float eigen_log1p(float value);

// Eigen vectorises Array::sin/cos/exp/log1p in 4-wide SSE packets and evaluates
// the final (count % 4) elements with the scalar std::sin/std::cos/std::log1p,
// which are correctly rounded and can land one ulp away from the packet
// polynomial. Element-wise ops therefore have to know their flat position.

[[nodiscard]] bool eigen_scalar_tail(std::size_t index, std::size_t count);

[[nodiscard]] float eigen_softplus(float value);

// Eigen's psin (psincos_float<true>): non-fused octant reduction and the
// sin/cos polynomials, matching ONNX Runtime's packet Sin exactly.
// Eigen's psincos_float switches to a Payne-Hanek reduction once |x| reaches
// huge_th (25966 for sin, 18838 for cos in the non-FMA branch); below that the
// four-term extended-precision reduction is used. Without this branch the
// polynomial keeps only the leading terms and drifts by ~1e-6 for the larger
// arguments that longer texts produce.

[[nodiscard]] float eigen_softplus_element(float value, std::size_t index, std::size_t count);

[[nodiscard]] float eigen_trig_reduce_huge(float value, std::uint32_t& quadrant);

[[nodiscard]] float eigen_sin(float value);

[[nodiscard]] float eigen_sin_element(float value, std::size_t index, std::size_t count);

[[nodiscard]] float eigen_cos(float value);

// Eigen's patan (3.4.90): atan(x) = x * P(x^2) / Q(x^2) for |x| <= 1,
// atan(x) = pi/2 - atan(1/x) otherwise, evaluated with non-fused multiply-add
// exactly like ONNX Runtime's packet kernel.

[[nodiscard]] float eigen_cos_element(float value, std::size_t index, std::size_t count);

// Snake activation (StyleTTS2): x + sin(alpha * x)^2 / alpha, with one learned
// alpha per channel.

[[nodiscard]] float eigen_atan(float value);

// Eigen's generic_fast_tanh_float, i.e. what ONNX Runtime's Tanh kernel emits
// (13/6-degree rational interpolant, FMA build). Needed because the harmonic
// source's tanh feeds a float32 DFT whose silent-bin phase is rounding-defined.

[[nodiscard]] float eigen_fast_tanh(float value);

[[nodiscard]] float ort_logistic(float value);

[[nodiscard]] float ort_tanh(float value);

// ONNX Runtime's Gelu<float> (approximation "none") calls MlasComputeGeluErf:
//   t = x * (1 / sqrt(2));  t = MlasComputeErf(t);  out = 0.5 * x * (t + 1)
// and MlasComputeErf on AMD64 runs MlasErfKernelFma3 (mlas/lib/x86_64/
// ErfKernelFma3.S), a split polynomial: |x| <= 0.921875 uses the small
// polynomial in |x|^2, larger values use the big polynomial feeding a
// self-contained exp, with a bitwise merge of the two and a sign copy. All
// steps below are the FMA forms from that kernel.

[[nodiscard]] float mlas_erf(float value);

// MLAS's Exp kernel (MlasComputeExpF32KernelFma3, TransKernelFma3.asm):
// clamp to [LowerRange, UpperRange], Eigen's rational range reduction with the
// magic rounding bias, the six polynomial terms, then the two-power scale that
// also reproduces subnormal results.

[[nodiscard]] float mlas_exp(float value);

// MlasGemm's non-packed K blocking (sgemm.cpp, MlasSgemmOperation): StrideK
// starts at MLAS_SGEMM_STRIDEK = 128, shrinks while N >= K, and doubles (with
// N halving) while N < K. Used for the attention MatMuls whose right operand
// is a runtime tensor instead of a prepacked constant weight.

[[nodiscard]] float mlas_sumexp_term(float value);

// SoftmaxCPU -> MlasComputeSoftmax for float: an exact maximum, the
// SumExpF32 kernel's 8-lane accumulation with the horizontal reduction
// ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), then exp * (1/sum).

void mlas_softmax_row(const float* input, float* output, std::size_t n);

[[nodiscard]] inline std::size_t mlas_gemm_kblock(std::size_t n, std::size_t k) {
    std::size_t kblock = 128;
    if (n >= k) {
        while (kblock > 1 && kblock / 2 >= k) {
            kblock /= 2;
        }
    } else {
        std::size_t stride_n = 128;
        while (stride_n > 16 && stride_n / 2 >= n) {
            kblock *= 2;
            stride_n /= 2;
        }
    }
    return kblock;
}

// How MLAS splits the N dimension of a convolution's GEMM, which decides each
// output column's StrideK (and therefore its K blocking):
//  - a pointwise convolution takes MlasConvAlgorithmGemmDirect, so the single
//    MlasGemm call splits N into 16-aligned blocks whenever N > M
//    (sgemm.cpp, MlasGemmBatch: ThreadCountN over ceil(N / 16) blocks);
//  - a general convolution with FilterCount <= OutputSize takes
//    MlasConvAlgorithmExpandThenGemmSegmented, which cuts N into ThreadStrideN
//    segments (convolve.cpp, MlasConvPrepare / MlasConvExpandThenGemmSegmented);
//  - FilterCount > OutputSize uses ExpandThenGemm, one range over all of N.
// Ranges are [start, count) column intervals.

[[nodiscard]] inline std::vector<std::pair<std::size_t, std::size_t>> mlas_conv_ranges(
    std::size_t out_channels, std::size_t out_len, std::size_t total_k, bool pointwise, std::size_t ort_threads) {
    constexpr std::size_t kThreadComplexity = 65536;  // MLAS_SGEMM_THREAD_COMPLEXITY
    constexpr std::size_t kMaximumThreadCount = 16;   // MLAS_MAXIMUM_THREAD_COUNT
    constexpr std::size_t kThreadAlign = 16;          // MLAS_SGEMM_STRIDEN_THREAD_ALIGN
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    const std::size_t complexity = out_channels * out_len * total_k;
    const std::size_t gemm_target = std::min<std::size_t>(complexity / kThreadComplexity + 1, ort_threads);
    if (pointwise && out_len > out_channels) {
        const std::size_t blocked = (out_len + kThreadAlign - 1) / kThreadAlign;
        const std::size_t threads = std::min(gemm_target, blocked);
        const std::size_t per_thread = blocked / std::max<std::size_t>(threads, 1);
        const std::size_t extra = blocked % std::max<std::size_t>(threads, 1);
        std::size_t start_block = 0;
        for (std::size_t i = 0; i < threads; ++i) {
            const std::size_t count_blocks = per_thread + (i < extra ? 1 : 0);
            const std::size_t start = start_block * kThreadAlign;
            const std::size_t count = std::min(out_len - start, count_blocks * kThreadAlign);
            ranges.emplace_back(start, count);
            start_block += count_blocks;
        }
        return ranges;
    }
    if (!pointwise && out_channels <= out_len) {
        // convolve.cpp caps the target at MLAS_MAXIMUM_THREAD_COUNT and only
        // scales with complexity below MLAS_SGEMM_THREAD_COMPLEXITY * 16.
        std::size_t target = complexity < kThreadComplexity * kMaximumThreadCount ? complexity / kThreadComplexity + 1
                                                                                  : kMaximumThreadCount;
        target = std::min(target, ort_threads);
        std::size_t stride_n = out_len / target;
        if (stride_n * target != out_len) {
            stride_n++;
        }
        if (target > 1) {
            stride_n = (stride_n + kThreadAlign - 1) & ~(kThreadAlign - 1);
            if (stride_n >= out_len) {
                target = 1;
            } else if (stride_n * (target - 1) >= out_len) {
                target--;
            }
        }
        if (stride_n < out_len) {
            for (std::size_t start = 0; start < out_len; start += stride_n) {
                ranges.emplace_back(start, std::min(stride_n, out_len - start));
            }
            return ranges;
        }
    }
    ranges.emplace_back(0, out_len);
    return ranges;
}

// One element of MlasComputeSumExpF32KernelFma3 (TransKernelFma3.asm): the
// range-reduced polynomial applied to x - max, with the exponent rebuilt from
// the magic-bias bits. MLAS clamps the sum-exp input at LowerRangeSumExp.

}  // namespace mnet::tts
