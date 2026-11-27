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

[[nodiscard]] inline EigenFrexp eigen_frexp(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t biased = (bits >> 23U) & 0xFFU;
    return {std::bit_cast<float>((bits & 0x807FFFFFU) | 0x3F000000U),
            static_cast<float>(static_cast<std::int32_t>(biased) - 126)};
}

[[nodiscard]] inline float eigen_log_positive(float value) {
    constexpr float kSqrtHalf = 0.707106781186547524F;
    constexpr float kLn2 = 0.693147180559945309417F;
    constexpr float kAlpha0 = 0.18256296349849254F;
    constexpr float kAlpha1 = 1.0000000190281063F;
    constexpr float kAlpha2 = 1.0000000190281136F;
    constexpr float kBeta0 = 0.049616247954120038F;
    constexpr float kBeta1 = 0.59923249590823520F;
    constexpr float kBeta2 = 1.4999999999999927F;
    constexpr float kBeta3 = 1.0F;
    const EigenFrexp fr = eigen_frexp(value);
    float x = fr.mantissa;
    float e = fr.exponent;
    if (x < kSqrtHalf) {
        const float tmp = x;
        x -= 1.0F;
        e -= 1.0F;
        x += tmp;
    } else {
        x -= 1.0F;
    }
    float p = kAlpha0 * x + kAlpha1;
    p = p * x + kAlpha2;
    p *= x;
    float q = kBeta0 * x + kBeta1;
    q = q * x + kBeta2;
    q = q * x + kBeta3;
    return e * kLn2 + (p / q);
}

[[nodiscard]] inline float eigen_exp(float value) {
    constexpr float kExpHigh = 88.723F;
    constexpr float kLog2E = 1.44269504088896341F;
    constexpr float kC1 = -0.693359375F;
    constexpr float kC2 = 2.12194440e-4F;
    constexpr float kP2 = 0.49999988079071044921875F;
    constexpr float kP3 = 0.16666518151760101318359375F;
    constexpr float kP4 = 4.166965186595916748046875e-2F;
    constexpr float kP5 = 8.36894474923610687255859375e-3F;
    constexpr float kP6 = 1.37449637986719608306884765625e-3F;
    const float x = std::min(value, kExpHigh);
    const float m = std::floor(x * kLog2E + 0.5F);
    float r = m * kC1 + x;
    r = m * kC2 + r;
    const float r2 = r * r;
    float even = r2 * kP6 + kP4;
    const float odd = r2 * kP5 + kP3;
    even = r2 * even + kP2;
    const float low = r + 1.0F;
    float y = r * odd + even;
    y = r2 * y + low;
    const std::int32_t exponent = std::min(std::max(static_cast<std::int32_t>(m) + 127, 0), 255);
    y *= std::bit_cast<float>(static_cast<std::uint32_t>(exponent) << 23U);
    return std::max(y, value);
}

[[nodiscard]] inline float eigen_log1p(float value) {
    const float xp1 = value + 1.0F;
    const float logarithm = eigen_log_positive(xp1);
    if (xp1 == 1.0F || xp1 == logarithm) {
        return value;
    }
    return value * (logarithm / (xp1 - 1.0F));
}

// Eigen vectorises Array::sin/cos/exp/log1p in 4-wide SSE packets and evaluates
// the final (count % 4) elements with the scalar std::sin/std::cos/std::log1p,
// which are correctly rounded and can land one ulp away from the packet
// polynomial. Element-wise ops therefore have to know their flat position.

[[nodiscard]] inline bool eigen_scalar_tail(std::size_t index, std::size_t count) {
    constexpr std::size_t kPacket = 4;
    const std::size_t tail = count % kPacket;
    return tail != 0 && index + tail >= count;
}

[[nodiscard]] inline float eigen_softplus(float value) {
    if (value > 0.0F) {
        return value + eigen_log1p(eigen_exp(-value));
    }
    return eigen_log1p(eigen_exp(value));
}

// Eigen's psin (psincos_float<true>): non-fused octant reduction and the
// sin/cos polynomials, matching ONNX Runtime's packet Sin exactly.
// Eigen's psincos_float switches to a Payne-Hanek reduction once |x| reaches
// huge_th (25966 for sin, 18838 for cos in the non-FMA branch); below that the
// four-term extended-precision reduction is used. Without this branch the
// polynomial keeps only the leading terms and drifts by ~1e-6 for the larger
// arguments that longer texts produce.

[[nodiscard]] inline float eigen_softplus_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        // Eigen's scalar fallback: (x > 0) ? x + log1p(exp(-x)) : log1p(exp(x)).
        return value > 0.0F ? value + std::log1p(std::exp(-value)) : std::log1p(std::exp(value));
    }
    return eigen_softplus(value);
}

[[nodiscard]] inline float eigen_trig_reduce_huge(float value, std::uint32_t& quadrant) {
    constexpr double kPiOverTwoPow62 = 3.4061215800865545e-19;  // pi/2 * 2^-62
    constexpr std::uint64_t kHalf = std::uint64_t(1) << 61U;    // 0.5 in 2.62 fixed point
    static constexpr std::uint32_t kTwoOverPi[26] = {
        0x00000028U, 0x000028beU, 0x0028be60U, 0x28be60dbU, 0xbe60db93U, 0x60db9391U, 0xdb939105U,
        0x9391054aU, 0x91054a7fU, 0x054a7f09U, 0x4a7f09d5U, 0x7f09d5f4U, 0x09d5f47dU, 0xd5f47d4dU,
        0xf47d4d37U, 0x7d4d3770U, 0x4d377036U, 0x377036d8U, 0x7036d8a5U, 0x36d8a566U, 0xd8a5664fU,
        0xa5664f10U, 0x664f10e4U, 0x4f10e410U, 0x10e41000U, 0xe4100000U};  // 192 bits of 2/pi
    std::uint32_t xi = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t e = (xi >> 23U) - 118U;  // -126 for the exponent, +8 for alignment
    xi = ((xi & 0x007FFFFFU) | 0x00800000U) << (e & 0x7U);
    const std::uint32_t i = e >> 3U;
    const std::uint32_t two_over_pi_1 = kTwoOverPi[i - 1U];
    const std::uint32_t two_over_pi_2 = kTwoOverPi[i + 3U];
    const std::uint32_t two_over_pi_3 = kTwoOverPi[i + 7U];
    std::uint64_t p = static_cast<std::uint64_t>(xi) * two_over_pi_3;
    p = static_cast<std::uint64_t>(xi) * two_over_pi_2 + (p >> 32U);
    p = (static_cast<std::uint64_t>(xi * two_over_pi_1) << 32U) + p;
    const std::uint64_t q = (p + kHalf) >> 62U;
    quadrant = static_cast<std::uint32_t>(static_cast<std::int64_t>(q));
    p -= q << 62U;
    return static_cast<float>(static_cast<double>(static_cast<std::int64_t>(p)) * kPiOverTwoPow62);
}

[[nodiscard]] inline float eigen_sin(float value) {
    constexpr float kTwoOverPi = 0.636619746685028076171875F;
    constexpr float kRoundingMagic = 12582912.0F;
    const float abs_value = std::fabs(value);
    float y = abs_value * kTwoOverPi;
    const float y_round = y + kRoundingMagic;
    std::uint32_t y_int = 0;
    std::memcpy(&y_int, &y_round, sizeof(y_int));
    y = y_round - kRoundingMagic;
    constexpr float kHugeThreshold = 25966.0F;
    float x = 0.0F;
    if (abs_value >= kHugeThreshold && std::isfinite(abs_value)) {
        // Payne-Hanek replaces the four-term reduction entirely for huge
        // arguments (Eigen overwrites x and the quadrant count).
        x = eigen_trig_reduce_huge(abs_value, y_int);
    } else {
        x = abs_value;
        x = y * -1.5703125F + x;
        x = y * -0.000483989715576171875F + x;
        x = y * 1.62865035235881805419921875e-07F + x;
        x = y * 5.5644315544167710640977020375430583953857421875e-11F + x;
    }
    const float x2 = x * x;
    float c = 2.4372266125283204019069671630859375e-05F;
    c = c * x2 - 0.00138865201734006404876708984375F;
    c = c * x2 + 0.041666619479656219482421875F;
    c = c * x2 - 0.5F;
    c = c * x2 + 1.0F;
    float s = -0.0001959234114083702898469196984621021329076029360294342041015625F;
    s = s * x2 + 0.0083326873655616851693794799871284340042620897293090820312500000F;
    s = s * x2 - 0.166666620398229825550373561782180331647396087646484375F;
    s = s * x2;
    s = s * x + x;
    const float result = (y_int & 1U) == 0U ? s : c;
    std::uint32_t value_bits = 0;
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    const std::uint32_t sign = (value_bits ^ (y_int << 30U)) & 0x80000000U;
    std::uint32_t out_bits = 0;
    std::memcpy(&out_bits, &result, sizeof(out_bits));
    out_bits ^= sign;
    float out = 0.0F;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

[[nodiscard]] inline float eigen_sin_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        return std::sin(value);
    }
    return eigen_sin(value);
}

[[nodiscard]] inline float eigen_cos(float value) {
    constexpr float kTwoOverPi = 0.636619746685028076171875F;
    constexpr float kRoundingMagic = 12582912.0F;
    const float abs_value = std::fabs(value);
    float y = abs_value * kTwoOverPi;
    const float y_round = y + kRoundingMagic;
    std::uint32_t y_int = 0;
    std::memcpy(&y_int, &y_round, sizeof(y_int));
    y = y_round - kRoundingMagic;
    constexpr float kHugeThreshold = 18838.0F;
    float x = 0.0F;
    if (abs_value >= kHugeThreshold && std::isfinite(abs_value)) {
        // Payne-Hanek replaces the four-term reduction entirely for huge
        // arguments (Eigen overwrites x and the quadrant count).
        x = eigen_trig_reduce_huge(abs_value, y_int);
    } else {
        x = abs_value;
        x = y * -1.5703125F + x;
        x = y * -0.000483989715576171875F + x;
        x = y * 1.62865035235881805419921875e-07F + x;
        x = y * 5.5644315544167710640977020375430583953857421875e-11F + x;
    }
    const float x2 = x * x;
    float c = 2.4372266125283204019069671630859375e-05F;
    c = c * x2 - 0.00138865201734006404876708984375F;
    c = c * x2 + 0.041666619479656219482421875F;
    c = c * x2 - 0.5F;
    c = c * x2 + 1.0F;
    float s = -0.0001959234114083702898469196984621021329076029360294342041015625F;
    s = s * x2 + 0.0083326873655616851693794799871284340042620897293090820312500000F;
    s = s * x2 - 0.166666620398229825550373561782180331647396087646484375F;
    s = s * x2;
    s = s * x + x;
    const float result = (y_int & 1U) == 0U ? c : s;
    const std::uint32_t sign = ((y_int + 1U) << 30U) & 0x80000000U;
    std::uint32_t out_bits = 0;
    std::memcpy(&out_bits, &result, sizeof(out_bits));
    out_bits ^= sign;
    float out = 0.0F;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

// Eigen's patan (3.4.90): atan(x) = x * P(x^2) / Q(x^2) for |x| <= 1,
// atan(x) = pi/2 - atan(1/x) otherwise, evaluated with non-fused multiply-add
// exactly like ONNX Runtime's packet kernel.

[[nodiscard]] inline float eigen_cos_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        return std::cos(value);
    }
    return eigen_cos(value);
}

// Snake activation (StyleTTS2): x + sin(alpha * x)^2 / alpha, with one learned
// alpha per channel.

[[nodiscard]] inline float eigen_atan(float value) {
    const float abs_value = std::fabs(value);
    const bool large = abs_value > 1.0F;
    const float x = large ? 1.0F / abs_value : abs_value;
    const float x2 = x * x;
    float p = 1.12026982009410858154296875e-01F;
    p = p * x2 + 7.296695709228515625e-01F;
    p = p * x2 + 8.109951019287109375e-01F;
    float q = 1.00917108356952667236328125e-02F;
    q = q * x2 + 2.8318560123443603515625e-01F;
    q = q * x2 + 1.0F;
    q = q * x2 + 8.109951019287109375e-01F;
    const float reduced = x * (p / q);
    const float result = large ? (1.5707964F - reduced) : reduced;
    return std::copysign(result, value);
}

// Eigen's generic_fast_tanh_float, i.e. what ONNX Runtime's Tanh kernel emits
// (13/6-degree rational interpolant, FMA build). Needed because the harmonic
// source's tanh feeds a float32 DFT whose silent-bin phase is rounding-defined.

[[nodiscard]] inline float eigen_fast_tanh(float value) {
    constexpr float kPlusClamp = 9.0F;
    constexpr float kMinusClamp = -9.0F;
    const float x = std::min(std::max(value, kMinusClamp), kPlusClamp);
    const float x2 = x * x;
    float p = std::fma(x2, -2.76076847742355e-16F, 2.00018790482477e-13F);
    p = std::fma(x2, p, -8.60467152213735e-11F);
    p = std::fma(x2, p, 5.12229709037114e-08F);
    p = std::fma(x2, p, 1.48572235717979e-05F);
    p = std::fma(x2, p, 6.37261928875436e-04F);
    p = std::fma(x2, p, 4.89352455891786e-03F);
    const float numerator = x * p;
    float q = std::fma(x2, 1.19825839466702e-06F, 1.18534705686654e-04F);
    q = std::fma(x2, q, 2.26843463243900e-03F);
    q = std::fma(x2, q, 4.89352518554385e-03F);
    return numerator / q;
}

[[nodiscard]] inline float ort_logistic(float value) {
    value = std::max(kLogisticLowerRange, value);
    value = std::min(kLogisticUpperRange, value);
    const float squared = value * value;
    float p = std::fma(squared, kLogisticAlpha9, kLogisticAlpha7);
    p = std::fma(p, squared, kLogisticAlpha5);
    p = std::fma(p, squared, kLogisticAlpha3);
    p = std::fma(p, squared, kLogisticAlpha1);
    p *= value;
    float q = std::fma(squared, kLogisticBeta10, kLogisticBeta8);
    q = std::fma(q, squared, kLogisticBeta6);
    q = std::fma(q, squared, kLogisticBeta4);
    q = std::fma(q, squared, kLogisticBeta2);
    q = std::fma(q, squared, kLogisticBeta0);
    return std::max(0.0F, (p / q) + kLogisticHalf);
}

[[nodiscard]] inline float ort_tanh(float value) {
    value = std::max(kTanhLowerRange, value);
    value = std::min(kTanhUpperRange, value);
    const float squared = value * value;
    float p = std::fma(squared, kTanhAlpha13, kTanhAlpha11);
    p = std::fma(p, squared, kTanhAlpha9);
    p = std::fma(p, squared, kTanhAlpha7);
    p = std::fma(p, squared, kTanhAlpha5);
    p = std::fma(p, squared, kTanhAlpha3);
    p = std::fma(p, squared, kTanhAlpha1);
    p *= value;
    float q = std::fma(squared, kTanhBeta6, kTanhBeta4);
    q = std::fma(q, squared, kTanhBeta2);
    q = std::fma(q, squared, kTanhBeta0);
    return p / q;
}

// ONNX Runtime's Gelu<float> (approximation "none") calls MlasComputeGeluErf:
//   t = x * (1 / sqrt(2));  t = MlasComputeErf(t);  out = 0.5 * x * (t + 1)
// and MlasComputeErf on AMD64 runs MlasErfKernelFma3 (mlas/lib/x86_64/
// ErfKernelFma3.S), a split polynomial: |x| <= 0.921875 uses the small
// polynomial in |x|^2, larger values use the big polynomial feeding a
// self-contained exp, with a bitwise merge of the two and a sign copy. All
// steps below are the FMA forms from that kernel.

[[nodiscard]] inline float mlas_erf(float value) {
    constexpr float kUpperAbsRange = 3.925F;
    constexpr float kSplitBoundary = 0.921875F;
    constexpr float kSmallP0 = -5.99104969e-4F;
    constexpr float kSmallP1 = 4.99339588e-3F;
    constexpr float kSmallP2 = -2.67667342e-2F;
    constexpr float kSmallP3 = 1.12818025e-1F;
    constexpr float kSmallP4 = -3.76124859e-1F;
    constexpr float kSmallP5MinusOne = 1.28379151e-1F;
    constexpr float kBigP0 = 1.72948930e-5F;
    constexpr float kBigP1 = -3.83208680e-4F;
    constexpr float kBigP2 = 3.88393435e-3F;
    constexpr float kBigP3 = -2.42545605e-2F;
    constexpr float kBigP4 = 1.06777847e-1F;
    constexpr float kBigP5 = 6.34846687e-1F;
    constexpr float kBigP6MinusOne = 1.28717512e-1F;
    constexpr float kExpLowerRange = -88.3762626647949F;
    constexpr float kExpLog2Reciprocal = 1.44269504088896341F;
    constexpr float kExpLog2High = -6.93145752e-1F;
    constexpr float kExpLog2Low = -1.42860677e-6F;
    constexpr float kExpP0 = 1.38319808e-3F;
    constexpr float kExpP1 = 8.37550033e-3F;
    constexpr float kExpP2 = 4.16689515e-2F;
    constexpr float kExpP3 = 1.66664466e-1F;
    constexpr float kExpP4 = 4.99999851e-1F;
    constexpr float kExpP5 = 1.0F;
    constexpr float kExpP6 = 1.0F;
    constexpr float kExpBias = 1.25829120e+7F;
    constexpr std::uint32_t kExponentBias = 127U;

    const std::uint32_t sign = std::bit_cast<std::uint32_t>(value) & 0x80000000U;
    float abs_value = std::fabs(value);
    abs_value = std::min(kUpperAbsRange, abs_value);
    const float square = abs_value * abs_value;
    float small = kSmallP0;
    small = std::fma(small, square, kSmallP1);
    small = std::fma(small, square, kSmallP2);
    small = std::fma(small, square, kSmallP3);
    small = std::fma(small, square, kSmallP4);
    small = std::fma(small, square, kSmallP5MinusOne);
    small = std::fma(small, abs_value, abs_value);
    const bool is_big = abs_value > kSplitBoundary;
    if (is_big) {
        small = 0.0F;
    }
    abs_value = is_big ? abs_value : 0.0F;
    float big = kBigP0;
    big = std::fma(big, abs_value, kBigP1);
    big = std::fma(big, abs_value, kBigP2);
    big = std::fma(big, abs_value, kBigP3);
    big = std::fma(big, abs_value, kBigP4);
    big = std::fma(big, abs_value, kBigP5);
    big = std::fma(big, abs_value, kBigP6MinusOne);
    big = std::fma(big, abs_value, abs_value);
    big = -big;
    big = std::max(big, kExpLowerRange);
    float r = std::fma(kExpLog2Reciprocal, big, kExpBias);
    r -= kExpBias;
    float fx = std::fma(r, kExpLog2High, big);
    fx = std::fma(r, kExpLog2Low, fx);
    float y = kExpP0;
    y = std::fma(y, fx, kExpP1);
    y = std::fma(y, fx, kExpP2);
    y = std::fma(y, fx, kExpP3);
    y = std::fma(y, fx, kExpP4);
    y = std::fma(y, fx, kExpP5);
    y = std::fma(y, fx, kExpP6);
    const std::uint32_t exponent = static_cast<std::uint32_t>(static_cast<std::int32_t>(r)) + kExponentBias;
    y *= std::bit_cast<float>(exponent << 23U);
    y = 1.0F - y;
    const std::uint32_t merged = std::bit_cast<std::uint32_t>(small) | std::bit_cast<std::uint32_t>(y);
    return std::bit_cast<float>(merged | sign);
}

// MLAS's Exp kernel (MlasComputeExpF32KernelFma3, TransKernelFma3.asm):
// clamp to [LowerRange, UpperRange], Eigen's rational range reduction with the
// magic rounding bias, the six polynomial terms, then the two-power scale that
// also reproduces subnormal results.

[[nodiscard]] inline float mlas_exp(float value) {
    constexpr float kLower = -103.9720840454F;
    constexpr float kUpper = 88.7762626647950F;
    constexpr float kRoundingBias = 12582912.0F;
    constexpr float kLog2Reciprocal = 1.44269504088896341F;
    constexpr float kLog2High = -6.93145752e-1F;
    constexpr float kLog2Low = -1.42860677e-6F;
    constexpr float kPoly0 = 0x1.694000p-10F;
    constexpr float kPoly1 = 0x1.125edcp-7F;
    constexpr float kPoly2 = 0x1.555b5ap-5F;
    constexpr float kPoly3 = 0x1.555450p-3F;
    constexpr float kPoly4 = 0x1.fffff6p-2F;
    constexpr float kPoly56 = 1.0F;
    constexpr std::int32_t kMinExponent = static_cast<std::int32_t>(0xC1000000U);
    constexpr std::int32_t kMaxExponent = static_cast<std::int32_t>(0x3F800000U);

    float x = std::max(kLower, std::min(kUpper, value));
    const float biased = std::fma(x, kLog2Reciprocal, kRoundingBias);
    const float m = biased - kRoundingBias;
    x = std::fma(m, kLog2High, x);
    x = std::fma(m, kLog2Low, x);
    float p = kPoly0;
    p = std::fma(p, x, kPoly1);
    p = std::fma(p, x, kPoly2);
    p = std::fma(p, x, kPoly3);
    p = std::fma(p, x, kPoly4);
    p = std::fma(p, x, kPoly56);

    const std::uint32_t exponent = std::bit_cast<std::uint32_t>(biased) << 23U;
    std::int32_t normal = std::min(static_cast<std::int32_t>(exponent), kMaxExponent);
    normal = std::max(normal, kMinExponent);
    const std::uint32_t overflow = exponent - static_cast<std::uint32_t>(normal);
    const std::uint32_t normal_scale = static_cast<std::uint32_t>(normal) + 0x3F800000U;
    const std::uint32_t overflow_scale = overflow + 0x3F800000U;
    const float overflow_value = std::bit_cast<float>(overflow_scale);
    p = std::fma(p, x * overflow_value, overflow_value);
    return p * std::bit_cast<float>(normal_scale);
}

// MlasGemm's non-packed K blocking (sgemm.cpp, MlasSgemmOperation): StrideK
// starts at MLAS_SGEMM_STRIDEK = 128, shrinks while N >= K, and doubles (with
// N halving) while N < K. Used for the attention MatMuls whose right operand
// is a runtime tensor instead of a prepacked constant weight.

[[nodiscard]] inline float mlas_sumexp_term(float value) {
    constexpr float kLowerRangeSumExp = -88.3762626647949F;
    constexpr float kRoundingBias = 12582912.0F;  // MLAS_ROUNDING_BIAS_MAGIC
    constexpr float kLog2Reciprocal = 1.44269504088896341F;
    constexpr float kLog2High = -6.93145752e-1F;
    constexpr float kLog2Low = -1.42860677e-6F;
    constexpr float kPoly0 = 0x1.694000p-10F;
    constexpr float kPoly1 = 0x1.125edcp-7F;
    constexpr float kPoly2 = 0x1.555b5ap-5F;
    constexpr float kPoly3 = 0x1.555450p-3F;
    constexpr float kPoly4 = 0x1.fffff6p-2F;
    constexpr float kPoly56 = 1.0F;
    value = std::max(kLowerRangeSumExp, value);
    const float biased = std::fma(value, kLog2Reciprocal, kRoundingBias);
    const float m = biased - kRoundingBias;
    float x = std::fma(m, kLog2High, value);
    x = std::fma(m, kLog2Low, x);
    float p = std::fma(kPoly0, x, kPoly1);
    p = std::fma(p, x, kPoly2);
    p = std::fma(p, x, kPoly3);
    p = std::fma(p, x, kPoly4);
    p = std::fma(p, x, kPoly56);
    p = std::fma(p, x, kPoly56);
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(biased);
    const std::uint32_t exponent = (bits << 23U) + 0x3F800000U;
    return p * std::bit_cast<float>(exponent);
}

// SoftmaxCPU -> MlasComputeSoftmax for float: an exact maximum, the
// SumExpF32 kernel's 8-lane accumulation with the horizontal reduction
// ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), then exp * (1/sum).

inline void mlas_softmax_row(const float* input, float* output, std::size_t n) {
    float maximum = input[0];
    for (std::size_t i = 1; i < n; ++i) {
        maximum = std::max(maximum, input[i]);
    }
    float lanes[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::size_t i = 0; i < n; ++i) {
        const float term = mlas_sumexp_term(input[i] - maximum);
        output[i] = term;
        lanes[i % 8] += term;
    }
    const float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    const float scale = 1.0F / sum;
    for (std::size_t i = 0; i < n; ++i) {
        output[i] *= scale;
    }
}

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
