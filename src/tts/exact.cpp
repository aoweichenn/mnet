// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <bit>
#include <thread>
#include <utility>
#include <vector>

#include <mnet/tts/tts_exact.hpp>

namespace mnet::tts {

[[nodiscard]] EigenFrexp eigen_frexp(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t biased = (bits >> 23U) & 0xFFU;
    return {std::bit_cast<float>((bits & 0x807FFFFFU) | 0x3F000000U),
            static_cast<float>(static_cast<std::int32_t>(biased) - 126)};
}

[[nodiscard]] float eigen_log_positive(float value) {
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

[[nodiscard]] float eigen_exp(float value) {
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

[[nodiscard]] float eigen_log1p(float value) {
    const float xp1 = value + 1.0F;
    const float logarithm = eigen_log_positive(xp1);
    if (xp1 == 1.0F || xp1 == logarithm) {
        return value;
    }
    return value * (logarithm / (xp1 - 1.0F));
}

[[nodiscard]] bool eigen_scalar_tail(std::size_t index, std::size_t count) {
    constexpr std::size_t kPacket = 4;
    const std::size_t tail = count % kPacket;
    return tail != 0 && index + tail >= count;
}

[[nodiscard]] float eigen_softplus(float value) {
    if (value > 0.0F) {
        return value + eigen_log1p(eigen_exp(-value));
    }
    return eigen_log1p(eigen_exp(value));
}

[[nodiscard]] float eigen_softplus_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        // Eigen's scalar fallback: (x > 0) ? x + log1p(exp(-x)) : log1p(exp(x)).
        return value > 0.0F ? value + std::log1p(std::exp(-value)) : std::log1p(std::exp(value));
    }
    return eigen_softplus(value);
}

[[nodiscard]] float eigen_trig_reduce_huge(float value, std::uint32_t& quadrant) {
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

[[nodiscard]] float eigen_sin(float value) {
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

[[nodiscard]] float eigen_sin_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        return std::sin(value);
    }
    return eigen_sin(value);
}

[[nodiscard]] float eigen_cos(float value) {
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

[[nodiscard]] float eigen_cos_element(float value, std::size_t index, std::size_t count) {
    if (eigen_scalar_tail(index, count)) {
        return std::cos(value);
    }
    return eigen_cos(value);
}

[[nodiscard]] float eigen_atan(float value) {
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

[[nodiscard]] float eigen_fast_tanh(float value) {
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

[[nodiscard]] float ort_logistic(float value) {
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

[[nodiscard]] float ort_tanh(float value) {
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

[[nodiscard]] float mlas_erf(float value) {
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

[[nodiscard]] float mlas_exp(float value) {
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

[[nodiscard]] float mlas_sumexp_term(float value) {
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

void mlas_softmax_row(const float* input, float* output, std::size_t n) {
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

}  // namespace mnet::tts
