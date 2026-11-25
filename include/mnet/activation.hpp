// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_ACTIVATION_HPP
#define MNET_ACTIVATION_HPP

#include <cmath>
#include <cstdint>

#include <algorithm>

#include <mnet/mat.hpp>

namespace mnet {

// Fused activation used by Convolution / InnerProduct. Mirrors the ncnn
// activation_type ABI: 0 none, 1 relu, 2 leaky, 3 clip, 4 sigmoid, 5 mish,
// 6 hardswish.
inline float activation_ss(float v, std::int32_t activation_type, const Mat& params) {
    switch (activation_type) {
        case 1:
            return std::max(v, 0.0F);
        case 2: {
            const float slope = params[0];
            return v > 0.0F ? v : v * slope;
        }
        case 3: {
            const float min_value = params[0];
            const float max_value = params[1];
            v = std::min(v, max_value);
            return std::max(v, min_value);
        }
        case 4: {
            v = std::min(v, 88.3762626647949F);
            v = std::max(v, -88.3762626647949F);
            return 1.0F / (1.0F + std::exp(-v));
        }
        case 5:
            return v * std::tanh(std::log(std::exp(v) + 1.0F));
        case 6: {
            const float alpha = params[0];
            const float beta = params[1];
            const float lower = -beta / alpha;
            const float upper = (1.0F / alpha) + lower;
            if (v < lower) {
                return 0.0F;
            }
            if (v > upper) {
                return v;
            }
            return v * (v * alpha + beta);
        }
        default:
            return v;
    }
}

}  // namespace mnet

#endif  // MNET_ACTIVATION_HPP
