// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/gguf/format.hpp>

#include <cmath>
#include <cstring>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include <gguf/internal.hpp>

namespace mnet::gguf {

bool& ggml_tinyblas_batch() {
    static bool batch = false;
    return batch;
}

[[nodiscard]] TypeInfo type_info(std::uint32_t type) {
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

[[nodiscard]] int nearest_int(float value) {
    // The barrier keeps the compiler from contracting the caller's `x * id` into
    // an fma with the +12582912 below: that would round the product+offset once
    // instead of rounding the product first, which flips exact .5 ties.
    volatile float rounded_input = value;
    const float val = rounded_input + 12582912.0F;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(bits));
    return static_cast<int>(bits & 0x007FFFFFU) - 0x00400000;
}

[[nodiscard]] std::uint16_t float_to_half(float value) {
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

[[nodiscard]] std::vector<float> quantize_q8_k(const float* input, std::size_t columns) {
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

[[nodiscard]] std::vector<float> quantize_q8_0(const float* input, std::size_t columns) {
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

[[nodiscard]] std::vector<float> round_bf16_input(const float* input, std::size_t columns) {
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

[[nodiscard]] bool uses_q8_k(std::uint32_t type) {
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

[[nodiscard]] bool uses_q8_0(std::uint32_t type) {
    return type == kGgmlQ8_0 || type == kGgmlNvfp4 || type == kGgmlQ2_0;
}

[[nodiscard]] Q8KBlocks make_q8_k(const float* input, std::size_t columns) {
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

[[nodiscard]] Q8_0Blocks make_q8_0_blocks(const float* input, std::size_t columns) {
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

[[nodiscard]] std::uint16_t float_to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 64U);
    }
    return static_cast<std::uint16_t>((bits + (0x7FFFU + ((bits >> 16U) & 1U))) >> 16U);
}

[[nodiscard]] float bf16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

[[nodiscard]] float half_to_float(std::uint16_t bits) {
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

[[nodiscard]] float e2m1_to_float(std::uint8_t nibble) {
    const float magnitude = kE2m1Magnitudes[nibble & 0x7U];
    return (nibble & 0x8U) != 0 ? -magnitude : magnitude;
}

[[nodiscard]] float ue4m3_to_float(std::uint8_t bits) {
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

void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    const auto byte = [q](int index) { return static_cast<std::uint32_t>(q[index]); };
    if (j < 4) {
        d = static_cast<std::uint8_t>(byte(j) & 63U);
        m = static_cast<std::uint8_t>(byte(j + 4) & 63U);
    } else {
        d = static_cast<std::uint8_t>((byte(j + 4) & 0x0FU) | ((byte(j - 4) >> 6U) << 4U));
        m = static_cast<std::uint8_t>((byte(j + 4) >> 4U) | ((byte(j) >> 6U) << 4U));
    }
}

std::uint64_t TensorInfo::n_bytes() const {
    const auto info = type_info(this->type);
    if (!info.supported || this->n_elements % info.block_elements != 0) {
        return 0;
    }
    return this->n_elements / info.block_elements * info.block_bytes;
}

}  // namespace mnet::gguf
