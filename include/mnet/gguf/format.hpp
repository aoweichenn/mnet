// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_GGUF_FORMAT_HPP
#define MNET_GGUF_FORMAT_HPP

// GGUF container value types, tensor descriptors, the fp16/bf16/ue4m3
// conversions and the Q8 activation quantisers. Only declarations live here;
// src/gguf/format.cpp holds the bodies.

#include <cstddef>
#include <cstdint>

#include <string>
#include <string_view>
#include <vector>

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
inline constexpr float kE2m1Magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

struct TensorInfo {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> dims;  // fastest-first, as stored by GGUF
    std::uint64_t offset = 0;
    std::uint64_t n_elements = 0;

    [[nodiscard]] std::uint64_t n_bytes() const;
};

struct Q8KBlocks {
    std::vector<std::int8_t> qs;
    std::vector<float> d;
    std::vector<std::int16_t> bsums;
};

struct Q8_0Blocks {
    std::vector<std::int8_t> qs;
    std::vector<float> d;
};

// --- conversions and activation quantisers -------------------------------

[[nodiscard]] TypeInfo type_info(std::uint32_t type);
[[nodiscard]] float half_to_float(std::uint16_t bits);
[[nodiscard]] float ue4m3_to_float(std::uint8_t bits);
[[nodiscard]] float e2m1_to_float(std::uint8_t nibble);
[[nodiscard]] std::int32_t nearest_int(float value);
[[nodiscard]] std::uint16_t float_to_half(float value);
[[nodiscard]] std::uint16_t float_to_bf16_bits(float value);
[[nodiscard]] float bf16_bits_to_float(std::uint16_t bits);
[[nodiscard]] std::vector<float> quantize_q8_k(const float* input, std::size_t columns);
[[nodiscard]] std::vector<float> quantize_q8_0(const float* input, std::size_t columns);
[[nodiscard]] std::vector<float> round_bf16_input(const float* input, std::size_t columns);
[[nodiscard]] Q8KBlocks make_q8_k(const float* input, std::size_t columns);
[[nodiscard]] Q8_0Blocks make_q8_0_blocks(const float* input, std::size_t columns);
[[nodiscard]] bool uses_q8_k(std::uint32_t type);
[[nodiscard]] bool uses_q8_0(std::uint32_t type);
[[nodiscard]] bool& ggml_tinyblas_batch();

}  // namespace mnet::gguf

#endif  // MNET_GGUF_FORMAT_HPP
