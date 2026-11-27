// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_GGUF_KERNELS_HPP
#define MNET_GGUF_KERNELS_HPP

// The vec_dot kernels used by Reader::matvec. Each one has a single entry
// point; the AVX2/FMA body and the lane-identical scalar body live in
// src/gguf/kernels.cpp, so this header only declares the contract.

#include <cstdint>

#include <mnet/gguf/format.hpp>

namespace mnet::gguf {

[[nodiscard]] inline bool has_int_q8k_dot(std::uint32_t type) {
    return type == kGgmlQ4_K || type == kGgmlQ5_K || type == kGgmlQ6_K || type == kGgmlIq1S || type == kGgmlIq2Xxs ||
           type == kGgmlIq2Xs || type == kGgmlIq2S || type == kGgmlIq3Xxs || type == kGgmlIq4Xs;
}

[[nodiscard]] inline bool has_int_q80_dot(std::uint32_t type) {
    return type == kGgmlQ2_0 || type == kGgmlQ8_0 || type == kGgmlNvfp4;
}

[[nodiscard]] float dot_q6_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq1_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq2_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq2_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq2_s_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq3_xxs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_iq4_xs_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_q4_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_q5_k_q8_k(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_q4_k_q8_k_8x8(const std::uint8_t* data, const Q8KBlocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_q2_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_q8_0_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_nvfp4_q8_0(const std::uint8_t* data, const Q8_0Blocks& q8, std::uint64_t blocks);
[[nodiscard]] float dot_f32_reference(const float* x, const float* y, std::uint64_t size);
[[nodiscard]] float dot_f32_reference_avx2(const float* x, const float* y, std::uint64_t size);
[[nodiscard]] float dot_bf16_bf16_reference(const std::uint8_t* data, const std::uint16_t* input,
                                            std::uint64_t columns);
[[nodiscard]] float dot_bf16_bf16_avx2(const std::uint8_t* data, const std::uint16_t* input, std::uint64_t columns);
[[nodiscard]] float dot_f32_tinyblas(const float* a, const float* b, std::uint64_t size);
[[nodiscard]] float dot_bf16_tinyblas(const std::uint8_t* data, const std::uint16_t* input, std::uint64_t size);

}  // namespace mnet::gguf

#endif  // MNET_GGUF_KERNELS_HPP
