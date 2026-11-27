// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/gguf/reader.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gguf/internal.hpp>
#include <mnet/detail/compute.hpp>
#include <mnet/detail/iq1s_grid.hpp>
#include <mnet/detail/iq_grids.hpp>
#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>

namespace mnet::gguf {

namespace {

void dequantize_q6_k(const std::uint8_t* data, float* out) {
    const auto* ql = data;
    const auto* qh = data + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(data + 192);
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[208] | (static_cast<std::uint16_t>(data[209]) << 8U));
    const float d = half_to_float(dBits);
    float* y = out;
    for (std::int32_t n = 0; n < 256; n += 128) {
        for (std::int32_t l = 0; l < 32; ++l) {
            const std::int32_t is = l / 16;
            const std::int32_t q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const std::int32_t q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const std::int32_t q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const std::int32_t q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * static_cast<float>(scales[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        scales += 8;
    }
}

void dequantize_q4_k(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const std::uint16_t minBits = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const float d = half_to_float(dBits);
    const float min = half_to_float(minBits);
    const std::uint8_t* scales = data + 4;
    const std::uint8_t* q = data + 16;
    std::int32_t is = 0;
    for (std::int32_t j = 0; j < 256; j += 64) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (std::int32_t l = 0; l < 32; ++l) {
            *out++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
        }
        for (std::int32_t l = 0; l < 32; ++l) {
            *out++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

void dequantize_q5_k(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const std::uint16_t minBits = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const float d = half_to_float(dBits);
    const float min = half_to_float(minBits);
    const std::uint8_t* scales = data + 4;
    const std::uint8_t* qh = data + 16;
    const std::uint8_t* ql = data + 48;
    std::int32_t is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (std::int32_t j = 0; j < 256; j += 64) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(m);
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(m);
        for (std::int32_t l = 0; l < 32; ++l) {
            *out++ = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
        }
        for (std::int32_t l = 0; l < 32; ++l) {
            *out++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

void dequantize_iq2_xxs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* q = data + 2;
    for (std::int32_t ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t lo =
            static_cast<std::uint32_t>(q[8 * ib32]) | (static_cast<std::uint32_t>(q[8 * ib32 + 1]) << 8U) |
            (static_cast<std::uint32_t>(q[8 * ib32 + 2]) << 16U) | (static_cast<std::uint32_t>(q[8 * ib32 + 3]) << 24U);
        const std::uint32_t hi =
            static_cast<std::uint32_t>(q[8 * ib32 + 4]) | (static_cast<std::uint32_t>(q[8 * ib32 + 5]) << 8U) |
            (static_cast<std::uint32_t>(q[8 * ib32 + 6]) << 16U) | (static_cast<std::uint32_t>(q[8 * ib32 + 7]) << 24U);
        const float db = d * (0.5F + static_cast<float>(hi >> 28U)) * 0.25F;
        for (std::int32_t l = 0; l < 4; ++l) {
            const std::uint64_t grid = detail::kIq2xxsGrid[(lo >> (8 * l)) & 0xFFU];
            const std::uint8_t signs = detail::kSignsIq2xs[(hi >> (7 * l)) & 127U];
            for (std::int32_t j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

void dequantize_iq2_xs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* scales = data + 66;
    for (std::int32_t ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F;
        const float db1 = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (std::int32_t l = 0; l < 4; ++l) {
            const std::uint16_t qv = static_cast<std::uint16_t>(
                qs[2 * (4 * ib32 + l)] | (static_cast<std::uint16_t>(qs[2 * (4 * ib32 + l) + 1]) << 8U));
            const std::uint64_t grid = detail::kIq2xsGrid[qv & 511U];
            const std::uint8_t signs = detail::kSignsIq2xs[qv >> 9U];
            const float db = l < 2 ? db0 : db1;
            for (std::int32_t j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

void dequantize_iq3_xxs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* scales_and_signs = qs + 64;
    for (std::int32_t ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t aux = static_cast<std::uint32_t>(scales_and_signs[4 * ib32]) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 1]) << 8U) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 2]) << 16U) |
                                  (static_cast<std::uint32_t>(scales_and_signs[4 * ib32 + 3]) << 24U);
        const float db = d * (0.5F + static_cast<float>(aux >> 28U)) * 0.5F;
        for (std::int32_t l = 0; l < 4; ++l) {
            const std::uint8_t signs = detail::kSignsIq2xs[(aux >> (7 * l)) & 127U];
            const std::uint32_t grid1 = detail::kIq3xxsGrid[qs[8 * ib32 + 2 * l]];
            const std::uint32_t grid2 = detail::kIq3xxsGrid[qs[8 * ib32 + 2 * l + 1]];
            for (std::int32_t j = 0; j < 4; ++j) {
                *out++ =
                    db * static_cast<float>((grid1 >> (8 * j)) & 0xFFU) * ((signs & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
            for (std::int32_t j = 0; j < 4; ++j) {
                *out++ = db * static_cast<float>((grid2 >> (8 * j)) & 0xFFU) *
                         ((signs & (1U << (j + 4))) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

void dequantize_iq2_s(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const std::uint8_t* qh = data + 66;
    const std::uint8_t* scales = data + 74;
    const std::uint8_t* signs = qs + 32;
    for (std::int32_t ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F;
        const float db1 = d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F;
        for (std::int32_t l = 0; l < 4; ++l) {
            const std::uint32_t index = static_cast<std::uint32_t>(qs[4 * ib32 + l]) |
                                        ((static_cast<std::uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300U);
            const std::uint64_t grid = detail::kIq2sGrid[index];
            const std::uint8_t sign = signs[4 * ib32 + l];
            const float db = l < 2 ? db0 : db1;
            for (std::int32_t j = 0; j < 8; ++j) {
                *out++ = db * static_cast<float>((grid >> (8 * j)) & 0xFFU) * ((sign & (1U << j)) != 0 ? -1.0F : 1.0F);
            }
        }
    }
}

void dequantize_iq1_s(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint8_t* qs = data + 2;
    const auto* qh = reinterpret_cast<const std::uint16_t*>(data + 34);
    for (std::int32_t ib = 0; ib < 8; ++ib) {
        const std::uint16_t high = qh[ib];
        const float dl = d * static_cast<float>(2 * ((high >> 12) & 7) + 1);
        const float delta = (high & 0x8000U) != 0 ? -kIq1sDelta : kIq1sDelta;
        for (std::int32_t l = 0; l < 4; ++l) {
            const std::uint16_t index = static_cast<std::uint16_t>(qs[ib * 4 + l] | (((high >> (3 * l)) & 7) << 8));
            const std::uint64_t grid = detail::kIq1sGrid[index];
            for (std::int32_t j = 0; j < 8; ++j) {
                const auto value = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFU);
                out[ib * 32 + l * 8 + j] = dl * (static_cast<float>(value) + delta);
            }
        }
    }
}

void dequantize_iq4_xs(const std::uint8_t* data, float* out) {
    const std::uint16_t dBits = static_cast<std::uint16_t>(data[0] | (static_cast<std::uint16_t>(data[1]) << 8U));
    const float d = half_to_float(dBits);
    const std::uint16_t scales_h = static_cast<std::uint16_t>(data[2] | (static_cast<std::uint16_t>(data[3]) << 8U));
    const std::uint8_t* scales_l = data + 4;
    const std::uint8_t* qs = data + 8;
    for (std::int32_t ib = 0; ib < 8; ++ib) {
        const std::int32_t ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (std::int32_t j = 0; j < 16; ++j) {
            out[j] = dl * static_cast<float>(kIq4NlValues[qs[j] & 0x0F]);
            out[j + 16] = dl * static_cast<float>(kIq4NlValues[qs[j] >> 4]);
        }
        out += 32;
        qs += 16;
    }
}

}  // namespace

Reader::Reader(std::string path) : path_(std::move(path)) {
    this->file_.open(this->path_, std::ios::binary);
    if (!this->file_) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("cannot open " + this->path_);
    }
    this->parse_header();
}

[[nodiscard]] std::vector<std::int32_t> Reader::signs_for_width(std::uint32_t width) const {
    const auto widths = this->int_arrays_.find("prism.hadamard.sign_widths");
    const auto values = this->int_arrays_.find("prism.hadamard.sign_values");
    if (widths == this->int_arrays_.end() || values == this->int_arrays_.end()) {
        return {};
    }
    std::size_t offset = 0;
    for (const std::int64_t candidate : widths->second) {
        if (candidate < 0) {
            return {};
        }
        const auto size = static_cast<std::size_t>(candidate);
        if (candidate == static_cast<std::int64_t>(width)) {
            if (offset + size > values->second.size()) {
                return {};
            }
            std::vector<std::int32_t> signs(size);
            for (std::size_t i = 0; i < size; ++i) {
                signs[i] = static_cast<std::int32_t>(values->second[offset + i]);
            }
            return signs;
        }
        offset += size;
    }
    return {};
}

[[nodiscard]] const TensorInfo& Reader::tensor(std::string_view name) const {
    for (const auto& item : this->tensors_) {
        if (item.name == name) {
            return item;
        }
    }
    // check-no-throw: allow invalid model input
    throw std::runtime_error("tensor not found: " + std::string(name));
}

[[nodiscard]] std::vector<std::uint8_t> Reader::read_raw(const TensorInfo& info, std::uint64_t byte_offset,
                                                         std::uint64_t byte_count) const {
    const std::uint64_t size = byte_count == 0 ? info.n_bytes() : byte_count;
    if (size == 0) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("unsupported or empty tensor: " +
                                 // check-no-throw: allow invalid model input
                                 info.name);
    }
    std::ifstream file(this->path_, std::ios::binary);
    if (!file) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("cannot open " + this->path_);
    }
    file.seekg(static_cast<std::streamoff>(this->data_start_ + info.offset + byte_offset));
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!file) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("short read for tensor: " + info.name);
    }
    return bytes;
}

[[nodiscard]] std::vector<float> Reader::dequantize(const TensorInfo& info, std::int32_t expert) const {
    const auto type = type_info(info.type);
    if (!type.supported) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("unsupported GGML type for tensor: " +
                                 // check-no-throw: allow invalid model input
                                 info.name);
    }
    std::uint64_t count = info.n_elements;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    if (info.dims.size() == 3) {
        if (expert < 0 || static_cast<std::uint64_t>(expert) >= info.dims[2]) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error("expert index required for 3-D tensor: " +
                                     // check-no-throw: allow invalid model input
                                     info.name);
        }
        const std::uint64_t per_expert = info.dims[0] * info.dims[1];
        if (per_expert % type.block_elements != 0) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error("expert slice is not block aligned: " +
                                     // check-no-throw: allow invalid model input
                                     info.name);
        }
        const std::uint64_t per_expert_bytes = per_expert / type.block_elements * type.block_bytes;
        byte_offset = static_cast<std::uint64_t>(expert) * per_expert_bytes;
        byte_count = per_expert_bytes;
        count = per_expert;
    }
    const std::vector<std::uint8_t> raw = this->read_raw(info, byte_offset, byte_count);
    std::vector<float> values = dequantize_block(info.type, raw, count);
    if (info.dims.size() == 3 && expert >= 0) {
        const std::string base = info.name.ends_with(".weight") ? info.name.substr(0, info.name.size() - 7) : info.name;
        const std::string scale_name = base + ".scale";
        for (const auto& candidate : this->tensors_) {
            if (candidate.name == scale_name && candidate.n_elements > static_cast<std::uint64_t>(expert)) {
                const auto scale_bytes = this->read_raw(candidate);
                const auto* scales = reinterpret_cast<const float*>(scale_bytes.data());
                const float factor = scales[expert];
                for (float& value : values) {
                    value *= factor;
                }
                break;
            }
        }
    }
    return values;
}

[[nodiscard]] Reader::Quantized Reader::quantized(const TensorInfo& info, std::int32_t expert) const {
    const auto type = type_info(info.type);
    if (!type.supported) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("unsupported GGML type for tensor: " +
                                 // check-no-throw: allow invalid model input
                                 info.name);
    }
    Quantized result;
    // Identity is derived from (reader, tensor offset, expert) instead of a
    // running counter: reloading the same tensor must hit the device cache
    // again, and two tensors must never share it.
    const auto salt = reinterpret_cast<std::uintptr_t>(this);
    result.identity = (info.offset + 0x9E3779B97F4A7C15ULL) * 0x100000001B3ULL ^
                      (static_cast<std::uint64_t>(expert + 1) * 0xC2B2AE3D27D4EB4FULL) ^ salt;
    result.type = info.type;
    result.rows = info.dims.size() >= 2 ? info.dims[1] : 1;
    result.columns = info.dims[0];
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    if (info.dims.size() == 3) {
        if (expert < 0 || static_cast<std::uint64_t>(expert) >= info.dims[2]) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error("expert index required for 3-D tensor: " +
                                     // check-no-throw: allow invalid model input
                                     info.name);
        }
        const std::uint64_t per_expert = info.dims[0] * info.dims[1];
        if (per_expert % type.block_elements != 0) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error("expert slice is not block aligned: " +
                                     // check-no-throw: allow invalid model input
                                     info.name);
        }
        const std::uint64_t per_expert_bytes = per_expert / type.block_elements * type.block_bytes;
        byte_offset = static_cast<std::uint64_t>(expert) * per_expert_bytes;
        byte_count = per_expert_bytes;
        result.rows = info.dims[1];
        const std::string base = info.name.ends_with(".weight") ? info.name.substr(0, info.name.size() - 7) : info.name;
        const std::string scale_name = base + ".scale";
        for (const auto& candidate : this->tensors_) {
            if (candidate.name == scale_name && candidate.n_elements > static_cast<std::uint64_t>(expert)) {
                const auto scale_bytes = this->read_raw(candidate);
                result.scale = reinterpret_cast<const float*>(scale_bytes.data())[expert];
                break;
            }
        }
    }
    result.blocks = this->read_raw(info, byte_offset, byte_count);
    return result;
}

void Reader::matvec(const Quantized& matrix, const float* input, float* output) {
    const auto info = type_info(matrix.type);
    if (matrix.columns % info.block_elements != 0) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("quantized row is not block aligned");
    }
    const std::size_t row_bytes = matrix.columns / info.block_elements * info.block_bytes;
    // Mirror llama.cpp: quantize the matmul input to the weight's
    // vec_dot_type before the dot product.
    Q8KBlocks q8k;
    Q8_0Blocks q80;
    std::vector<float> converted;
    const float* source = input;
    const bool int_q8k = uses_q8_k(matrix.type) && has_int_q8k_dot(matrix.type);
    const bool int_q80 = uses_q8_0(matrix.type) && has_int_q80_dot(matrix.type);
    if (int_q8k) {
        q8k = make_q8_k(input, matrix.columns);
    } else if (int_q80) {
        q80 = make_q8_0_blocks(input, matrix.columns);
    } else if (uses_q8_k(matrix.type)) {
        converted = quantize_q8_k(input, matrix.columns);
        source = converted.data();
    } else if (uses_q8_0(matrix.type)) {
        converted = quantize_q8_0(input, matrix.columns);
        source = converted.data();
    } else if (matrix.type == kGgmlBf16) {
        converted = round_bf16_input(input, matrix.columns);
        source = converted.data();
    }
    const std::uint64_t blocks = matrix.columns / info.block_elements;
    if (int_q80) {
        // Device-neutral: the compute layer uses a GPU when one is registered,
        // otherwise it runs the same CPU loop the reader used before.
        mnet::detail::compute::matvec_q8_0(matrix.type, matrix.blocks.data(), matrix.rows, matrix.columns, row_bytes,
                                           matrix.scale, q80.qs.data(), q80.d.data(), output, matrix.identity);
        return;
    }
    // Rows are independent and keep a fixed accumulation order, so the
    // result is identical for any thread count.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (matrix.rows >= 128)
#endif
    for (std::uint64_t row = 0; row < matrix.rows; ++row) {
        const std::uint8_t* data = matrix.blocks.data() + row * row_bytes;
        float value = 0.0F;
        if (int_q8k) {
            switch (matrix.type) {
                case kGgmlQ4_K:
                    value =
                        matrix.rows % 8 == 0 ? dot_q4_k_q8_k_8x8(data, q8k, blocks) : dot_q4_k_q8_k(data, q8k, blocks);
                    break;
                case kGgmlQ5_K:
                    value = dot_q5_k_q8_k(data, q8k, blocks);
                    break;
                case kGgmlQ6_K:
                    value = dot_q6_k_q8_k(data, q8k, blocks);
                    break;
                case kGgmlIq1S:
                    value = dot_iq1_s_q8_k(data, q8k, blocks);
                    break;
                case kGgmlIq2Xxs:
                    value = dot_iq2_xxs_q8_k(data, q8k, blocks);
                    break;
                case kGgmlIq2Xs:
                    value = dot_iq2_xs_q8_k(data, q8k, blocks);
                    break;
                case kGgmlIq2S:
                    value = dot_iq2_s_q8_k(data, q8k, blocks);
                    break;
                case kGgmlIq3Xxs:
                    value = dot_iq3_xxs_q8_k(data, q8k, blocks);
                    break;
                default:
                    value = dot_iq4_xs_q8_k(data, q8k, blocks);
                    break;
            }
        } else {
            value = dot_row(matrix.type, data, source, matrix.columns);
        }
        output[row] = matrix.scale * value;
    }
}

void Reader::matvec_batch(std::initializer_list<std::pair<const Quantized*, float*>> jobs, const float* input) {
    if (jobs.size() == 0) {
        return;
    }
    if (std::getenv("MNET_NO_BATCH") != nullptr) {
        for (const auto& job : jobs) {
            matvec(*job.first, input, job.second);
        }
        return;
    }
    const Quantized* first = jobs.begin()->first;
    const auto info = type_info(first->type);
    const bool q8_0_family = uses_q8_0(first->type) && has_int_q80_dot(first->type);
    bool compatible = q8_0_family;
    for (const auto& job : jobs) {
        compatible = compatible && job.first->type == first->type && job.first->columns == first->columns &&
                     uses_q8_0(job.first->type) && has_int_q80_dot(job.first->type) &&
                     job.first->columns % info.block_elements == 0;
    }
    if (!compatible) {
        for (const auto& job : jobs) {
            matvec(*job.first, input, job.second);
        }
        return;
    }

    const std::uint64_t row_bytes = first->columns / info.block_elements * info.block_bytes;
    const auto q80 = make_q8_0_blocks(input, first->columns);
    std::vector<mnet::detail::compute::MatvecQ8_0Job> requests;
    requests.reserve(jobs.size());
    for (const auto& job : jobs) {
        requests.push_back({job.first->type, job.first->blocks.data(), job.first->rows, row_bytes, job.first->scale,
                            job.second, job.first->identity});
    }
    mnet::detail::compute::matvec_q8_0_batch(requests.data(), requests.size(), first->columns, q80.qs.data(),
                                             q80.d.data());
}

void Reader::matmul(const Quantized& matrix, const float* input, float* output, std::size_t tokens,
                    std::size_t input_stride, std::size_t output_stride) {
    for (std::size_t t = 0; t < tokens; ++t) {
        matvec(matrix, input + t * input_stride, output + t * output_stride);
    }
}

std::uint16_t Reader::read_u16(std::istream& in) {
    std::uint8_t b[2] = {};
    in.read(reinterpret_cast<char*>(b), 2);
    return static_cast<std::uint16_t>(b[0] | (static_cast<std::uint16_t>(b[1]) << 8));
}

std::uint32_t Reader::read_u32(std::istream& in) {
    std::uint8_t b[4] = {};
    in.read(reinterpret_cast<char*>(b), 4);
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t Reader::read_u64(std::istream& in) {
    std::uint64_t lo = read_u32(in);
    std::uint64_t hi = read_u32(in);
    return lo | (hi << 32);
}

std::string Reader::read_string(std::istream& in) {
    const std::uint64_t length = read_u64(in);
    std::string value(static_cast<std::size_t>(length), '\0');
    in.read(value.data(), static_cast<std::streamsize>(length));
    return value;
}

std::int64_t Reader::read_scalar_int(std::istream& in, std::uint32_t type) {
    switch (type) {
        case kTypeUint8:
            return static_cast<std::int64_t>(in.get());
        case kTypeInt8:
            return static_cast<std::int64_t>(static_cast<std::int8_t>(in.get()));
        case kTypeUint16:
            return static_cast<std::int64_t>(read_u16(in));
        case kTypeInt16:
            return static_cast<std::int64_t>(static_cast<std::int16_t>(read_u16(in)));
        case kTypeUint32:
            return static_cast<std::int64_t>(read_u32(in));
        case kTypeInt32:
            return static_cast<std::int64_t>(static_cast<std::int32_t>(read_u32(in)));
        case kTypeUint64:
            return static_cast<std::int64_t>(read_u64(in));
        case kTypeInt64:
            return static_cast<std::int64_t>(read_u64(in));
        case kTypeBool:
            return static_cast<std::int64_t>(in.get());
        default:
            // check-no-throw: allow invalid model input
            throw std::runtime_error("not an integer GGUF metadata type");
    }
}

void Reader::skip_value(std::istream& in, std::uint32_t type) {
    switch (type) {
        case kTypeString:
            static_cast<void>(read_string(in));
            return;
        case kTypeUint8:
        case kTypeInt8:
        case kTypeBool:
            in.seekg(1, std::ios::cur);
            return;
        case kTypeUint16:
        case kTypeInt16:
            in.seekg(2, std::ios::cur);
            return;
        case kTypeUint32:
        case kTypeInt32:
        case kTypeFloat32:
            in.seekg(4, std::ios::cur);
            return;
        case kTypeUint64:
        case kTypeInt64:
        case kTypeFloat64:
            in.seekg(8, std::ios::cur);
            return;
        case kTypeArray: {
            const std::uint32_t element_type = read_u32(in);
            const std::uint64_t count = read_u64(in);
            for (std::uint64_t i = 0; i < count; ++i) {
                skip_value(in, element_type);
            }
            return;
        }
        default:
            // check-no-throw: allow invalid model input
            throw std::runtime_error("unknown GGUF metadata type");
    }
}

void Reader::parse_header() {
    const std::uint32_t magic = read_u32(this->file_);
    if (magic != kGgufMagic) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("not a GGUF file");
    }
    this->version_ = read_u32(this->file_);
    const std::uint64_t tensor_count = read_u64(this->file_);
    const std::uint64_t kv_count = read_u64(this->file_);
    if (tensor_count > 1'000'000ULL || kv_count > 1'000'000ULL) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error("implausible GGUF header counts");
    }

    for (std::uint64_t i = 0; i < kv_count; ++i) {
        const std::string key = read_string(this->file_);
        const std::uint32_t type = read_u32(this->file_);
        if (type == kTypeString) {
            this->strings_[key] = read_string(this->file_);
        } else if (type == kTypeUint32 || type == kTypeInt32) {
            this->integers_[key] = read_u32(this->file_);
        } else if (type == kTypeUint64 || type == kTypeInt64) {
            this->integers_[key] = read_u64(this->file_);
        } else if (type == kTypeBool) {
            this->integers_[key] = static_cast<std::uint64_t>(this->file_.get());
        } else if (type == kTypeFloat32) {
            const std::uint32_t bits = read_u32(this->file_);
            float value = 0.0F;
            std::memcpy(&value, &bits, sizeof(value));
            this->floats_[key] = static_cast<double>(value);
        } else if (type == kTypeArray) {
            const std::uint32_t element_type = read_u32(this->file_);
            const std::uint64_t count = read_u64(this->file_);
            if (element_type == kTypeString) {
                auto& values = this->string_arrays_[key];
                values.reserve(static_cast<std::size_t>(count));
                for (std::uint64_t index = 0; index < count; ++index) {
                    values.push_back(read_string(this->file_));
                }
            } else if (element_type == kTypeUint8 || element_type == kTypeInt8 || element_type == kTypeBool ||
                       element_type == kTypeUint16 || element_type == kTypeInt16 || element_type == kTypeUint32 ||
                       element_type == kTypeInt32 || element_type == kTypeUint64 || element_type == kTypeInt64) {
                auto& values = this->int_arrays_[key];
                values.reserve(static_cast<std::size_t>(count));
                for (std::uint64_t index = 0; index < count; ++index) {
                    values.push_back(read_scalar_int(this->file_, element_type));
                }
            } else {
                for (std::uint64_t index = 0; index < count; ++index) {
                    skip_value(this->file_, element_type);
                }
            }
        } else {
            skip_value(this->file_, type);
        }
    }

    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        info.name = read_string(this->file_);
        const std::uint32_t n_dims = read_u32(this->file_);
        if (n_dims == 0 || n_dims > 4) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error("invalid tensor rank: " + info.name);
        }
        info.dims.resize(n_dims);
        info.n_elements = 1;
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            info.dims[d] = read_u64(this->file_);
            info.n_elements *= info.dims[d];
        }
        info.type = read_u32(this->file_);
        info.offset = read_u64(this->file_);
        this->tensors_.push_back(std::move(info));
    }

    const std::streampos header_end = this->file_.tellg();
    const std::uint64_t alignment =
        this->integers_.contains("general.alignment") ? this->integers_.at("general.alignment") : 32ULL;
    const std::uint64_t position = static_cast<std::uint64_t>(header_end);
    this->data_start_ = (position + alignment - 1) / alignment * alignment;
}

inline float Reader::dot_row(std::uint32_t type, const std::uint8_t* data, const float* input, std::uint64_t columns) {
    switch (type) {
        case kGgmlF32: {
            const auto* values = reinterpret_cast<const float*>(data);
            if (ggml_tinyblas_batch()) {
                return dot_f32_tinyblas(values, input, columns);
            }
#if defined(__AVX2__) && defined(__FMA__)
            return dot_f32_reference_avx2(values, input, columns);
#else
            return dot_f32_reference(values, input, columns);
#endif
        }
        case kGgmlF16: {
            const auto* values = reinterpret_cast<const std::uint16_t*>(data);
            double sum = 0.0;
            for (std::uint64_t i = 0; i < columns; ++i) {
                sum += static_cast<double>(half_to_float(values[i])) * static_cast<double>(input[i]);
            }
            return static_cast<float>(sum);
        }
        case kGgmlBf16: {
            // llama.cpp: the activation is converted to bf16 (vec_dot_type), then
            // ggml_vec_dot_bf16 runs on bf16 x bf16.
            static thread_local std::vector<std::uint16_t> converted;
            converted.resize(columns);
            for (std::uint64_t i = 0; i < columns; ++i) {
                converted[i] = float_to_bf16_bits(input[i]);
            }
            if (ggml_tinyblas_batch()) {
                return dot_bf16_tinyblas(data, converted.data(), columns);
            }
#if defined(__AVX2__) && defined(__FMA__)
            return dot_bf16_bf16_avx2(data, converted.data(), columns);
#else
            return dot_bf16_bf16_reference(data, converted.data(), columns);
#endif
        }
        case kGgmlQ8_0: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 32;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 34;
                const float scale =
                    half_to_float(static_cast<std::uint16_t>(base[0] | (static_cast<std::uint16_t>(base[1]) << 8U)));
                double block_sum = 0.0;
                for (std::uint64_t j = 0; j < 32; ++j) {
                    block_sum += static_cast<double>(static_cast<std::int8_t>(base[2 + j])) *
                                 static_cast<double>(input[b * 32 + j]);
                }
                sum += block_sum * static_cast<double>(scale);
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ6_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q6_k(data + b * 210, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq4Xs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq4_xs(data + b * 136, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ4_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q4_k(data + b * 144, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ5_K: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_q5_k(data + b * 176, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2Xxs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_xxs(data + b * 66, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2Xs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_xs(data + b * 74, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq3Xxs: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq3_xxs(data + b * 98, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq2S: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq2_s(data + b * 82, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlIq1S: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 256;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                float values[256];
                dequantize_iq1_s(data + b * 50, values);
                for (std::uint64_t j = 0; j < 256; ++j) {
                    sum += static_cast<double>(values[j]) * static_cast<double>(input[b * 256 + j]);
                }
            }
            return static_cast<float>(sum);
        }
        case kGgmlQ2_0: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 64;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 18;
                const float scale =
                    half_to_float(static_cast<std::uint16_t>(base[0] | (static_cast<std::uint16_t>(base[1]) << 8U)));
                double block_sum = 0.0;
                for (std::uint64_t j = 0; j < 64; ++j) {
                    const std::uint8_t byte = base[2 + j / 4];
                    const auto q = static_cast<std::int32_t>((byte >> ((j % 4) * 2U)) & 0x03U);
                    block_sum += static_cast<double>(q - 1) * static_cast<double>(input[b * 64 + j]);
                }
                sum += block_sum * static_cast<double>(scale);
            }
            return static_cast<float>(sum);
        }
        case kGgmlNvfp4: {
            double sum = 0.0;
            const std::uint64_t blocks = columns / 64;
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* base = data + b * 36;
                for (std::uint64_t s = 0; s < 4; ++s) {
                    const float scale = ue4m3_to_float(base[s]);
                    if (scale == 0.0F) {
                        continue;
                    }
                    double block_sum = 0.0;
                    const std::uint64_t offset = b * 64 + s * 16;
                    for (std::uint64_t j = 0; j < 8; ++j) {
                        const std::uint8_t byte = base[4 + s * 8 + j];
                        block_sum += static_cast<double>(e2m1_to_float(static_cast<std::uint8_t>(byte & 0x0FU))) *
                                     static_cast<double>(input[offset + j]);
                        block_sum += static_cast<double>(e2m1_to_float(static_cast<std::uint8_t>(byte >> 4U))) *
                                     static_cast<double>(input[offset + j + 8]);
                    }
                    sum += block_sum * static_cast<double>(scale);
                }
            }
            return static_cast<float>(sum);
        }
        default:
            // check-no-throw: allow invalid model input
            throw std::runtime_error("unsupported GGML type " +
                                     // check-no-throw: allow invalid model input
                                     std::to_string(type));
    }
}

inline std::vector<float> Reader::dequantize_block(std::uint32_t type, const std::vector<std::uint8_t>& raw,
                                                   std::uint64_t count) {
    std::vector<float> out(static_cast<std::size_t>(count), 0.0F);
    switch (type) {
        case kGgmlF32:
            std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
            return out;
        case kGgmlF16: {
            const auto* data = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = half_to_float(data[i]);
            }
            return out;
        }
        case kGgmlBf16: {
            const auto* data = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < out.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(data[i]) << 16U;
                std::memcpy(&out[i], &bits, sizeof(float));
            }
            return out;
        }
        case kGgmlQ8_0: {
            if (count % 32 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "Q8_0 element count not a multiple of 32");
            }
            const std::size_t blocks = out.size() / 32;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 34;
                const float scale = half_to_float(
                    static_cast<std::uint16_t>(raw[base] | (static_cast<std::uint16_t>(raw[base + 1]) << 8U)));
                for (std::size_t j = 0; j < 32; ++j) {
                    out[b * 32 + j] = static_cast<float>(static_cast<std::int8_t>(raw[base + 2 + j])) * scale;
                }
            }
            return out;
        }
        case kGgmlQ6_K: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "Q6_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q6_k(raw.data() + b * 210, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq4Xs: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ4_XS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq4_xs(raw.data() + b * 136, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ4_K: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "Q4_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q4_k(raw.data() + b * 144, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ5_K: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "Q5_K element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_q5_k(raw.data() + b * 176, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2Xxs: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ2_XXS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_xxs(raw.data() + b * 66, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2Xs: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ2_XS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_xs(raw.data() + b * 74, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq3Xxs: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ3_XXS element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq3_xxs(raw.data() + b * 98, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq2S: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ2_S element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq2_s(raw.data() + b * 82, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlIq1S: {
            if (count % 256 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "IQ1_S element count not a multiple of 256");
            }
            for (std::size_t b = 0; b < out.size() / 256; ++b) {
                dequantize_iq1_s(raw.data() + b * 50, out.data() + b * 256);
            }
            return out;
        }
        case kGgmlQ2_0: {
            if (count % 64 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "Q2_0 element count not a multiple of 64");
            }
            const std::size_t blocks = out.size() / 64;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 18;
                const float scale = half_to_float(
                    static_cast<std::uint16_t>(raw[base] | (static_cast<std::uint16_t>(raw[base + 1]) << 8U)));
                for (std::size_t j = 0; j < 64; ++j) {
                    const std::uint8_t byte = raw[base + 2 + j / 4];
                    const auto q = static_cast<std::int32_t>((byte >> ((j % 4) * 2U)) & 0x03U);
                    out[b * 64 + j] = static_cast<float>(q - 1) * scale;
                }
            }
            return out;
        }
        case kGgmlNvfp4: {
            if (count % 64 != 0) {
                // check-no-throw: allow invalid model input
                throw std::runtime_error(
                    // check-no-throw: allow invalid model input
                    "NVFP4 element count not a multiple of 64");
            }
            const std::size_t blocks = out.size() / 64;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t base = b * 36;
                for (std::size_t s = 0; s < 4; ++s) {
                    const float scale = ue4m3_to_float(raw[base + s]);
                    if (scale == 0.0F) {
                        continue;
                    }
                    const std::size_t sub = b * 64 + s * 16;
                    for (std::size_t j = 0; j < 8; ++j) {
                        const std::uint8_t byte = raw[base + 4 + s * 8 + j];
                        out[sub + j] = e2m1_to_float(static_cast<std::uint8_t>(byte & 0x0FU)) * scale;
                        out[sub + j + 8] = e2m1_to_float(static_cast<std::uint8_t>(byte >> 4U)) * scale;
                    }
                }
            }
            return out;
        }
        default:
            // check-no-throw: allow invalid model input
            throw std::runtime_error("unsupported GGML type " +
                                     // check-no-throw: allow invalid model input
                                     std::to_string(type));
    }
}

}  // namespace mnet::gguf
