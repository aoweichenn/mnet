// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_GGUF_READER_HPP
#define MNET_GGUF_READER_HPP

// GGUF container reader: parses the header, hands out tensor bytes and runs
// the quantised matvec through <mnet/gguf/kernels.hpp>. The heavy methods live
// in src/gguf/reader.cpp; the small accessors stay inline.

#include <cstdint>

#include <fstream>
#include <istream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mnet/gguf/format.hpp>

namespace mnet::gguf {

class Reader {
public:
    explicit Reader(std::string path);

    [[nodiscard]] const std::string& path() const { return this->path_; }
    [[nodiscard]] std::uint32_t version() const { return this->version_; }
    [[nodiscard]] std::uint64_t data_start() const { return this->data_start_; }
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return this->tensors_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const { return this->strings_; }
    [[nodiscard]] const std::unordered_map<std::string, std::uint64_t>& integers() const { return this->integers_; }
    [[nodiscard]] const std::unordered_map<std::string, double>& floats() const { return this->floats_; }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::int64_t>>& int_arrays() const {
        return this->int_arrays_;
    }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::string>>& string_arrays() const {
        return this->string_arrays_;
    }

    // Sign vector for a Hadamard-folded weight width, concatenated as stored in
    // `prism.hadamard.sign_widths` / `sign_values`.
    [[nodiscard]] std::vector<std::int32_t> signs_for_width(std::uint32_t width) const;

    [[nodiscard]] const TensorInfo& tensor(std::string_view name) const;

    [[nodiscard]] std::vector<std::uint8_t> read_raw(const TensorInfo& info, std::uint64_t byte_offset = 0,
                                                     std::uint64_t byte_count = 0) const;

    // Dequantize one tensor to fp32. `expert` slices a 3-D MoE weight and
    // applies the per-expert `<name>.scale` when present.
    [[nodiscard]] std::vector<float> dequantize(const TensorInfo& info, std::int32_t expert = -1) const;

    [[nodiscard]] static std::vector<float> dequantize_block(std::uint32_t type, const std::vector<std::uint8_t>& raw,
                                                             std::uint64_t count);
    [[nodiscard]] static float dot_row(std::uint32_t type, const std::uint8_t* data, const float* input,
                                       std::uint64_t columns);

    // A tensor kept in its GGUF block-quantized form. `matvec` dequantizes each
    // block directly into the dot product, so the fp32 copy of the weight is
    // never materialized (a 16x saving for Q2_0 versus fp32).
    struct Quantized {
        std::uint32_t type = kGgmlF32;
        std::uint64_t rows = 0;     // output features
        std::uint64_t columns = 0;  // input features
        float scale = 1.0F;         // per-expert MoE scale, folded at the output
        std::vector<std::uint8_t> blocks;

        [[nodiscard]] bool empty() const { return this->blocks.empty(); }
    };

    [[nodiscard]] Quantized quantized(const TensorInfo& info, std::int32_t expert = -1) const;

    // y = W x with W kept quantized. Deterministic: each output element is
    // accumulated left to right in double, independent of thread count.
    static void matvec(const Quantized& matrix, const float* input, float* output);

    static void matmul(const Quantized& matrix, const float* input, float* output, std::size_t tokens,
                       std::size_t input_stride, std::size_t output_stride);

private:
    static std::uint16_t read_u16(std::istream& in);
    static std::uint32_t read_u32(std::istream& in);
    static std::uint64_t read_u64(std::istream& in);
    static std::string read_string(std::istream& in);
    static std::int64_t read_scalar_int(std::istream& in, std::uint32_t type);

    static void skip_value(std::istream& in, std::uint32_t type);

    void parse_header();

    std::string path_;
    mutable std::ifstream file_;
    std::uint32_t version_ = 0;
    std::uint64_t data_start_ = 0;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, std::uint64_t> integers_;
    std::unordered_map<std::string, double> floats_;
    std::unordered_map<std::string, std::vector<std::int64_t>> int_arrays_;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays_;
};

}  // namespace mnet::gguf

#endif  // MNET_GGUF_READER_HPP
