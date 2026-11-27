// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Native byte-level BPE over a GGUF vocabulary. Implements the GPT-2 byte map,
// merge ranks, and the Qwen35 / GPT4O pre-tokenizers used by the supported
// models. No llama.cpp code is called.

#ifndef MNET_BPE_HPP
#define MNET_BPE_HPP

#include <cstdint>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mnet/detail/unicode_flags.hpp>
#include <mnet/gguf.hpp>

namespace mnet::bpe {

enum class PreTokenizer { Qwen35, Gpt4o };

[[nodiscard]] inline std::uint32_t utf8_decode(std::string_view text, std::size_t& offset) {
    const auto byte = static_cast<unsigned char>(text[offset]);
    if (byte < 0x80U) {
        ++offset;
        return byte;
    }
    std::uint32_t cpt = 0;
    std::size_t length = 0;
    if ((byte & 0xE0U) == 0xC0U) {
        cpt = byte & 0x1FU;
        length = 2;
    } else if ((byte & 0xF0U) == 0xE0U) {
        cpt = byte & 0x0FU;
        length = 3;
    } else if ((byte & 0xF8U) == 0xF0U) {
        cpt = byte & 0x07U;
        length = 4;
    } else {
        ++offset;
        return byte;
    }
    if (offset + length > text.size()) {
        ++offset;
        return byte;
    }
    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[offset + i]);
        if ((next & 0xC0U) != 0x80U) {
            ++offset;
            return byte;
        }
        cpt = (cpt << 6U) | (next & 0x3FU);
    }
    offset += length;
    return cpt;
}

inline void utf8_append(std::string& out, std::uint32_t cpt) {
    if (cpt < 0x80U) {
        out.push_back(static_cast<char>(cpt));
    } else if (cpt < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (cpt >> 6U)));
        out.push_back(static_cast<char>(0x80U | (cpt & 0x3FU)));
    } else if (cpt < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (cpt >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cpt >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cpt & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cpt >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cpt >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cpt >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cpt & 0x3FU)));
    }
}

[[nodiscard]] inline std::vector<std::uint32_t> utf8_codepoints(std::string_view text) {
    std::vector<std::uint32_t> out;
    out.reserve(text.size());
    std::size_t offset = 0;
    while (offset < text.size()) {
        out.push_back(utf8_decode(text, offset));
    }
    return out;
}

class Encoder {
public:
    static Encoder from_reader(const mnet::gguf::Reader& reader);

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

    [[nodiscard]] std::vector<std::string> pre_tokenize(std::string_view text) const;

    [[nodiscard]] std::string decode(const std::vector<std::int32_t>& ids) const;

private:
    void build_byte_maps();

    [[nodiscard]] std::size_t contraction_length(const std::vector<std::uint32_t>& cpts, std::size_t pos) const;

    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> split_qwen35(
        const std::vector<std::uint32_t>& cpts) const;

    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> split_gpt4o(
        const std::vector<std::uint32_t>& cpts) const;

    void encode_plain(std::string_view text, std::vector<std::int32_t>& ids) const;

    void encode_word(const std::string& raw, std::vector<std::int32_t>& ids) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, std::int32_t> vocab_;
    std::unordered_map<std::string, std::int32_t> ranks_;
    std::vector<std::pair<std::string, std::int32_t>> specials_;
    std::array<std::string, 256> byte_to_utf8_;
    std::unordered_map<std::uint32_t, std::uint8_t> utf8_to_byte_;
    PreTokenizer pre_ = PreTokenizer::Qwen35;
    bool ignore_merges_ = false;
};

}  // namespace mnet::bpe

#endif  // MNET_BPE_HPP
