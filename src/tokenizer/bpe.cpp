// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/tokenizer/bpe.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mnet/detail/unicode_flags.hpp>
#include <mnet/gguf.hpp>

namespace mnet::bpe {

Encoder Encoder::from_reader(const mnet::gguf::Reader& reader) {
    Encoder encoder;
    const auto& arrays = reader.string_arrays();
    const auto tokens = arrays.find("tokenizer.ggml.tokens");
    if (tokens == arrays.end()) {
        // check-no-throw: allow invalid model input
        throw std::runtime_error(
            // check-no-throw: allow invalid model input
            "tokenizer: tokenizer.ggml.tokens missing");
    }
    encoder.tokens_ = tokens->second;
    encoder.vocab_.reserve(encoder.tokens_.size());
    for (std::size_t i = 0; i < encoder.tokens_.size(); ++i) {
        encoder.vocab_.emplace(encoder.tokens_[i], static_cast<std::int32_t>(i));
    }
    const auto merges = arrays.find("tokenizer.ggml.merges");
    if (merges != arrays.end()) {
        encoder.ranks_.reserve(merges->second.size());
        for (std::size_t rank = 0; rank < merges->second.size(); ++rank) {
            const auto& merge = merges->second[rank];
            const auto space = merge.find(' ');
            if (space == std::string::npos) {
                continue;
            }
            encoder.ranks_.emplace(merge.substr(0, space) + '\0' + merge.substr(space + 1),
                                   static_cast<std::int32_t>(rank));
        }
    }
    const auto pre = reader.metadata().find("tokenizer.ggml.pre");
    if (pre != reader.metadata().end() && pre->second == "inkling2") {
        encoder.pre_ = PreTokenizer::Gpt4o;
        encoder.ignore_merges_ = true;
    } else {
        encoder.pre_ = PreTokenizer::Qwen35;
    }
    const auto types = reader.int_arrays().find("tokenizer.ggml.token_type");
    if (types != reader.int_arrays().end()) {
        for (std::size_t i = 0; i < types->second.size() && i < encoder.tokens_.size(); ++i) {
            const auto type = types->second[i];
            if (type == 3 || type == 4) {  // control / user-defined
                encoder.specials_.emplace_back(encoder.tokens_[i], static_cast<std::int32_t>(i));
            }
        }
        std::sort(encoder.specials_.begin(), encoder.specials_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
    encoder.build_byte_maps();
    return encoder;
}

[[nodiscard]] std::vector<std::int32_t> Encoder::encode(std::string_view text) const {
    std::vector<std::int32_t> ids;
    std::size_t start = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        bool matched = false;
        for (const auto& special : this->specials_) {
            if (text.compare(offset, special.first.size(), special.first) == 0) {
                this->encode_plain(text.substr(start, offset - start), ids);
                ids.push_back(special.second);
                offset += special.first.size();
                start = offset;
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }
        std::size_t probe = offset;
        static_cast<void>(utf8_decode(text, probe));
        offset = probe;
    }
    this->encode_plain(text.substr(start), ids);
    return ids;
}

[[nodiscard]] std::vector<std::string> Encoder::pre_tokenize(std::string_view text) const {
    std::vector<std::string> out;
    const auto cpts = utf8_codepoints(text);
    const auto words = this->pre_ == PreTokenizer::Gpt4o ? this->split_gpt4o(cpts) : this->split_qwen35(cpts);
    for (const auto& [start, length] : words) {
        std::string raw;
        for (std::size_t i = 0; i < length; ++i) {
            utf8_append(raw, cpts[start + i]);
        }
        out.push_back(raw);
    }
    return out;
}

[[nodiscard]] std::string Encoder::decode(const std::vector<std::int32_t>& ids) const {
    std::string out;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= this->tokens_.size()) {
            continue;
        }
        const auto& token = this->tokens_[static_cast<std::size_t>(id)];
        bool special = false;
        for (const auto& entry : this->specials_) {
            if (entry.second == id) {
                special = true;
                break;
            }
        }
        if (special) {
            out += token;
            continue;
        }
        const auto cpts = utf8_codepoints(token);
        for (const auto cpt : cpts) {
            const auto it = this->utf8_to_byte_.find(cpt);
            if (it != this->utf8_to_byte_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                utf8_append(out, cpt);
            }
        }
    }
    return out;
}

void Encoder::build_byte_maps() {
    bool used[256] = {};
    for (std::int32_t ch = 0x21; ch <= 0x7E; ++ch) {
        used[ch] = true;
    }
    for (std::int32_t ch = 0xA1; ch <= 0xAC; ++ch) {
        used[ch] = true;
    }
    for (std::int32_t ch = 0xAE; ch <= 0xFF; ++ch) {
        used[ch] = true;
    }
    std::int32_t extra = 0;
    for (std::int32_t ch = 0; ch < 256; ++ch) {
        const std::uint32_t cpt = used[ch] ? static_cast<std::uint32_t>(ch) : static_cast<std::uint32_t>(256 + extra++);
        this->byte_to_utf8_[static_cast<std::size_t>(ch)] = std::string();
        utf8_append(this->byte_to_utf8_[static_cast<std::size_t>(ch)], cpt);
        this->utf8_to_byte_[cpt] = static_cast<std::uint8_t>(ch);
    }
}

[[nodiscard]] std::size_t Encoder::contraction_length(const std::vector<std::uint32_t>& cpts, std::size_t pos) const {
    if (pos >= cpts.size() || cpts[pos] != '\'' || pos + 1 >= cpts.size()) {
        return 0;
    }
    auto lower = [](std::uint32_t cpt) { return (cpt >= 'A' && cpt <= 'Z') ? cpt + ('a' - 'A') : cpt; };
    const std::uint32_t a = lower(cpts[pos + 1]);
    const std::uint32_t b = pos + 2 < cpts.size() ? lower(cpts[pos + 2]) : 0;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        return 2;
    }
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
        return 3;
    }
    return 0;
}

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> Encoder::split_qwen35(
    const std::vector<std::uint32_t>& cpts) const {
    std::vector<std::pair<std::size_t, std::size_t>> words;
    const std::size_t n = cpts.size();
    auto flags = [&](std::size_t i) -> std::uint16_t { return i < n ? mnet::detail::unicode_flags(cpts[i]) : 0; };
    auto is_letter = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniLetter) != 0; };
    auto is_mark = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniMark) != 0; };
    auto is_number = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniNumber) != 0; };
    auto is_space = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniWhitespace) != 0; };
    auto add = [&](std::size_t start, std::size_t end) {
        if (end > start) {
            words.emplace_back(start, end - start);
        }
    };
    for (std::size_t pos = 0; pos < n;) {
        const std::uint32_t cpt = cpts[pos];
        const std::size_t contr = this->contraction_length(cpts, pos);
        if (contr != 0) {
            pos += contr;
            add(pos - contr, pos);
            continue;
        }
        if (!(cpt == '\r' || cpt == '\n' || is_number(pos))) {
            if (is_letter(pos) || is_mark(pos) || is_mark(pos + 1) || is_letter(pos + 1)) {
                const std::size_t begin = pos;
                ++pos;
                while (is_letter(pos) || is_mark(pos)) {
                    ++pos;
                }
                add(begin, pos);
                continue;
            }
        }
        if (is_number(pos)) {
            ++pos;
            add(pos - 1, pos);
            continue;
        }
        const bool cpt_space = cpt == ' ';
        const bool next_in_word = !(is_space(pos + 1) || is_letter(pos + 1) || is_mark(pos + 1) || is_number(pos + 1));
        if (cpt_space ? next_in_word
                      : (flags(pos) != 0 && !(is_space(pos) || is_letter(pos) || is_mark(pos) || is_number(pos)))) {
            std::size_t begin = pos;
            pos += cpt_space ? 1 : 0;
            while (pos < n && flags(pos) != 0 && !(is_space(pos) || is_letter(pos) || is_mark(pos) || is_number(pos))) {
                ++pos;
            }
            while (pos < n && (cpts[pos] == '\r' || cpts[pos] == '\n')) {
                ++pos;
            }
            add(begin, pos);
            continue;
        }
        std::size_t spaces = 0;
        std::size_t last_newline = 0;
        while (is_space(pos + spaces)) {
            if (cpts[pos + spaces] == '\r' || cpts[pos + spaces] == '\n') {
                last_newline = pos + spaces + 1;
            }
            ++spaces;
        }
        if (last_newline > 0) {
            add(pos, last_newline);
            pos = last_newline;
            continue;
        }
        if (spaces > 1 && pos + spaces < n) {
            const std::size_t begin = pos;
            pos += spaces - 1;
            add(begin, pos);
            continue;
        }
        if (spaces > 0) {
            const std::size_t begin = pos;
            pos += spaces;
            add(begin, pos);
            continue;
        }
        add(pos, pos + 1);
        ++pos;
    }
    return words;
}

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> Encoder::split_gpt4o(
    const std::vector<std::uint32_t>& cpts) const {
    std::vector<std::pair<std::size_t, std::size_t>> words;
    const std::size_t n = cpts.size();
    auto flags = [&](std::size_t i) -> std::uint16_t { return i < n ? mnet::detail::unicode_flags(cpts[i]) : 0; };
    auto is_letter = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniLetter) != 0; };
    auto is_mark = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniMark) != 0; };
    auto is_number = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniNumber) != 0; };
    auto is_space = [&](std::size_t i) { return (flags(i) & mnet::detail::kUniWhitespace) != 0; };
    auto is_upper = [&](std::size_t i) { return (is_letter(i) || is_mark(i)) && !(cpts[i] >= 'a' && cpts[i] <= 'z'); };
    auto is_lower = [&](std::size_t i) { return (is_letter(i) || is_mark(i)) && !(cpts[i] >= 'A' && cpts[i] <= 'Z'); };
    auto add = [&](std::size_t start, std::size_t end) {
        if (end > start) {
            words.emplace_back(start, end - start);
        }
    };
    for (std::size_t pos = 0; pos < n;) {
        const std::size_t contr = this->contraction_length(cpts, pos);
        if (contr != 0) {
            add(pos, pos + contr);
            pos += contr;
            continue;
        }
        // optional leading non-CR/LF/letter/number character
        std::size_t p = pos;
        if (p < n && !(cpts[p] == '\r' || cpts[p] == '\n' || is_letter(p) || is_number(p))) {
            ++p;
        }
        std::size_t end = 0;
        {
            std::size_t q = p;
            while (is_upper(q)) {
                ++q;
            }
            if (is_lower(q)) {
                while (is_lower(q)) {
                    ++q;
                }
                end = q;
            }
        }
        if (end == 0) {
            std::size_t q = p;
            while (is_upper(q)) {
                ++q;
            }
            if (q > p) {
                while (is_lower(q)) {
                    ++q;
                }
                end = q;
            }
        }
        if (end > 0) {
            end += this->contraction_length(cpts, end);
            add(pos, end);
            pos = end;
            continue;
        }
        if (is_number(pos)) {
            std::size_t q = pos;
            std::size_t count = 0;
            while (count < 3 && is_number(q)) {
                ++q;
                ++count;
            }
            add(pos, q);
            pos = q;
            continue;
        }
        {
            std::size_t q = pos;
            if (q < n && cpts[q] == ' ') {
                ++q;
            }
            if (q < n && !(is_space(q) || is_letter(q) || is_number(q))) {
                while (q < n && !(is_space(q) || is_letter(q) || is_number(q))) {
                    ++q;
                }
                while (q < n && (cpts[q] == '\r' || cpts[q] == '\n' || cpts[q] == '/')) {
                    ++q;
                }
                add(pos, q);
                pos = q;
                continue;
            }
        }
        std::size_t spaces = 0;
        std::size_t last_newline = 0;
        while (is_space(pos + spaces)) {
            if (cpts[pos + spaces] == '\r' || cpts[pos + spaces] == '\n') {
                last_newline = pos + spaces + 1;
            }
            ++spaces;
        }
        if (last_newline > 0) {
            add(pos, last_newline);
            pos = last_newline;
            continue;
        }
        if (spaces > 1 && pos + spaces < n) {
            add(pos, pos + spaces - 1);
            pos += spaces - 1;
            continue;
        }
        if (spaces > 0) {
            add(pos, pos + spaces);
            pos += spaces;
            continue;
        }
        add(pos, pos + 1);
        ++pos;
    }
    return words;
}

void Encoder::encode_plain(std::string_view text, std::vector<std::int32_t>& ids) const {
    if (text.empty()) {
        return;
    }
    const auto cpts = utf8_codepoints(text);
    const auto words = this->pre_ == PreTokenizer::Gpt4o ? this->split_gpt4o(cpts) : this->split_qwen35(cpts);
    for (const auto& [start, length] : words) {
        std::string raw;
        for (std::size_t i = 0; i < length; ++i) {
            utf8_append(raw, cpts[start + i]);
        }
        this->encode_word(raw, ids);
    }
}

void Encoder::encode_word(const std::string& raw, std::vector<std::int32_t>& ids) const {
    std::string encoded;
    for (const char c : raw) {
        encoded += this->byte_to_utf8_[static_cast<std::uint8_t>(c)];
    }
    if (this->ignore_merges_) {
        const auto it = this->vocab_.find(encoded);
        if (it != this->vocab_.end()) {
            ids.push_back(it->second);
            return;
        }
    }
    std::vector<std::string> symbols;
    std::size_t offset = 0;
    while (offset < encoded.size()) {
        const std::size_t begin = offset;
        static_cast<void>(utf8_decode(encoded, offset));
        symbols.emplace_back(encoded.substr(begin, offset - begin));
    }
    while (symbols.size() > 1) {
        std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();
        std::size_t best = symbols.size();
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = this->ranks_.find(symbols[i] + '\0' + symbols[i + 1]);
            if (it != this->ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == symbols.size()) {
            break;
        }
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    for (const auto& symbol : symbols) {
        const auto it = this->vocab_.find(symbol);
        if (it == this->vocab_.end()) {
            // check-no-throw: allow invalid model input
            throw std::runtime_error(
                // check-no-throw: allow invalid model input
                "tokenizer: byte symbol not in vocabulary");
        }
        ids.push_back(it->second);
    }
}

}  // namespace mnet::bpe
