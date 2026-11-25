// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_TOKENIZER_HPP
#define MNET_TOKENIZER_HPP

#include <cstdint>

#include <memory>
#include <string>
#include <vector>

namespace mnet {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Load a byte-level BPE vocabulary. Special tokens default to Qwen-style.
    std::int32_t load(const char* vocab_path, const char* merges_path);

    [[nodiscard]] std::vector<std::int32_t> encode(const std::string& text) const;
    [[nodiscard]] std::string decode(const std::vector<std::int32_t>& ids) const;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

}  // namespace mnet

#endif  // MNET_TOKENIZER_HPP
