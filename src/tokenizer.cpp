// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/tokenizer.hpp>

#include <mnet/detail/bpe_tokenizer.h>

#include <optional>

namespace mnet {

struct Tokenizer::Impl {
    std::unique_ptr<BpeTokenizer> bpe;
};

Tokenizer::Tokenizer() : d(new Impl()) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::int32_t Tokenizer::load(const char* vocab_path, const char* merges_path) {
    if (vocab_path == nullptr || merges_path == nullptr) {
        return -1;
    }
    SpecialTokensConfig config;
    config.bos_token = "";
    config.eos_token = "<|im_end|>";
    config.unk_token = "<unk>";
    config.pad_token = "<|endoftext|>";
    this->d->bpe =
        std::make_unique<BpeTokenizer>(BpeTokenizer::LoadFromFiles(vocab_path, merges_path, config, true, true, true));
    return 0;
}

std::vector<std::int32_t> Tokenizer::encode(const std::string& text) const {
    const std::vector<int> ids = this->d->bpe->encode(text, false, false, false, false);
    return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::vector<int> ids_int(ids.begin(), ids.end());
    return this->d->bpe->decode(ids_int, true);
}

}  // namespace mnet
