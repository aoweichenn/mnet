#pragma once
#include <cstdint>
#include <string>
#include <optional>

struct SpecialTokensConfig {
    std::optional<std::string> bos_token;
    std::optional<std::string> eos_token;
    std::optional<std::string> unk_token;
    std::optional<std::string> sep_token;
    std::optional<std::string> pad_token;
    std::optional<std::string> cls_token;
    std::optional<std::string> mask_token;
};

struct SpecialTokenIds {
    std::int32_t bos_id = -1;
    std::int32_t eos_id = -1;
    std::int32_t unk_id = -1;
    std::int32_t sep_id = -1;
    std::int32_t pad_id = -1;
    std::int32_t cls_id = -1;
    std::int32_t mask_id = -1;
};