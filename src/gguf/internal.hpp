// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_SRC_GGUF_INTERNAL_HPP
#define MNET_SRC_GGUF_INTERNAL_HPP

// Helpers shared by src/gguf/format.cpp, kernels.cpp and reader.cpp. This
// header is private to the library and is not installed.

#include <cstdint>

namespace mnet::gguf {

void get_scale_min_k4(std::int32_t j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m);

}  // namespace mnet::gguf

#endif  // MNET_SRC_GGUF_INTERNAL_HPP
