// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_OPTION_HPP
#define MNET_OPTION_HPP

#include <cstdint>

#include <mmemory/allocator.hpp>

namespace mnet {

// Runtime knobs shared by every layer. Only CPU is implemented for now.
struct Option {
    std::int32_t num_threads = 1;
    mmemory::Allocator* blob_allocator = nullptr;
    mmemory::Allocator* workspace_allocator = nullptr;

    // Release a blob once its last consumer has run (ncnn light mode).
    bool use_light_mode = true;
    bool use_packing_layout = false;
    bool use_int8_inference = false;
    bool use_bf16_storage = false;
    bool use_fp16_storage = false;
    bool use_fp16_packed = false;
    bool use_fp16_arithmetic = false;
    bool use_vulkan_compute = false;

    mmemory::Allocator* kvcache_allocator = nullptr;
    std::int32_t kvcache_max_seqlen_hint = 0;
};

}  // namespace mnet

#endif  // MNET_OPTION_HPP
