// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <atomic>
#include <bit>
#include <fstream>
#include <map>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mnet/tts/tts_tensor.hpp>

namespace mnet::tts {

void account_bytes(std::size_t bytes) {
    const std::size_t live = g_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::size_t peak = g_peak_bytes.load(std::memory_order_relaxed);
    while (live > peak && !g_peak_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
}

void release_bytes(std::size_t bytes) {
    g_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

void reserve_bytes(std::size_t bytes) {
    const std::size_t live = g_live_bytes.load(std::memory_order_relaxed);
    if (g_mem_budget_bytes != 0 && live + bytes > g_mem_budget_bytes) {
        throw std::bad_alloc();
    }
    account_bytes(bytes);
}

}  // namespace mnet::tts
