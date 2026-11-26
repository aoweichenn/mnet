// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_NET_HPP
#define MNET_NET_HPP

#include <cstddef>
#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include <mnet/layer.hpp>
#include <mnet/mat.hpp>
#include <mnet/option.hpp>

namespace mnet {

struct NetPrivate;

// Process default memory budget: MNET_MAX_MEM_MB when set, otherwise 50% of
// MemTotal on Linux (WSL boxes are small and an OOM kill takes the parent
// shell with it). Returns 0 when no safe default is known (macOS/Android).
[[nodiscard]] std::size_t default_memory_budget_bytes();

// Standalone CPU runtime: graph loading through mgraph/mformat and layer
// execution through mlayer. It never includes or links ncnn.
class Net {
public:
    Net();
    ~Net();

    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

    std::int32_t load_param(const char* param_path);
    // Parse a null-terminated text param from memory (ncnn load_param_mem).
    std::int32_t load_param_mem(const char* mem);
    // Parse a binary param blob from memory.
    std::int32_t load_param_bin_mem(const void* data, std::size_t size);
    std::int32_t load_model(const char* bin_path);
    // Load weights from memory.
    std::int32_t load_model_mem(const void* data, std::size_t size);
    std::int32_t load(const char* param_path, const char* bin_path);

    // Hard cap for activation/workspace bytes (0 = unlimited). Weight bytes
    // (the model file size) are subtracted from it, so a graph whose weights
    // alone exceed the budget fails before loading. Exceeding the budget makes
    // the offending allocation return nullptr, i.e. a normal layer failure
    // instead of an OS OOM kill.
    void set_memory_budget_bytes(std::size_t bytes);
    [[nodiscard]] std::size_t memory_budget_bytes() const;
    [[nodiscard]] std::size_t memory_peak_bytes() const;
    [[nodiscard]] std::size_t memory_current_bytes() const;
    [[nodiscard]] bool memory_budget_exhausted() const;

    [[nodiscard]] Option& opt() { return this->opt_; }
    [[nodiscard]] const Option& opt() const { return this->opt_; }

    class Extractor {
    public:
        std::int32_t input(const char* blob_name, const Mat& in);
        std::int32_t input(std::int32_t blob_index, const Mat& in);
        std::int32_t extract(const char* blob_name, Mat& out);
        std::int32_t extract(std::int32_t blob_index, Mat& out);

        // Per-extractor overrides, mirroring ncnn's Extractor API.
        void set_num_threads(std::int32_t num_threads) { this->opt_.num_threads = num_threads; }
        void set_light_mode(bool enable) { this->opt_.use_light_mode = enable; }
        void set_blob_allocator(mmemory::Allocator* allocator) { this->opt_.blob_allocator = allocator; }
        void set_workspace_allocator(mmemory::Allocator* allocator) { this->opt_.workspace_allocator = allocator; }

    private:
        friend class Net;
        explicit Extractor(const Net* net);

        const Net* net_;
        Option opt_;
        std::vector<Mat> blob_mats_;
    };

    [[nodiscard]] Extractor create_extractor() const { return Extractor(this); }

    [[nodiscard]] std::vector<std::string> input_names() const;
    [[nodiscard]] std::vector<std::string> output_names() const;

private:
    std::int32_t load_param_bytes(const std::uint8_t* data, std::size_t size, bool binary);

    Option opt_;
    std::unique_ptr<NetPrivate> d;
};

}  // namespace mnet

#endif  // MNET_NET_HPP
