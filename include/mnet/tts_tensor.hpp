#pragma once

// Activation tensors and the flattened parameter blob used by the TTS runners,
// plus the tracked memory budget that MNET_MAX_MEM_MB enforces.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <new>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mnet::tts {

// MNET_MAX_MEM_MB budgets the bytes mnet_tts itself keeps live (parameter blob
// plus every activation tensor). Tracked explicitly because RLIMIT_AS is a poor
// proxy: per-thread malloc arenas reserve gigabytes of address space that never
// become resident. An over-budget allocation throws std::bad_alloc, which main()
// turns into a clean exit(1).
inline std::size_t g_mem_budget_bytes = 0;
inline std::atomic<std::size_t> g_live_bytes{0};

inline std::atomic<std::size_t> g_peak_bytes{0};

inline void account_bytes(std::size_t bytes) {
    const std::size_t live = g_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::size_t peak = g_peak_bytes.load(std::memory_order_relaxed);
    while (live > peak && !g_peak_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
}

inline void release_bytes(std::size_t bytes) {
    g_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

inline void reserve_bytes(std::size_t bytes) {
    const std::size_t live = g_live_bytes.load(std::memory_order_relaxed);
    if (g_mem_budget_bytes != 0 && live + bytes > g_mem_budget_bytes) {
        throw std::bad_alloc();
    }
    account_bytes(bytes);
}

inline std::size_t live_bytes() {
    return g_live_bytes.load(std::memory_order_relaxed);
}

// [channels][length], the layout of the ONNX activations ([1, C, T] flattened).
struct Act {
    std::size_t channels = 0;
    std::size_t length = 0;
    std::vector<float> data;

    Act() = default;
    Act(std::size_t c, std::size_t t) : channels(c), length(t) {
        reserve_bytes(c * t * sizeof(float));
        this->data.assign(c * t, 0.0F);
    }
    Act(std::size_t c, std::size_t t, std::vector<float> values) : channels(c), length(t), data(std::move(values)) {
        reserve_bytes(this->data.size() * sizeof(float));
    }
    Act(const Act& other) : channels(other.channels), length(other.length), data(other.data) {
        reserve_bytes(this->data.size() * sizeof(float));
    }
    Act(Act&& other) noexcept : channels(other.channels), length(other.length), data(std::move(other.data)) {
        other.reset();
    }
    Act& operator=(const Act& other) {
        if (this != &other) {
            reserve_bytes(other.data.size() * sizeof(float));
            release_bytes(this->data.size() * sizeof(float));
            this->channels = other.channels;
            this->length = other.length;
            this->data = other.data;
        }
        return *this;
    }
    Act& operator=(Act&& other) noexcept {
        if (this != &other) {
            release_bytes(this->data.size() * sizeof(float));
            this->channels = other.channels;
            this->length = other.length;
            this->data = std::move(other.data);
            other.reset();
        }
        return *this;
    }
    ~Act() { release_bytes(this->data.size() * sizeof(float)); }

    void reset() {
        this->channels = 0;
        this->length = 0;
        this->data.clear();
    }

    [[nodiscard]] float at(std::size_t c, std::size_t t) const { return this->data[c * this->length + t]; }
    [[nodiscard]] float& at(std::size_t c, std::size_t t) { return this->data[c * this->length + t]; }
};

// Weights are views into the single flat blob the loader reads, so the model
// costs one copy of the parameters rather than two (Kokoro's 325 MB blob used
// to be duplicated, doubling the process footprint).
struct Weight {
    std::span<const float> values;
    std::vector<std::size_t> shape;

    [[nodiscard]] std::size_t size(std::size_t index) const { return this->shape.at(index); }
};

class Weights {
public:
    [[nodiscard]] bool load(const std::string& manifest_path, const std::string& blob_path) {
        std::ifstream manifest(manifest_path);
        if (!manifest) {
            std::fprintf(stderr, "cannot open %s\n", manifest_path.c_str());
            return false;
        }
        std::size_t count = 0;
        manifest >> count;
        std::ifstream blob(blob_path, std::ios::binary | std::ios::ate);
        if (!blob) {
            std::fprintf(stderr, "cannot open %s\n", blob_path.c_str());
            return false;
        }
        const auto bytes = static_cast<std::size_t>(blob.tellg());
        blob.seekg(0);
        reserve_bytes(bytes);
        this->flat_.resize(bytes / sizeof(float));
        blob.read(reinterpret_cast<char*>(this->flat_.data()), static_cast<std::streamsize>(bytes));
        if (!blob) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            std::string name;
            std::size_t elements = 0;
            std::size_t offset = 0;
            std::string shape;
            if (!(manifest >> name >> elements >> offset >> shape)) {
                return false;
            }
            const auto existing = this->by_offset_.find(offset);
            if (existing != this->by_offset_.end()) {
                this->index_.emplace(std::move(name), existing->second);
                continue;
            }
            Weight weight;
            weight.values = std::span<const float>(this->flat_.data() + offset, elements);
            std::size_t start = 0;
            while (start <= shape.size()) {
                const std::size_t next = shape.find('x', start);
                const auto part = shape.substr(start, next == std::string::npos ? std::string::npos : next - start);
                if (!part.empty()) {
                    weight.shape.push_back(std::stoul(part));
                }
                if (next == std::string::npos) {
                    break;
                }
                start = next + 1;
            }
            this->by_offset_.emplace(offset, this->weights_.size());
            this->weights_.push_back(std::move(weight));
            this->index_.emplace(std::move(name), this->weights_.size() - 1);
        }
        return true;
    }

    [[nodiscard]] const Weight& at(const std::string& name) const {
        const auto it = this->index_.find(name);
        if (it == this->index_.end()) {
            throw std::runtime_error("missing weight: " + name);
        }
        return this->weights_[it->second];
    }

    [[nodiscard]] bool has(const std::string& name) const { return this->index_.find(name) != this->index_.end(); }

private:
    std::vector<float> flat_;
    std::vector<Weight> weights_;
    std::map<std::string, std::size_t> index_;
    std::map<std::size_t, std::size_t> by_offset_;
};

}  // namespace mnet::tts
