// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

#include <mnet/detail/compute.hpp>
#include <mnet/detail/txn_log.hpp>
#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>

namespace mnet::detail::compute {
namespace {

std::atomic<MatvecQ8_0Fn>& backend_slot() {
    static std::atomic<MatvecQ8_0Fn> slot{nullptr};
    return slot;
}

std::atomic<MatvecQ8_0BatchFn>& batch_slot() {
    static std::atomic<MatvecQ8_0BatchFn> slot{nullptr};
    return slot;
}

std::atomic<FfnSwigluFn>& ffn_slot() {
    static std::atomic<FfnSwigluFn> slot{nullptr};
    return slot;
}

std::atomic<FfnSwigluBatchFn>& ffn_batch_slot() {
    static std::atomic<FfnSwigluBatchFn> slot{nullptr};
    return slot;
}

std::atomic<AttentionQ2_0Fn>& attention_slot() {
    static std::atomic<AttentionQ2_0Fn> slot{nullptr};
    return slot;
}

std::atomic<SsmQ2_0Fn>& ssm_slot() {
    static std::atomic<SsmQ2_0Fn> slot{nullptr};
    return slot;
}

struct Stats {
    std::atomic<std::uint64_t> backend_ns{0};
    std::atomic<std::uint64_t> cpu_ns{0};
    std::atomic<std::uint64_t> backend_calls{0};
    std::atomic<std::uint64_t> cpu_calls{0};
    std::atomic<std::uint64_t> ffn_ns{0};
    std::atomic<std::uint64_t> ffn_calls{0};
    std::atomic<std::uint64_t> attn_ns{0};
    std::atomic<std::uint64_t> attn_calls{0};
    std::atomic<std::uint64_t> ssm_ns{0};
    std::atomic<std::uint64_t> ssm_calls{0};

    ~Stats() {
        if (std::getenv("MNET_COMPUTE_STATS") == nullptr) {
            return;
        }
        char line[256];
        std::snprintf(line, sizeof(line),
                      "mnet: compute backend %" PRIu64 " calls %.3f s, cpu %" PRIu64 " calls %.3f s, fused ffn %" PRIu64
                      " calls %.3f s, fused attn %" PRIu64 " calls %.3f s, fused ssm %" PRIu64 " calls %.3f s\n",
                      this->backend_calls.load(), static_cast<double>(this->backend_ns.load()) * 1.0e-9,
                      this->cpu_calls.load(), static_cast<double>(this->cpu_ns.load()) * 1.0e-9, this->ffn_calls.load(),
                      static_cast<double>(this->ffn_ns.load()) * 1.0e-9, this->attn_calls.load(),
                      static_cast<double>(this->attn_ns.load()) * 1.0e-9, this->ssm_calls.load(),
                      static_cast<double>(this->ssm_ns.load()) * 1.0e-9);
        mnet::detail::write_diagnostic(std::string(line));
    }
};

Stats& stats() {
    static Stats instance;
    return instance;
}

}  // namespace

#if !defined(MNET_METAL)
void register_platform_backends() {}
#endif

void set_matvec_q8_0_backend(MatvecQ8_0Fn backend) {
    backend_slot().store(backend);
}

void set_matvec_q8_0_batch_backend(MatvecQ8_0BatchFn backend) {
    batch_slot().store(backend);
}

void set_ffn_swiglu_backend(FfnSwigluFn backend) {
    ffn_slot().store(backend);
}

void set_attention_backend(AttentionQ2_0Fn backend) {
    attention_slot().store(backend);
}

void set_ssm_backend(SsmQ2_0Fn backend) {
    ssm_slot().store(backend);
}

void set_ffn_swiglu_batch_backend(FfnSwigluBatchFn backend) {
    ffn_batch_slot().store(backend);
}

namespace {

void cpu_matvec(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales, float* out) {
    mnet::gguf::Q8_0Blocks activations;
    activations.qs.assign(q8, q8 + static_cast<std::ptrdiff_t>(columns));
    activations.d.assign(q8_scales, q8_scales + static_cast<std::ptrdiff_t>(columns / 32U));
    const std::uint64_t block_elements = type == mnet::gguf::kGgmlQ8_0 ? 32U : 64U;
    const std::uint64_t blocks = columns / block_elements;

    // Rows are independent and keep a fixed accumulation order, so the result
    // is identical for any thread count.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (rows >= 128)
#endif
    for (std::uint64_t row = 0; row < rows; ++row) {
        const std::uint8_t* row_data = data + row * row_bytes;
        float value = 0.0F;
        if (type == mnet::gguf::kGgmlNvfp4) {
            value = mnet::gguf::dot_nvfp4_q8_0(row_data, activations, blocks);
        } else if (type == mnet::gguf::kGgmlQ2_0) {
            value = mnet::gguf::dot_q2_0_q8_0(row_data, activations, blocks);
        } else {
            value = mnet::gguf::dot_q8_0_q8_0(row_data, activations, blocks);
        }
        out[row] = scale * value;
    }
}

}  // namespace

void matvec_q8_0(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                 std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales, float* out,
                 std::uint64_t identity, const std::uint8_t* region, std::uint64_t region_bytes) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const MatvecQ8_0Fn backend = backend_slot().load();
    if (backend != nullptr) {
        const auto begin = std::chrono::steady_clock::now();
        const bool handled =
            backend(type, data, rows, columns, row_bytes, scale, q8, q8_scales, out, identity, region, region_bytes);
        if (handled) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            stats().backend_ns +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            ++stats().backend_calls;
            return;
        }
    }
    const auto cpu_begin = std::chrono::steady_clock::now();
    cpu_matvec(type, data, rows, columns, row_bytes, scale, q8, q8_scales, out);
    const auto cpu_elapsed = std::chrono::steady_clock::now() - cpu_begin;
    stats().cpu_ns +=
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_elapsed).count());
    ++stats().cpu_calls;
}

void matvec_q8_0_batch(const MatvecQ8_0Job* jobs, std::size_t count, std::uint64_t columns, const std::int8_t* q8,
                       const float* q8_scales) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const MatvecQ8_0BatchFn backend = batch_slot().load();
    if (backend != nullptr) {
        const auto begin = std::chrono::steady_clock::now();
        const bool handled = backend(jobs, count, columns, q8, q8_scales);
        if (handled) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            stats().backend_ns +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            stats().backend_calls += count;
            return;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        cpu_matvec(jobs[i].type, jobs[i].data, jobs[i].rows, columns, jobs[i].row_bytes, jobs[i].scale, q8, q8_scales,
                   jobs[i].out);
    }
    stats().cpu_calls += count;
}

bool ffn_swiglu_q2_0(const MatvecQ8_0Job& gate, const MatvecQ8_0Job& up, const MatvecQ8_0Job& down,
                     std::uint64_t columns, std::uint64_t hidden, const float* input, const std::int32_t* signs,
                     std::uint32_t block_size) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const FfnSwigluFn backend = ffn_slot().load();
    if (backend == nullptr) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool handled = backend(gate, up, down, columns, hidden, input, signs, block_size);
    if (handled) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        stats().ffn_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        ++stats().ffn_calls;
    }
    return handled;
}

bool attention_q2_0(const AttentionQ2_0Job& job) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const AttentionQ2_0Fn backend = attention_slot().load();
    if (backend == nullptr) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool handled = backend(job);
    if (handled) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        stats().attn_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        ++stats().attn_calls;
    }
    return handled;
}

bool ffn_swiglu_batch(const FfnSwigluEntry* entries, std::size_t count, std::uint64_t columns, std::uint64_t hidden,
                      const std::int32_t* signs, std::uint32_t block_size) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const FfnSwigluBatchFn backend = ffn_batch_slot().load();
    if (backend == nullptr) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool handled = backend(entries, count, columns, hidden, signs, block_size);
    if (handled) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        stats().ffn_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        stats().ffn_calls += count;
    }
    return handled;
}

bool ssm_q2_0(const SsmQ2_0Job& job) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const SsmQ2_0Fn backend = ssm_slot().load();
    if (backend == nullptr) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool handled = backend(job);
    if (handled) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        stats().ssm_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        ++stats().ssm_calls;
    }
    return handled;
}

bool fast_path_enabled() {
    static const bool enabled = std::getenv("MNET_METAL_FAST") != nullptr;
    return enabled;
}

bool fused_ffn_enabled() {
    // On by default with the fast path; MNET_NO_FUSED_FFN forces the
    // step-by-step block for A/B tests.
    static const bool disabled = std::getenv("MNET_NO_FUSED_FFN") != nullptr;
    return !disabled;
}

bool fused_attention_enabled() {
    // On by default with the fast path; MNET_NO_FUSED_ATTN forces the
    // step-by-step block for A/B tests.
    static const bool disabled = std::getenv("MNET_NO_FUSED_ATTN") != nullptr;
    return !disabled;
}

bool fused_ssm_enabled() {
    // On by default with the fast path; MNET_NO_FUSED_SSM forces the
    // step-by-step block for A/B tests.
    static const bool disabled = std::getenv("MNET_NO_FUSED_SSM") != nullptr;
    return !disabled;
}

}  // namespace mnet::detail::compute
