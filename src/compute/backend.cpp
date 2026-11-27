// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/detail/compute.hpp>

#include <atomic>
#include <mutex>
#include <vector>

#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>

namespace mnet::detail::compute {
namespace {

std::atomic<MatvecQ8_0Fn>& backend_slot() {
    static std::atomic<MatvecQ8_0Fn> slot{nullptr};
    return slot;
}

}  // namespace

#if !defined(MNET_METAL)
void register_platform_backends() {}
#endif

void set_matvec_q8_0_backend(MatvecQ8_0Fn backend) { backend_slot().store(backend); }

void matvec_q8_0(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                 std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales, float* out) {
    static std::once_flag once;
    std::call_once(once, [] { register_platform_backends(); });
    const MatvecQ8_0Fn backend = backend_slot().load();
    if (backend != nullptr && backend(type, data, rows, columns, row_bytes, scale, q8, q8_scales, out)) {
        return;
    }

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

}  // namespace mnet::detail::compute
