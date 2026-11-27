// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_QUANTIZE_HPP
#define MNET_QUANTIZE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <mnet/core/mat.hpp>
#include <mnet/core/option.hpp>

namespace mnet {

// ncnn-compatible int8 quantization used by the int8 layer kernels:
// q = round(v * scale), clamped to [-127, 127]. `scale_data` holds one scale
// for the whole blob or one scale per channel (ncnn's Quantize layer).
inline void quantize_int8(const Mat& src, Mat& dst, const Mat& scale_data, const Option& opt) {
    const std::int32_t dims = src.dims;
    const std::int32_t w = src.w;
    const std::int32_t h = src.h;
    const std::int32_t d = src.d;
    const std::int32_t channels = src.c;
    const std::int32_t scale_count = scale_data.w;

    const auto quantize_row = [](const float* ptr, std::int8_t* out, float scale, std::int32_t size) {
        for (std::int32_t i = 0; i < size; ++i) {
            const auto v = static_cast<std::int32_t>(std::round(ptr[i] * scale));
            out[i] = static_cast<std::int8_t>(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    };

    if (dims == 1) {
        dst.create(w, static_cast<std::size_t>(1u), opt.blob_allocator);
        if (dst.empty()) {
            return;
        }
        quantize_row(src, dst, scale_data[0], w);
        return;
    }

    if (dims == 2) {
        dst.create(w, h, static_cast<std::size_t>(1u), opt.blob_allocator);
        if (dst.empty()) {
            return;
        }
#pragma omp parallel for num_threads(opt.num_threads)
        for (std::int32_t i = 0; i < h; ++i) {
            const float scale = scale_count == 1 ? scale_data[0] : scale_data[static_cast<std::size_t>(i)];
            quantize_row(src.row(i), dst.row<std::int8_t>(i), scale, w);
        }
        return;
    }

    if (dims == 3) {
        dst.create(w, h, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    } else {
        dst.create(w, h, d, channels, static_cast<std::size_t>(1u), opt.blob_allocator);
    }
    if (dst.empty()) {
        return;
    }
#pragma omp parallel for num_threads(opt.num_threads)
    for (std::int32_t q = 0; q < channels; ++q) {
        const float scale = scale_count == 1 ? scale_data[0] : scale_data[static_cast<std::size_t>(q)];
        quantize_row(src.channel(q), dst.channel(q), scale, w * h * d);
    }
}

}  // namespace mnet

#endif  // MNET_QUANTIZE_HPP
