// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_BORDER_HPP
#define MNET_BORDER_HPP

#include <cstdint>
#include <cstring>

#include <mnet/mat.hpp>
#include <mnet/option.hpp>

namespace mnet {

// Copy `src` into a zero/constant padded `dst`. Currently float32 only, which
// covers the CPU fp32 layer set.
inline void copy_make_border(const Mat& src, Mat& dst, std::int32_t top, std::int32_t bottom, std::int32_t left,
                             std::int32_t right, std::int32_t /*type*/, float v, const Option& opt) {
    if (top == 0 && bottom == 0 && left == 0 && right == 0) {
        dst = src;
        return;
    }
    if (src.elemsize != 4u) {
        dst = src;
        return;
    }

    const std::int32_t w = src.w;
    const std::int32_t h = src.h;
    const std::int32_t c = src.c;
    const std::int32_t outw = w + left + right;
    const std::int32_t outh = h + top + bottom;

    // dst may alias src through a shallow copy; release first so create()
    // cannot take its "same shape" fast path and reuse the shared buffer.
    dst.release();
    dst.create(outw, outh, c, 4u, opt.workspace_allocator);
    if (dst.empty()) {
        return;
    }
    dst.fill(v);

    for (std::int32_t q = 0; q < c; ++q) {
        for (std::int32_t y = 0; y < h; ++y) {
            std::memcpy(dst.channel(q).row(y + top) + left, src.channel(q).row(y),
                        static_cast<std::size_t>(w) * sizeof(float));
        }
    }
}

// Flatten all channels into a 1D Mat without copying more than once per channel.
inline void flatten(const Mat& src, Mat& dst, const Option& opt) {
    if (src.dims == 1) {
        dst = src;
        return;
    }

    const std::int32_t size = src.w * src.h * src.d;
    dst.release();
    dst.create(size * src.c, src.elemsize, opt.blob_allocator);
    if (dst.empty()) {
        return;
    }

    auto* outptr = static_cast<unsigned char*>(dst);
    for (std::int32_t q = 0; q < src.c; ++q) {
        std::memcpy(outptr, static_cast<const unsigned char*>(src.channel(q)),
                    static_cast<std::size_t>(size) * src.elemsize);
        outptr += static_cast<std::size_t>(size) * src.elemsize;
    }
}

}  // namespace mnet

#endif  // MNET_BORDER_HPP
