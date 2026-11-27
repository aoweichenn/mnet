// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#include <gtest/gtest.h>
#include <mnet/core/layer.hpp>
#include <mnet/ops/border.hpp>

TEST(ConvolutionKernel, CopyMakeBorderZeroPadPreservesData) {
    mnet::Mat src(4, 3, 2);
    for (std::int32_t q = 0; q < src.c; ++q) {
        float* p = src.channel(q);
        for (std::int32_t i = 0; i < src.w * src.h; ++i) {
            p[i] = static_cast<float>(q * 100 + i);
        }
    }

    mnet::Option opt;
    mnet::Mat dst;
    mnet::copy_make_border(src, dst, 0, 0, 0, 0, 0, -3.402823466e+38F, opt);

    ASSERT_EQ(dst.w, src.w);
    ASSERT_EQ(dst.h, src.h);
    ASSERT_EQ(dst.c, src.c);
    for (std::int32_t q = 0; q < src.c; ++q) {
        const float* a = src.channel(q);
        const float* b = dst.channel(q);
        for (std::int32_t i = 0; i < src.w * src.h; ++i) {
            EXPECT_FLOAT_EQ(a[i], b[i]) << q << " " << i;
        }
    }
}
