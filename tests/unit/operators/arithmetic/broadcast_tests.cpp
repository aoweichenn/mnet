// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/binaryop.hpp>
#include <mlayer/compare.hpp>
#include <mlayer/where.hpp>
#include <mnet/layer.hpp>

TEST(ArithmeticKernel, BinaryOpAddAndScalarMul) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;

    mformat::ParamDict pd;
    pd.set(0, 0);
    BinOp add;
    ASSERT_EQ(add.load_param(pd), 0);

    mnet::Mat a(4);
    mnet::Mat b(4);
    for (std::int32_t i = 0; i < 4; ++i) {
        a[static_cast<std::size_t>(i)] = static_cast<float>(i);
        b[static_cast<std::size_t>(i)] = static_cast<float>(10 + i);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    opt.num_threads = 2;
    ASSERT_EQ(add.forward(bottoms, tops, opt), 0);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(tops[0][static_cast<std::size_t>(i)], static_cast<float>(10 + 2 * i));
    }

    mformat::ParamDict pd2;
    pd2.set(0, 2);
    pd2.set(1, 1);
    pd2.set(2, 3.0F);
    BinOp mul;
    ASSERT_EQ(mul.load_param(pd2), 0);
    mnet::Mat c = a.clone();
    ASSERT_EQ(mul.forward_inplace(c, opt), 0);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c[static_cast<std::size_t>(i)], static_cast<float>(i) * 3.0F);
    }
}

TEST(ArithmeticKernel, CompareMasksWithPerChannelBroadcast) {
    using Cmp = mlayer::Compare<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 4);  // greater
    Cmp gt;
    ASSERT_EQ(gt.load_param(pd), 0);

    constexpr std::int32_t w = 4;
    constexpr std::int32_t h = 3;
    mnet::Mat a(w, h, 1);
    mnet::Mat b(1, h, 1);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            a.channel(0).row(y)[x] = static_cast<float>(x);
        }
        b.channel(0).row(y)[0] = static_cast<float>(y);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(gt.forward(bottoms, tops, opt), 0);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            const float expect = x > y ? 1.0F : 0.0F;
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], expect) << "x=" << x << " y=" << y;
        }
    }
}

TEST(ArithmeticKernel, WhereSelectsBroadcastCondition) {
    using Wh = mlayer::Where<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    Wh where;
    ASSERT_EQ(where.load_param(pd), 0);

    constexpr std::int32_t w = 4;
    constexpr std::int32_t h = 2;
    mnet::Mat cond(w, h, 1);
    mnet::Mat a(1, h, 1);
    mnet::Mat b(1, h, 1);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            cond.channel(0).row(y)[x] = (x % 2 == 0) ? 1.0F : 0.0F;
        }
        a.channel(0).row(y)[0] = 10.0F + static_cast<float>(y);
        b.channel(0).row(y)[0] = 20.0F + static_cast<float>(y);
    }
    std::vector<mnet::Mat> bottoms = {cond, a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(where.forward(bottoms, tops, opt), 0);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            const float expect = (x % 2 == 0) ? 10.0F + static_cast<float>(y) : 20.0F + static_cast<float>(y);
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], expect) << "x=" << x << " y=" << y;
        }
    }
}

// A per-channel vector must broadcast along the height axis, not the width:
// ONNX [1, C, 1] against [1, C, L] is the most common transformer pattern.
TEST(ArithmeticKernel, BinaryOpBroadcastHeightVector) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 0);
    BinOp add;
    ASSERT_EQ(add.load_param(pd), 0);

    constexpr std::int32_t w = 8;
    constexpr std::int32_t h = 4;
    mnet::Mat a(w, h, 1);
    mnet::Mat b(1, h, 1);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            a.channel(0).row(y)[x] = static_cast<float>(x) + 100.0F * static_cast<float>(y);
        }
        b.channel(0).row(y)[0] = 1000.0F + static_cast<float>(y);
    }

    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    opt.num_threads = 2;
    ASSERT_EQ(add.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, w);
    ASSERT_EQ(tops[0].h, h);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            const float expect =
                static_cast<float>(x) + 100.0F * static_cast<float>(y) + 1000.0F + static_cast<float>(y);
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], expect) << "at x=" << x << " y=" << y;
        }
    }

    // Same values but B as a 1-D blob: the emitter words constants that way.
    mnet::Mat flat(h);
    for (std::int32_t y = 0; y < h; ++y) {
        flat[static_cast<std::size_t>(y)] = 1000.0F + static_cast<float>(y);
    }
    std::vector<mnet::Mat> bottoms_flat = {a, flat};
    std::vector<mnet::Mat> tops_flat(1);
    ASSERT_EQ(add.forward(bottoms_flat, tops_flat, opt), 0);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            const float expect =
                static_cast<float>(x) + 100.0F * static_cast<float>(y) + 1000.0F + static_cast<float>(y);
            EXPECT_FLOAT_EQ(tops_flat[0].channel(0).row(y)[x], expect) << "flat at x=" << x << " y=" << y;
        }
    }
}

// A 2-D (w,h) blob meeting a 3-D (w,h,1) blob must stay elementwise.
TEST(ArithmeticKernel, BinaryOpRowVectorTimesChannelVector) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 2);
    BinOp mul;
    ASSERT_EQ(mul.load_param(pd), 0);
    constexpr std::int32_t t = 8;
    constexpr std::int32_t c = 3;
    mnet::Mat a(1, t);     // (w=1,h=T)
    mnet::Mat b(c, 1, 1);  // (w=C,h=1,c=1)
    for (std::int32_t i = 0; i < t; ++i) {
        a.row(i)[0] = static_cast<float>(i + 1);
    }
    for (std::int32_t i = 0; i < c; ++i) {
        b.channel(0).row(0)[i] = static_cast<float>(i + 2);
    }
    std::vector<mnet::Mat> bottoms = {a, b}, tops(1);
    mnet::Option opt;
    ASSERT_EQ(mul.forward(bottoms, tops, opt), 0);
    EXPECT_EQ(tops[0].w, c);
    EXPECT_EQ(tops[0].h, t);
    EXPECT_EQ(tops[0].c, 1);
    for (std::int32_t y = 0; y < t; ++y) {
        for (std::int32_t x = 0; x < c; ++x) {
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], static_cast<float>((x + 2) * (y + 1)));
        }
    }
}

TEST(ArithmeticKernel, BinaryOpMixesTwoAndThreeDimensionalBlobs) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 0);
    BinOp add;
    ASSERT_EQ(add.load_param(pd), 0);

    constexpr std::int32_t w = 5;
    constexpr std::int32_t h = 3;
    mnet::Mat a(w, h);     // dims == 2
    mnet::Mat b(w, h, 1);  // dims == 3
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            a.row(y)[x] = static_cast<float>(x + y);
            b.channel(0).row(y)[x] = static_cast<float>(10 * x + y);
        }
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(add.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, w);
    ASSERT_EQ(tops[0].h, h);
    ASSERT_EQ(tops[0].c, 1);
    for (std::int32_t y = 0; y < h; ++y) {
        for (std::int32_t x = 0; x < w; ++x) {
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], static_cast<float>(11 * x + 2 * y));
        }
    }
}
