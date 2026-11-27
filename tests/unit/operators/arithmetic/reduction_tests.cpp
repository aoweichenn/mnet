// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/cumulativesum.hpp>
#include <mlayer/reduction.hpp>
#include <mlayer/unaryop.hpp>
#include <mnet/core/layer.hpp>

TEST(ArithmeticKernel, UnarySqrtScalarAndVector) {
    using U = mlayer::UnaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 5);
    U op;
    ASSERT_EQ(op.load_param(pd), 0);
    mnet::Option opt;
    mnet::Mat one(1);
    one[0] = 0.07177F;
    ASSERT_EQ(op.forward_inplace(one, opt), 0);
    EXPECT_NEAR(one[0], 0.267900F, 1e-5F);
    mnet::Mat vec(4);
    vec[0] = 1;
    vec[1] = 4;
    vec[2] = 9;
    vec[3] = 16;
    ASSERT_EQ(op.forward_inplace(vec, opt), 0);
    EXPECT_FLOAT_EQ(vec[0], 1);
    EXPECT_FLOAT_EQ(vec[1], 2);
    EXPECT_FLOAT_EQ(vec[2], 3);
    EXPECT_FLOAT_EQ(vec[3], 4);
}

TEST(ArithmeticKernel, ReductionScalarAxisSumsWidthToVector) {
    using Red = mlayer::Reduction<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 0);  // sum
    pd.set(6, 1);  // reduce width (2-D: 0=h, 1=w)
    Red red;
    ASSERT_EQ(red.load_param(pd), 0);
    mnet::Mat in(4, 3);
    for (std::int32_t y = 0; y < 3; ++y) {
        for (std::int32_t x = 0; x < 4; ++x) {
            in.row(y)[x] = static_cast<float>(y * 10 + x + 1);
        }
    }
    mnet::Mat out;
    mnet::Option opt;
    ASSERT_EQ(red.forward(in, out, opt), 0);
    EXPECT_EQ(out.dims, 1);
    EXPECT_EQ(out.w, 3);
    EXPECT_FLOAT_EQ(out[0], 10.0F);
    EXPECT_FLOAT_EQ(out[1], 50.0F);
    EXPECT_FLOAT_EQ(out[2], 90.0F);
}

TEST(ArithmeticKernel, CumulativeSumAxisMatchesHeight) {
    constexpr std::int32_t MNET_CUMULATIVE_SUM_TEST_WIDTH = 3;
    constexpr std::int32_t MNET_CUMULATIVE_SUM_TEST_HEIGHT = 4;
    using CumSum = mlayer::CumulativeSum<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    mformat::ParamDict pd;
    pd.set(0, 1);
    CumSum sum;
    ASSERT_EQ(sum.load_param(pd), 0);

    mnet::Mat input(MNET_CUMULATIVE_SUM_TEST_WIDTH, MNET_CUMULATIVE_SUM_TEST_HEIGHT, 1);
    for (std::int32_t y = 0; y < MNET_CUMULATIVE_SUM_TEST_HEIGHT; ++y) {
        for (std::int32_t x = 0; x < MNET_CUMULATIVE_SUM_TEST_WIDTH; ++x) {
            input.channel(0).row(y)[x] = static_cast<float>(y * MNET_CUMULATIVE_SUM_TEST_WIDTH + x + 1);
        }
    }

    mnet::Option opt;
    ASSERT_EQ(sum.forward_inplace(input, opt), 0);
    for (std::int32_t x = 0; x < MNET_CUMULATIVE_SUM_TEST_WIDTH; ++x) {
        float expected = 0.0F;
        for (std::int32_t y = 0; y < MNET_CUMULATIVE_SUM_TEST_HEIGHT; ++y) {
            expected += static_cast<float>(y * MNET_CUMULATIVE_SUM_TEST_WIDTH + x + 1);
            EXPECT_FLOAT_EQ(input.channel(0).row(y)[x], expected);
        }
    }
}

TEST(ArithmeticKernel, UnaryOpTanhAndSquare) {
    using UO = mlayer::UnaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mnet::Option opt;
    mnet::Mat data(3);
    data[0] = -1.0F;
    data[1] = 0.0F;
    data[2] = 2.0F;

    mformat::ParamDict square_pd;
    square_pd.set(0, static_cast<std::int32_t>(UO::Operation_SQUARE));
    UO square;
    ASSERT_EQ(square.load_param(square_pd), 0);
    mnet::Mat squared = data;
    ASSERT_EQ(square.forward_inplace(squared, opt), 0);
    EXPECT_FLOAT_EQ(squared[0], 1.0F);
    EXPECT_FLOAT_EQ(squared[1], 0.0F);
    EXPECT_FLOAT_EQ(squared[2], 4.0F);

    mformat::ParamDict tanh_pd;
    tanh_pd.set(0, static_cast<std::int32_t>(UO::Operation_TANH));
    UO tanh;
    ASSERT_EQ(tanh.load_param(tanh_pd), 0);
    mnet::Mat th(3);
    th[0] = -1.0F;
    th[1] = 0.0F;
    th[2] = 2.0F;
    ASSERT_EQ(tanh.forward_inplace(th, opt), 0);
    EXPECT_NEAR(th[0], std::tanh(-1.0F), 1e-6F);
    EXPECT_NEAR(th[2], std::tanh(2.0F), 1e-6F);
}

TEST(ArithmeticKernel, ReductionMaxAndMean) {
    using RD = mlayer::Reduction<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mnet::Option opt;

    mformat::ParamDict max_pd;
    max_pd.set(0, static_cast<std::int32_t>(RD::ReductionOp_MAX));
    max_pd.set(1, 1);
    RD max_op;
    ASSERT_EQ(max_op.load_param(max_pd), 0);
    mnet::Mat input(3, 2);
    const float values[6] = {1.0F, -4.0F, 3.0F, 2.0F, 0.0F, 5.0F};
    std::memcpy(input.data, values, sizeof(values));
    mnet::Mat out;
    ASSERT_EQ(max_op.forward(input, out, opt), 0);
    ASSERT_EQ(out.dims, 1);
    ASSERT_EQ(out.w, 1);
    EXPECT_FLOAT_EQ(out[0], 5.0F);

    mformat::ParamDict mean_pd;
    mean_pd.set(0, static_cast<std::int32_t>(RD::ReductionOp_MEAN));
    mean_pd.set(1, 1);
    RD mean_op;
    ASSERT_EQ(mean_op.load_param(mean_pd), 0);
    ASSERT_EQ(mean_op.forward(input, out, opt), 0);
    EXPECT_NEAR(out[0], (1.0F - 4.0F + 3.0F + 2.0F + 0.0F + 5.0F) / 6.0F, 1e-6F);
}

TEST(ArithmeticKernel, ReductionLastAxisOn4dBlob) {
    using RD = mlayer::Reduction<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, static_cast<std::int32_t>(RD::ReductionOp_MAX));
    pd.set(1, 0);
    mnet::Mat axes(1);
    static_cast<std::int32_t*>(static_cast<void*>(axes.data))[0] = -1;
    pd.set(3, axes);
    pd.set(4, 0);
    pd.set(5, 1);
    RD op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat in(8, 3, 2, 4);
    in.fill(1.0F);
    mnet::Option opt;
    mnet::Mat out;
    ASSERT_EQ(op.forward(in, out, opt), 0);
    // w is reduced; h, d, c are preserved as a 3D blob (h, d, c).
    ASSERT_EQ(out.dims, 3);
    ASSERT_EQ(out.w, 3);
    ASSERT_EQ(out.h, 2);
    ASSERT_EQ(out.c, 4);
}

TEST(ArithmeticKernel, ReductionLastAxisOn3dBlob) {
    using RD = mlayer::Reduction<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, static_cast<std::int32_t>(RD::ReductionOp_MAX));
    pd.set(1, 0);
    mnet::Mat axes(1);
    static_cast<std::int32_t*>(static_cast<void*>(axes.data))[0] = -1;
    pd.set(3, axes);
    pd.set(4, 0);
    pd.set(5, 1);
    RD op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat in(5, 3, 2);
    for (std::int32_t q = 0; q < 2; ++q) {
        for (std::int32_t i = 0; i < 15; ++i) {
            in.channel(q)[static_cast<std::size_t>(i)] = static_cast<float>(q * 100 + i);
        }
    }
    mnet::Option opt;
    mnet::Mat out;
    ASSERT_EQ(op.forward(in, out, opt), 0);
    ASSERT_EQ(out.dims, 2);
    ASSERT_EQ(out.w, 3);
    ASSERT_EQ(out.h, 2);
    const float expect[6] = {4.0F, 9.0F, 14.0F, 104.0F, 109.0F, 114.0F};
    for (std::int32_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(i)], expect[i]);
    }
}
