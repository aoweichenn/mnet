// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstring>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/gateddeltarule.hpp>
#include <mlayer/shortconv.hpp>
#include <mnet/layer.hpp>

// Causal depthwise short convolution: y[t] = sum_k xs[t + k] * w[k] with the
// K-1 history tokens as prefix; activation 0 disables the Qwen3.5 SiLU.
TEST(SequenceKernel, ShortConvCausalKernel) {
    using C = mlayer::ShortConv<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 0);  // no activation
    C conv;
    ASSERT_EQ(conv.load_param(pd), 0);

    const std::int32_t kernel = 3;
    const std::int32_t groups = 2;
    const std::int32_t tokens = 2;

    mnet::Mat weight(kernel, 1, groups);
    for (std::int32_t g = 0; g < groups; ++g) {
        float* w = weight.channel(g);
        for (std::int32_t k = 0; k < kernel; ++k) {
            w[k] = 0.5F * static_cast<float>(k + 1) + static_cast<float>(g);
        }
    }

    mnet::Mat x(groups, tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        float* row = x.row(t);
        for (std::int32_t g = 0; g < groups; ++g) {
            row[g] = static_cast<float>(g * 10 + t + 1);
        }
    }

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {weight, x};
    std::vector<mnet::Mat> top(2);
    ASSERT_EQ(conv.forward(bottom, top, opt), 0);

    for (std::int32_t g = 0; g < groups; ++g) {
        const float* w = weight.channel(g);
        const float x0 = x.row(0)[g];
        const float x1 = x.row(1)[g];
        const float y0 = x0 * w[2];  // xs = [0, 0, x0, x1]
        const float y1 = x0 * w[1] + x1 * w[2];
        EXPECT_NEAR(top[0].row(0)[g], y0, 1e-6F);
        EXPECT_NEAR(top[0].row(1)[g], y1, 1e-6F);
    }

    // The rolling state keeps the last K inputs: [0, x0, x1].
    ASSERT_EQ(top[1].w, kernel);
    ASSERT_EQ(top[1].h, groups);
    for (std::int32_t g = 0; g < groups; ++g) {
        EXPECT_NEAR(top[1].row(g)[0], 0.0F, 1e-6F);
        EXPECT_NEAR(top[1].row(g)[1], x.row(0)[g], 1e-6F);
        EXPECT_NEAR(top[1].row(g)[2], x.row(1)[g], 1e-6F);
    }
}

TEST(SequenceKernel, ShortConvCausalState) {
    using SC = mlayer::ShortConv<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mnet::Option opt;

    mnet::Mat weight(3, 1, 2);
    const float w0[3] = {1.0F, 2.0F, 3.0F};
    const float w1[3] = {0.5F, 0.5F, 0.5F};
    std::memcpy(weight.channel(0), w0, sizeof(w0));
    std::memcpy(weight.channel(1), w1, sizeof(w1));

    mnet::Mat mixed(2, 2);
    mixed.row(0)[0] = 1.0F;
    mixed.row(0)[1] = 3.0F;
    mixed.row(1)[0] = 2.0F;
    mixed.row(1)[1] = 4.0F;

    std::vector<mnet::Mat> bottoms = {weight, mixed};
    std::vector<mnet::Mat> tops(2);
    SC layer;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);

    const float silu3 = 3.0F / (1.0F + std::exp(-3.0F));
    const float silu8 = 8.0F / (1.0F + std::exp(-8.0F));
    const float silu15 = 1.5F / (1.0F + std::exp(-1.5F));
    const float silu35 = 3.5F / (1.0F + std::exp(-3.5F));
    EXPECT_NEAR(tops[0].row(0)[0], silu3, 1e-5F);
    EXPECT_NEAR(tops[0].row(1)[0], silu8, 1e-5F);
    EXPECT_NEAR(tops[0].row(0)[1], silu15, 1e-5F);
    EXPECT_NEAR(tops[0].row(1)[1], silu35, 1e-5F);

    ASSERT_EQ(tops[1].w, 3);
    ASSERT_EQ(tops[1].h, 2);
    EXPECT_FLOAT_EQ(tops[1].row(0)[0], 0.0F);
    EXPECT_FLOAT_EQ(tops[1].row(0)[1], 1.0F);
    EXPECT_FLOAT_EQ(tops[1].row(0)[2], 2.0F);
}

TEST(SequenceKernel, GatedDeltaRuleSingleStep) {
    using GDR = mlayer::GatedDeltaRule<mnet::Layer, mnet::Mat, mnet::Option>;
    mnet::Option opt;

    mnet::Mat A_log(1);
    A_log[0] = 0.0F;
    mnet::Mat dt_bias(1);
    dt_bias[0] = 0.0F;
    mnet::Mat b(1, 1);
    b[0] = 0.0F;
    mnet::Mat a(1, 1);
    a[0] = 0.0F;

    mnet::Mat query(2, 1, 1);
    query[0] = 1.0F;
    query[1] = 0.0F;
    mnet::Mat key(2, 1, 1);
    key[0] = 1.0F;
    key[1] = 0.0F;
    mnet::Mat value(2, 1, 1);
    value[0] = 1.0F;
    value[1] = 2.0F;

    std::vector<mnet::Mat> bottoms = {A_log, dt_bias, b, a, query, key, value};
    std::vector<mnet::Mat> tops(2);
    GDR layer;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);

    const float scale = 1.0F / std::sqrt(2.0F);
    EXPECT_NEAR(tops[0][0], 0.5F * scale, 1e-5F);
    EXPECT_NEAR(tops[0][1], 1.0F * scale, 1e-5F);
    EXPECT_NEAR(tops[1][0], 0.5F, 1e-5F);
    EXPECT_NEAR(tops[1][1], 1.0F, 1e-5F);
    EXPECT_NEAR(tops[1][2], 0.0F, 1e-5F);
    EXPECT_NEAR(tops[1][3], 0.0F, 1e-5F);
}
