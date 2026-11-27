// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/situ.hpp>
#include <mnet/layer.hpp>

// SiTU gated activation (Inkling-2-Mini MoE): beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb).
TEST(ActivationKernel, SituMatchesReference) {
    using L = mlayer::Situ<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 4.0F);
    pd.set(1, 25.0F);
    L situ;
    ASSERT_EQ(situ.load_param(pd), 0);

    const float gates[4] = {0.0F, 1.0F, -1.0F, 2.5F};
    const float ups[4] = {0.0F, 1.0F, -2.0F, 3.5F};
    mnet::Mat gate(4);
    mnet::Mat up(4);
    for (std::int32_t i = 0; i < 4; ++i) {
        gate[static_cast<std::size_t>(i)] = gates[i];
        up[static_cast<std::size_t>(i)] = ups[i];
    }

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {gate, up};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(situ.forward(bottom, top, opt), 0);

    for (std::int32_t i = 0; i < 4; ++i) {
        const float beta = 4.0F;
        const float lb = 25.0F;
        const float act = beta * std::tanh(gates[i] / beta) / (1.0F + std::exp(-gates[i]));
        const float expected = act * (lb * std::tanh(ups[i] / lb));
        EXPECT_NEAR(top[0][static_cast<std::size_t>(i)], expected, 1e-6F);
    }
}
