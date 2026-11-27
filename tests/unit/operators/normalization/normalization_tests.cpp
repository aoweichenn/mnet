// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/lrn.hpp>
#include <mlayer/rmsnorm.hpp>
#include <mnet/core/layer.hpp>

TEST(NormalizationKernel, RMSNormAffine) {
    using Norm = mlayer::RMSNorm<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;

    mformat::ParamDict pd;
    pd.set(0, 4);
    pd.set(1, 1e-5F);
    pd.set(2, 1);
    mnet::Mat weights[1];
    weights[0].create(4);
    for (std::int32_t i = 0; i < 4; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.5F + 0.1F * static_cast<float>(i);
    }

    Norm norm;
    ASSERT_EQ(norm.load_param(pd), 0);
    ASSERT_EQ(norm.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(4);
    for (std::int32_t i = 0; i < 4; ++i) {
        input[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    mnet::Mat out = input.clone();

    mnet::Option opt;
    opt.num_threads = 1;
    ASSERT_EQ(norm.forward_inplace(out, opt), 0);

    float sqsum = 0.0F;
    for (std::int32_t i = 0; i < 4; ++i) {
        sqsum += input[static_cast<std::size_t>(i)] * input[static_cast<std::size_t>(i)];
    }
    const float scale = 1.0F / std::sqrt(sqsum / 4.0F + 1e-5F);
    for (std::int32_t i = 0; i < 4; ++i) {
        const float ref = input[static_cast<std::size_t>(i)] * scale * weights[0][static_cast<std::size_t>(i)];
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], ref, 1e-5F);
    }
}

TEST(NormalizationKernel, LrnMatchesReference) {
    using LRN = mlayer::LRN<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    for (std::int32_t region_type = 0; region_type < 2; ++region_type) {
        mformat::ParamDict pd;
        pd.set(0, region_type);
        pd.set(1, 3);
        pd.set(2, 0.001F);
        pd.set(3, 0.75F);
        pd.set(4, 1.0F);
        LRN op;
        ASSERT_EQ(op.load_param(pd), 0);

        const std::int32_t w = 5;
        const std::int32_t h = 4;
        const std::int32_t c = 3;
        mnet::Mat in(w, h, c);
        std::vector<float> src(static_cast<std::size_t>(w * h * c));
        for (std::int32_t q = 0; q < c; ++q) {
            for (std::int32_t i = 0; i < w * h; ++i) {
                const float v = std::sin(0.43F * static_cast<float>(q * w * h + i) + 0.2F);
                src[static_cast<std::size_t>(q * w * h + i)] = v;
                in.channel(q)[static_cast<std::size_t>(i)] = v;
            }
        }

        mnet::Mat out = in;
        mnet::Option opt;
        ASSERT_EQ(op.forward_inplace(out, opt), 0);

        for (std::int32_t q = 0; q < c; ++q) {
            for (std::int32_t y = 0; y < h; ++y) {
                for (std::int32_t x = 0; x < w; ++x) {
                    float sum = 0.0F;
                    if (region_type == 0) {
                        for (std::int32_t p = q - 1; p <= q + 1; ++p) {
                            if (p < 0 || p >= c) {
                                continue;
                            }
                            const float v = src[static_cast<std::size_t>((p * h + y) * w + x)];
                            sum += v * v;
                        }
                        sum *= 0.001F / 3.0F;
                    } else {
                        for (std::int32_t ky = -1; ky <= 1; ++ky) {
                            for (std::int32_t kx = -1; kx <= 1; ++kx) {
                                const std::int32_t sy = y + ky;
                                const std::int32_t sx = x + kx;
                                if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                                    continue;
                                }
                                const float v = src[static_cast<std::size_t>((q * h + sy) * w + sx)];
                                sum += v * v;
                            }
                        }
                        sum *= 0.001F / 9.0F;
                    }
                    const float scale = std::pow(1.0F + sum, -0.75F);
                    const float expect = src[static_cast<std::size_t>((q * h + y) * w + x)] * scale;
                    EXPECT_NEAR(out.channel(q)[static_cast<std::size_t>(y * w + x)], expect, 1e-6F)
                        << "region=" << region_type << " q=" << q << " y=" << y << " x=" << x;
                }
            }
        }
    }
}
