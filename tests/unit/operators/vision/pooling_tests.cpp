// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/interp.hpp>
#include <mlayer/pooling.hpp>
#include <mlayer/proposal.hpp>
#include <mlayer/psroipooling.hpp>
#include <mlayer/roipooling.hpp>
#include <mnet/border.hpp>
#include <mnet/layer.hpp>

TEST(VisionKernel, PoolingMaxNoPadMatchesEveryWindow) {
    using Pool =
        mlayer::Pooling<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false, mnet::copy_make_border>;

    mformat::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(2, 2);
    pd.set(12, 2);

    Pool pool;
    ASSERT_EQ(pool.load_param(pd), 0);

    mnet::Mat input(113, 113, 64);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < input.w * input.h; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    mnet::Mat out;
    ASSERT_EQ(pool.forward(input, out, opt), 0);
    ASSERT_EQ(out.w, 56);
    ASSERT_EQ(out.h, 56);
    ASSERT_EQ(out.c, input.c);
    for (std::int32_t q = 0; q < out.c; ++q) {
        const auto source = input.channel(q);
        const auto output = out.channel(q);
        for (std::int32_t y = 0; y < out.h; ++y) {
            for (std::int32_t x = 0; x < out.w; ++x) {
                auto expected = -std::numeric_limits<float>::infinity();
                for (std::int32_t ky = 0; ky < 3; ++ky) {
                    for (std::int32_t kx = 0; kx < 3; ++kx) {
                        expected = std::max(expected, source.row(y * 2 + ky)[x * 2 + kx]);
                    }
                }
                EXPECT_FLOAT_EQ(output.row(y)[x], expected) << q << " " << y << " " << x;
            }
        }
    }
}

TEST(VisionKernel, PSROIPoolingAveragesRoi) {
    using PS = mlayer::PSROIPooling<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 2);     // pooled_width
    pd.set(1, 2);     // pooled_height
    pd.set(2, 1.0F);  // spatial_scale
    pd.set(3, 1);     // output_dim
    PS layer;
    ASSERT_EQ(layer.load_param(pd), 0);

    mnet::Mat feature(4, 4, 4);
    for (std::int32_t q = 0; q < 4; ++q) {
        for (std::int32_t i = 0; i < 16; ++i) {
            feature.channel(q)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1 + q * 16);
        }
    }
    mnet::Mat roi(4);
    roi[0] = 0.0F;
    roi[1] = 0.0F;
    roi[2] = 3.0F;
    roi[3] = 3.0F;

    std::vector<mnet::Mat> bottoms = {feature, roi};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 2);
    ASSERT_EQ(tops[0].h, 2);
    ASSERT_EQ(tops[0].c, 1);
    // Each output bin reads a different channel, whose base increases by 16.
    EXPECT_NEAR(tops[0][0], 3.5F, 1e-5F);
    EXPECT_NEAR(tops[0][1], 21.5F, 1e-5F);
    EXPECT_NEAR(tops[0][2], 43.5F, 1e-5F);
    EXPECT_NEAR(tops[0][3], 61.5F, 1e-5F);
}

TEST(VisionKernel, ProposalProducesClampedRois) {
    using PR = mlayer::Proposal<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    PR layer;
    ASSERT_EQ(layer.load_param(mformat::ParamDict()), 0);
    ASSERT_EQ(layer.anchors.h, 9);

    const std::int32_t w = 2;
    const std::int32_t h = 2;
    const std::int32_t num_anchors = layer.anchors.h;
    mnet::Mat scores(2 * num_anchors, w * h);
    scores.fill(1.0F);
    mnet::Mat bbox(4 * num_anchors, w * h);
    bbox.fill(0.0F);
    mnet::Mat im_info(3);
    im_info[0] = 600.0F;
    im_info[1] = 600.0F;
    im_info[2] = 1.0F;

    std::vector<mnet::Mat> bottoms = {scores, bbox, im_info};
    std::vector<mnet::Mat> tops(2);
    mnet::Option opt;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].dims, 3);
    ASSERT_EQ(tops[0].w, 4);
    ASSERT_EQ(tops[0].h, 1);
    ASSERT_GT(tops[0].c, 0);
    for (std::int32_t i = 0; i < tops[0].c; ++i) {
        const float* r = tops[0].channel(i);
        EXPECT_GE(r[0], 0.0F);
        EXPECT_GE(r[1], 0.0F);
        EXPECT_LE(r[2], 599.0F);
        EXPECT_LE(r[3], 599.0F);
        EXPECT_LE(r[0], r[2]);
        EXPECT_LE(r[1], r[3]);
    }
    EXPECT_EQ(tops[1].c, tops[0].c);
}

TEST(VisionKernel, InterpBilinearEdgeClamps) {
    using IP = mlayer::Interp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 2);  // bilinear
    pd.set(3, 4);  // output height
    pd.set(4, 4);  // output width
    IP op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat in(2, 2);
    in[0] = 1.0F;
    in[1] = 2.0F;
    in[2] = 3.0F;
    in[3] = 4.0F;
    mnet::Option opt;
    mnet::Mat out;
    ASSERT_EQ(op.forward(in, out, opt), 0);
    ASSERT_EQ(out.w, 4);
    ASSERT_EQ(out.h, 4);
    const float expect[16] = {1.0F, 1.25F, 1.75F, 2.0F, 1.5F, 1.75F, 2.25F, 2.5F,
                              2.5F, 2.75F, 3.25F, 3.5F, 3.0F, 3.25F, 3.75F, 4.0F};
    for (std::int32_t i = 0; i < 16; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], expect[i], 1e-6F);
    }
}

TEST(VisionKernel, RoiPoolingMatchesReference) {
    using ROI = mlayer::ROIPooling<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    mformat::ParamDict pd;
    pd.set(0, 3);
    pd.set(1, 2);
    pd.set(2, 0.5F);
    ROI op;
    ASSERT_EQ(op.load_param(pd), 0);

    const std::int32_t w = 8;
    const std::int32_t h = 6;
    const std::int32_t c = 2;
    mnet::Mat feat(w, h, c);
    std::vector<float> src(static_cast<std::size_t>(w * h * c));
    for (std::int32_t i = 0; i < w * h * c; ++i) {
        src[static_cast<std::size_t>(i)] = std::sin(0.37F * static_cast<float>(i) - 0.4F);
        feat.channel(i / (w * h))[static_cast<std::size_t>(i % (w * h))] = src[static_cast<std::size_t>(i)];
    }
    mnet::Mat rois(4, 1);
    rois[0] = 1.0F;
    rois[1] = 2.0F;
    rois[2] = 12.0F;
    rois[3] = 9.0F;

    std::vector<mnet::Mat> bottoms = {feat, rois};
    std::vector<mnet::Mat> tops = {mnet::Mat()};
    mnet::Option opt;
    ASSERT_EQ(op.forward(bottoms, tops, opt), 0);
    const mnet::Mat& out = tops[0];
    ASSERT_EQ(out.w, 3);
    ASSERT_EQ(out.h, 2);
    ASSERT_EQ(out.c, c);

    const std::int32_t roi_x1 = static_cast<std::int32_t>(std::round(rois[0] * 0.5F));
    const std::int32_t roi_y1 = static_cast<std::int32_t>(std::round(rois[1] * 0.5F));
    const std::int32_t roi_x2 = static_cast<std::int32_t>(std::round(rois[2] * 0.5F));
    const std::int32_t roi_y2 = static_cast<std::int32_t>(std::round(rois[3] * 0.5F));
    const std::int32_t roi_w = std::max(roi_x2 - roi_x1 + 1, 1);
    const std::int32_t roi_h = std::max(roi_y2 - roi_y1 + 1, 1);
    const float bin_w = static_cast<float>(roi_w) / 3.0F;
    const float bin_h = static_cast<float>(roi_h) / 2.0F;

    for (std::int32_t q = 0; q < c; ++q) {
        for (std::int32_t ph = 0; ph < 2; ++ph) {
            for (std::int32_t pw = 0; pw < 3; ++pw) {
                const float phf = static_cast<float>(ph);
                const float pwf = static_cast<float>(pw);
                std::int32_t hstart = static_cast<std::int32_t>(static_cast<float>(roi_y1) + std::floor(phf * bin_h));
                std::int32_t wstart = static_cast<std::int32_t>(static_cast<float>(roi_x1) + std::floor(pwf * bin_w));
                std::int32_t hend =
                    static_cast<std::int32_t>(static_cast<float>(roi_y1) + std::ceil((phf + 1.0F) * bin_h));
                std::int32_t wend =
                    static_cast<std::int32_t>(static_cast<float>(roi_x1) + std::ceil((pwf + 1.0F) * bin_w));
                hstart = std::min(std::max(hstart, 0), h);
                wstart = std::min(std::max(wstart, 0), w);
                hend = std::min(std::max(hend, 0), h);
                wend = std::min(std::max(wend, 0), w);
                float expect = 0.0F;
                bool first = true;
                for (std::int32_t y = hstart; y < hend; ++y) {
                    for (std::int32_t x = wstart; x < wend; ++x) {
                        const float v = src[static_cast<std::size_t>((q * h + y) * w + x)];
                        expect = first ? v : std::max(expect, v);
                        first = false;
                    }
                }
                EXPECT_FLOAT_EQ(out.channel(q)[static_cast<std::size_t>(ph * 3 + pw)], first ? 0.0F : expect);
            }
        }
    }
}
