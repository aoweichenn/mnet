// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/convolution.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/layer.hpp>
#include <mnet/quantize.hpp>

namespace {

using Conv = mlayer::Convolution<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin, false,
                                 false, mnet::activation_ss, mnet::quantize_int8, mnet::copy_make_border>;

}  // namespace

TEST(Kernel, Convolution1x1MultiChannel) {
    const std::int32_t w = 4;
    const std::int32_t h = 3;
    const std::int32_t inch = 3;
    const std::int32_t outch = 2;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 1);
    pd.set(11, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch);

    mnet::Mat weights[2];
    weights[0].create(outch * inch);
    weights[0][0] = 1.0F;
    weights[0][1] = 2.0F;
    weights[0][2] = 3.0F;
    weights[0][3] = -1.0F;
    weights[0][4] = -2.0F;
    weights[0][5] = -3.0F;
    weights[1].create(outch);
    weights[1][0] = 0.5F;
    weights[1][1] = -0.5F;

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = static_cast<float>(i + q);
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);
    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);

    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t i = 0; i < w * h; ++i) {
            float ref = weights[1][p];
            for (std::int32_t q = 0; q < inch; ++q) {
                ref += input.channel(q)[i] * weights[0][p * inch + q];
            }
            EXPECT_TRUE(std::isfinite(outptr[i])) << p << " " << i;
            EXPECT_NEAR(outptr[i], ref, 1e-5F) << p << " " << i;
        }
    }
}

#include <cstdio>
#include <cstdlib>

#include <string>

#include <mformat/data_reader.hpp>

TEST(Kernel, ModelBinReadsFp16WeightsAndBias) {
    const char* dir_env = std::getenv("MNET_SQUEEZENET_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_SQUEEZENET_DIR to run the model-bin test";
    }
    const std::string path = std::string(dir_env) + "/squeezenet_v1.1.bin";
    std::FILE* file = std::fopen(path.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    mformat::DataReaderFromStdio reader(file);
    mformat::ModelBinFromDataReader mb(reader);

    mnet::Mat conv1_w = mb.load(1728, 0);
    mnet::Mat conv1_b = mb.load(64, 1);
    mnet::Mat squeeze_w = mb.load(1024, 0);
    mnet::Mat squeeze_b = mb.load(16, 1);
    std::fclose(file);

    ASSERT_EQ(conv1_w.w, 1728);
    ASSERT_EQ(conv1_b.w, 64);
    ASSERT_EQ(squeeze_w.w, 1024);
    ASSERT_EQ(squeeze_b.w, 16);

    for (std::int32_t i = 0; i < squeeze_w.w; ++i) {
        ASSERT_TRUE(std::isfinite(squeeze_w[i])) << i;
    }
    for (std::int32_t i = 0; i < squeeze_b.w; ++i) {
        ASSERT_TRUE(std::isfinite(squeeze_b[i])) << i;
    }
    EXPECT_NEAR(squeeze_w[64], -0.17248535F, 1e-5F);
}

TEST(Kernel, Convolution1x1WithModelBinWeights) {
    const char* dir_env = std::getenv("MNET_SQUEEZENET_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_SQUEEZENET_DIR to run the model-bin test";
    }
    const std::string path = std::string(dir_env) + "/squeezenet_v1.1.bin";
    std::FILE* file = std::fopen(path.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    mformat::DataReaderFromStdio reader(file);
    mformat::ModelBinFromDataReader mb(reader);
    mnet::Mat conv1_w = mb.load(1728, 0);
    mnet::Mat conv1_b = mb.load(64, 1);
    mnet::Mat squeeze_w = mb.load(1024, 0);
    mnet::Mat squeeze_b = mb.load(16, 1);
    std::fclose(file);

    (void)conv1_w;
    (void)conv1_b;

    mformat::ParamDict pd;
    pd.set(0, 16);
    pd.set(1, 1);
    pd.set(11, 1);
    pd.set(5, 1);
    pd.set(6, 1024);
    pd.set(9, 1);

    mnet::Mat weights[2];
    weights[0] = squeeze_w;
    weights[1] = squeeze_b;

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(56, 56, 64);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < 56 * 56; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    for (std::int32_t p = 0; p < out.c; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t i = 0; i < out.w * out.h; ++i) {
            ASSERT_TRUE(std::isfinite(outptr[i])) << p << " " << i;
        }
    }
}

TEST(Kernel, CopyMakeBorderZeroPadPreservesData) {
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

#include <mlayer/pooling.hpp>

TEST(Kernel, PoolingMaxNoPadKeepsFinite) {
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
    for (std::int32_t q = 0; q < out.c; ++q) {
        const float* p = out.channel(q);
        for (std::int32_t i = 0; i < out.w * out.h; ++i) {
            ASSERT_TRUE(std::isfinite(p[i])) << q << " " << i << " " << p[i];
        }
    }
}
