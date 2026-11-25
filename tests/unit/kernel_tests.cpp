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
    weights[1][static_cast<std::size_t>(0)] = 0.5F;
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
            float ref = weights[1][static_cast<std::size_t>(p)];
            for (std::int32_t q = 0; q < inch; ++q) {
                ref +=
                    input.channel(q)[static_cast<std::size_t>(i)] * weights[0][static_cast<std::size_t>(p * inch + q)];
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
        ASSERT_TRUE(std::isfinite(squeeze_w[static_cast<std::size_t>(i)])) << i;
    }
    for (std::int32_t i = 0; i < squeeze_b.w; ++i) {
        ASSERT_TRUE(std::isfinite(squeeze_b[static_cast<std::size_t>(i)])) << i;
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

#include <mlayer/binaryop.hpp>

TEST(Kernel, BinaryOpAddAndScalarMul) {
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

#include <mlayer/permute.hpp>
#include <mlayer/rmsnorm.hpp>

TEST(Kernel, PermuteSwapsWidthAndHeight) {
    using Perm = mlayer::Permute<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;

    mformat::ParamDict pd;
    pd.set(0, 1);
    Perm perm;
    ASSERT_EQ(perm.load_param(pd), 0);

    mnet::Mat input(4, 3, 2);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < input.w * input.h; ++i) {
            p[i] = static_cast<float>(q * 100 + i);
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(perm.forward(input, out, opt), 0);
    ASSERT_EQ(out.w, 3);
    ASSERT_EQ(out.h, 4);
    ASSERT_EQ(out.c, 2);

    for (std::int32_t q = 0; q < out.c; ++q) {
        const float* s = input.channel(q);
        const float* d = out.channel(q);
        for (std::int32_t y = 0; y < out.h; ++y) {
            for (std::int32_t x = 0; x < out.w; ++x) {
                EXPECT_FLOAT_EQ(d[static_cast<std::size_t>(y * out.w + x)],
                                s[static_cast<std::size_t>(x * input.w + y)]);
            }
        }
    }
}

TEST(Kernel, RMSNormAffine) {
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

#include <mlayer/interp.hpp>
#include <mlayer/matmul.hpp>
#include <mlayer/reshape.hpp>

TEST(Kernel, ReshapeNumericAndExpression) {
    using RS = mlayer::Reshape<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    mformat::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 3);
    pd.set(2, 4);
    RS op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat input(4, 3, 2);
    for (std::int32_t i = 0; i < 24; ++i) {
        static_cast<float*>(input)[i] = static_cast<float>(i);
    }
    mnet::Option opt;
    mnet::Mat out;
    ASSERT_EQ(op.forward(input, out, opt), 0);
    ASSERT_EQ(out.dims, 3);
    ASSERT_EQ(out.w, 2);
    ASSERT_EQ(out.h, 3);
    ASSERT_EQ(out.c, 4);

    mformat::ParamDict pd2;
    pd2.set(6, std::string("1w,1h,4"));
    RS op2;
    ASSERT_EQ(op2.load_param(pd2), 0);
    mnet::Mat shape(3, 2);
    std::vector<mnet::Mat> bottoms = {input, shape};
    std::vector<mnet::Mat> tops(1);
    ASSERT_EQ(op2.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 3);
    ASSERT_EQ(tops[0].h, 2);
    ASSERT_EQ(tops[0].c, 4);
}

TEST(Kernel, MatMul2DAndInterpNearest) {
    using MM = mlayer::MatMul<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    mformat::ParamDict pd;
    pd.set(0, 0);
    MM mm;
    ASSERT_EQ(mm.load_param(pd), 0);

    mnet::Mat a(3, 2);
    mnet::Mat b(4, 3);
    for (std::int32_t i = 0; i < 6; ++i) {
        static_cast<float*>(a)[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    for (std::int32_t i = 0; i < 12; ++i) {
        static_cast<float*>(b)[i] = static_cast<float>(i);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(mm.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 4);
    ASSERT_EQ(tops[0].h, 2);
    for (std::int32_t i = 0; i < 2; ++i) {
        for (std::int32_t j = 0; j < 4; ++j) {
            float ref = 0.0F;
            for (std::int32_t k = 0; k < 3; ++k) {
                ref += a.row(i)[k] * b.row(k)[j];
            }
            EXPECT_FLOAT_EQ(tops[0].row(i)[j], ref);
        }
    }

    using IP = mlayer::Interp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd2;
    pd2.set(0, 1);
    pd2.set(1, 2.0F);
    pd2.set(2, 2.0F);
    IP ip;
    ASSERT_EQ(ip.load_param(pd2), 0);
    mnet::Mat small(2, 2, 1);
    small.channel(0)[0] = 1.0F;
    small.channel(0)[1] = 2.0F;
    small.channel(0)[2] = 3.0F;
    small.channel(0)[3] = 4.0F;
    mnet::Mat up;
    ASSERT_EQ(ip.forward(small, up, opt), 0);
    ASSERT_EQ(up.w, 4);
    ASSERT_EQ(up.h, 4);
    EXPECT_FLOAT_EQ(up.channel(0)[0], 1.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[3], 2.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[12], 3.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[15], 4.0F);
}

#include <mlayer/gemm.hpp>
#include <mlayer/rotaryembed.hpp>
#include <mlayer/sdpa.hpp>

TEST(Kernel, GemmConstantBAndC) {
    using G = mlayer::Gemm<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;

    mformat::ParamDict pd;
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 2);
    pd.set(8, 4);
    pd.set(9, 3);
    pd.set(10, 0);
    G gemm;
    ASSERT_EQ(gemm.load_param(pd), 0);

    mnet::Mat weights[2];
    weights[0].create(3, 4);
    for (std::int32_t i = 0; i < 12; ++i) {
        static_cast<float*>(weights[0])[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    weights[1].create(1);
    weights[1][static_cast<std::size_t>(0)] = 0.5F;
    ASSERT_EQ(gemm.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat a(3, 2);
    for (std::int32_t i = 0; i < 6; ++i) {
        static_cast<float*>(a)[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(gemm.forward(a, out, opt), 0);
    ASSERT_EQ(out.w, 4);
    ASSERT_EQ(out.h, 2);
    for (std::int32_t i = 0; i < 2; ++i) {
        for (std::int32_t j = 0; j < 4; ++j) {
            float ref = 0.5F;
            for (std::int32_t k = 0; k < 3; ++k) {
                ref += a.row(i)[k] * weights[0].row(j)[k];
            }
            EXPECT_NEAR(out.row(i)[j], ref, 1e-5F);
        }
    }
}

TEST(Kernel, RotaryEmbedAndSdpaSingleStep) {
    using RE = mlayer::RotaryEmbed<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    RE rope;
    ASSERT_EQ(rope.load_param(pd), 0);

    mnet::Mat x(4, 1, 1);
    x.channel(0)[0] = 1.0F;
    x.channel(0)[1] = 0.0F;
    x.channel(0)[2] = 0.0F;
    x.channel(0)[3] = 1.0F;
    mnet::Mat cos_cache(2, 1);
    mnet::Mat sin_cache(2, 1);
    cos_cache.row(0)[0] = 0.0F;
    cos_cache.row(0)[1] = 0.0F;
    sin_cache.row(0)[0] = 1.0F;
    sin_cache.row(0)[1] = 1.0F;

    std::vector<mnet::Mat> rb = {x, cos_cache, sin_cache};
    std::vector<mnet::Mat> rt(1);
    mnet::Option opt;
    ASSERT_EQ(rope.forward(rb, rt, opt), 0);
    EXPECT_NEAR(rt[0].channel(0)[0], 0.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[1], -1.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[2], 1.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[3], 0.0F, 1e-6F);

    using AP = mlayer::SDPA<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict spd;
    spd.set(5, 0);
    spd.set(7, 0);
    AP sdpa;
    ASSERT_EQ(sdpa.load_param(spd), 0);

    mnet::Mat q(4, 1, 1);
    mnet::Mat k(4, 1, 1);
    mnet::Mat v(4, 1, 1);
    for (std::int32_t i = 0; i < 4; ++i) {
        q.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        k.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        v.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(10 + i);
    }
    std::vector<mnet::Mat> sb = {q, k, v};
    std::vector<mnet::Mat> st(1);
    ASSERT_EQ(sdpa.forward(sb, st, opt), 0);
    ASSERT_EQ(st[0].w, 4);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(st[0].channel(0)[static_cast<std::size_t>(i)], v.channel(0)[static_cast<std::size_t>(i)], 1e-5F);
    }
}

#include <cstdio>

#include <chrono>

TEST(Kernel, Convolution3x3Benchmark) {
    const std::int32_t w = 56;
    const std::int32_t h = 56;
    const std::int32_t c = 64;
    const std::int32_t outch = 64;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(2, 1);
    pd.set(12, 1);
    pd.set(3, 1);
    pd.set(13, 1);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * c * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * c * 9);
    for (std::int32_t i = 0; i < outch * c * 9; ++i) {
        static_cast<float*>(weights[0])[static_cast<std::size_t>(i)] = 0.001F * static_cast<float>(i % 97);
    }
    weights[1].create(outch);
    weights[1].fill(0.0F);

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, c);
    for (std::int32_t q = 0; q < c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 5; ++i) {
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "conv3x3=%0.2fms\\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0);
}
