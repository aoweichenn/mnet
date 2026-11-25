// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/convolution.hpp>
#include <mlayer/crop.hpp>
#include <mlayer/einsum.hpp>
#include <mlayer/expanddims.hpp>
#include <mlayer/gateddeltarule.hpp>
#include <mlayer/multiheadattention.hpp>
#include <mlayer/proposal.hpp>
#include <mlayer/psroipooling.hpp>
#include <mlayer/reduction.hpp>
#include <mlayer/shortconv.hpp>
#include <mlayer/squeeze.hpp>
#include <mlayer/unaryop.hpp>
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

#include <mlayer/concat.hpp>
#include <mmemory/allocator.hpp>

TEST(Kernel, ConcatBenchmark) {
    using C = mlayer::Concat<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    C concat;
    ASSERT_EQ(concat.load_param(pd), 0);

    mnet::Mat a(80, 80, 64);
    mnet::Mat b(80, 80, 32);
    a.fill(1.0F);
    b.fill(2.0F);
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    opt.num_threads = 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 20; ++i) {
        ASSERT_EQ(concat.forward(bottoms, tops, opt), 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "concat_reuse=%.2fms\\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0);

    mnet::Option pool_opt;
    pool_opt.num_threads = 4;
    mmemory::UnlockedPoolAllocator pool;
    ASSERT_TRUE(pool.set_size_compare_ratio(0.0F));
    pool_opt.blob_allocator = &pool;
    const auto t2 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 20; ++i) {
        std::vector<mnet::Mat> fresh(1);
        ASSERT_EQ(concat.forward(bottoms, fresh, pool_opt), 0);
    }
    const auto t3 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "concat_fresh_pool=%.2fms\\n",
                 std::chrono::duration<double, std::milli>(t3 - t2).count() / 20.0);
}

TEST(Kernel, Convolution3x3Im2colMatchesReference) {
    const std::int32_t w = 6;
    const std::int32_t h = 5;
    const std::int32_t inch = 2;
    const std::int32_t outch = 3;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.01F * static_cast<float>((i * 7) % 31) - 0.15F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.1F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.1F * static_cast<float>(i + q * 5));
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                max_err = std::fmax(max_err, std::fabs(sum - outptr[oy * w + ox]));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}

TEST(Kernel, ReshapeLargeChannel) {
    using RS = mlayer::Reshape<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 1);
    pd.set(2, -1);
    RS op;
    ASSERT_EQ(op.load_param(pd), 0);
    mnet::Mat input(20, 20, 256);
    input.fill(1.0F);
    mnet::Option opt;
    mnet::Mat out;
    ASSERT_EQ(op.forward(input, out, opt), 0);
    ASSERT_FALSE(out.empty());
    ASSERT_EQ(out.dims, 3);
    ASSERT_EQ(out.w, 1);
    ASSERT_EQ(out.h, 1);
    ASSERT_EQ(out.c, 102400);
}

TEST(Kernel, UnaryOpTanhAndSquare) {
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

TEST(Kernel, ReductionMaxAndMean) {
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

TEST(Kernel, ExpandDimsThenSqueezeRoundTrips) {
    using ED = mlayer::ExpandDims<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    using SQ = mlayer::Squeeze<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict ed_pd;
    ed_pd.set(3, mnet::Mat(1));
    static_cast<std::int32_t*>(static_cast<void*>(ed_pd.get(3, mnet::Mat()).data))[0] = 1;
    // ParamDict copies the Mat, so set the axis through a local Mat first.
    mnet::Mat axes(1);
    static_cast<std::int32_t*>(static_cast<void*>(axes.data))[0] = 1;
    ed_pd.set(3, axes);

    ED expand;
    ASSERT_EQ(expand.load_param(ed_pd), 0);
    mnet::Mat input(4, 3);
    input.fill(2.0F);
    mnet::Option opt;
    mnet::Mat expanded;
    ASSERT_EQ(expand.forward(input, expanded, opt), 0);
    ASSERT_EQ(expanded.dims, 3);

    mformat::ParamDict sq_pd;
    sq_pd.set(3, axes);
    SQ squeeze;
    ASSERT_EQ(squeeze.load_param(sq_pd), 0);
    mnet::Mat squeezed;
    ASSERT_EQ(squeeze.forward(expanded, squeezed, opt), 0);
    ASSERT_EQ(squeezed.dims, 2);
    ASSERT_EQ(squeezed.w, 4);
    ASSERT_EQ(squeezed.h, 3);
}

TEST(Kernel, EinsumMatMulMatchesReference) {
    using ES = mlayer::Einsum<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    // "ik,kj->ij"
    const char* equation = "ik,kj->ij";
    mnet::Mat eq(static_cast<std::int32_t>(std::strlen(equation)));
    for (std::int32_t i = 0; i < eq.w; ++i) {
        static_cast<std::int32_t*>(static_cast<void*>(eq.data))[i] = equation[i];
    }
    mformat::ParamDict pd;
    pd.set(0, eq);
    ES op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat a(2, 3);
    mnet::Mat b(3, 2);
    for (std::int32_t i = 0; i < 6; ++i) {
        a[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        b[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(op.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 3);
    ASSERT_EQ(tops[0].h, 3);
    // a = [[1,2],[3,4],[5,6]], b = [[1,2,3],[4,5,6]]
    const float expect[9] = {9.0F, 12.0F, 15.0F, 19.0F, 26.0F, 33.0F, 29.0F, 40.0F, 51.0F};
    for (std::int32_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(tops[0][static_cast<std::size_t>(i)], expect[i]);
    }
}

TEST(Kernel, CropByReferenceShape) {
    using CR = mlayer::Crop<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    CR op;
    ASSERT_EQ(op.load_param(pd), 0);
    ASSERT_FALSE(op.one_blob_only);

    mnet::Mat src(4, 3, 2);
    for (std::int32_t q = 0; q < src.c; ++q) {
        for (std::int32_t i = 0; i < src.w * src.h; ++i) {
            src.channel(q)[static_cast<std::size_t>(i)] = static_cast<float>(q * 100 + i);
        }
    }
    mnet::Mat ref(2, 2, 1);
    std::vector<mnet::Mat> bottoms = {src, ref};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(op.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 2);
    ASSERT_EQ(tops[0].h, 2);
    EXPECT_FLOAT_EQ(tops[0].channel(0)[0], 0.0F);
    EXPECT_FLOAT_EQ(tops[0].channel(0)[1], 1.0F);
    EXPECT_FLOAT_EQ(tops[0].channel(0)[2], 4.0F);
    EXPECT_FLOAT_EQ(tops[0].channel(0)[3], 5.0F);
}

TEST(Kernel, ShortConvCausalState) {
    using SC = mlayer::ShortConv<mnet::Layer, mnet::Mat, mnet::Option>;
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

TEST(Kernel, GatedDeltaRuleSingleStep) {
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

TEST(Kernel, MultiHeadAttentionSelfAttention) {
    using MHA = mlayer::MultiHeadAttention<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    mformat::ParamDict pd;
    pd.set(0, 2);  // embed_dim
    pd.set(1, 1);  // num_heads
    pd.set(2, 4);  // weight_data_size = qdim * embed_dim
    pd.set(3, 2);  // kdim
    pd.set(4, 2);  // vdim
    MHA layer;
    ASSERT_EQ(layer.load_param(pd), 0);

    mnet::Mat identity(4);
    identity[0] = 1.0F;
    identity[1] = 0.0F;
    identity[2] = 0.0F;
    identity[3] = 1.0F;
    layer.q_weight_data = identity;
    layer.k_weight_data = identity;
    layer.v_weight_data = identity;
    layer.out_weight_data = identity;
    layer.q_bias_data = mnet::Mat(2);
    layer.k_bias_data = mnet::Mat(2);
    layer.v_bias_data = mnet::Mat(2);
    layer.out_bias_data = mnet::Mat(2);
    layer.q_bias_data.fill(0.0F);
    layer.k_bias_data.fill(0.0F);
    layer.v_bias_data.fill(0.0F);
    layer.out_bias_data.fill(0.0F);

    mnet::Mat input(2, 1);
    input[0] = 1.0F;
    input[1] = 0.0F;

    std::vector<mnet::Mat> bottoms = {input};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 2);
    ASSERT_EQ(tops[0].h, 1);
    EXPECT_NEAR(tops[0].row(0)[0], 1.0F, 1e-5F);
    EXPECT_NEAR(tops[0].row(0)[1], 0.0F, 1e-5F);
}

TEST(Kernel, PSROIPoolingAveragesRoi) {
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
            feature.channel(q)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
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
    // Each 2x2 bin of the 4x4 map: {1,2,5,6}, {3,4,7,8}, {9,10,13,14}, {11,12,15,16}
    EXPECT_NEAR(tops[0][0], 3.5F, 1e-5F);
    EXPECT_NEAR(tops[0][1], 5.5F, 1e-5F);
    EXPECT_NEAR(tops[0][2], 11.5F, 1e-5F);
    EXPECT_NEAR(tops[0][3], 13.5F, 1e-5F);
}

TEST(Kernel, ProposalProducesClampedRois) {
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

TEST(Kernel, Convolution3x3WinogradMatchesReference) {
    const std::int32_t w = 7;
    const std::int32_t h = 6;
    const std::int32_t inch = 4;
    const std::int32_t outch = 8;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.01F * static_cast<float>((i * 7) % 31) - 0.15F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.1F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.1F * static_cast<float>(i + q * 5));
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                max_err = std::fmax(max_err, std::fabs(sum - outptr[oy * w + ox]));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}
