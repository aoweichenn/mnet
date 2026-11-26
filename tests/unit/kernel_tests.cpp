// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstring>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/convolution.hpp>
#include <mlayer/convolutiondepthwise.hpp>
#include <mlayer/crop.hpp>
#include <mlayer/einsum.hpp>
#include <mlayer/expanddims.hpp>
#include <mlayer/gateddeltarule.hpp>
#include <mlayer/gemm.hpp>
#include <mlayer/lrn.hpp>
#include <mlayer/multiheadattention.hpp>
#include <mlayer/proposal.hpp>
#include <mlayer/psroipooling.hpp>
#include <mlayer/reduction.hpp>
#include <mlayer/roipooling.hpp>
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

#include <mlayer/expanddims.hpp>

TEST(Kernel, ExpandDimsScalarAxisMappings) {
    using ED = mlayer::ExpandDims<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    const auto run = [](const mnet::Mat& in, std::int32_t axis, std::int32_t ew, std::int32_t eh,
                        std::int32_t ec) {
        mformat::ParamDict pd;
        pd.set(0, axis);
        ED ed;
        EXPECT_EQ(ed.load_param(pd), 0);
        std::vector<mnet::Mat> tops(1);
        mnet::Option opt;
        EXPECT_EQ(ed.forward(in, tops[0], opt), 0);
        EXPECT_EQ(tops[0].dims, 3);
        EXPECT_EQ(tops[0].w, ew);
        EXPECT_EQ(tops[0].h, eh);
        EXPECT_EQ(tops[0].c, ec);
    };

    mnet::Mat mat(4, 3);
    run(mat, 0, 4, 3, 1);       // expand c
    run(mat, 1, 4, 1, 3);       // expand h
    run(mat, 2, 1, 4, 3);       // expand w

}

#include <mlayer/reduction.hpp>

TEST(Kernel, UnarySqrtScalarAndVector) {
    using U = mlayer::UnaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd; pd.set(0, 5);
    U op; ASSERT_EQ(op.load_param(pd), 0);
    mnet::Option opt;
    mnet::Mat one(1); one[0] = 0.07177F;
    ASSERT_EQ(op.forward_inplace(one, opt), 0);
    EXPECT_NEAR(one[0], 0.267900F, 1e-5F);
    mnet::Mat vec(4); vec[0]=1; vec[1]=4; vec[2]=9; vec[3]=16;
    ASSERT_EQ(op.forward_inplace(vec, opt), 0);
    EXPECT_FLOAT_EQ(vec[0],1); EXPECT_FLOAT_EQ(vec[1],2);
    EXPECT_FLOAT_EQ(vec[2],3); EXPECT_FLOAT_EQ(vec[3],4);
}

TEST(Kernel, ReductionScalarAxisSumsWidthToVector) {
    using Red = mlayer::Reduction<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 0);   // sum
    pd.set(6, 1);   // reduce width (2-D: 0=h, 1=w)
    Red red; ASSERT_EQ(red.load_param(pd), 0);
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

#include <mlayer/reshape.hpp>

TEST(Kernel, ReshapeExpandDimsCropChain) {
    using Reshape = mlayer::Reshape<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    using ED = mlayer::ExpandDims<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    using Crop = mlayer::Crop<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mnet::Option opt;
    mnet::Mat flat(1024);
    for (std::int32_t i = 0; i < 1024; ++i) {
        flat[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    mformat::ParamDict p_reshape;
    p_reshape.set(0, 1024); p_reshape.set(1, 1);
    Reshape reshape; ASSERT_EQ(reshape.load_param(p_reshape), 0);
    std::vector<mnet::Mat> r_out(1);
    ASSERT_EQ(reshape.forward(flat, r_out[0], opt), 0);
    EXPECT_EQ(r_out[0].w, 1024); EXPECT_EQ(r_out[0].h, 1);

    mformat::ParamDict p_ed; p_ed.set(0, 2);   // prepend w
    ED ed; ASSERT_EQ(ed.load_param(p_ed), 0);
    std::vector<mnet::Mat> e_out(1);
    ASSERT_EQ(ed.forward(r_out[0], e_out[0], opt), 0);
    EXPECT_EQ(e_out[0].w, 1); EXPECT_EQ(e_out[0].h, 1024); EXPECT_EQ(e_out[0].c, 1);

    mformat::ParamDict p_crop; p_crop.set(1, 0); p_crop.set(4, 512);
    Crop crop; ASSERT_EQ(crop.load_param(p_crop), 0);
    std::vector<mnet::Mat> bottoms = {e_out[0]}; std::vector<mnet::Mat> c_out(1);
    ASSERT_EQ(crop.forward(bottoms, c_out, opt), 0);
    EXPECT_EQ(c_out[0].w, 1); EXPECT_EQ(c_out[0].h, 512);
    EXPECT_FLOAT_EQ(c_out[0].channel(0).row(511)[0], 511.0F);
}

#include <mlayer/compare.hpp>
#include <mlayer/where.hpp>

TEST(Kernel, CompareMasksWithPerChannelBroadcast) {
    using Cmp = mlayer::Compare<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 4);   // greater
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

TEST(Kernel, WhereSelectsBroadcastCondition) {
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
TEST(Kernel, BinaryOpBroadcastHeightVector) {
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
            const float expect = static_cast<float>(x) + 100.0F * static_cast<float>(y) + 1000.0F + static_cast<float>(y);
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
            const float expect = static_cast<float>(x) + 100.0F * static_cast<float>(y) + 1000.0F + static_cast<float>(y);
            EXPECT_FLOAT_EQ(tops_flat[0].channel(0).row(y)[x], expect) << "flat at x=" << x << " y=" << y;
        }
    }
}

// A 2-D (w,h) blob meeting a 3-D (w,h,1) blob must stay elementwise.
TEST(Kernel, BinaryOpRowVectorTimesChannelVector) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd; pd.set(0, 2);
    BinOp mul; ASSERT_EQ(mul.load_param(pd), 0);
    constexpr std::int32_t t = 8;
    constexpr std::int32_t c = 3;
    mnet::Mat a(1, t);          // (w=1,h=T)
    mnet::Mat b(c, 1, 1);       // (w=C,h=1,c=1)
    for (std::int32_t i=0;i<t;++i) a.row(i)[0] = static_cast<float>(i+1);
    for (std::int32_t i=0;i<c;++i) b.channel(0).row(0)[i] = static_cast<float>(i+2);
    std::vector<mnet::Mat> bottoms={a,b}, tops(1);
    mnet::Option opt;
    ASSERT_EQ(mul.forward(bottoms,tops,opt),0);
    EXPECT_EQ(tops[0].w,c); EXPECT_EQ(tops[0].h,t); EXPECT_EQ(tops[0].c,1);
    for (std::int32_t y=0;y<t;++y)
        for (std::int32_t x=0;x<c;++x)
            EXPECT_FLOAT_EQ(tops[0].channel(0).row(y)[x], static_cast<float>((x+2)*(y+1)));
}

TEST(Kernel, BinaryOpMixesTwoAndThreeDimensionalBlobs) {
    using BinOp = mlayer::BinaryOp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 0);
    BinOp add;
    ASSERT_EQ(add.load_param(pd), 0);

    constexpr std::int32_t w = 5;
    constexpr std::int32_t h = 3;
    mnet::Mat a(w, h);        // dims == 2
    mnet::Mat b(w, h, 1);     // dims == 3
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
#include <mlayer/shortconv.hpp>
#include <mlayer/relposbias.hpp>
#include <mlayer/gather.hpp>
#include <mlayer/scatternd.hpp>
#include <mlayer/situ.hpp>
#include <mlayer/stft.hpp>

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

// The int8 KV cache must produce the same attention output as the fp32 cache
// within quantization noise, while keeping the cache at one byte per value.
TEST(Kernel, SdpaInt8KvCacheMatchesFp32) {
    using AP = mlayer::SDPA<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict spd;
    spd.set(5, 0);
    spd.set(7, 1);
    AP sdpa;
    ASSERT_EQ(sdpa.load_param(spd), 0);

    const std::int32_t heads = 2;
    const std::int32_t group = 2;
    const std::int32_t dim = 32;
    const std::int32_t steps[5] = {3, 1, 1, 1, 1};

    auto run = [&](bool int8_kv, std::vector<float>& flat, mnet::Mat& last_key) {
        mnet::Option opt;
        opt.num_threads = 2;
        opt.use_int8_kv_cache = int8_kv;

        std::uint32_t seed = 20260101u;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (static_cast<float>(seed >> 8) / 16777216.0F) * 2.0F - 1.0F;
        };

        mnet::Mat past_k;
        mnet::Mat past_v;
        for (std::int32_t step = 0; step < 5; ++step) {
            const std::int32_t n = steps[step];
            mnet::Mat q(dim, n, heads);
            mnet::Mat k(dim, n, group);
            mnet::Mat v(dim, n, group);
            for (std::int32_t c = 0; c < heads; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        q.channel(c).row(h)[w] = rnd();
                    }
                }
            }
            for (std::int32_t c = 0; c < group; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        k.channel(c).row(h)[w] = rnd();
                        v.channel(c).row(h)[w] = rnd();
                    }
                }
            }

            std::vector<mnet::Mat> bottom = {q, k, v, past_k, past_v};
            std::vector<mnet::Mat> top(3);
            ASSERT_EQ(sdpa.forward(bottom, top, opt), 0);
            for (std::int32_t c = 0; c < heads; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        flat.push_back(top[0].channel(c).row(h)[w]);
                    }
                }
            }
            past_k = top[1];
            past_v = top[2];
        }
        last_key = past_k;
    };

    std::vector<float> ref;
    std::vector<float> quantized;
    mnet::Mat fp32_key;
    mnet::Mat int8_key;
    run(false, ref, fp32_key);
    run(true, quantized, int8_key);

    ASSERT_EQ(ref.size(), quantized.size());
    float max_diff = 0.0F;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ref[i] - quantized[i]));
    }
    EXPECT_LT(max_diff, 0.05F);

    EXPECT_EQ(int8_key.elemsize, 1u);
    EXPECT_GT(fp32_key.total() * 4, int8_key.total() * 3);
}

// SiTU gated activation (Inkling-2-Mini MoE): beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb).
TEST(Kernel, SituMatchesReference) {
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

// Causal depthwise short convolution: y[t] = sum_k xs[t + k] * w[k] with the
// K-1 history tokens as prefix; activation 0 disables the Qwen3.5 SiLU.
TEST(Kernel, ShortConvCausalKernel) {
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
        const float y0 = x0 * w[2];                 // xs = [0, 0, x0, x1]
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

// Relative-position bias + attention sink: bias[h,q,k] = dot(rel_bias[:,d], R[:,h,q]) - sink[h,q].
TEST(Kernel, RelPosBiasMatchesReference) {
    using B = mlayer::RelPosBias<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    B bias;
    ASSERT_EQ(bias.load_param(mformat::ParamDict()), 0);

    const std::int32_t n_rel = 2;
    const std::int32_t extent = 3;
    const std::int32_t heads = 2;
    const std::int32_t tokens = 2;
    const std::int32_t n_kv = 4;

    mnet::Mat rel(n_rel, tokens, heads);
    mnet::Mat table(n_rel, extent);
    mnet::Mat sink(heads, tokens);
    mnet::Mat rel_pos(n_kv, tokens);

    for (std::int32_t h = 0; h < heads; ++h) {
        for (std::int32_t q = 0; q < tokens; ++q) {
            rel.channel(h).row(q)[0] = 1.0F + static_cast<float>(h) + 0.1F * static_cast<float>(q);
            rel.channel(h).row(q)[1] = -0.5F + static_cast<float>(q);
            sink.row(q)[h] = 0.25F * static_cast<float>(h + 1);
        }
    }
    for (std::int32_t d = 0; d < extent; ++d) {
        table.row(d)[0] = 0.1F * static_cast<float>(d);
        table.row(d)[1] = 1.0F - 0.2F * static_cast<float>(d);
    }
    // distances: key 0..3 for query 0 -> 0,1,2,extent(out of window)
    rel_pos.row(0)[0] = 0.0F;
    rel_pos.row(0)[1] = 1.0F;
    rel_pos.row(0)[2] = 2.0F;
    rel_pos.row(0)[3] = static_cast<float>(extent);
    rel_pos.row(1)[0] = 1.0F;
    rel_pos.row(1)[1] = 100.0F;  // clamps out of window
    rel_pos.row(1)[2] = 0.0F;
    rel_pos.row(1)[3] = 2.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {rel, table, sink, rel_pos};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(bias.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].w, n_kv);
    ASSERT_EQ(top[0].h, tokens);
    ASSERT_EQ(top[0].c, heads);

    for (std::int32_t h = 0; h < heads; ++h) {
        for (std::int32_t q = 0; q < tokens; ++q) {
            for (std::int32_t k = 0; k < n_kv; ++k) {
                const float d = rel_pos.row(q)[k];
                float sum = 0.0F;
                if (d >= 0.0F && d < static_cast<float>(extent)) {
                    const auto di = static_cast<std::int32_t>(d);
                    sum = table.row(di)[0] * rel.channel(h).row(q)[0] + table.row(di)[1] * rel.channel(h).row(q)[1];
                }
                EXPECT_NEAR(top[0].channel(h).row(q)[k], sum - sink.row(q)[h], 1e-6F);
            }
        }
    }
}

// One-sided STFT: framing + window + DFT, output (bins, frames, 2).
TEST(Kernel, StftMatchesReference) {
    using S = mlayer::Stft<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    S stft;
    ASSERT_EQ(stft.load_param(mformat::ParamDict()), 0);

    // 12 samples, frame 4, step 2 -> 5 frames.
    const float samples[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    mnet::Mat signal(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        signal[static_cast<std::size_t>(i)] = samples[i];
    }
    mnet::Mat window(4);
    window.fill(1.0F);
    mnet::Mat step(1);
    step[0] = 2.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {signal, window, step};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(stft.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].w, 3);   // bins = 4/2 + 1
    ASSERT_EQ(top[0].h, 5);   // frames
    ASSERT_EQ(top[0].c, 2);

    for (std::int32_t f = 0; f < 5; ++f) {
        for (std::int32_t k = 0; k < 3; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (std::int32_t n = 0; n < 4; ++n) {
                const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(k) *
                                     static_cast<double>(n) / 4.0;
                const double v = static_cast<double>(samples[f * 2 + n]);
                re += v * std::cos(angle);
                im -= v * std::sin(angle);
            }
            EXPECT_NEAR(top[0].channel(0).row(f)[k], re, 1e-5F);
            EXPECT_NEAR(top[0].channel(1).row(f)[k], im, 1e-5F);
        }
    }

    // DC bin is the frame sum, Nyquist alternates sign: [1,0,0,0] -> [1,1,1].
    EXPECT_NEAR(top[0].channel(0).row(0)[0], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(0).row(0)[1], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(0).row(0)[2], 1.0F, 1e-5F);
    // frame 1 is samples[2..5] = [0,0,0,1] -> X = [1, +i, -1]
    EXPECT_NEAR(top[0].channel(0).row(1)[1], 0.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(1).row(1)[1], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(0).row(1)[2], -1.0F, 1e-5F);
}

// Gather along axis 0 (1-D and 2-D) and axis 1 (2-D).
TEST(Kernel, GatherIndices) {
    using G = mlayer::Gather<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 0);
    G gather;
    ASSERT_EQ(gather.load_param(pd), 0);

    mnet::Mat data(4);
    data[0] = 10.0F;
    data[1] = 20.0F;
    data[2] = 30.0F;
    data[3] = 40.0F;
    mnet::Mat idx(3);
    idx[0] = 2.0F;
    idx[1] = 0.0F;
    idx[2] = 3.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {data, idx};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(gather.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].w, 3);
    EXPECT_FLOAT_EQ(top[0][0], 30.0F);
    EXPECT_FLOAT_EQ(top[0][1], 10.0F);
    EXPECT_FLOAT_EQ(top[0][2], 40.0F);
}

// ScatterND for the iSTFT overlap-add: flat 1-D data, [n,1] indices, n updates.
TEST(Kernel, ScatterNdOverlapAdd) {
    using S = mlayer::ScatterND<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    S scatter;
    ASSERT_EQ(scatter.load_param(mformat::ParamDict()), 0);

    mnet::Mat data(6);
    for (std::int32_t i = 0; i < 6; ++i) {
        data[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    mnet::Mat indices(2, 1);
    indices[0] = 1.0F;
    indices[1] = 4.0F;
    mnet::Mat updates(2);
    updates[0] = -1.0F;
    updates[1] = -4.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {data, indices, updates};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(scatter.forward(bottom, top, opt), 0);

    EXPECT_FLOAT_EQ(top[0][0], 0.0F);
    EXPECT_FLOAT_EQ(top[0][1], -1.0F);
    EXPECT_FLOAT_EQ(top[0][2], 2.0F);
    EXPECT_FLOAT_EQ(top[0][3], 3.0F);
    EXPECT_FLOAT_EQ(top[0][4], -4.0F);
    EXPECT_FLOAT_EQ(top[0][5], 5.0F);
    // input must stay intact
    EXPECT_FLOAT_EQ(data[1], 1.0F);
}

// ScatterND with (row, col) index pairs, the shape Kokoro's iSTFT tail uses.
TEST(Kernel, ScatterNdRankedIndices) {
    using S = mlayer::ScatterND<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    S scatter;
    ASSERT_EQ(scatter.load_param(mformat::ParamDict()), 0);

    mnet::Mat data(4, 2);  // 2 rows of 4
    for (std::int32_t y = 0; y < 2; ++y) {
        for (std::int32_t x = 0; x < 4; ++x) {
            data.row(y)[x] = static_cast<float>(y * 10 + x);
        }
    }
    mnet::Mat indices(2, 3);  // three (row, col) pairs
    indices.row(0)[0] = 0.0F;
    indices.row(0)[1] = 2.0F;
    indices.row(1)[0] = 1.0F;
    indices.row(1)[1] = 0.0F;
    indices.row(2)[0] = 1.0F;
    indices.row(2)[1] = 3.0F;
    mnet::Mat updates(3);
    updates[0] = -1.0F;
    updates[1] = -2.0F;
    updates[2] = -3.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {data, indices, updates};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(scatter.forward(bottom, top, opt), 0);
    EXPECT_FLOAT_EQ(top[0].row(0)[2], -1.0F);
    EXPECT_FLOAT_EQ(top[0].row(1)[0], -2.0F);
    EXPECT_FLOAT_EQ(top[0].row(1)[3], -3.0F);
    EXPECT_FLOAT_EQ(top[0].row(0)[0], 0.0F);
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

TEST(Kernel, MultiHeadAttentionKvCacheMatchesFullSequence) {
    using MHA = mlayer::MultiHeadAttention<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    mformat::ParamDict pd;
    pd.set(0, 4);   // embed_dim
    pd.set(1, 2);   // num_heads
    pd.set(2, 16);  // weight_data_size
    pd.set(3, 4);   // kdim
    pd.set(4, 4);   // vdim
    pd.set(7, 1);   // kv_cache
    MHA layer;
    ASSERT_EQ(layer.load_param(pd), 0);

    const std::int32_t E = 4;
    std::vector<mnet::Mat> weights;
    for (std::int32_t t = 0; t < 8; ++t) {
        mnet::Mat w = (t % 2 == 0) ? mnet::Mat(E * E) : mnet::Mat(E);
        for (std::int32_t i = 0; i < w.w; ++i) {
            w[static_cast<std::size_t>(i)] = 0.11F * static_cast<float>((i * 7 + t * 3) % 17) - 0.7F;
        }
        weights.push_back(w);
    }
    ASSERT_EQ(layer.load_model(mformat::ModelBinFromMatArray(weights.data())), 0);

    auto make_x = [&](std::int32_t seq, std::int32_t seed) {
        mnet::Mat x(E, seq);
        for (std::int32_t i = 0; i < E * seq; ++i) {
            x[static_cast<std::size_t>(i)] = std::sin(0.3F * static_cast<float>(i + seed));
        }
        return x;
    };

    mformat::ParamDict full_pd;
    full_pd.set(0, 4);
    full_pd.set(1, 2);
    full_pd.set(2, 16);
    full_pd.set(3, 4);
    full_pd.set(4, 4);
    MHA full_layer;
    ASSERT_EQ(full_layer.load_param(full_pd), 0);
    ASSERT_EQ(full_layer.load_model(mformat::ModelBinFromMatArray(weights.data())), 0);

    mnet::Option opt;
    mnet::Mat full = make_x(2, 1);
    std::vector<mnet::Mat> full_bottoms = {full};
    std::vector<mnet::Mat> full_tops(1);
    ASSERT_EQ(full_layer.forward(full_bottoms, full_tops, opt), 0);

    std::vector<mnet::Mat> top0(3);
    std::vector<mnet::Mat> b0 = {make_x(1, 1), mnet::Mat(), mnet::Mat()};
    ASSERT_EQ(layer.forward(b0, top0, opt), 0);
    ASSERT_EQ(top0[1].dims, 3);
    ASSERT_EQ(top0[1].w, 2);
    ASSERT_EQ(top0[1].h, 1);
    ASSERT_EQ(top0[1].c, 2);

    std::vector<mnet::Mat> top1(3);
    std::vector<mnet::Mat> b1 = {make_x(1, 1 + E), top0[1], top0[2]};
    ASSERT_EQ(layer.forward(b1, top1, opt), 0);
    ASSERT_EQ(top1[1].h, 2);

    for (std::int32_t i = 0; i < E; ++i) {
        EXPECT_NEAR(top1[0].row(0)[i], full_tops[0].row(1)[i], 1e-5F);
    }
}

TEST(Kernel, Convolution3x3SmallInputChannelsMatchesReference) {
    // Exercises the im2col path for K = inch * 9 = 27 (below the old >= 64 gate).
    const std::int32_t w = 9;
    const std::int32_t h = 8;
    const std::int32_t inch = 3;
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
        weights[0][static_cast<std::size_t>(i)] = 0.02F * static_cast<float>((i * 5) % 23) - 0.2F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        for (std::int32_t i = 0; i < w * h; ++i) {
            input.channel(q)[static_cast<std::size_t>(i)] = std::cos(0.08F * static_cast<float>(i + q * 7));
        }
    }

    mnet::Option opt;
    opt.num_threads = 3;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
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
                max_err = std::fmax(max_err, std::fabs(sum - out.channel(p)[static_cast<std::size_t>(oy * w + ox)]));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}

TEST(Kernel, Convolution3x3Winograd43MatchesReference) {
    // 64x64 output with inch=32 selects F(4x4, 3x3).
    const std::int32_t w = 64;
    const std::int32_t h = 64;
    const std::int32_t inch = 32;
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
        weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        for (std::int32_t i = 0; i < w * h; ++i) {
            input.channel(q)[static_cast<std::size_t>(i)] = std::sin(0.05F * static_cast<float>(i + q * 31));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        for (std::int32_t oy = 0; oy < h; oy += 7) {
            for (std::int32_t ox = 0; ox < w; ox += 7) {
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
                max_err = std::fmax(max_err, std::fabs(sum - out.channel(p)[static_cast<std::size_t>(oy * w + ox)]));
            }
        }
    }
    EXPECT_LT(max_err, 1e-4F);
}

TEST(Kernel, InterpBilinearEdgeClamps) {
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

TEST(Kernel, ReductionLastAxisOn4dBlob) {
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

TEST(Kernel, ReductionLastAxisOn3dBlob) {
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

TEST(Kernel, LrnMatchesReference) {
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

TEST(Kernel, ConvolutionGemmPackOddShapes) {
    // Exercises the pack8 GEMM tail paths: pixel count not a multiple of the
    // 32-pixel tile and output channels not a multiple of the 4-channel block.
    struct Shape {
        std::int32_t kw;
        std::int32_t stride;
        std::int32_t pad;
    };
    const Shape shapes[3] = {{1, 1, 0}, {3, 1, 1}, {3, 2, 1}};

    for (const Shape& shape : shapes) {
        const std::int32_t w = 7;
        const std::int32_t h = 5;
        const std::int32_t inch = 5;
        const std::int32_t outch = 6;
        const std::int32_t maxk = shape.kw * shape.kw;

        mformat::ParamDict pd;
        pd.set(0, outch);
        pd.set(1, shape.kw);
        pd.set(11, shape.kw);
        pd.set(3, shape.stride);
        pd.set(13, shape.stride);
        pd.set(4, shape.pad);
        pd.set(14, shape.pad);
        pd.set(15, shape.pad);
        pd.set(16, shape.pad);
        pd.set(5, 1);
        pd.set(6, outch * inch * maxk);
        pd.set(9, 1);  // ReLU

        mnet::Mat weights[2];
        weights[0].create(outch * inch * maxk);
        weights[1].create(outch);
        std::vector<float> wv(static_cast<std::size_t>(outch * inch * maxk));
        for (std::int32_t i = 0; i < outch * inch * maxk; ++i) {
            wv[static_cast<std::size_t>(i)] = std::sin(0.37F * static_cast<float>(i)) * 0.4F;
            weights[0][static_cast<std::size_t>(i)] = wv[static_cast<std::size_t>(i)];
        }
        for (std::int32_t i = 0; i < outch; ++i) {
            weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i) - 0.1F;
        }

        Conv conv;
        ASSERT_EQ(conv.load_param(pd), 0);
        ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

        mnet::Mat input(w, h, inch);
        std::vector<float> in(static_cast<std::size_t>(w * h * inch));
        for (std::int32_t i = 0; i < w * h * inch; ++i) {
            in[static_cast<std::size_t>(i)] = std::sin(0.53F * static_cast<float>(i) + 0.2F);
            input.channel(i / (w * h))[static_cast<std::size_t>(i % (w * h))] = in[static_cast<std::size_t>(i)];
        }

        mnet::Option opt;
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
        ASSERT_EQ(out.w, (w + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.h, (h + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.c, outch);

        for (std::int32_t p = 0; p < outch; ++p) {
            const float* wp = wv.data() + static_cast<std::size_t>(p * inch * maxk);
            for (std::int32_t y = 0; y < out.h; ++y) {
                for (std::int32_t x = 0; x < out.w; ++x) {
                    float sum = weights[1][static_cast<std::size_t>(p)];
                    for (std::int32_t q = 0; q < inch; ++q) {
                        for (std::int32_t ky = 0; ky < shape.kw; ++ky) {
                            for (std::int32_t kx = 0; kx < shape.kw; ++kx) {
                                const std::int32_t sy = y * shape.stride + ky - shape.pad;
                                const std::int32_t sx = x * shape.stride + kx - shape.pad;
                                if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                                    continue;
                                }
                                sum += wp[static_cast<std::size_t>((q * maxk) + ky * shape.kw + kx)] *
                                       in[static_cast<std::size_t>((q * h + sy) * w + sx)];
                            }
                        }
                    }
                    const float expect = sum > 0.0F ? sum : 0.0F;
                    EXPECT_NEAR(out.channel(p)[static_cast<std::size_t>(y * out.w + x)], expect, 1e-5F)
                        << "k=" << shape.kw << " s=" << shape.stride << " p=" << p << " y=" << y << " x=" << x;
                }
            }
        }
    }
}

TEST(Kernel, RoiPoolingMatchesReference) {
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

TEST(Kernel, ConvolutionDepthWiseMatchesReference) {
    using ConvDW = mlayer::ConvolutionDepthWise<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict,
                                                mformat::ModelBin, false, false, mnet::activation_ss,
                                                mnet::quantize_int8, mnet::copy_make_border, mnet::flatten>;

    struct Shape {
        std::int32_t kw;
        std::int32_t stride;
        std::int32_t pad;
        std::int32_t activation;
    };
    const Shape shapes[3] = {{3, 1, 1, 0}, {3, 2, 1, 0}, {3, 1, 1, 1}};

    for (const Shape& shape : shapes) {
        const std::int32_t w = 11;
        const std::int32_t h = 9;
        const std::int32_t channels = 6;
        const std::int32_t maxk = shape.kw * shape.kw;

        mformat::ParamDict pd;
        pd.set(0, channels);
        pd.set(1, shape.kw);
        pd.set(11, shape.kw);
        pd.set(3, shape.stride);
        pd.set(13, shape.stride);
        pd.set(4, shape.pad);
        pd.set(14, shape.pad);
        pd.set(15, shape.pad);
        pd.set(16, shape.pad);
        pd.set(7, channels);
        pd.set(5, 1);
        pd.set(6, channels * maxk);
        pd.set(9, shape.activation);

        mnet::Mat weights[2];
        weights[0].create(channels * maxk);
        weights[1].create(channels);
        std::vector<float> wv(static_cast<std::size_t>(channels * maxk));
        for (std::int32_t i = 0; i < channels * maxk; ++i) {
            wv[static_cast<std::size_t>(i)] = std::sin(0.29F * static_cast<float>(i) + 0.1F) * 0.5F;
            weights[0][static_cast<std::size_t>(i)] = wv[static_cast<std::size_t>(i)];
        }
        for (std::int32_t i = 0; i < channels; ++i) {
            weights[1][static_cast<std::size_t>(i)] = 0.03F * static_cast<float>(i) - 0.05F;
        }

        ConvDW conv;
        ASSERT_EQ(conv.load_param(pd), 0);
        ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

        mnet::Mat input(w, h, channels);
        std::vector<float> in(static_cast<std::size_t>(w * h * channels));
        for (std::int32_t i = 0; i < w * h * channels; ++i) {
            in[static_cast<std::size_t>(i)] = std::sin(0.41F * static_cast<float>(i) - 0.7F);
            input.channel(i / (w * h))[static_cast<std::size_t>(i % (w * h))] = in[static_cast<std::size_t>(i)];
        }

        mnet::Option opt;
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
        ASSERT_EQ(out.w, (w + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.h, (h + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.c, channels);

        for (std::int32_t q = 0; q < channels; ++q) {
            const float* kptr = wv.data() + static_cast<std::size_t>(q * maxk);
            for (std::int32_t y = 0; y < out.h; ++y) {
                for (std::int32_t x = 0; x < out.w; ++x) {
                    float sum = weights[1][static_cast<std::size_t>(q)];
                    for (std::int32_t ky = 0; ky < shape.kw; ++ky) {
                        for (std::int32_t kx = 0; kx < shape.kw; ++kx) {
                            const std::int32_t sy = y * shape.stride + ky - shape.pad;
                            const std::int32_t sx = x * shape.stride + kx - shape.pad;
                            if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                                continue;
                            }
                            sum += kptr[static_cast<std::size_t>(ky * shape.kw + kx)] *
                                   in[static_cast<std::size_t>((q * h + sy) * w + sx)];
                        }
                    }
                    const float expect = shape.activation == 1 ? std::max(sum, 0.0F) : sum;
                    EXPECT_NEAR(out.channel(q)[static_cast<std::size_t>(y * out.w + x)], expect, 1e-5F)
                        << "k=" << shape.kw << " s=" << shape.stride << " q=" << q << " y=" << y << " x=" << x;
                }
            }
        }
    }
}

TEST(Kernel, GemmFastInt4MatchesInt8Reference) {
    constexpr std::int32_t N = 4;
    constexpr std::int32_t K = 128;
    constexpr std::int32_t block = 64;
    constexpr std::int32_t M = 3;

    // 4-bit two's-complement weights in ncnn's little-endian packing.
    std::vector<std::int8_t> q(static_cast<std::size_t>(N) * K);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t k = 0; k < K; ++k) {
            q[static_cast<std::size_t>(j * K + k)] = static_cast<std::int8_t>(((j * 7 + k * 3) % 15) - 7);
        }
    }
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(N) * (K / 2), 0);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t k = 0; k < K; ++k) {
            const auto value = static_cast<std::uint32_t>(q[static_cast<std::size_t>(j * K + k)]) & 0x0Fu;
            const std::size_t byte = static_cast<std::size_t>(j) * (K / 2) + static_cast<std::size_t>(k / 2);
            packed[byte] |= static_cast<std::uint8_t>(value << ((k % 2) * 4));
        }
    }

    mnet::Mat w4;
    w4.create(static_cast<std::int32_t>(packed.size()), static_cast<std::size_t>(1u));
    std::memcpy(w4.data, packed.data(), packed.size());
    mnet::Mat w8;
    w8.create(N * K, static_cast<std::size_t>(1u));
    std::memcpy(w8.data, q.data(), q.size());

    mnet::Mat scales;
    scales.create(K / block, N, static_cast<std::size_t>(4u), nullptr);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t b = 0; b < K / block; ++b) {
            scales.row(j)[b] = 9.0F + static_cast<float>(j) + static_cast<float>(b);
        }
    }

    using G = mlayer::Gemm<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    auto make_params = [](std::int32_t term) {
        mformat::ParamDict pd;
        pd.set(3, 1);
        pd.set(5, 1);
        pd.set(6, 1);
        pd.set(8, N);
        pd.set(9, K);
        pd.set(10, -1);
        pd.set(18, term);
        return pd;
    };

    G gemm4;
    ASSERT_EQ(gemm4.load_param(make_params(401)), 0);
    mnet::Mat weights4[2] = {w4, scales};
    ASSERT_EQ(gemm4.load_model(mformat::ModelBinFromMatArray(weights4)), 0);

    G gemm8;
    ASSERT_EQ(gemm8.load_param(make_params(801)), 0);
    mnet::Mat weights8[2] = {w8, scales};
    ASSERT_EQ(gemm8.load_model(mformat::ModelBinFromMatArray(weights8)), 0);

    mnet::Mat in;
    in.create(K, M);
    auto* ptr = static_cast<float*>(in.data);
    for (std::int32_t i = 0; i < K * M; ++i) {
        ptr[i] = std::sin(0.017F * static_cast<float>(i) + 0.12F);
    }

    mnet::Option opt_fast;
    opt_fast.use_fast_int4_gemm = true;
    mnet::Option opt_ref;

    mnet::Mat out4;
    mnet::Mat out8;
    ASSERT_EQ(gemm4.forward(in, out4, opt_fast), 0);
    ASSERT_EQ(gemm8.forward(in, out8, opt_ref), 0);
    ASSERT_EQ(out4.w, N);
    ASSERT_EQ(out4.h, M);
    for (std::int32_t i = 0; i < M; ++i) {
        for (std::int32_t j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(out4.row(i)[j], out8.row(i)[j]) << "i=" << i << " j=" << j;
        }
    }
}
