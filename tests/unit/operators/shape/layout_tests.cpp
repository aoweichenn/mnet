// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/crop.hpp>
#include <mlayer/expanddims.hpp>
#include <mlayer/permute.hpp>
#include <mlayer/reshape.hpp>
#include <mlayer/squeeze.hpp>
#include <mnet/core/layer.hpp>

TEST(ShapeKernel, ExpandDimsScalarAxisMappings) {
    using ED = mlayer::ExpandDims<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    const auto run = [](const mnet::Mat& in, std::int32_t axis, std::int32_t ew, std::int32_t eh, std::int32_t ec) {
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
    run(mat, 0, 4, 3, 1);  // expand c
    run(mat, 1, 4, 1, 3);  // expand h
    run(mat, 2, 1, 4, 3);  // expand w
}

TEST(ShapeKernel, ReshapeExpandDimsCropChain) {
    using Reshape = mlayer::Reshape<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    using ED = mlayer::ExpandDims<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    using Crop = mlayer::Crop<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mnet::Option opt;
    mnet::Mat flat(1024);
    for (std::int32_t i = 0; i < 1024; ++i) {
        flat[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    mformat::ParamDict p_reshape;
    p_reshape.set(0, 1024);
    p_reshape.set(1, 1);
    Reshape reshape;
    ASSERT_EQ(reshape.load_param(p_reshape), 0);
    std::vector<mnet::Mat> r_out(1);
    ASSERT_EQ(reshape.forward(flat, r_out[0], opt), 0);
    EXPECT_EQ(r_out[0].w, 1024);
    EXPECT_EQ(r_out[0].h, 1);

    mformat::ParamDict p_ed;
    p_ed.set(0, 2);  // prepend w
    ED ed;
    ASSERT_EQ(ed.load_param(p_ed), 0);
    std::vector<mnet::Mat> e_out(1);
    ASSERT_EQ(ed.forward(r_out[0], e_out[0], opt), 0);
    EXPECT_EQ(e_out[0].w, 1);
    EXPECT_EQ(e_out[0].h, 1024);
    EXPECT_EQ(e_out[0].c, 1);

    mformat::ParamDict p_crop;
    p_crop.set(1, 0);
    p_crop.set(4, 512);
    Crop crop;
    ASSERT_EQ(crop.load_param(p_crop), 0);
    std::vector<mnet::Mat> bottoms = {e_out[0]};
    std::vector<mnet::Mat> c_out(1);
    ASSERT_EQ(crop.forward(bottoms, c_out, opt), 0);
    EXPECT_EQ(c_out[0].w, 1);
    EXPECT_EQ(c_out[0].h, 512);
    EXPECT_FLOAT_EQ(c_out[0].channel(0).row(511)[0], 511.0F);
}

TEST(ShapeKernel, PermuteForced4DFrom3D) {
    using Perm = mlayer::Permute<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, false>;
    mformat::ParamDict pd;
    pd.set(0, 114);
    Perm perm;
    ASSERT_EQ(perm.load_param(pd), 0);
    mnet::Mat in(4, 3, 2);
    for (std::int32_t t = 0; t < 2; ++t) {
        for (std::int32_t c = 0; c < 3; ++c) {
            for (std::int32_t d = 0; d < 4; ++d) {
                in.channel(t).row(c)[d] = static_cast<float>(t * 100 + c * 10 + d);
            }
        }
    }
    mnet::Mat out;
    mnet::Option opt;
    ASSERT_EQ(perm.forward(in, out, opt), 0);
    EXPECT_EQ(out.dims, 4);
    EXPECT_EQ(out.w, 4);
    EXPECT_EQ(out.h, 2);
    EXPECT_EQ(out.d, 1);
    EXPECT_EQ(out.c, 3);
    for (std::int32_t c = 0; c < 3; ++c) {
        for (std::int32_t t = 0; t < 2; ++t) {
            for (std::int32_t d = 0; d < 4; ++d) {
                EXPECT_FLOAT_EQ(out.channel(c).depth(0).row(t)[d], static_cast<float>(t * 100 + c * 10 + d));
            }
        }
    }
}

TEST(ShapeKernel, PermuteSwapsWidthAndHeight) {
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

TEST(ShapeKernel, ReshapeNumericAndExpression) {
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

TEST(ShapeKernel, ReshapeLargeChannel) {
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

TEST(ShapeKernel, ExpandDimsThenSqueezeRoundTrips) {
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

TEST(ShapeKernel, CropByReferenceShape) {
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
