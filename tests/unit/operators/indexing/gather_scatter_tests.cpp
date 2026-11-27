// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/gather.hpp>
#include <mlayer/gatherelements.hpp>
#include <mlayer/gathernd.hpp>
#include <mlayer/nonzero.hpp>
#include <mlayer/range.hpp>
#include <mlayer/scatternd.hpp>
#include <mnet/layer.hpp>

// Gather along axis 0 (1-D and 2-D) and axis 1 (2-D).
TEST(IndexingKernel, GatherIndices) {
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

TEST(IndexingKernel, GatherSqueezedVectorIndex) {
    using G = mlayer::Gather<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    pd.set(0, 0);
    G gather;
    ASSERT_EQ(gather.load_param(pd), 0);

    mnet::Mat data(3, 1, 1);
    data[0] = 1.0F;
    data[1] = 2.0F;
    data[2] = 3.0F;
    mnet::Mat index(1);
    index[0] = 1.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {data, index};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(gather.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].w, 1);
    EXPECT_FLOAT_EQ(top[0][0], 2.0F);
}

TEST(IndexingKernel, RangeAndDynamicGatherOps) {
    using R = mlayer::Range<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    R range;
    ASSERT_EQ(range.load_param(mformat::ParamDict()), 0);
    mnet::Mat start(1);
    start[0] = 2.0F;
    mnet::Mat limit(1);
    limit[0] = 7.0F;
    mnet::Mat delta(1);
    delta[0] = 2.0F;
    mnet::Option opt;
    std::vector<mnet::Mat> range_bottom = {start, limit, delta};
    std::vector<mnet::Mat> range_top(1);
    ASSERT_EQ(range.forward(range_bottom, range_top, opt), 0);
    ASSERT_EQ(range_top[0].w, 3);
    EXPECT_FLOAT_EQ(range_top[0][0], 2.0F);
    EXPECT_FLOAT_EQ(range_top[0][1], 4.0F);
    EXPECT_FLOAT_EQ(range_top[0][2], 6.0F);

    mnet::Mat mask(3, 2, 1);
    mask.channel(0).row(0)[0] = 0.0F;
    mask.channel(0).row(0)[1] = 1.0F;
    mask.channel(0).row(0)[2] = 0.0F;
    mask.channel(0).row(1)[0] = 2.0F;
    mask.channel(0).row(1)[1] = 0.0F;
    mask.channel(0).row(1)[2] = 3.0F;
    using NZ = mlayer::NonZero<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    NZ nonzero;
    ASSERT_EQ(nonzero.load_param(mformat::ParamDict()), 0);
    mnet::Mat nz;
    ASSERT_EQ(nonzero.forward(mask, nz, opt), 0);
    ASSERT_EQ(nz.w, 3);
    ASSERT_EQ(nz.h, 3);
    EXPECT_FLOAT_EQ(nz.row(0)[0], 0.0F);
    EXPECT_FLOAT_EQ(nz.row(1)[1], 1.0F);
    EXPECT_FLOAT_EQ(nz.row(2)[2], 2.0F);

    mnet::Mat indices(3, 2);
    indices.row(0)[0] = 0.0F;
    indices.row(0)[1] = 0.0F;
    indices.row(0)[2] = 1.0F;
    indices.row(1)[0] = 0.0F;
    indices.row(1)[1] = 1.0F;
    indices.row(1)[2] = 2.0F;
    using GND = mlayer::GatherND<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    GND gather_nd;
    ASSERT_EQ(gather_nd.load_param(mformat::ParamDict()), 0);
    std::vector<mnet::Mat> gnd_bottom = {mask, indices};
    std::vector<mnet::Mat> gnd_top(1);
    ASSERT_EQ(gather_nd.forward(gnd_bottom, gnd_top, opt), 0);
    ASSERT_EQ(gnd_top[0].w, 2);
    EXPECT_FLOAT_EQ(gnd_top[0][0], 1.0F);
    EXPECT_FLOAT_EQ(gnd_top[0][1], 3.0F);

    mnet::Mat elements(3, 2);
    elements.row(0)[0] = 10.0F;
    elements.row(0)[1] = 20.0F;
    elements.row(0)[2] = 30.0F;
    elements.row(1)[0] = 40.0F;
    elements.row(1)[1] = 50.0F;
    elements.row(1)[2] = 60.0F;
    mnet::Mat element_indices(1, 2);
    element_indices.row(0)[0] = 2.0F;
    element_indices.row(1)[0] = 0.0F;
    using GE = mlayer::GatherElements<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    GE gather_elements;
    mformat::ParamDict ge_pd;
    ge_pd.set(0, 0);
    ASSERT_EQ(gather_elements.load_param(ge_pd), 0);
    std::vector<mnet::Mat> ge_bottom = {elements, element_indices};
    std::vector<mnet::Mat> ge_top(1);
    ASSERT_EQ(gather_elements.forward(ge_bottom, ge_top, opt), 0);
    ASSERT_EQ(ge_top[0].w, 1);
    ASSERT_EQ(ge_top[0].h, 2);
    EXPECT_FLOAT_EQ(ge_top[0].row(0)[0], 30.0F);
    EXPECT_FLOAT_EQ(ge_top[0].row(1)[0], 40.0F);
}

// ScatterND for the iSTFT overlap-add: flat 1-D data, [n,1] indices, n updates.
TEST(IndexingKernel, ScatterNdOverlapAdd) {
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
TEST(IndexingKernel, ScatterNdRankedIndices) {
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
