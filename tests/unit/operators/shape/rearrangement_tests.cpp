// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(ShuffleChannel, InterleavesGroupsAndReversesThePermutation) {
    const auto grouped = tensor({3, 1, 1, 1, 6}, {1, 2, 3, 4, 5, 6});
    const auto interleaved = tensor({3, 1, 1, 1, 6}, {1, 4, 2, 5, 3, 6});
    expect_forward({"ShuffleChannel", {grouped}, {interleaved}, params<std::int32_t>({{0, 2}})});
    expect_forward({"ShuffleChannel", {interleaved}, {grouped}, params<std::int32_t>({{0, 2}, {1, 1}})});
}

// assertions-in-helper: expect_forward
TEST(PixelShuffle, WritesSubpixelsInSpatialOrder) {
    expect_forward({"PixelShuffle",
                    {tensor({3, 2, 1, 1, 4}, {1, 2, 3, 4, 5, 6, 7, 8})},
                    {tensor({3, 4, 2, 1, 1}, {1, 3, 2, 4, 5, 7, 6, 8})},
                    params<std::int32_t>({{0, 2}})});
}

// assertions-in-helper: expect_forward
TEST(Reorg, PacksSpatialOffsetsIntoChannels) {
    expect_forward({"Reorg",
                    {tensor({3, 4, 2, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8})},
                    {tensor({3, 2, 1, 1, 4}, {1, 3, 2, 4, 5, 7, 6, 8})},
                    params<std::int32_t>({{0, 2}})});
}

// assertions-in-helper: expect_forward
TEST(YoloV5Focus, OrdersOffsetsByColumnBeforeRow) {
    expect_forward({"YoloV5Focus",
                    {tensor({3, 4, 2, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8})},
                    {tensor({3, 2, 1, 1, 4}, {1, 3, 5, 7, 2, 4, 6, 8})}});
}

TEST(Packing, PacksChannelsWithoutChangingTheirOrder) {
    const auto input = tensor({3, 2, 1, 1, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("Packing"));
    ASSERT_NE(layer, nullptr);
    constexpr std::int32_t MNET_TEST_CHANNELS_PER_PACK = 4;
    ASSERT_EQ(layer->load_param(params<std::int32_t>({{0, MNET_TEST_CHANNELS_PER_PACK}})), 0);
    auto packed = mnet::Mat{};
    ASSERT_EQ(layer->forward(input, packed, mnet::Option{}), 0);
    ASSERT_FALSE(packed.empty());
    EXPECT_EQ(packed.elempack, MNET_TEST_CHANNELS_PER_PACK);
    EXPECT_EQ(packed.elemsize, MNET_TEST_CHANNELS_PER_PACK * sizeof(float));
    const auto expected_shape = mnet::test::TensorShape{3, 2, 1, 1, 1};
    ASSERT_EQ(mnet::test::shape_of(packed), expected_shape);
    constexpr std::array<float, 8> MNET_TEST_PACKED_CHANNEL_ORDER{1, 3, 5, 7, 2, 4, 6, 8};
    for (std::size_t i = 0; i < MNET_TEST_PACKED_CHANNEL_ORDER.size(); ++i) {
        EXPECT_FLOAT_EQ(packed[i], MNET_TEST_PACKED_CHANNEL_ORDER[i]);
    }
}

}  // namespace
