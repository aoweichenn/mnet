// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::expect_rejected;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Concat, JoinsUnequalWidthsInEachRow) {
    expect_forward({"Concat",
                    {tensor({2, 2, 2}, {1, 2, 3, 4}), tensor({2, 1, 2}, {5, 6})},
                    {tensor({2, 3, 2}, {1, 2, 5, 3, 4, 6})},
                    params<std::int32_t>({{0, -1}})});
}

// assertions-in-helper: expect_forward
TEST(Slice, SplitsEachRowAtUnequalWidths) {
    auto parameters = params<std::int32_t>({{1, -1}});
    parameters.set(0, mnet::test::indices({1, 2}));
    expect_forward({"Slice",
                    {tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6})},
                    {tensor({2, 1, 2}, {1, 4}), tensor({2, 2, 2}, {2, 3, 5, 6})},
                    parameters});
}

// assertions-in-helper: expect_forward
TEST(Split, DeliversFullInputToEveryOutput) {
    const auto data = vector({-1, 3, 2});
    expect_forward({"Split", {data}, {vector({-1, 3, 2}), vector({-1, 3, 2}), vector({-1, 3, 2})}});
}

// assertions-in-helper: expect_forward
TEST(Noop, PassesSeveralInputsThroughInOrder) {
    expect_forward({"Noop", {vector({3, -1}), vector({7})}, {vector({3, -1}), vector({7})}});
}

// assertions-in-helper: expect_forward
TEST(CopyTo, OverwritesOnlyTheRequestedSubmatrix) {
    const auto destination = tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6});
    expect_forward({"CopyTo",
                    {destination, tensor({2, 1, 2}, {8, 9})},
                    {tensor({2, 3, 2}, {1, 8, 3, 4, 9, 6})},
                    params<std::int32_t>({{0, 1}})});
    mnet::test::expect_tensor(destination, tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6}), 0);
}

TEST(DeepCopy, ProducesIndependentStorage) {
    const auto input = tensor({3, 3, 1, 1, 2}, {1, 2, 3, 4, 5, 6});
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("DeepCopy"));
    ASSERT_NE(layer, nullptr);
    auto output = mnet::Mat{};
    ASSERT_EQ(layer->forward(input, output, mnet::Option{}), 0);
    mnet::test::expect_tensor(output, input, 0);
    ASSERT_NE(output.data, input.data);
    output[0] = 0;
    EXPECT_FLOAT_EQ(input[0], 1);
}

// assertions-in-helper: expect_forward
TEST(Flatten, FlattensInChannelOrderWithoutReadingAlignmentPadding) {
    expect_forward({"Flatten", {tensor({3, 3, 1, 1, 2}, {1, 2, 3, 4, 5, 6})}, {vector({1, 2, 3, 4, 5, 6})}});
}

// assertions-in-helper: expect_forward
TEST(Tile, RepeatsTheWholeRowRatherThanEachElement) {
    expect_forward({"Tile",
                    {tensor({2, 2, 2}, {1, 2, 3, 4})},
                    {tensor({2, 6, 2}, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4})},
                    params<mnet::Mat>({{2, mnet::test::indices({1, 3})}})});
}

// assertions-in-helper: expect_forward
TEST(Flip, ReversesTwoAxesWithNegativeAxisNotation) {
    expect_forward({"Flip",
                    {tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6})},
                    {tensor({2, 3, 2}, {6, 5, 4, 3, 2, 1})},
                    params<mnet::Mat>({{0, mnet::test::indices({0, -1})}})});
}

// assertions-in-helper: expect_forward
TEST(Padding, ReflectsWithoutRepeatingEdgeSamples) {
    expect_forward(
        {"Padding", {vector({2, 5, 9})}, {vector({9, 5, 2, 5, 9, 5})}, params<std::int32_t>({{2, 2}, {3, 1}, {4, 2}})});
}

// assertions-in-helper: expect_forward
TEST(OnnxPad, CropsOneSideAndPadsOppositeSideWithDynamicValue) {
    expect_forward({"OnnxPad",
                    {tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6}), vector({1, -1, 0, 1}), vector({-8})},
                    {tensor({2, 3, 3}, {-8, -8, -8, 2, 3, -8, 5, 6, -8})},
                    params<std::int32_t>({{0, 2}})});
}

// assertions-in-helper: expect_rejected
TEST(OnnxPad, RejectsWrongPadCountAndCroppingPastTheInput) {
    const auto data = vector({1, 2, 3});
    const auto parameters = params<std::int32_t>({{0, 1}});
    expect_rejected("OnnxPad", {data, vector({1})}, parameters);
    expect_rejected("OnnxPad", {data, vector({-4, 0})}, parameters);
}

}  // namespace
