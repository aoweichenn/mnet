// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Shape, ReportsLogicalOnnxOrderWithImplicitLeadingDimensions) {
    constexpr std::int32_t MNET_TEST_SHAPE_WIDTH = 12;
    constexpr std::int32_t MNET_TEST_SHAPE_HEIGHT = 192;
    auto data = mnet::Mat(MNET_TEST_SHAPE_WIDTH, MNET_TEST_SHAPE_HEIGHT, 1);
    data.fill(0.0F);
    expect_forward({"Shape",
                    {data},
                    {vector({1, MNET_TEST_SHAPE_HEIGHT, MNET_TEST_SHAPE_WIDTH})},
                    params<std::int32_t>({{0, 3}})});
    expect_forward(
        {"Shape", {tensor({4, 3, 1, 2, 1}, {1, 2, 3, 4, 5, 6})}, {vector({2, 1, 3})}, params<std::int32_t>({{0, 3}})});
}

// assertions-in-helper: expect_forward
TEST(OnnxSlice, ReversesRowsWithOnnxOpenEndSentinel) {
    constexpr float MNET_TEST_ONNX_OPEN_END = -9.0e18F;
    const auto data = tensor({2, 4, 2}, {1, 2, 3, 4, -1, -2, -3, -4});
    expect_forward({"OnnxSlice",
                    {data, vector({-1}), vector({MNET_TEST_ONNX_OPEN_END}), vector({0}), vector({-1})},
                    {tensor({2, 4, 2}, {-1, -2, -3, -4, 1, 2, 3, 4})},
                    params<std::int32_t>({{0, 2}})});
}

// assertions-in-helper: expect_forward
TEST(ConstantOfShape, FillsEveryElementOfRuntimeShape) {
    expect_forward(
        {"ConstantOfShape", {vector({2, 3})}, {tensor({2, 3, 2}, {7, 7, 7, 7, 7, 7})}, params<float>({{0, 7}})});
}

// assertions-in-helper: expect_forward
TEST(ReshapeDynamic, PreservesElementOrderWhenIntroducingExplicitBatchAndDepth) {
    constexpr std::int32_t MNET_TEST_RESHAPE_WIDTH = 12;
    constexpr std::int32_t MNET_TEST_RESHAPE_INPUT_HEIGHT = 192;
    constexpr std::int32_t MNET_TEST_RESHAPE_OUTPUT_DEPTH = 2;
    constexpr std::int32_t MNET_TEST_RESHAPE_OUTPUT_HEIGHT =
        MNET_TEST_RESHAPE_INPUT_HEIGHT / MNET_TEST_RESHAPE_OUTPUT_DEPTH;
    constexpr std::int32_t MNET_TEST_RESHAPE_ELEMENTS = MNET_TEST_RESHAPE_WIDTH * MNET_TEST_RESHAPE_INPUT_HEIGHT;
    auto data = mnet::Mat(MNET_TEST_RESHAPE_WIDTH, MNET_TEST_RESHAPE_INPUT_HEIGHT, 1);
    auto expected =
        mnet::Mat(MNET_TEST_RESHAPE_WIDTH, MNET_TEST_RESHAPE_OUTPUT_HEIGHT, MNET_TEST_RESHAPE_OUTPUT_DEPTH, 1);
    for (auto i = 0; i < MNET_TEST_RESHAPE_ELEMENTS; ++i) {
        const auto index = static_cast<std::size_t>(i);
        data[index] = static_cast<float>(i);
        expected[index] = static_cast<float>(i);
    }
    expect_forward(
        {"ReshapeDynamic",
         {data, vector({1, MNET_TEST_RESHAPE_OUTPUT_DEPTH, MNET_TEST_RESHAPE_OUTPUT_HEIGHT, MNET_TEST_RESHAPE_WIDTH})},
         {expected}});
}

// assertions-in-helper: expect_forward
TEST(ExpandDynamic, RepeatsAVectorAsRows) {
    expect_forward({"ExpandDynamic", {vector({1, 2, 3}), vector({2, 3})}, {tensor({2, 3, 2}, {1, 2, 3, 1, 2, 3})}});
}

}  // namespace
