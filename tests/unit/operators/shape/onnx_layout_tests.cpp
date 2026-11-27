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
TEST(SqueezeOnnx, RemovesOnlyTheRequestedSingletonAxis) {
    expect_forward({"SqueezeOnnx",
                    {tensor({4, 3, 2, 1, 2}, {1, -2, 3, 4, 5, -6, 7, 8, 9, -10, 11, 12})},
                    {tensor({3, 3, 2, 1, 2}, {1, -2, 3, 4, 5, -6, 7, 8, 9, -10, 11, 12})},
                    params<std::int32_t>({{0, 4}, {1, 1}})});
}

// assertions-in-helper: expect_rejected
TEST(SqueezeOnnx, RejectsRemovalOfAnAxisThatIsNotSingleton) {
    expect_rejected("SqueezeOnnx", {tensor({2, 2, 2}, {1, 2, 3, 4})}, params<std::int32_t>({{0, 2}, {1, 1}}));
}

// assertions-in-helper: expect_forward
TEST(UnsqueezeOnnx, RestoresImplicitBatchAxisAndInsertsTrailingSingleton) {
    expect_forward({"UnsqueezeOnnx",
                    {tensor({2, 3, 2}, {1, -2, 3, 4, 5, -6})},
                    {tensor({4, 1, 3, 2, 1}, {1, -2, 3, 4, 5, -6})},
                    params<std::int32_t>({{0, 3}, {1, 3}})});
}

// assertions-in-helper: expect_forward
TEST(ReshapeDynamic, CopiesZeroDimensionAndInfersRemainingExtent) {
    const auto data = tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6});
    expect_forward({"ReshapeDynamic", {data, vector({0, -1})}, {tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6})}});
    expect_forward({"ReshapeDynamic", {data, vector({3, -1})}, {tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6})}});
}

// assertions-in-helper: expect_rejected
TEST(ReshapeDynamic, RejectsAmbiguousAndIncompatibleShapes) {
    const auto data = vector({1, 2, 3, 4, 5, 6});
    expect_rejected("ReshapeDynamic", {data, vector({-1, -1})});
    expect_rejected("ReshapeDynamic", {data, vector({4, -1})});
    expect_rejected("ReshapeDynamic", {data, vector({2, 2})});
}

// assertions-in-helper: expect_forward
TEST(ExpandDynamic, BroadcastsMultipleAxesWithoutChangingElementOrder) {
    const auto data = tensor({2, 1, 2}, {3, -1});
    const auto expected = tensor({3, 3, 2, 1, 2}, {3, 3, 3, -1, -1, -1, 3, 3, 3, -1, -1, -1});
    expect_forward({"ExpandDynamic", {data, vector({2, 2, 3})}, {expected}});
}

// assertions-in-helper: expect_rejected
TEST(ExpandDynamic, RejectsExpansionOfIncompatibleNonSingletonAxis) {
    expect_rejected("ExpandDynamic", {vector({1, 2}), vector({2, 3})});
}

// assertions-in-helper: expect_rejected
TEST(ConstantOfShape, RejectsNonPositiveDimensions) {
    expect_rejected("ConstantOfShape", {vector({2, -1})});
}

// assertions-in-helper: expect_rejected
TEST(OnnxSlice, RejectsZeroStepAndOutOfRangeAxis) {
    const auto data = vector({1, 2, 3});
    const auto parameters = params<std::int32_t>({{0, 1}});
    expect_rejected("OnnxSlice", {data, vector({0}), vector({2}), vector({0}), vector({0})}, parameters);
    expect_rejected("OnnxSlice", {data, vector({0}), vector({2}), vector({1}), vector({1})}, parameters);
}

}  // namespace
