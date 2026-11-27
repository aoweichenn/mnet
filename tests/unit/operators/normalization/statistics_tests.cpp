// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(InstanceNorm, UsesPerInstanceStatisticsInsteadOfGlobalStatistics) {
    auto parameters = params<std::int32_t>({{0, 2}});
    parameters.set(1, 0.0F);
    expect_forward({"InstanceNorm",
                    {tensor({3, 2, 1, 1, 2}, {1, 5, 12, 6})},
                    {tensor({3, 2, 1, 1, 2}, {-1, 3, 2, 8})},
                    parameters,
                    {vector({2, -3}), vector({1, 5})}});
}

// assertions-in-helper: expect_forward
TEST(GroupNorm, SharesStatisticsWithinButNotAcrossGroups) {
    auto parameters = params<std::int32_t>({{0, 2}, {1, 4}, {3, 0}});
    parameters.set(2, 0.0F);
    expect_forward({"GroupNorm",
                    {tensor({3, 1, 1, 1, 4}, {2, 4, 20, 10})},
                    {tensor({3, 1, 1, 1, 4}, {-1, 1, 1, -1})},
                    parameters});
}

// assertions-in-helper: expect_forward
TEST(LayerNorm, NormalizesEachRowAndHandlesZeroVariance) {
    auto parameters = params<std::int32_t>({{0, 2}});
    parameters.set(1, 1.0F);
    expect_forward({"LayerNorm",
                    {tensor({2, 2, 2}, {2, 4, 8, 8})},
                    {tensor({2, 2, 2}, {-0.41421356F, -3.12132034F, 1, -1})},
                    parameters,
                    {vector({2, -3}), vector({1, -1})}});
}

// assertions-in-helper: expect_forward
TEST(MVN, CentersChannelsIndependentlyWithoutScalingWhenDisabled) {
    expect_forward({"MVN", {tensor({3, 2, 1, 1, 2}, {1, 5, 12, 6})}, {tensor({3, 2, 1, 1, 2}, {-2, 2, 3, -3})}});
}

}  // namespace
