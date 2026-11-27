// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Bias, AddsASeparateBiasToEachChannel) {
    expect_forward({"Bias",
                    {tensor({3, 3, 1, 1, 2}, {-5, 0, 3, -8, 1, 7})},
                    {tensor({3, 3, 1, 1, 2}, {-3, 2, 5, -11, -2, 4})},
                    params<std::int32_t>({{0, 2}}),
                    {vector({2, -3})}});
}

// assertions-in-helper: expect_forward
TEST(Scale, ScalesBeforeAddingBias) {
    expect_forward({"Scale",
                    {tensor({3, 2, 1, 1, 2}, {-3, 5, -7, 2})},
                    {tensor({3, 2, 1, 1, 2}, {-5, 11, 11, -7})},
                    params<std::int32_t>({{0, 2}, {1, 1}}),
                    {vector({2, -2}), vector({1, -3})}});
}

// assertions-in-helper: expect_forward
TEST(BatchNorm, UsesStoredChannelStatisticsAndAffineWeights) {
    expect_forward({"BatchNorm",
                    {tensor({3, 2, 1, 1, 2}, {1, 5, -1, 5})},
                    {tensor({3, 2, 1, 1, 2}, {-1, 5, 6, -2})},
                    params<std::int32_t>({{0, 2}}),
                    {vector({3, -4}), vector({3, 2}), vector({4, 9}), vector({2, 2})}});
}

// assertions-in-helper: expect_forward
TEST(PReLU, UsesPerChannelSlopesOnlyForNegativeValues) {
    expect_forward({"PReLU",
                    {tensor({3, 3, 1, 1, 2}, {-4, 0, 3, -4, 0, 3})},
                    {tensor({3, 3, 1, 1, 2}, {-1, 0, 3, -2, 0, 3})},
                    params<std::int32_t>({{0, 2}}),
                    {vector({0.25F, 0.5F})}});
}

// assertions-in-helper: expect_forward
TEST(Normalize, UsesJointL2NormThenScalesEachChannel) {
    auto parameters = params<std::int32_t>({{0, 1}, {3, 2}});
    parameters.set(2, 0.0F);
    expect_forward({"Normalize",
                    {tensor({3, 2, 1, 1, 2}, {3, 0, 0, 4})},
                    {tensor({3, 2, 1, 1, 2}, {1.2F, 0, 0, -2.4F})},
                    parameters,
                    {vector({2, -3})}});
}

}  // namespace
