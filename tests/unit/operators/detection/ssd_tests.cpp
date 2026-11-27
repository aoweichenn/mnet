// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(PriorBox, NormalizesAndClipsAnchorsAndWritesVariances) {
    auto parameters = params<std::int32_t>({{8, 1}, {9, 8}, {10, 4}});
    parameters.set(0, vector({4}));
    parameters.set(13, 0.5F);
    const auto feature_map = tensor({2, 2, 1}, {0, 0});
    auto image = mnet::Mat(8, 4);
    image.fill(0.0F);
    const auto expected =
        tensor({2, 8, 2}, {0, 0, 0.5F, 1, 0.5F, 0, 1, 1, 0.1F, 0.1F, 0.2F, 0.2F, 0.1F, 0.1F, 0.2F, 0.2F});
    expect_forward({"PriorBox", {feature_map}, {expected}, parameters});
    expect_forward({"PriorBox", {feature_map, image}, {expected}, parameters});
}

// assertions-in-helper: expect_forward
TEST(DetectionOutput, DecodesOffsetAndSuppressesLowerScoringOverlap) {
    const auto offsets = vector({1, 0, 0, 0, 0, 0, 0, 0});
    const auto scores = vector({0.1F, 0.9F, 0.2F, 0.8F});
    const auto priors = tensor(
        {2, 8, 2}, {0.1F, 0.2F, 0.5F, 0.6F, 0.1F, 0.2F, 0.5F, 0.6F, 0.1F, 0.1F, 0.2F, 0.2F, 0.1F, 0.1F, 0.2F, 0.2F});
    expect_forward({"DetectionOutput",
                    {offsets, scores, priors},
                    {tensor({2, 6, 1}, {1, 0.9F, 0.14F, 0.2F, 0.54F, 0.6F})},
                    params<std::int32_t>({{0, 2}})});
}

}  // namespace
