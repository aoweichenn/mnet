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
TEST(YoloDetectionOutput, CombinesSoftmaxAndObjectnessAndSuppressesOverlappingBoxes) {
    auto parameters = params<std::int32_t>({{0, 2}, {1, 2}});
    parameters.set(4, vector({0.5F, 1, 0.5F, 1}));
    constexpr float MNET_TEST_LOG_THREE = 1.09861229F;
    const auto logits = tensor({3, 1, 1, 1, 14}, {0, 0, 0, 0, MNET_TEST_LOG_THREE, 0, MNET_TEST_LOG_THREE, 0, 0, 0, 0,
                                                  0, 0, MNET_TEST_LOG_THREE});
    expect_forward(
        {"YoloDetectionOutput", {logits}, {tensor({2, 6, 1}, {2, 0.5625F, 0.25F, 0, 0.75F, 1})}, parameters});
}

// assertions-in-helper: expect_forward
TEST(Yolov3DetectionOutput, UsesSelectedAnchorAndNetworkScale) {
    auto parameters = params<std::int32_t>({{0, 1}, {1, 1}});
    parameters.set(4, vector({1, 1, 2, 4}));
    parameters.set(5, vector({1}));
    parameters.set(6, vector({8}));
    const auto logits = tensor({3, 1, 1, 1, 6}, {0, 0, 0, 0, 0, 0});
    expect_forward({"Yolov3DetectionOutput",
                    {logits},
                    {tensor({2, 6, 1}, {1, 1.0F / 3.0F, 0.375F, 0.25F, 0.625F, 0.75F})},
                    parameters});
}

// assertions-in-helper: expect_rejected
TEST(YoloDetectionOutput, RejectsUnexpectedChannels) {
    auto parameters = params<std::int32_t>({{0, 1}, {1, 1}});
    parameters.set(4, vector({1, 1}));
    expect_rejected("YoloDetectionOutput", {tensor({3, 1, 1, 1, 2}, {0, 0})}, parameters);
}

}  // namespace
