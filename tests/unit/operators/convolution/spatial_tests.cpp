// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Convolution3D, AccumulatesAcrossDepthAndInputChannels) {
    const auto input = tensor({4, 2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    expect_forward({"Convolution3D",
                    {input},
                    {tensor({4, 2, 1, 1, 1}, {3, 6})},
                    params<std::int32_t>({{0, 1}, {1, 1}, {21, 2}, {5, 1}, {6, 4}}),
                    {vector({2, 3, -1, -1}), vector({4})}});
}

// assertions-in-helper: expect_forward
TEST(ConvolutionDepthWise1D, UsesIndependentFiltersWithDilation) {
    expect_forward({"ConvolutionDepthWise1D",
                    {tensor({2, 4, 2}, {1, 2, 3, 4, 6, 8, 10, 12})},
                    {tensor({2, 2, 2}, {7, 10, -2, -4})},
                    params<std::int32_t>({{0, 2}, {1, 2}, {2, 2}, {6, 4}, {7, 2}}),
                    {vector({1, 2, -2, 1})}});
}

// assertions-in-helper: expect_forward
TEST(ConvolutionDepthWise3D, UsesIndependentFiltersAcrossDepth) {
    expect_forward({"ConvolutionDepthWise3D",
                    {tensor({4, 2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8})},
                    {tensor({4, 2, 1, 1, 2}, {7, 10, -3, -4})},
                    params<std::int32_t>({{0, 2}, {1, 1}, {21, 2}, {6, 4}, {7, 2}}),
                    {vector({1, 2, -2, 1})}});
}

// assertions-in-helper: expect_forward
TEST(DeformableConv2D, InterpolatesFractionalOffsetsAndAppliesMask) {
    const auto input = tensor({3, 2, 2, 1, 1}, {2, 6, 10, 18});
    const auto offsets = tensor({3, 2, 2, 1, 2}, {0.5F, 0, 0, 0, 0.5F, 0, 0, 2});
    const auto mask = tensor({3, 2, 2, 1, 1}, {0.5F, 1, 0, 1});
    expect_forward({"DeformableConv2D",
                    {input, offsets, mask},
                    {tensor({3, 2, 2, 1, 1}, {9, 12, 0, 0})},
                    params<std::int32_t>({{0, 1}, {1, 1}, {6, 1}}),
                    {vector({2})}});
}

}  // namespace
