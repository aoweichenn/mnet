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
TEST(GridSample, InterpolatesInteriorAndUsesZeroOutsideTheGrid) {
    const auto image = tensor({3, 2, 2, 1, 1}, {2, 6, 10, 18});
    const auto grid = tensor({3, 2, 3, 1, 1}, {0, 0, 1, -1, 4, 0});
    expect_forward({"GridSample", {image, grid}, {tensor({3, 3, 1, 1, 1}, {9, 6, 0})}, params<std::int32_t>({{2, 1}})});
}

// assertions-in-helper: expect_rejected
TEST(GridSample, RequiresGridInput) {
    expect_rejected("GridSample", {tensor({3, 1, 1, 1, 1}, {1})});
}

// assertions-in-helper: expect_rejected
TEST(ROIAlign, RequiresFeatureMapAndRoiInputs) {
    expect_rejected("ROIAlign", {});
    expect_rejected("ROIAlign", {tensor({3, 2, 2, 1, 1}, {1, 2, 3, 4})});
}

// assertions-in-helper: expect_forward
TEST(ROIAlign, SamplesFractionalBinCentersIndependentlyPerChannel) {
    const auto image = tensor({3, 2, 2, 1, 2}, {2, 6, 10, 18, 0, 4, 8, 12});
    expect_forward({"ROIAlign",
                    {image, vector({0, 0, 1, 1})},
                    {tensor({3, 1, 1, 1, 2}, {9, 6})},
                    params<std::int32_t>({{0, 1}, {1, 1}, {3, 1}})});
}

// assertions-in-helper: expect_forward
TEST(Unfold, EmitsKernelOffsetsAsRowsAndSpatialPositionsAsColumns) {
    expect_forward({"Unfold",
                    {tensor({3, 3, 2, 1, 1}, {1, 2, 3, 4, 5, 6})},
                    {tensor({2, 2, 4}, {1, 2, 2, 3, 4, 5, 5, 6})},
                    params<std::int32_t>({{1, 2}})});
}

// assertions-in-helper: expect_forward
TEST(Fold, SumsOverlappingPatchesInsteadOfOverwritingThem) {
    expect_forward({"Fold",
                    {tensor({2, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8})},
                    {tensor({3, 3, 2, 1, 1}, {1, 5, 4, 5, 13, 8})},
                    params<std::int32_t>({{1, 2}, {20, 3}, {21, 2}})});
}

}  // namespace
