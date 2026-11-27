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
TEST(Range, DescendsWithExclusiveLimitAndCanProduceEmptyOutput) {
    expect_forward({"Range", {vector({5}), vector({-1}), vector({-2})}, {vector({5, 3, 1})}});
    expect_forward({"Range", {vector({1}), vector({4}), vector({-2})}, {mnet::Mat{}}});
}

// assertions-in-helper: expect_rejected
TEST(Range, RejectsZeroStep) {
    expect_rejected("Range", {vector({1}), vector({3}), vector({0})});
}

// assertions-in-helper: expect_forward
TEST(NonZero, ReturnsEveryCoordinateInRowMajorOrderAndHandlesZeros) {
    const auto data = tensor({2, 3, 2}, {0, -2, 0, 7, 0, 5});
    expect_forward({"NonZero", {data}, {tensor({2, 3, 2}, {0, 1, 1, 1, 0, 2})}});
    expect_forward({"NonZero", {vector({0, 0, 0})}, {mnet::Mat{}}});
}

// assertions-in-helper: expect_forward
TEST(GatherND, GathersEntireTrailingRowsAndAllowsRepeatedIndices) {
    const auto data = tensor({2, 3, 2}, {1, -2, 3, 4, 5, 6});
    const auto rows = tensor({2, 1, 3}, {1, 0, 1});
    expect_forward({"GatherND", {data, rows}, {tensor({2, 3, 3}, {4, 5, 6, 1, -2, 3, 4, 5, 6})}});
}

// assertions-in-helper: expect_rejected
TEST(Gather, RejectsMissingDataAndIndexInputs) {
    expect_rejected("Gather", {});
    expect_rejected("Gather", {tensor({2, 2, 2}, {1, 2, 3, 4})});
}

// assertions-in-helper: expect_rejected
TEST(GatherElements, RejectsOutOfRangeIndicesAndUnsupportedAxis) {
    const auto data = tensor({2, 3, 2}, {1, 2, 3, 4, 5, 6});
    expect_rejected("GatherElements", {data, tensor({2, 1, 1}, {3})});
    expect_rejected("GatherElements", {data, tensor({2, 1, 1}, {0})}, params<std::int32_t>({{0, 2}}));
}

}  // namespace
