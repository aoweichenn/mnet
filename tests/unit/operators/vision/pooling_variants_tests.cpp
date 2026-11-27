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
TEST(SPP, MatchesCaffeBinsOnThreeByFiveMap) {
    // Independent Caffe/ncnn bin computation: level 1 keeps the whole map, level 2
    // splits 3x5 into ceil-sized 2x3 windows with odd padding. Expectations were
    // produced by a separate script, not by this layer.
    const auto input = tensor({3, 5, 3, 1, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    expect_forward({"SPP",
                    {input},
                    {tensor({3, 5, 1, 1, 2}, {15, 2, 5, 12, 15, 15, 2, 5, 12, 15})},
                    params<std::int32_t>({{0, 0}, {1, 2}})});
    expect_forward({"SPP",
                    {input},
                    {tensor({3, 5, 1, 1, 2}, {8, 0.5F, 2, 6, 11.5F, 8, 0.5F, 2, 6, 11.5F})},
                    params<std::int32_t>({{0, 1}, {1, 2}}),
                    {},
                    1e-5F});
}

// assertions-in-helper: expect_forward
TEST(SPP, MatchesCaffeBinsOnFourByFourMapWithThreeLevels) {
    // 4x4 divides evenly at every level, so each bin is exercised exactly once
    // and the expected values are a plain prefix/identity sequence.
    const auto input = tensor({3, 4, 4, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    expect_forward(
        {"SPP",
         {input},
         {tensor({3, 21, 1, 1, 2}, {15, 5, 7, 13, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                    15, 5, 7, 13, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})},
         params<std::int32_t>({{0, 0}, {1, 3}})});
    expect_forward({"SPP",
                    {input},
                    {tensor({3, 21, 1, 1, 2},
                            {7.5F, 2.5F, 4.5F, 10.5F, 12.5F, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                             7.5F, 2.5F, 4.5F, 10.5F, 12.5F, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})},
                    params<std::int32_t>({{0, 1}, {1, 3}}),
                    {},
                    1e-5F});
}

// assertions-in-helper: expect_rejected
TEST(SPP, RejectsNonPositivePyramidHeight) {
    expect_rejected("SPP", {tensor({3, 1, 1, 1, 1}, {1})}, params<std::int32_t>({{1, 0}}));
}

}  // namespace
