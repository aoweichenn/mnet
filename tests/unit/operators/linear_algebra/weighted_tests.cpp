// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(InnerProduct, AppliesIndependentWeightsAndBiasToEveryOutput) {
    expect_forward({"InnerProduct",
                    {vector({2, -1, 3})},
                    {vector({4, -9})},
                    params<std::int32_t>({{0, 2}, {1, 1}, {2, 6}}),
                    {vector({1, 2, 1, -2, 1, 0}), vector({1, -4})}});
}

// assertions-in-helper: expect_forward
TEST(Embed, LooksUpOutOfOrderAndRepeatedTokensWithBias) {
    expect_forward({"Embed",
                    {mnet::test::indices({2, 0, 2, 1})},
                    {tensor({2, 2, 4}, {6, -4, 2, -8, 6, -4, 4, -6})},
                    params<std::int32_t>({{0, 2}, {1, 3}, {2, 1}, {3, 6}}),
                    {vector({1, -1, 3, 1, 5, 3}), vector({1, -7})}});
}

}  // namespace
