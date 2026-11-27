// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(RNN, AppliesPreviousHiddenStateInReverseSequenceOrder) {
    const auto last = std::tanh(1.0F);
    const auto middle = std::tanh(last);
    const auto first = std::tanh(-1.0F + middle);
    expect_forward({"RNN",
                    {tensor({2, 1, 3}, {-1, 0, 1})},
                    {tensor({2, 1, 3}, {first, middle, last})},
                    params<std::int32_t>({{0, 1}, {1, 1}, {2, 1}}),
                    {vector({1}), vector({0}), vector({1})}});
}

// assertions-in-helper: expect_forward
TEST(GRU, RetainsAndUpdatesHiddenStateAtEachStep) {
    constexpr float MNET_TEST_GATE = 0.5F;
    const auto candidate = std::tanh(1.0F);
    const auto h0 = MNET_TEST_GATE * candidate;
    const auto h1 = MNET_TEST_GATE * h0;
    const auto h2 = MNET_TEST_GATE * (h1 - candidate);
    expect_forward({"GRU",
                    {tensor({2, 1, 3}, {1, 0, -1})},
                    {tensor({2, 1, 3}, {h0, h1, h2})},
                    params<std::int32_t>({{0, 1}, {1, 3}}),
                    {vector({0, 0, 1}), vector({0, 0, 0, 0}), vector({0, 0, 0})}});
}

// assertions-in-helper: expect_forward
TEST(LSTM, CarriesTheCellStateAcrossThreeTimesteps) {
    constexpr float MNET_TEST_GATE = 0.5F;
    const auto candidate = std::tanh(1.0F);
    const auto c0 = MNET_TEST_GATE * candidate;
    const auto c1 = MNET_TEST_GATE * c0;
    const auto c2 = MNET_TEST_GATE * (c1 - candidate);
    const auto output = tensor(
        {2, 1, 3}, {MNET_TEST_GATE * std::tanh(c0), MNET_TEST_GATE * std::tanh(c1), MNET_TEST_GATE * std::tanh(c2)});
    expect_forward({"LSTM",
                    {tensor({2, 1, 3}, {1, 0, -1})},
                    {output},
                    params<std::int32_t>({{0, 1}, {1, 4}}),
                    {vector({0, 0, 0, 1}), vector({0, 0, 0, 0}), vector({0, 0, 0, 0})}});
}

}  // namespace
