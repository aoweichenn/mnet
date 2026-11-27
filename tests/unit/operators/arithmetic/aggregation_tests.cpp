// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::expect_rejected;
using mnet::test::params;
using mnet::test::ReferenceTensor;
using mnet::test::tensor;
using mnet::test::topk_reference;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(ArgMax, ReturnsTopKValuesAndOriginalPositions) {
    expect_forward({"ArgMax",
                    {vector({-4, 5, -1, 8, 2})},
                    {tensor({2, 3, 2}, {8, 5, 2, 3, 1, 4})},
                    params<std::int32_t>({{0, 1}, {1, 3}})});
}

// assertions-in-helper: expect_rejected
TEST(ArgMax, RejectsTopKOutsideInputLength) {
    expect_rejected("ArgMax", {vector({1})}, params<std::int32_t>({{1, 0}}));
    const auto input = vector({1});
    const auto beyond_storage = static_cast<std::int32_t>(input.total()) + 1;
    expect_rejected("ArgMax", {input}, params<std::int32_t>({{1, beyond_storage}}));
}

// assertions-in-helper: expect_rejected
TEST(ArgMax, RejectsTopKThatOnlyFitsInAlignmentPadding) {
    const auto input = vector({1});
    const auto beyond_logical_length = input.w + 1;
    expect_rejected("ArgMax", {input}, params<std::int32_t>({{1, beyond_logical_length}}));
}

// assertions-in-helper: expect_forward
TEST(ArgMax, IgnoresAlignmentPaddingBetweenChannels) {
    constexpr float MNET_TEST_ARGMAX_PADDING = 1000.0F;
    auto input = tensor({3, 3, 1, 1, 2}, {1, 8, 3, -2, 7, -9});
    const auto plane = static_cast<std::size_t>(input.w * input.h * input.d);
    auto logical = std::vector<float>(static_cast<std::size_t>(input.c) * plane);
    for (std::int32_t channel = 0; channel < input.c; ++channel) {
        const auto view = input.channel(channel);
        std::copy_n(static_cast<const float*>(view.data), plane,
                    logical.data() + static_cast<std::size_t>(channel) * plane);
        auto writable = input.channel(channel);
        std::fill_n(static_cast<float*>(writable) + static_cast<std::ptrdiff_t>(plane),
                    static_cast<std::ptrdiff_t>(writable.total() - plane), MNET_TEST_ARGMAX_PADDING);
    }
    const auto reference = ReferenceTensor({static_cast<std::size_t>(input.c), 1, plane}, logical);
    expect_forward({"ArgMax",
                    {input},
                    {mnet::test::convert(topk_reference(reference, 4, true))},
                    params<std::int32_t>({{0, 1}, {1, 4}})});
    expect_forward({"ArgMax",
                    {input},
                    {mnet::test::convert(topk_reference(reference, 4, false))},
                    params<std::int32_t>({{0, 0}, {1, 4}})});
}

// assertions-in-helper: expect_forward
TEST(ArgMax, KeepsTheFirstIndexForTies) {
    constexpr float MNET_TEST_ARGMAX_TIE = 5.0F;
    expect_forward({"ArgMax",
                    {vector({MNET_TEST_ARGMAX_TIE, MNET_TEST_ARGMAX_TIE, 1})},
                    {tensor({2, 2, 2}, {MNET_TEST_ARGMAX_TIE, MNET_TEST_ARGMAX_TIE, 0, 1})},
                    params<std::int32_t>({{0, 1}, {1, 2}})});
}

// assertions-in-helper: expect_rejected
TEST(ArgMax, RejectsTopKBeyondLogicalElements) {
    expect_rejected("ArgMax", {tensor({3, 2, 1, 1, 2}, {1, 2, 3, 4})}, params<std::int32_t>({{1, 5}}));
}

// assertions-in-helper: expect_forward
TEST(ArgMax, MatchesOnnxTopKAcrossInputRanks) {
    const auto shapes = std::vector<std::vector<std::size_t>>{{5}, {2, 3}, {2, 2, 3}, {2, 2, 2, 2}, {3, 65, 2}};
    auto ordinal = std::size_t{0};
    for (const auto& shape : shapes) {
        SCOPED_TRACE(shape.size());
        auto values = std::vector<float>{};
        const auto count = ReferenceTensor::element_count(shape);
        for (std::size_t i = 0; i < count; ++i) {
            values.push_back(static_cast<float>((i * 7 + ordinal) % 5));
        }
        const auto reference = ReferenceTensor(shape, values);
        for (const auto top_k : {1, 2, 4}) {
            if (static_cast<std::size_t>(top_k) > reference.element_count()) {
                continue;
            }
            SCOPED_TRACE(top_k);
            const auto indices = topk_reference(reference, top_k, false);
            expect_forward({"ArgMax",
                            {mnet::test::convert(reference)},
                            {mnet::test::convert(indices)},
                            params<std::int32_t>({{0, 0}, {1, top_k}})});
            const auto ranked = topk_reference(reference, top_k, true);
            expect_forward({"ArgMax",
                            {mnet::test::convert(reference)},
                            {mnet::test::convert(ranked)},
                            params<std::int32_t>({{0, 1}, {1, top_k}})});
        }
        ++ordinal;
    }
}

// assertions-in-helper: expect_forward
TEST(Eltwise, AppliesEachCoefficientToItsOwnInput) {
    auto parameters = params<std::int32_t>({{0, 1}});
    parameters.set(1, vector({2, -1, 0.5F}));
    expect_forward(
        {"Eltwise", {vector({1, 2, 3}), vector({4, 5, 6}), vector({8, 6, 4})}, {vector({2, 2, 2})}, parameters});
    expect_forward(
        {"Eltwise", {vector({1, 2, 3}), vector({4, 5, 6})}, {vector({4, 5, 6})}, params<std::int32_t>({{0, 2}})});
}

// assertions-in-helper: expect_forward
TEST(Diag, PlacesValuesOnTheRequestedUpperDiagonal) {
    expect_forward(
        {"Diag", {vector({5, -3})}, {tensor({2, 3, 3}, {0, 5, 0, 0, 0, -3, 0, 0, 0})}, params<std::int32_t>({{0, 1}})});
}

// assertions-in-helper: expect_forward
TEST(GLU, GatesTheFirstHalfUsingTheSecondHalf) {
    expect_forward({"GLU", {vector({2, -4, 0, 1})}, {vector({1, -2.92423431F})}});
}

// assertions-in-helper: expect_rejected
TEST(GLU, RejectsInputWithNoRoomForGateAndValue) {
    expect_rejected("GLU", {vector({1})});
}

}  // namespace
