// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <limits>

#include <mnet/core/net.hpp>
#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::ReferenceTensor;
using mnet::test::tensor;

// assertions-in-helper: expect_forward
TEST(SDPA, MatchesReferenceForMultipleHeadsAndCausalMask) {
    constexpr std::int32_t MNET_TEST_SDPA_HEADS = 1;
    constexpr std::int32_t MNET_TEST_SDPA_GROUPS = 1;
    constexpr std::int32_t MNET_TEST_SDPA_HEAD_DIM = 4;
    constexpr std::int32_t MNET_TEST_SDPA_TOKENS = 3;
    constexpr float MNET_TEST_SDPA_NEGATIVE_INFINITY = -std::numeric_limits<float>::infinity();
    const auto width = MNET_TEST_SDPA_HEADS * MNET_TEST_SDPA_HEAD_DIM;
    const auto expected_shape = std::vector<std::size_t>{1, MNET_TEST_SDPA_TOKENS, static_cast<std::size_t>(width)};
    auto queries = mnet::Mat(width, MNET_TEST_SDPA_TOKENS);
    auto keys = mnet::Mat(width, MNET_TEST_SDPA_TOKENS, MNET_TEST_SDPA_GROUPS);
    auto values = mnet::Mat(width, MNET_TEST_SDPA_TOKENS, MNET_TEST_SDPA_GROUPS);
    queries.fill(0.0F);
    keys.fill(0.0F);
    values.fill(0.0F);
    auto mask = mnet::Mat(MNET_TEST_SDPA_TOKENS, MNET_TEST_SDPA_TOKENS);
    mask.fill(0.0F);
    for (std::int32_t token = 0; token < MNET_TEST_SDPA_TOKENS; ++token) {
        const auto query_row = queries.row(token);
        const auto key_row = keys.row(token);
        const auto value_row = values.row(token);
        for (std::int32_t index = 0; index < width; ++index) {
            query_row[index] = static_cast<float>(index % 3) - 1;
            key_row[index] = static_cast<float>(index % 5) - 2;
            value_row[index] = static_cast<float>(token * 10 + index);
        }
        for (std::int32_t key_token = 0; key_token < MNET_TEST_SDPA_TOKENS; ++key_token) {
            mask.row(token)[key_token] = key_token <= token ? 0.0F : MNET_TEST_SDPA_NEGATIVE_INFINITY;
        }
    }
    auto expected_values = std::vector<float>{};
    const auto scale = 1.0 / std::sqrt(static_cast<double>(MNET_TEST_SDPA_HEAD_DIM));
    constexpr float MNET_TEST_SDPA_MAX_SCORE = 2.0F;
    for (std::int32_t token = 0; token < MNET_TEST_SDPA_TOKENS; ++token) {
        const auto query_row = queries.row(token);
        auto scores = std::vector<double>(MNET_TEST_SDPA_TOKENS, 0.0);
        auto denominator = 0.0;
        for (std::int32_t key_token = 0; key_token <= token; ++key_token) {
            for (std::int32_t dimension = 0; dimension < MNET_TEST_SDPA_HEAD_DIM; ++dimension) {
                scores[static_cast<std::size_t>(key_token)] +=
                    static_cast<double>(query_row[dimension]) * static_cast<double>(keys.row(key_token)[dimension]);
            }
            scores[static_cast<std::size_t>(key_token)] *= scale;
            denominator +=
                std::exp(scores[static_cast<std::size_t>(key_token)] - static_cast<double>(MNET_TEST_SDPA_MAX_SCORE));
        }
        for (std::int32_t dimension = 0; dimension < MNET_TEST_SDPA_HEAD_DIM; ++dimension) {
            auto value = 0.0;
            for (std::int32_t key_token = 0; key_token <= token; ++key_token) {
                value += std::exp(scores[static_cast<std::size_t>(key_token)] -
                                  static_cast<double>(MNET_TEST_SDPA_MAX_SCORE)) *
                         static_cast<double>(values.row(key_token)[dimension]);
            }
            expected_values.push_back(static_cast<float>(value / denominator));
        }
    }
    const auto expected = ReferenceTensor(expected_shape, expected_values);
    auto parameters = params<std::int32_t>({{5, 1}});
    expect_forward({"SDPA", {queries, keys, values, mask}, {mnet::test::convert(expected)}, parameters, {}, 1e-4F});
}

}  // namespace
