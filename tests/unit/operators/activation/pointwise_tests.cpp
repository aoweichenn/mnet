// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <array>

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

struct ActivationCase {
    const char* type;
    std::array<float, 5> expected;
    mformat::ParamDict parameters{};
};

class Activation : public testing::TestWithParam<ActivationCase> {};

std::string activation_name(const testing::TestParamInfo<ActivationCase>& info) {
    return info.param.type;
}

// assertions-in-helper: expect_forward
TEST_P(Activation, MatchesReferenceAcrossNegativeZeroAndPositiveInputs) {
    const auto& [type, y, parameters] = GetParam();
    const auto input = tensor({3, 5, 1, 1, 2}, {-4, -1, 0, 1, 4, 4, 1, 0, -1, -4});
    const auto expected = tensor({3, 5, 1, 1, 2}, {y[0], y[1], y[2], y[3], y[4], y[4], y[3], y[2], y[1], y[0]});
    expect_forward({type, {input}, {expected}, parameters});
    mnet::test::expect_tensor(input, tensor({3, 5, 1, 1, 2}, {-4, -1, 0, 1, 4, 4, 1, 0, -1, -4}), 0);
}

// Expected values were evaluated from the scalar equations at x = [-4, -1, 0, 1, 4].
INSTANTIATE_TEST_SUITE_P(
    KnownAnswers, Activation,
    testing::Values(ActivationCase{"AbsVal", {4, 1, 0, 1, 4}},
                    ActivationCase{"BNLL", {0.01814993F, 0.31326169F, 0.69314718F, 1.31326169F, 4.01814993F}},
                    ActivationCase{"CELU", {-1.72932943F, -0.78693868F, 0, 1, 4}, params<float>({{0, 2}})},
                    ActivationCase{"Clip", {-1, -1, 0, 1, 2}, params<float>({{0, -1}, {1, 2}})},
                    ActivationCase{"Dropout", {-2, -0.5F, 0, 0.5F, 2}, params<float>({{0, 0.5F}})},
                    ActivationCase{"ELU", {-0.49084218F, -0.31606028F, 0, 1, 4}, params<float>({{0, 0.5F}})},
                    ActivationCase{"Erf", {-0.99999998F, -0.84270079F, 0, 0.84270079F, 0.99999998F}},
                    ActivationCase{"Exp", {0.25F, 0.70710678F, 1, 1.41421356F, 4}, params<float>({{0, 2}, {1, 0.5F}})},
                    ActivationCase{"GELU", {-0.00012668F, -0.15865525F, 0, 0.84134475F, 3.99987332F}},
                    ActivationCase{"HardSigmoid", {0, 0.3F, 0.5F, 0.7F, 1}},
                    ActivationCase{"HardSwish", {0, -0.3F, 0, 0.7F, 4}},
                    ActivationCase{"Mish", {-0.07259174F, -0.30340146F, 0, 0.86509839F, 3.99741281F}},
                    ActivationCase{"Power", {49, 1, 1, 9, 81}, params<float>({{0, 2}, {1, 2}, {2, 1}})},
                    ActivationCase{"ReLU", {-1, -0.25F, 0, 1, 4}, params<float>({{0, 0.25F}})},
                    ActivationCase{"SELU", {-0.98168436F, -0.63212056F, 0, 2, 8}, params<float>({{0, 0.5F}, {1, 2}})},
                    ActivationCase{"Shrink", {-3.5F, 0, 0, 0, 3.5F}, params<float>({{0, 0.5F}, {1, 1}})},
                    ActivationCase{"Sigmoid", {0.01798621F, 0.26894142F, 0.5F, 0.73105858F, 0.98201379F}},
                    ActivationCase{"Softplus", {0.01814993F, 0.31326169F, 0.69314718F, 1.31326169F, 4.01814993F}},
                    ActivationCase{"Swish", {-0.07194484F, -0.26894142F, 0, 0.73105858F, 3.92805516F}},
                    ActivationCase{"TanH", {-0.99932930F, -0.76159416F, 0, 0.76159416F, 0.99932930F}},
                    ActivationCase{"Threshold", {0, 0, 0, 0, 1}, params<float>({{0, 1}})}),
    activation_name);

// assertions-in-helper: expect_forward
TEST(Log, AppliesBaseScaleAndShift) {
    // log2(1 + 2*x) where the argument runs through successive powers of two.
    expect_forward(
        {"Log", {vector({0, 0.5F, 1.5F, 3.5F})}, {vector({0, 1, 2, 3})}, params<float>({{0, 2}, {1, 2}, {2, 1}})});
}

// assertions-in-helper: expect_forward
TEST(Softmax, NormalizesRowsWithoutOverflow) {
    const auto input = tensor({2, 3, 2}, {1000, 1001, 1002, -1000, -999, -998});
    const auto expected =
        tensor({2, 3, 2}, {0.09003057F, 0.24472847F, 0.66524096F, 0.09003057F, 0.24472847F, 0.66524096F});
    expect_forward({"Softmax", {input}, {expected}, params<std::int32_t>({{0, 1}, {1, 1}})});
}

}  // namespace
