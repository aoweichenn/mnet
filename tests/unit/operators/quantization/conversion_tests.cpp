// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::indices;
using mnet::test::params;
using mnet::test::shape_of;
using mnet::test::vector;

void expect_int8(const mnet::Mat& output, std::initializer_list<std::int8_t> expected) {
    ASSERT_FALSE(output.empty());
    ASSERT_EQ(output.elempack, 1);
    ASSERT_EQ(output.elemsize, sizeof(std::int8_t));
    const auto shape = mnet::test::TensorShape{1, static_cast<std::int32_t>(expected.size())};
    ASSERT_EQ(shape_of(output), shape);
    const auto* values = static_cast<const std::int8_t*>(output.data);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(values[i], expected.begin()[i]) << "Index: " << i;
    }
}

TEST(Cast, RoundsTiesAwayFromZeroAndClampsToInt8Range) {
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("Cast"));
    ASSERT_NE(layer, nullptr);
    constexpr std::int32_t MNET_TEST_CAST_TYPE_INT8 = 3;
    ASSERT_EQ(layer->load_param(params<std::int32_t>({{1, MNET_TEST_CAST_TYPE_INT8}})), 0);
    auto output = mnet::Mat{};
    ASSERT_EQ(layer->forward(vector({-200, -1.5F, -0.49F, 0.5F, 1.5F, 200}), output, mnet::Option{}), 0);
    expect_int8(output, {-128, -2, 0, 1, 2, 127});
}

TEST(Quantize, ScalesThenRoundsAndSaturatesSymmetrically) {
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("Quantize"));
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->load_param(mformat::ParamDict{}), 0);
    const auto scale = vector({2});
    ASSERT_EQ(layer->load_model(mformat::ModelBinFromMatArray(&scale)), 0);
    auto output = mnet::Mat{};
    ASSERT_EQ(layer->forward(vector({-100, -0.75F, -0.24F, 0.25F, 1.25F, 100}), output, mnet::Option{}), 0);
    expect_int8(output, {-127, -2, 0, 1, 3, 127});
}

// assertions-in-helper: expect_forward
TEST(Dequantize, AppliesScaleThenBiasToSignedAccumulators) {
    expect_forward({"Dequantize",
                    {indices({-10, 0, 4, 200})},
                    {vector({-1.5F, 1, 2, 51})},
                    params<std::int32_t>({{1, 1}}),
                    {vector({0.25F}), vector({1})}});
}

TEST(Requantize, AppliesBiasAndActivationBeforeOutputScaleAndSaturation) {
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("Requantize"));
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->load_param(params<std::int32_t>({{2, 1}, {3, 1}})), 0);
    const auto weights = std::array{vector({0.5F}), vector({3}), vector({1})};
    ASSERT_EQ(layer->load_model(mformat::ModelBinFromMatArray(weights.data())), 0);
    auto output = mnet::Mat{};
    ASSERT_EQ(layer->forward(indices({-8, -1, 0, 3, 200}), output, mnet::Option{}), 0);
    expect_int8(output, {0, 2, 3, 8, 127});
}

}  // namespace
