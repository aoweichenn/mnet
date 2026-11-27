// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <numeric>

#include <mnet/net.hpp>
#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::expect_tensor;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::values_of;
using mnet::test::vector;

TEST(Input, ExposesNamedGraphInputAndPreservesEveryValue) {
    constexpr char MNET_TEST_INPUT_GRAPH[] = "7767517\n1 1\nInput source 0 1 samples\n";
    auto net = mnet::Net{};
    ASSERT_EQ(net.load_param_mem(MNET_TEST_INPUT_GRAPH), 0);
    const auto expected_inputs = std::vector<std::string>{"samples"};
    ASSERT_EQ(net.input_names(), expected_inputs);
    auto extractor = net.create_extractor();
    const auto samples = tensor({2, 3, 2}, {-1, 0, 1, 4, 2, 7});
    EXPECT_NE(extractor.input("missing", samples), 0);
    ASSERT_EQ(extractor.input("samples", samples), 0);
    auto output = mnet::Mat{};
    ASSERT_EQ(extractor.extract("samples", output), 0);
    expect_tensor(output, samples, 0);
}

// assertions-in-helper: expect_forward
TEST(MemoryData, ReturnsConstantTensorWithoutAnyInputs) {
    expect_forward({"MemoryData",
                    {},
                    {tensor({2, 2, 3}, {1, -2, 3, 4, 0, 6})},
                    params<std::int32_t>({{0, 2}, {1, 3}}),
                    {vector({1, -2, 3, 4, 0, 6})}});
}

TEST(RandomNormalLike, UsesSeedAndShapeWithoutDependingOnInputValues) {
    constexpr std::int32_t MNET_TEST_RANDOM_SAMPLES = 256;
    constexpr float MNET_TEST_RANDOM_MEAN = 8.0F;
    constexpr float MNET_TEST_RANDOM_SCALE = 2.0F;
    constexpr std::int32_t MNET_TEST_RANDOM_SEED = 42;
    auto first_input = mnet::Mat(MNET_TEST_RANDOM_SAMPLES);
    auto second_input = mnet::Mat(MNET_TEST_RANDOM_SAMPLES);
    first_input.fill(0.0F);
    second_input.fill(1.0F);
    auto layer = std::unique_ptr<mnet::Layer>(mnet::create_layer("RandomNormalLike"));
    ASSERT_NE(layer, nullptr);
    auto parameters = params<float>({{0, MNET_TEST_RANDOM_MEAN}, {1, MNET_TEST_RANDOM_SCALE}});
    parameters.set(2, MNET_TEST_RANDOM_SEED);
    ASSERT_EQ(layer->load_param(parameters), 0);
    auto first = mnet::Mat{};
    auto repeated = mnet::Mat{};
    auto changed_seed = mnet::Mat{};
    const auto opt = mnet::Option{};
    ASSERT_EQ(layer->forward(first_input, first, opt), 0);
    ASSERT_EQ(layer->forward(second_input, repeated, opt), 0);
    ASSERT_EQ(mnet::test::shape_of(first), mnet::test::shape_of(first_input));
    ASSERT_EQ(first.elemsize, sizeof(float));
    ASSERT_EQ(first.elempack, 1);
    expect_tensor(repeated, first, 0);
    parameters.set(2, MNET_TEST_RANDOM_SEED + 1);
    ASSERT_EQ(layer->load_param(parameters), 0);
    ASSERT_EQ(layer->forward(first_input, changed_seed, opt), 0);
    ASSERT_FALSE(changed_seed.empty());
    ASSERT_EQ(changed_seed.elemsize, sizeof(float));
    ASSERT_EQ(changed_seed.elempack, 1);
    const auto samples = values_of(first);
    EXPECT_NE(values_of(changed_seed), samples);
    const auto count = static_cast<float>(samples.size());
    const auto mean = std::accumulate(samples.begin(), samples.end(), 0.0F) / count;
    const auto variance = std::accumulate(samples.begin(), samples.end(), 0.0F,
                                          [mean](float sum, float sample) {
                                              const auto delta = sample - mean;
                                              return sum + delta * delta;
                                          }) /
                          count;
    // The C++ library controls the exact sequence, so compare moments with ample margin.
    EXPECT_NEAR(mean, MNET_TEST_RANDOM_MEAN, 0.75F);
    EXPECT_NEAR(std::sqrt(variance), MNET_TEST_RANDOM_SCALE, 0.75F);
}

}  // namespace
