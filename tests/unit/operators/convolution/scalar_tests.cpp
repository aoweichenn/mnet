// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>

#include <array>
#include <initializer_list>
#include <vector>

#include <operator_test.hpp>

namespace {

using mnet::test::ReferenceTensor;

inline std::vector<float> convolution_reference(const ReferenceTensor& input, const ReferenceTensor& weights,
                                                const ReferenceTensor& bias, std::int32_t stride) {
    const auto input_channels = static_cast<std::int32_t>(input.shape[0]);
    const auto input_height = static_cast<std::int32_t>(input.shape[1]);
    const auto input_width = static_cast<std::int32_t>(input.shape[2]);
    const auto output_channels = static_cast<std::int32_t>(weights.shape[0]);
    constexpr std::int32_t MNET_TEST_KERNEL_SIZE = 3;
    constexpr std::int32_t MNET_TEST_KERNEL_HALF = 1;
    const auto output_height = (input_height + stride - 1) / stride;
    const auto output_width = (input_width + stride - 1) / stride;
    auto output = std::vector<float>(static_cast<std::size_t>(output_channels * output_height * output_width), 0.0F);
    for (std::int32_t channel = 0; channel < output_channels; ++channel) {
        for (std::int32_t y = 0; y < output_height; ++y) {
            for (std::int32_t x = 0; x < output_width; ++x) {
                auto sum = bias.at(static_cast<std::size_t>(channel));
                for (std::int32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
                    for (std::int32_t kernel_y = 0; kernel_y < MNET_TEST_KERNEL_SIZE; ++kernel_y) {
                        const auto source_y = y * stride + kernel_y - MNET_TEST_KERNEL_HALF;
                        if (source_y < 0 || source_y >= input_height) {
                            continue;
                        }
                        for (std::int32_t kernel_x = 0; kernel_x < MNET_TEST_KERNEL_SIZE; ++kernel_x) {
                            const auto source_x = x * stride + kernel_x - MNET_TEST_KERNEL_HALF;
                            if (source_x < 0 || source_x >= input_width) {
                                continue;
                            }
                            const auto step =
                                static_cast<std::size_t>(input_channel) * static_cast<std::size_t>(input_height) *
                                    static_cast<std::size_t>(input_width) +
                                static_cast<std::size_t>(source_y) * static_cast<std::size_t>(input_width) +
                                static_cast<std::size_t>(source_x);
                            const auto weight_index = static_cast<std::size_t>(channel) *
                                                          static_cast<std::size_t>(input_channels) *
                                                          MNET_TEST_KERNEL_SIZE * MNET_TEST_KERNEL_SIZE +
                                                      static_cast<std::size_t>(input_channel) * MNET_TEST_KERNEL_SIZE *
                                                          MNET_TEST_KERNEL_SIZE +
                                                      static_cast<std::size_t>(kernel_y) * MNET_TEST_KERNEL_SIZE +
                                                      static_cast<std::size_t>(kernel_x);
                            sum += input.at(step) * weights.at(weight_index);
                        }
                    }
                }
                const auto output_offset =
                    static_cast<std::size_t>(channel * output_height + y) * static_cast<std::size_t>(output_width) +
                    static_cast<std::size_t>(x);
                output[output_offset] = sum;
            }
        }
    }
    return output;
}

// assertions-in-helper: expect_forward
TEST(Convolution, MatchesNhwcReferenceAcrossOutputBlockAndTailSizes) {
    constexpr std::int32_t MNET_TEST_CONV_HEIGHT = 5;
    constexpr std::int32_t MNET_TEST_CONV_WIDTH = 7;
    constexpr std::int32_t MNET_TEST_CONV_INPUT_CHANNELS = 2;
    auto ordinal = std::size_t{0};
    for (const auto output_channels : {7, 8, 9, 16}) {
        for (const auto stride : {1, 2}) {
            SCOPED_TRACE(output_channels);
            SCOPED_TRACE(stride);
            const auto input_shape =
                std::vector<std::size_t>{MNET_TEST_CONV_INPUT_CHANNELS, MNET_TEST_CONV_HEIGHT, MNET_TEST_CONV_WIDTH};
            const auto weight_shape =
                std::vector<std::size_t>{static_cast<std::size_t>(output_channels), MNET_TEST_CONV_INPUT_CHANNELS, 9};
            const auto bias_shape = std::vector<std::size_t>{static_cast<std::size_t>(output_channels)};
            const auto input_count = ReferenceTensor::full(input_shape, 0.0F).element_count();
            const auto weight_count = ReferenceTensor::full(weight_shape, 0.0F).element_count();
            auto input_values = std::vector<float>{};
            auto weight_values = std::vector<float>{};
            auto bias_values = std::vector<float>{};
            for (std::size_t i = 0; i < input_count; ++i) {
                input_values.push_back(static_cast<float>(static_cast<std::int32_t>((i * 3 + ordinal) % 11) - 5) / 8);
            }
            for (std::size_t i = 0; i < weight_count; ++i) {
                weight_values.push_back(static_cast<float>(static_cast<std::int32_t>((i * 5 + ordinal) % 13) - 6) / 16);
            }
            for (std::size_t i = 0; i < static_cast<std::size_t>(output_channels); ++i) {
                bias_values.push_back(static_cast<float>(static_cast<std::int32_t>(i % 3) - 1) / 4);
            }
            const auto input = ReferenceTensor(input_shape, input_values);
            const auto weights = ReferenceTensor(weight_shape, weight_values);
            const auto bias = ReferenceTensor(bias_shape, bias_values);
            const auto expected_values = convolution_reference(input, weights, bias, stride);
            const auto output_height = (MNET_TEST_CONV_HEIGHT + stride - 1) / stride;
            const auto output_width = (MNET_TEST_CONV_WIDTH + stride - 1) / stride;
            const auto expected =
                ReferenceTensor({static_cast<std::size_t>(output_channels), static_cast<std::size_t>(output_height),
                                 static_cast<std::size_t>(output_width)},
                                expected_values);
            mformat::ParamDict parameters;
            parameters.set(0, output_channels);
            parameters.set(1, 3);
            parameters.set(11, 3);
            parameters.set(3, stride);
            parameters.set(13, stride);
            parameters.set(4, 1);
            parameters.set(15, 1);
            parameters.set(14, 1);
            parameters.set(16, 1);
            parameters.set(5, 1);
            parameters.set(6, static_cast<std::int32_t>(weight_values.size()));
            mnet::test::expect_forward({"Convolution",
                                        {mnet::test::convert(input)},
                                        {mnet::test::convert(expected)},
                                        parameters,
                                        {mnet::test::vector(weight_values), mnet::test::vector(bias_values)},
                                        1e-4F});
            ++ordinal;
        }
    }
}

}  // namespace
