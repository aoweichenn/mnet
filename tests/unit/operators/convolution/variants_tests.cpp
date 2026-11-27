// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/convolution1d.hpp>
#include <mlayer/convolutiondepthwise.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/layer.hpp>
#include <mnet/quantize.hpp>

TEST(ConvolutionKernel, Convolution1DMatchesNumpy) {
    constexpr std::int32_t MNET_CONVOLUTION1D_TEST_WIDTH = 4;
    constexpr std::int32_t MNET_CONVOLUTION1D_TEST_CHANNELS = 2;
    constexpr std::int32_t MNET_CONVOLUTION1D_TEST_KERNEL = 3;
    constexpr std::int32_t MNET_CONVOLUTION1D_TEST_PAD = 1;
    constexpr std::int32_t MNET_CONVOLUTION1D_TEST_WEIGHT_COUNT =
        MNET_CONVOLUTION1D_TEST_CHANNELS * MNET_CONVOLUTION1D_TEST_CHANNELS * MNET_CONVOLUTION1D_TEST_KERNEL;
    constexpr float MNET_CONVOLUTION1D_TEST_FIRST_BIAS = 0.5F;
    constexpr float MNET_CONVOLUTION1D_TEST_SECOND_BIAS = -1.0F;

    using Conv = mlayer::Convolution1D<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin,
                                       false, mnet::activation_ss, mnet::copy_make_border, mnet::flatten>;

    mformat::ParamDict pd;
    pd.set(0, MNET_CONVOLUTION1D_TEST_CHANNELS);
    pd.set(1, MNET_CONVOLUTION1D_TEST_KERNEL);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, MNET_CONVOLUTION1D_TEST_PAD);
    pd.set(5, 1);
    pd.set(6, MNET_CONVOLUTION1D_TEST_WEIGHT_COUNT);

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    conv.weight_data.create(MNET_CONVOLUTION1D_TEST_WEIGHT_COUNT);
    for (std::int32_t i = 0; i < MNET_CONVOLUTION1D_TEST_WEIGHT_COUNT; ++i) {
        conv.weight_data[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    conv.bias_data.create(MNET_CONVOLUTION1D_TEST_CHANNELS);
    conv.bias_data[0] = MNET_CONVOLUTION1D_TEST_FIRST_BIAS;
    conv.bias_data[1] = MNET_CONVOLUTION1D_TEST_SECOND_BIAS;

    mnet::Mat input(MNET_CONVOLUTION1D_TEST_WIDTH, MNET_CONVOLUTION1D_TEST_CHANNELS);
    for (std::int32_t channel = 0; channel < MNET_CONVOLUTION1D_TEST_CHANNELS; ++channel) {
        for (std::int32_t position = 0; position < MNET_CONVOLUTION1D_TEST_WIDTH; ++position) {
            input.row(channel)[position] =
                static_cast<float>(channel * (MNET_CONVOLUTION1D_TEST_WIDTH + 1) + position + 1);
        }
    }

    mnet::Option opt;
    mnet::Mat output;
    ASSERT_EQ(conv.forward(input, output, opt), 0);
    ASSERT_EQ(output.w, MNET_CONVOLUTION1D_TEST_WIDTH);
    ASSERT_EQ(output.h, MNET_CONVOLUTION1D_TEST_CHANNELS);

    for (std::int32_t output_channel = 0; output_channel < MNET_CONVOLUTION1D_TEST_CHANNELS; ++output_channel) {
        for (std::int32_t position = 0; position < MNET_CONVOLUTION1D_TEST_WIDTH; ++position) {
            float expected =
                output_channel == 0 ? MNET_CONVOLUTION1D_TEST_FIRST_BIAS : MNET_CONVOLUTION1D_TEST_SECOND_BIAS;
            for (std::int32_t input_channel = 0; input_channel < MNET_CONVOLUTION1D_TEST_CHANNELS; ++input_channel) {
                for (std::int32_t kernel = 0; kernel < MNET_CONVOLUTION1D_TEST_KERNEL; ++kernel) {
                    const auto input_position = position + kernel - MNET_CONVOLUTION1D_TEST_PAD;
                    if (input_position >= 0 && input_position < MNET_CONVOLUTION1D_TEST_WIDTH) {
                        const auto weight_index = (output_channel * MNET_CONVOLUTION1D_TEST_CHANNELS + input_channel) *
                                                      MNET_CONVOLUTION1D_TEST_KERNEL +
                                                  kernel;
                        expected += input.row(input_channel)[input_position] *
                                    conv.weight_data[static_cast<std::size_t>(weight_index)];
                    }
                }
            }
            EXPECT_FLOAT_EQ(output.row(output_channel)[position], expected);
        }
    }
}

TEST(ConvolutionKernel, ConvolutionDepthWiseMatchesReference) {
    using ConvDW = mlayer::ConvolutionDepthWise<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict,
                                                mformat::ModelBin, false, false, mnet::activation_ss,
                                                mnet::quantize_int8, mnet::copy_make_border, mnet::flatten>;

    struct Shape {
        std::int32_t kw;
        std::int32_t stride;
        std::int32_t pad;
        std::int32_t activation;
    };
    const Shape shapes[3] = {{3, 1, 1, 0}, {3, 2, 1, 0}, {3, 1, 1, 1}};

    for (const Shape& shape : shapes) {
        const std::int32_t w = 11;
        const std::int32_t h = 9;
        const std::int32_t channels = 6;
        const std::int32_t maxk = shape.kw * shape.kw;

        mformat::ParamDict pd;
        pd.set(0, channels);
        pd.set(1, shape.kw);
        pd.set(11, shape.kw);
        pd.set(3, shape.stride);
        pd.set(13, shape.stride);
        pd.set(4, shape.pad);
        pd.set(14, shape.pad);
        pd.set(15, shape.pad);
        pd.set(16, shape.pad);
        pd.set(7, channels);
        pd.set(5, 1);
        pd.set(6, channels * maxk);
        pd.set(9, shape.activation);

        mnet::Mat weights[2];
        weights[0].create(channels * maxk);
        weights[1].create(channels);
        std::vector<float> wv(static_cast<std::size_t>(channels * maxk));
        for (std::int32_t i = 0; i < channels * maxk; ++i) {
            wv[static_cast<std::size_t>(i)] = std::sin(0.29F * static_cast<float>(i) + 0.1F) * 0.5F;
            weights[0][static_cast<std::size_t>(i)] = wv[static_cast<std::size_t>(i)];
        }
        for (std::int32_t i = 0; i < channels; ++i) {
            weights[1][static_cast<std::size_t>(i)] = 0.03F * static_cast<float>(i) - 0.05F;
        }

        ConvDW conv;
        ASSERT_EQ(conv.load_param(pd), 0);
        ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

        mnet::Mat input(w, h, channels);
        std::vector<float> in(static_cast<std::size_t>(w * h * channels));
        for (std::int32_t i = 0; i < w * h * channels; ++i) {
            in[static_cast<std::size_t>(i)] = std::sin(0.41F * static_cast<float>(i) - 0.7F);
            input.channel(i / (w * h))[static_cast<std::size_t>(i % (w * h))] = in[static_cast<std::size_t>(i)];
        }

        mnet::Option opt;
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
        ASSERT_EQ(out.w, (w + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.h, (h + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.c, channels);

        for (std::int32_t q = 0; q < channels; ++q) {
            const float* kptr = wv.data() + static_cast<std::size_t>(q * maxk);
            for (std::int32_t y = 0; y < out.h; ++y) {
                for (std::int32_t x = 0; x < out.w; ++x) {
                    float sum = weights[1][static_cast<std::size_t>(q)];
                    for (std::int32_t ky = 0; ky < shape.kw; ++ky) {
                        for (std::int32_t kx = 0; kx < shape.kw; ++kx) {
                            const std::int32_t sy = y * shape.stride + ky - shape.pad;
                            const std::int32_t sx = x * shape.stride + kx - shape.pad;
                            if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                                continue;
                            }
                            sum += kptr[static_cast<std::size_t>(ky * shape.kw + kx)] *
                                   in[static_cast<std::size_t>((q * h + sy) * w + sx)];
                        }
                    }
                    const float expect = shape.activation == 1 ? std::max(sum, 0.0F) : sum;
                    EXPECT_NEAR(out.channel(q)[static_cast<std::size_t>(y * out.w + x)], expect, 1e-5F)
                        << "k=" << shape.kw << " s=" << shape.stride << " q=" << q << " y=" << y << " x=" << x;
                }
            }
        }
    }
}
