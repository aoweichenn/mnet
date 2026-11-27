// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mformat/data_reader.hpp>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/convolution.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/layer.hpp>
#include <mnet/quantize.hpp>

namespace {

using Conv = mlayer::Convolution<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin, false,
                                 false, mnet::activation_ss, mnet::quantize_int8, mnet::copy_make_border>;

}  // namespace

TEST(ConvolutionKernel, Convolution1x1MultiChannel) {
    const std::int32_t w = 4;
    const std::int32_t h = 3;
    const std::int32_t inch = 3;
    const std::int32_t outch = 2;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 1);
    pd.set(11, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch);

    mnet::Mat weights[2];
    weights[0].create(outch * inch);
    weights[0][0] = 1.0F;
    weights[0][1] = 2.0F;
    weights[0][2] = 3.0F;
    weights[0][3] = -1.0F;
    weights[0][4] = -2.0F;
    weights[0][5] = -3.0F;
    weights[1].create(outch);
    weights[1][static_cast<std::size_t>(0)] = 0.5F;
    weights[1][1] = -0.5F;

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = static_cast<float>(i + q);
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);
    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);

    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t i = 0; i < w * h; ++i) {
            float ref = weights[1][static_cast<std::size_t>(p)];
            for (std::int32_t q = 0; q < inch; ++q) {
                ref +=
                    input.channel(q)[static_cast<std::size_t>(i)] * weights[0][static_cast<std::size_t>(p * inch + q)];
            }
            EXPECT_TRUE(std::isfinite(outptr[i])) << p << " " << i;
            EXPECT_NEAR(outptr[i], ref, 1e-5F) << p << " " << i;
        }
    }
}

TEST(ConvolutionKernel, Convolution1x1WithModelBinWeights) {
    const char* dir_env = std::getenv("MNET_SQUEEZENET_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_SQUEEZENET_DIR to run the model-bin test";
    }
    const std::string path = std::string(dir_env) + "/squeezenet_v1.1.bin";
    std::FILE* file = std::fopen(path.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    mformat::DataReaderFromStdio reader(file);
    mformat::ModelBinFromDataReader mb(reader);
    mnet::Mat conv1_w = mb.load(1728, 0);
    mnet::Mat conv1_b = mb.load(64, 1);
    mnet::Mat squeeze_w = mb.load(1024, 0);
    mnet::Mat squeeze_b = mb.load(16, 1);
    std::fclose(file);

    (void)conv1_w;
    (void)conv1_b;

    mformat::ParamDict pd;
    pd.set(0, 16);
    pd.set(1, 1);
    pd.set(11, 1);
    pd.set(5, 1);
    pd.set(6, 1024);
    pd.set(9, 1);

    mnet::Mat weights[2];
    weights[0] = squeeze_w;
    weights[1] = squeeze_b;

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(56, 56, 64);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < 56 * 56; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    for (std::int32_t p = 0; p < out.c; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t i = 0; i < out.w * out.h; ++i) {
            ASSERT_TRUE(std::isfinite(outptr[i])) << p << " " << i;
        }
    }
}

TEST(ConvolutionKernel, Convolution3x3Im2colMatchesReference) {
    const std::int32_t w = 6;
    const std::int32_t h = 5;
    const std::int32_t inch = 2;
    const std::int32_t outch = 3;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.01F * static_cast<float>((i * 7) % 31) - 0.15F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.1F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.1F * static_cast<float>(i + q * 5));
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);
    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                const auto actual = outptr[oy * w + ox];
                ASSERT_TRUE(std::isfinite(actual)) << p << " " << oy << " " << ox;
                max_err = std::fmax(max_err, std::fabs(sum - actual));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}

TEST(ConvolutionKernel, Convolution3x3WithFullOutputBlockMatchesReference) {
    const std::int32_t w = 7;
    const std::int32_t h = 6;
    const std::int32_t inch = 4;
    const std::int32_t outch = 8;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.01F * static_cast<float>((i * 7) % 31) - 0.15F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.1F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.1F * static_cast<float>(i + q * 5));
        }
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);
    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        const float* outptr = out.channel(p);
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                const auto actual = outptr[oy * w + ox];
                ASSERT_TRUE(std::isfinite(actual)) << p << " " << oy << " " << ox;
                max_err = std::fmax(max_err, std::fabs(sum - actual));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}

TEST(ConvolutionKernel, Convolution3x3SmallInputChannelsMatchesReference) {
    // Exercises the im2col path for K = inch * 9 = 27 (below the old >= 64 gate).
    const std::int32_t w = 9;
    const std::int32_t h = 8;
    const std::int32_t inch = 3;
    const std::int32_t outch = 8;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.02F * static_cast<float>((i * 5) % 23) - 0.2F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        for (std::int32_t i = 0; i < w * h; ++i) {
            input.channel(q)[static_cast<std::size_t>(i)] = std::cos(0.08F * static_cast<float>(i + q * 7));
        }
    }

    mnet::Option opt;
    opt.num_threads = 3;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);
    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                const auto actual = out.channel(p)[static_cast<std::size_t>(oy * w + ox)];
                ASSERT_TRUE(std::isfinite(actual)) << p << " " << oy << " " << ox;
                max_err = std::fmax(max_err, std::fabs(sum - actual));
            }
        }
    }
    EXPECT_LT(max_err, 1e-5F);
}

TEST(ConvolutionKernel, Convolution3x3OnLargeFeatureMapMatchesReference) {
    const std::int32_t w = 64;
    const std::int32_t h = 64;
    const std::int32_t inch = 32;
    const std::int32_t outch = 8;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * inch * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * inch * 9);
    for (std::int32_t i = 0; i < outch * inch * 9; ++i) {
        weights[0][static_cast<std::size_t>(i)] = 0.01F * static_cast<float>((i * 7) % 31) - 0.15F;
    }
    weights[1].create(outch);
    for (std::int32_t i = 0; i < outch; ++i) {
        weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i);
    }

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, inch);
    for (std::int32_t q = 0; q < inch; ++q) {
        for (std::int32_t i = 0; i < w * h; ++i) {
            input.channel(q)[static_cast<std::size_t>(i)] = std::sin(0.05F * static_cast<float>(i + q * 31));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    mnet::Mat out;
    ASSERT_EQ(conv.forward(input, out, opt), 0);

    ASSERT_EQ(out.w, w);
    ASSERT_EQ(out.h, h);
    ASSERT_EQ(out.c, outch);
    float max_err = 0.0F;
    for (std::int32_t p = 0; p < outch; ++p) {
        for (std::int32_t oy = 0; oy < h; ++oy) {
            for (std::int32_t ox = 0; ox < w; ++ox) {
                float sum = weights[1][static_cast<std::size_t>(p)];
                for (std::int32_t q = 0; q < inch; ++q) {
                    for (std::int32_t ky = 0; ky < 3; ++ky) {
                        const std::int32_t iy = oy + ky - 1;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (std::int32_t kx = 0; kx < 3; ++kx) {
                            const std::int32_t ix = ox + kx - 1;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            sum += input.channel(q)[static_cast<std::size_t>(iy * w + ix)] *
                                   weights[0][static_cast<std::size_t>((p * inch + q) * 9 + ky * 3 + kx)];
                        }
                    }
                }
                const auto actual = out.channel(p)[static_cast<std::size_t>(oy * w + ox)];
                ASSERT_TRUE(std::isfinite(actual)) << p << " " << oy << " " << ox;
                max_err = std::fmax(max_err, std::fabs(sum - actual));
            }
        }
    }
    EXPECT_LT(max_err, 1e-4F);
}

TEST(ConvolutionKernel, ConvolutionGemmPackOddShapes) {
    // Exercises the pack8 GEMM tail paths: pixel count not a multiple of the
    // 32-pixel tile and output channels not a multiple of the 4-channel block.
    struct Shape {
        std::int32_t kw;
        std::int32_t stride;
        std::int32_t pad;
    };
    const Shape shapes[3] = {{1, 1, 0}, {3, 1, 1}, {3, 2, 1}};

    for (const Shape& shape : shapes) {
        const std::int32_t w = 7;
        const std::int32_t h = 5;
        const std::int32_t inch = 5;
        const std::int32_t outch = 6;
        const std::int32_t maxk = shape.kw * shape.kw;

        mformat::ParamDict pd;
        pd.set(0, outch);
        pd.set(1, shape.kw);
        pd.set(11, shape.kw);
        pd.set(3, shape.stride);
        pd.set(13, shape.stride);
        pd.set(4, shape.pad);
        pd.set(14, shape.pad);
        pd.set(15, shape.pad);
        pd.set(16, shape.pad);
        pd.set(5, 1);
        pd.set(6, outch * inch * maxk);
        pd.set(9, 1);  // ReLU

        mnet::Mat weights[2];
        weights[0].create(outch * inch * maxk);
        weights[1].create(outch);
        std::vector<float> wv(static_cast<std::size_t>(outch * inch * maxk));
        for (std::int32_t i = 0; i < outch * inch * maxk; ++i) {
            wv[static_cast<std::size_t>(i)] = std::sin(0.37F * static_cast<float>(i)) * 0.4F;
            weights[0][static_cast<std::size_t>(i)] = wv[static_cast<std::size_t>(i)];
        }
        for (std::int32_t i = 0; i < outch; ++i) {
            weights[1][static_cast<std::size_t>(i)] = 0.05F * static_cast<float>(i) - 0.1F;
        }

        Conv conv;
        ASSERT_EQ(conv.load_param(pd), 0);
        ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

        mnet::Mat input(w, h, inch);
        std::vector<float> in(static_cast<std::size_t>(w * h * inch));
        for (std::int32_t i = 0; i < w * h * inch; ++i) {
            in[static_cast<std::size_t>(i)] = std::sin(0.53F * static_cast<float>(i) + 0.2F);
            input.channel(i / (w * h))[static_cast<std::size_t>(i % (w * h))] = in[static_cast<std::size_t>(i)];
        }

        mnet::Option opt;
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
        ASSERT_EQ(out.w, (w + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.h, (h + 2 * shape.pad - shape.kw) / shape.stride + 1);
        ASSERT_EQ(out.c, outch);

        for (std::int32_t p = 0; p < outch; ++p) {
            const float* wp = wv.data() + static_cast<std::size_t>(p * inch * maxk);
            for (std::int32_t y = 0; y < out.h; ++y) {
                for (std::int32_t x = 0; x < out.w; ++x) {
                    float sum = weights[1][static_cast<std::size_t>(p)];
                    for (std::int32_t q = 0; q < inch; ++q) {
                        for (std::int32_t ky = 0; ky < shape.kw; ++ky) {
                            for (std::int32_t kx = 0; kx < shape.kw; ++kx) {
                                const std::int32_t sy = y * shape.stride + ky - shape.pad;
                                const std::int32_t sx = x * shape.stride + kx - shape.pad;
                                if (sy < 0 || sy >= h || sx < 0 || sx >= w) {
                                    continue;
                                }
                                sum += wp[static_cast<std::size_t>((q * maxk) + ky * shape.kw + kx)] *
                                       in[static_cast<std::size_t>((q * h + sy) * w + sx)];
                            }
                        }
                    }
                    const float expect = sum > 0.0F ? sum : 0.0F;
                    EXPECT_NEAR(out.channel(p)[static_cast<std::size_t>(y * out.w + x)], expect, 1e-5F)
                        << "k=" << shape.kw << " s=" << shape.stride << " p=" << p << " y=" << y << " x=" << x;
                }
            }
        }
    }
}
