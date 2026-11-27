// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/param_dict.hpp>
#include <mlayer/stft.hpp>
#include <mnet/layer.hpp>

// One-sided STFT: framing + window + DFT, output (bins, frames, 2).
TEST(SignalKernel, StftMatchesReference) {
    using S = mlayer::Stft<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    S stft;
    ASSERT_EQ(stft.load_param(mformat::ParamDict()), 0);

    // 12 samples, frame 4, step 2 -> 5 frames.
    const float samples[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    mnet::Mat signal(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        signal[static_cast<std::size_t>(i)] = samples[i];
    }
    mnet::Mat window(4);
    window.fill(1.0F);
    mnet::Mat step(1);
    step[0] = 2.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {signal, window, step};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(stft.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].dims, 4);
    ASSERT_EQ(top[0].w, 2);  // real/imag
    ASSERT_EQ(top[0].h, 3);  // bins = 4/2 + 1
    ASSERT_EQ(top[0].d, 1);
    ASSERT_EQ(top[0].c, 5);  // frames

    for (std::int32_t f = 0; f < 5; ++f) {
        for (std::int32_t k = 0; k < 3; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (std::int32_t n = 0; n < 4; ++n) {
                const double angle =
                    2.0 * 3.14159265358979323846 * static_cast<double>(k) * static_cast<double>(n) / 4.0;
                const double v = static_cast<double>(samples[f * 2 + n]);
                re += v * std::cos(angle);
                im -= v * std::sin(angle);
            }
            EXPECT_NEAR(top[0].channel(f).row(k)[0], re, 1e-5F);
            EXPECT_NEAR(top[0].channel(f).row(k)[1], im, 1e-5F);
        }
    }

    // DC bin is the frame sum, Nyquist alternates sign: [1,0,0,0] -> [1,1,1].
    EXPECT_NEAR(top[0].channel(0).row(0)[0], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(0).row(1)[0], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(0).row(2)[0], 1.0F, 1e-5F);
    // frame 1 is samples[2..5] = [0,0,0,1] -> X = [1, +i, -1]
    EXPECT_NEAR(top[0].channel(1).row(1)[0], 0.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(1).row(1)[1], 1.0F, 1e-5F);
    EXPECT_NEAR(top[0].channel(1).row(2)[0], -1.0F, 1e-5F);
}

TEST(SignalKernel, StftBluesteinMatchesDirectDft) {
    using S = mlayer::Stft<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    constexpr std::int32_t MNET_STFT_BLUESTEIN_FRAME = 20;
    S stft;
    ASSERT_EQ(stft.load_param(mformat::ParamDict()), 0);

    mnet::Mat signal(MNET_STFT_BLUESTEIN_FRAME);
    for (std::int32_t i = 0; i < MNET_STFT_BLUESTEIN_FRAME; ++i) {
        signal[static_cast<std::size_t>(i)] = static_cast<float>((i * 7) % 11) - 5.0F;
    }
    mnet::Mat window(MNET_STFT_BLUESTEIN_FRAME);
    window.fill(1.0F);
    mnet::Mat step(1);
    step[0] = static_cast<float>(MNET_STFT_BLUESTEIN_FRAME);

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {signal, window, step};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(stft.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].c, 1);

    for (std::int32_t k = 0; k < MNET_STFT_BLUESTEIN_FRAME / 2 + 1; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (std::int32_t n = 0; n < MNET_STFT_BLUESTEIN_FRAME; ++n) {
            const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(k) * static_cast<double>(n) /
                                 static_cast<double>(MNET_STFT_BLUESTEIN_FRAME);
            const auto value = static_cast<double>(signal[static_cast<std::size_t>(n)]);
            re += value * std::cos(angle);
            im -= value * std::sin(angle);
        }
        EXPECT_NEAR(top[0].channel(0).row(k)[0], re, 1e-4F);
        EXPECT_NEAR(top[0].channel(0).row(k)[1], im, 1e-4F);
    }
}
