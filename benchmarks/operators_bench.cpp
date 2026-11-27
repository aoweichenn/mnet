// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <chrono>
#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/concat.hpp>
#include <mlayer/convolution.hpp>
#include <mmemory/allocator.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/layer.hpp>
#include <mnet/quantize.hpp>

namespace {

using Conv = mlayer::Convolution<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin, false,
                                 false, mnet::activation_ss, mnet::quantize_int8, mnet::copy_make_border>;

}  // namespace

TEST(OperatorBenchmarks, Convolution3x3) {
    const std::int32_t w = 56;
    const std::int32_t h = 56;
    const std::int32_t c = 64;
    const std::int32_t outch = 64;

    mformat::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 3);
    pd.set(11, 3);
    pd.set(2, 1);
    pd.set(12, 1);
    pd.set(3, 1);
    pd.set(13, 1);
    pd.set(4, 1);
    pd.set(15, 1);
    pd.set(14, 1);
    pd.set(16, 1);
    pd.set(5, 1);
    pd.set(6, outch * c * 9);

    mnet::Mat weights[2];
    weights[0].create(outch * c * 9);
    for (std::int32_t i = 0; i < outch * c * 9; ++i) {
        static_cast<float*>(weights[0])[static_cast<std::size_t>(i)] = 0.001F * static_cast<float>(i % 97);
    }
    weights[1].create(outch);
    weights[1].fill(0.0F);

    Conv conv;
    ASSERT_EQ(conv.load_param(pd), 0);
    ASSERT_EQ(conv.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat input(w, h, c);
    for (std::int32_t q = 0; q < c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q));
        }
    }

    mnet::Option opt;
    opt.num_threads = 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 5; ++i) {
        mnet::Mat out;
        ASSERT_EQ(conv.forward(input, out, opt), 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "conv3x3=%0.2fms\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0);
}

TEST(OperatorBenchmarks, Concat) {
    using C = mlayer::Concat<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    C concat;
    ASSERT_EQ(concat.load_param(pd), 0);

    mnet::Mat a(80, 80, 64);
    mnet::Mat b(80, 80, 32);
    a.fill(1.0F);
    b.fill(2.0F);
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    opt.num_threads = 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 20; ++i) {
        ASSERT_EQ(concat.forward(bottoms, tops, opt), 0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "concat_reuse=%.2fms\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0);

    mnet::Option pool_opt;
    pool_opt.num_threads = 4;
    mmemory::UnlockedPoolAllocator pool;
    ASSERT_TRUE(pool.set_size_compare_ratio(0.0F));
    pool_opt.blob_allocator = &pool;
    const auto t2 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < 20; ++i) {
        std::vector<mnet::Mat> fresh(1);
        ASSERT_EQ(concat.forward(bottoms, fresh, pool_opt), 0);
    }
    const auto t3 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "concat_fresh_pool=%.2fms\n",
                 std::chrono::duration<double, std::milli>(t3 - t2).count() / 20.0);
}
