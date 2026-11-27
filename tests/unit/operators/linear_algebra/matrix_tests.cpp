// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstring>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/einsum.hpp>
#include <mlayer/gemm.hpp>
#include <mlayer/interp.hpp>
#include <mlayer/matmul.hpp>
#include <mnet/layer.hpp>

TEST(LinearAlgebraKernel, MatMul2DAndInterpNearest) {
    using MM = mlayer::MatMul<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;

    mformat::ParamDict pd;
    pd.set(0, 0);
    MM mm;
    ASSERT_EQ(mm.load_param(pd), 0);

    mnet::Mat a(3, 2);
    mnet::Mat b(4, 3);
    for (std::int32_t i = 0; i < 6; ++i) {
        static_cast<float*>(a)[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    for (std::int32_t i = 0; i < 12; ++i) {
        static_cast<float*>(b)[i] = static_cast<float>(i);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(mm.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 4);
    ASSERT_EQ(tops[0].h, 2);
    for (std::int32_t i = 0; i < 2; ++i) {
        for (std::int32_t j = 0; j < 4; ++j) {
            float ref = 0.0F;
            for (std::int32_t k = 0; k < 3; ++k) {
                ref += a.row(i)[k] * b.row(k)[j];
            }
            EXPECT_FLOAT_EQ(tops[0].row(i)[j], ref);
        }
    }

    using IP = mlayer::Interp<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd2;
    pd2.set(0, 1);
    pd2.set(1, 2.0F);
    pd2.set(2, 2.0F);
    IP ip;
    ASSERT_EQ(ip.load_param(pd2), 0);
    mnet::Mat small(2, 2, 1);
    small.channel(0)[0] = 1.0F;
    small.channel(0)[1] = 2.0F;
    small.channel(0)[2] = 3.0F;
    small.channel(0)[3] = 4.0F;
    mnet::Mat up;
    ASSERT_EQ(ip.forward(small, up, opt), 0);
    ASSERT_EQ(up.w, 4);
    ASSERT_EQ(up.h, 4);
    EXPECT_FLOAT_EQ(up.channel(0)[0], 1.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[3], 2.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[12], 3.0F);
    EXPECT_FLOAT_EQ(up.channel(0)[15], 4.0F);
}

TEST(LinearAlgebraKernel, GemmConstantBAndC) {
    using G = mlayer::Gemm<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;

    mformat::ParamDict pd;
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 2);
    pd.set(8, 4);
    pd.set(9, 3);
    pd.set(10, 0);
    G gemm;
    ASSERT_EQ(gemm.load_param(pd), 0);

    mnet::Mat weights[2];
    weights[0].create(3, 4);
    for (std::int32_t i = 0; i < 12; ++i) {
        static_cast<float*>(weights[0])[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    weights[1].create(1);
    weights[1][static_cast<std::size_t>(0)] = 0.5F;
    ASSERT_EQ(gemm.load_model(mformat::ModelBinFromMatArray(weights)), 0);

    mnet::Mat a(3, 2);
    for (std::int32_t i = 0; i < 6; ++i) {
        static_cast<float*>(a)[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }

    mnet::Option opt;
    opt.num_threads = 2;
    mnet::Mat out;
    ASSERT_EQ(gemm.forward(a, out, opt), 0);
    ASSERT_EQ(out.w, 4);
    ASSERT_EQ(out.h, 2);
    for (std::int32_t i = 0; i < 2; ++i) {
        for (std::int32_t j = 0; j < 4; ++j) {
            float ref = 0.5F;
            for (std::int32_t k = 0; k < 3; ++k) {
                ref += a.row(i)[k] * weights[0].row(j)[k];
            }
            EXPECT_NEAR(out.row(i)[j], ref, 1e-5F);
        }
    }
}

TEST(LinearAlgebraKernel, EinsumMatMulMatchesReference) {
    using ES = mlayer::Einsum<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    // "ik,kj->ij"
    const char* equation = "ik,kj->ij";
    mnet::Mat eq(static_cast<std::int32_t>(std::strlen(equation)));
    for (std::int32_t i = 0; i < eq.w; ++i) {
        static_cast<std::int32_t*>(static_cast<void*>(eq.data))[i] = equation[i];
    }
    mformat::ParamDict pd;
    pd.set(0, eq);
    ES op;
    ASSERT_EQ(op.load_param(pd), 0);

    mnet::Mat a(2, 3);
    mnet::Mat b(3, 2);
    for (std::int32_t i = 0; i < 6; ++i) {
        a[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        b[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    std::vector<mnet::Mat> bottoms = {a, b};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(op.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 3);
    ASSERT_EQ(tops[0].h, 3);
    // a = [[1,2],[3,4],[5,6]], b = [[1,2,3],[4,5,6]]
    const float expect[9] = {9.0F, 12.0F, 15.0F, 19.0F, 26.0F, 33.0F, 29.0F, 40.0F, 51.0F};
    for (std::int32_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(tops[0][static_cast<std::size_t>(i)], expect[i]);
    }
}

TEST(LinearAlgebraKernel, GemmFastInt4MatchesInt8Reference) {
    constexpr std::int32_t N = 4;
    constexpr std::int32_t K = 128;
    constexpr std::int32_t block = 64;
    constexpr std::int32_t M = 3;

    // 4-bit two's-complement weights in ncnn's little-endian packing.
    std::vector<std::int8_t> q(static_cast<std::size_t>(N) * K);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t k = 0; k < K; ++k) {
            q[static_cast<std::size_t>(j * K + k)] = static_cast<std::int8_t>(((j * 7 + k * 3) % 15) - 7);
        }
    }
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(N) * (K / 2), 0);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t k = 0; k < K; ++k) {
            const auto value = static_cast<std::uint32_t>(q[static_cast<std::size_t>(j * K + k)]) & 0x0Fu;
            const std::size_t byte = static_cast<std::size_t>(j) * (K / 2) + static_cast<std::size_t>(k / 2);
            packed[byte] |= static_cast<std::uint8_t>(value << ((k % 2) * 4));
        }
    }

    mnet::Mat w4;
    w4.create(static_cast<std::int32_t>(packed.size()), static_cast<std::size_t>(1u));
    std::memcpy(w4.data, packed.data(), packed.size());
    mnet::Mat w8;
    w8.create(N * K, static_cast<std::size_t>(1u));
    std::memcpy(w8.data, q.data(), q.size());

    mnet::Mat scales;
    scales.create(K / block, N, static_cast<std::size_t>(4u), nullptr);
    for (std::int32_t j = 0; j < N; ++j) {
        for (std::int32_t b = 0; b < K / block; ++b) {
            scales.row(j)[b] = 9.0F + static_cast<float>(j) + static_cast<float>(b);
        }
    }

    using G = mlayer::Gemm<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    auto make_params = [](std::int32_t term) {
        mformat::ParamDict pd;
        pd.set(3, 1);
        pd.set(5, 1);
        pd.set(6, 1);
        pd.set(8, N);
        pd.set(9, K);
        pd.set(10, -1);
        pd.set(18, term);
        return pd;
    };

    G gemm4;
    ASSERT_EQ(gemm4.load_param(make_params(401)), 0);
    mnet::Mat weights4[2] = {w4, scales};
    ASSERT_EQ(gemm4.load_model(mformat::ModelBinFromMatArray(weights4)), 0);

    G gemm8;
    ASSERT_EQ(gemm8.load_param(make_params(801)), 0);
    mnet::Mat weights8[2] = {w8, scales};
    ASSERT_EQ(gemm8.load_model(mformat::ModelBinFromMatArray(weights8)), 0);

    mnet::Mat in;
    in.create(K, M);
    auto* ptr = static_cast<float*>(in.data);
    for (std::int32_t i = 0; i < K * M; ++i) {
        ptr[i] = std::sin(0.017F * static_cast<float>(i) + 0.12F);
    }

    mnet::Option opt_fast;
    opt_fast.use_fast_int4_gemm = true;
    mnet::Option opt_ref;

    mnet::Mat out4;
    mnet::Mat out8;
    ASSERT_EQ(gemm4.forward(in, out4, opt_fast), 0);
    ASSERT_EQ(gemm8.forward(in, out8, opt_ref), 0);
    ASSERT_EQ(out4.w, N);
    ASSERT_EQ(out4.h, M);
    for (std::int32_t i = 0; i < M; ++i) {
        for (std::int32_t j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(out4.row(i)[j], out8.row(i)[j]) << "i=" << i << " j=" << j;
        }
    }
}
