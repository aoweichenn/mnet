// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mlayer/multiheadattention.hpp>
#include <mlayer/relposbias.hpp>
#include <mlayer/rotaryembed.hpp>
#include <mlayer/sdpa.hpp>
#include <mnet/layer.hpp>

TEST(SequenceKernel, RotaryEmbedAndSdpaSingleStep) {
    using RE = mlayer::RotaryEmbed<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict pd;
    RE rope;
    ASSERT_EQ(rope.load_param(pd), 0);

    mnet::Mat x(4, 1, 1);
    x.channel(0)[0] = 1.0F;
    x.channel(0)[1] = 0.0F;
    x.channel(0)[2] = 0.0F;
    x.channel(0)[3] = 1.0F;
    mnet::Mat cos_cache(2, 1);
    mnet::Mat sin_cache(2, 1);
    cos_cache.row(0)[0] = 0.0F;
    cos_cache.row(0)[1] = 0.0F;
    sin_cache.row(0)[0] = 1.0F;
    sin_cache.row(0)[1] = 1.0F;

    std::vector<mnet::Mat> rb = {x, cos_cache, sin_cache};
    std::vector<mnet::Mat> rt(1);
    mnet::Option opt;
    ASSERT_EQ(rope.forward(rb, rt, opt), 0);
    EXPECT_NEAR(rt[0].channel(0)[0], 0.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[1], -1.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[2], 1.0F, 1e-6F);
    EXPECT_NEAR(rt[0].channel(0)[3], 0.0F, 1e-6F);

    using AP = mlayer::SDPA<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict spd;
    spd.set(5, 0);
    spd.set(7, 0);
    AP sdpa;
    ASSERT_EQ(sdpa.load_param(spd), 0);

    mnet::Mat q(4, 1, 1);
    mnet::Mat k(4, 1, 1);
    mnet::Mat v(4, 1, 1);
    for (std::int32_t i = 0; i < 4; ++i) {
        q.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        k.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        v.channel(0)[static_cast<std::size_t>(i)] = static_cast<float>(10 + i);
    }
    std::vector<mnet::Mat> sb = {q, k, v};
    std::vector<mnet::Mat> st(1);
    ASSERT_EQ(sdpa.forward(sb, st, opt), 0);
    ASSERT_EQ(st[0].w, 4);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(st[0].channel(0)[static_cast<std::size_t>(i)], v.channel(0)[static_cast<std::size_t>(i)], 1e-5F);
    }
}

// The int8 KV cache must produce the same attention output as the fp32 cache
// within quantization noise, while keeping the cache at one byte per value.
TEST(SequenceKernel, SdpaInt8KvCacheMatchesFp32) {
    using AP = mlayer::SDPA<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    mformat::ParamDict spd;
    spd.set(5, 0);
    spd.set(7, 1);
    AP sdpa;
    ASSERT_EQ(sdpa.load_param(spd), 0);

    const std::int32_t heads = 2;
    const std::int32_t group = 2;
    const std::int32_t dim = 32;
    const std::int32_t steps[5] = {3, 1, 1, 1, 1};

    auto run = [&](bool int8_kv, std::vector<float>& flat, mnet::Mat& last_key) {
        mnet::Option opt;
        opt.num_threads = 2;
        opt.use_int8_kv_cache = int8_kv;

        std::uint32_t seed = 20260101u;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (static_cast<float>(seed >> 8) / 16777216.0F) * 2.0F - 1.0F;
        };

        mnet::Mat past_k;
        mnet::Mat past_v;
        for (std::int32_t step = 0; step < 5; ++step) {
            const std::int32_t n = steps[step];
            mnet::Mat q(dim, n, heads);
            mnet::Mat k(dim, n, group);
            mnet::Mat v(dim, n, group);
            for (std::int32_t c = 0; c < heads; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        q.channel(c).row(h)[w] = rnd();
                    }
                }
            }
            for (std::int32_t c = 0; c < group; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        k.channel(c).row(h)[w] = rnd();
                        v.channel(c).row(h)[w] = rnd();
                    }
                }
            }

            std::vector<mnet::Mat> bottom = {q, k, v, past_k, past_v};
            std::vector<mnet::Mat> top(3);
            ASSERT_EQ(sdpa.forward(bottom, top, opt), 0);
            for (std::int32_t c = 0; c < heads; ++c) {
                for (std::int32_t h = 0; h < n; ++h) {
                    for (std::int32_t w = 0; w < dim; ++w) {
                        flat.push_back(top[0].channel(c).row(h)[w]);
                    }
                }
            }
            past_k = top[1];
            past_v = top[2];
        }
        last_key = past_k;
    };

    std::vector<float> ref;
    std::vector<float> quantized;
    mnet::Mat fp32_key;
    mnet::Mat int8_key;
    run(false, ref, fp32_key);
    run(true, quantized, int8_key);

    ASSERT_EQ(ref.size(), quantized.size());
    float max_diff = 0.0F;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ref[i] - quantized[i]));
    }
    EXPECT_LT(max_diff, 0.05F);

    EXPECT_EQ(int8_key.elemsize, 1u);
    EXPECT_GT(fp32_key.total() * 4, int8_key.total() * 3);
}

// Relative-position bias + attention sink: bias[h,q,k] = dot(rel_bias[:,d], R[:,h,q]) - sink[h,q].
TEST(SequenceKernel, RelPosBiasMatchesReference) {
    using B = mlayer::RelPosBias<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict>;
    B bias;
    ASSERT_EQ(bias.load_param(mformat::ParamDict()), 0);

    const std::int32_t n_rel = 2;
    const std::int32_t extent = 3;
    const std::int32_t heads = 2;
    const std::int32_t tokens = 2;
    const std::int32_t n_kv = 4;

    mnet::Mat rel(n_rel, tokens, heads);
    mnet::Mat table(n_rel, extent);
    mnet::Mat sink(heads, tokens);
    mnet::Mat rel_pos(n_kv, tokens);

    for (std::int32_t h = 0; h < heads; ++h) {
        for (std::int32_t q = 0; q < tokens; ++q) {
            rel.channel(h).row(q)[0] = 1.0F + static_cast<float>(h) + 0.1F * static_cast<float>(q);
            rel.channel(h).row(q)[1] = -0.5F + static_cast<float>(q);
            sink.row(q)[h] = 0.25F * static_cast<float>(h + 1);
        }
    }
    for (std::int32_t d = 0; d < extent; ++d) {
        table.row(d)[0] = 0.1F * static_cast<float>(d);
        table.row(d)[1] = 1.0F - 0.2F * static_cast<float>(d);
    }
    // distances: key 0..3 for query 0 -> 0,1,2,extent(out of window)
    rel_pos.row(0)[0] = 0.0F;
    rel_pos.row(0)[1] = 1.0F;
    rel_pos.row(0)[2] = 2.0F;
    rel_pos.row(0)[3] = static_cast<float>(extent);
    rel_pos.row(1)[0] = 1.0F;
    rel_pos.row(1)[1] = 100.0F;  // clamps out of window
    rel_pos.row(1)[2] = 0.0F;
    rel_pos.row(1)[3] = 2.0F;

    mnet::Option opt;
    std::vector<mnet::Mat> bottom = {rel, table, sink, rel_pos};
    std::vector<mnet::Mat> top(1);
    ASSERT_EQ(bias.forward(bottom, top, opt), 0);
    ASSERT_EQ(top[0].w, n_kv);
    ASSERT_EQ(top[0].h, tokens);
    ASSERT_EQ(top[0].c, heads);

    for (std::int32_t h = 0; h < heads; ++h) {
        for (std::int32_t q = 0; q < tokens; ++q) {
            for (std::int32_t k = 0; k < n_kv; ++k) {
                const float d = rel_pos.row(q)[k];
                float sum = 0.0F;
                if (d >= 0.0F && d < static_cast<float>(extent)) {
                    const auto di = static_cast<std::int32_t>(d);
                    sum = table.row(di)[0] * rel.channel(h).row(q)[0] + table.row(di)[1] * rel.channel(h).row(q)[1];
                }
                EXPECT_NEAR(top[0].channel(h).row(q)[k], sum - sink.row(q)[h], 1e-6F);
            }
        }
    }
}

TEST(SequenceKernel, MultiHeadAttentionSelfAttention) {
    using MHA = mlayer::MultiHeadAttention<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    mformat::ParamDict pd;
    pd.set(0, 2);  // embed_dim
    pd.set(1, 1);  // num_heads
    pd.set(2, 4);  // weight_data_size = qdim * embed_dim
    pd.set(3, 2);  // kdim
    pd.set(4, 2);  // vdim
    MHA layer;
    ASSERT_EQ(layer.load_param(pd), 0);

    mnet::Mat identity(4);
    identity[0] = 1.0F;
    identity[1] = 0.0F;
    identity[2] = 0.0F;
    identity[3] = 1.0F;
    layer.q_weight_data = identity;
    layer.k_weight_data = identity;
    layer.v_weight_data = identity;
    layer.out_weight_data = identity;
    layer.q_bias_data = mnet::Mat(2);
    layer.k_bias_data = mnet::Mat(2);
    layer.v_bias_data = mnet::Mat(2);
    layer.out_bias_data = mnet::Mat(2);
    layer.q_bias_data.fill(0.0F);
    layer.k_bias_data.fill(0.0F);
    layer.v_bias_data.fill(0.0F);
    layer.out_bias_data.fill(0.0F);

    mnet::Mat input(2, 1);
    input[0] = 1.0F;
    input[1] = 0.0F;

    std::vector<mnet::Mat> bottoms = {input};
    std::vector<mnet::Mat> tops(1);
    mnet::Option opt;
    ASSERT_EQ(layer.forward(bottoms, tops, opt), 0);
    ASSERT_EQ(tops[0].w, 2);
    ASSERT_EQ(tops[0].h, 1);
    EXPECT_NEAR(tops[0].row(0)[0], 1.0F, 1e-5F);
    EXPECT_NEAR(tops[0].row(0)[1], 0.0F, 1e-5F);
}

TEST(SequenceKernel, MultiHeadAttentionKvCacheMatchesFullSequence) {
    using MHA = mlayer::MultiHeadAttention<mnet::Layer, mnet::Mat, mnet::Option, mformat::ParamDict, mformat::ModelBin>;
    mformat::ParamDict pd;
    pd.set(0, 4);   // embed_dim
    pd.set(1, 2);   // num_heads
    pd.set(2, 16);  // weight_data_size
    pd.set(3, 4);   // kdim
    pd.set(4, 4);   // vdim
    pd.set(7, 1);   // kv_cache
    MHA layer;
    ASSERT_EQ(layer.load_param(pd), 0);

    const std::int32_t E = 4;
    std::vector<mnet::Mat> weights;
    for (std::int32_t t = 0; t < 8; ++t) {
        mnet::Mat w = (t % 2 == 0) ? mnet::Mat(E * E) : mnet::Mat(E);
        for (std::int32_t i = 0; i < w.w; ++i) {
            w[static_cast<std::size_t>(i)] = 0.11F * static_cast<float>((i * 7 + t * 3) % 17) - 0.7F;
        }
        weights.push_back(w);
    }
    ASSERT_EQ(layer.load_model(mformat::ModelBinFromMatArray(weights.data())), 0);

    auto make_x = [&](std::int32_t seq, std::int32_t seed) {
        mnet::Mat x(E, seq);
        for (std::int32_t i = 0; i < E * seq; ++i) {
            x[static_cast<std::size_t>(i)] = std::sin(0.3F * static_cast<float>(i + seed));
        }
        return x;
    };

    mformat::ParamDict full_pd;
    full_pd.set(0, 4);
    full_pd.set(1, 2);
    full_pd.set(2, 16);
    full_pd.set(3, 4);
    full_pd.set(4, 4);
    MHA full_layer;
    ASSERT_EQ(full_layer.load_param(full_pd), 0);
    ASSERT_EQ(full_layer.load_model(mformat::ModelBinFromMatArray(weights.data())), 0);

    mnet::Option opt;
    mnet::Mat full = make_x(2, 1);
    std::vector<mnet::Mat> full_bottoms = {full};
    std::vector<mnet::Mat> full_tops(1);
    ASSERT_EQ(full_layer.forward(full_bottoms, full_tops, opt), 0);

    std::vector<mnet::Mat> top0(3);
    std::vector<mnet::Mat> b0 = {make_x(1, 1), mnet::Mat(), mnet::Mat()};
    ASSERT_EQ(layer.forward(b0, top0, opt), 0);
    ASSERT_EQ(top0[1].dims, 3);
    ASSERT_EQ(top0[1].w, 2);
    ASSERT_EQ(top0[1].h, 1);
    ASSERT_EQ(top0[1].c, 2);

    std::vector<mnet::Mat> top1(3);
    std::vector<mnet::Mat> b1 = {make_x(1, 1 + E), top0[1], top0[2]};
    ASSERT_EQ(layer.forward(b1, top1, opt), 0);
    ASSERT_EQ(top1[1].h, 2);

    for (std::int32_t i = 0; i < E; ++i) {
        EXPECT_NEAR(top1[0].row(0)[i], full_tops[0].row(1)[i], 1e-5F);
    }
}
