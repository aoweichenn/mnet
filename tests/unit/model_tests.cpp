// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdlib>

#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <mnet/net.hpp>

TEST(Net, RunsSqueezenetEndToEnd) {
    const char* dir_env = std::getenv("MNET_SQUEEZENET_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_SQUEEZENET_DIR to run the end-to-end model test";
    }
    const std::filesystem::path dir(dir_env);
    const std::string param = (dir / "squeezenet_v1.1.param").string();
    const std::string bin = (dir / "squeezenet_v1.1.bin").string();

    mnet::Net net;
    net.opt().num_threads = 4;
    ASSERT_EQ(net.load(param.c_str(), bin.c_str()), 0);

    mnet::Mat input(227, 227, 3);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < 227 * 227; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Net::Extractor ex = net.create_extractor();
    ASSERT_EQ(ex.input("data", input), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract("prob", out), 0);
    ASSERT_EQ(out.dims, 1);
    ASSERT_EQ(out.w, 1000);

    float sum = 0.0F;
    for (std::int32_t i = 0; i < out.w; ++i) {
        EXPECT_GE(out[static_cast<std::size_t>(i)], 0.0F);
        sum += out[static_cast<std::size_t>(i)];
    }
    EXPECT_NEAR(sum, 1.0F, 1e-3F);
}

TEST(Net, RunsShufflenetV2EndToEnd) {
    const char* dir_env = std::getenv("MNET_SHUFFLENET_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_SHUFFLENET_DIR to run the end-to-end model test";
    }
    const std::filesystem::path dir(dir_env);
    const std::string param = (dir / "shufflenet_v2_x0.5.param").string();
    const std::string bin = (dir / "shufflenet_v2_x0.5.bin").string();

    mnet::Net net;
    net.opt().num_threads = 4;
    ASSERT_EQ(net.load(param.c_str(), bin.c_str()), 0);

    mnet::Mat input(224, 224, 3);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < 224 * 224; ++i) {
            p[i] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Net::Extractor ex = net.create_extractor();
    ASSERT_EQ(ex.input("data", input), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract("fc", out), 0);

    std::int32_t bad = 0;
    if (out.dims == 1) {
        ASSERT_EQ(out.w, 1000);
        for (std::int32_t i = 0; i < out.w; ++i) {
            if (!std::isfinite(out[static_cast<std::size_t>(i)])) {
                ++bad;
            }
        }
    } else {
        ASSERT_EQ(out.w, 1);
        ASSERT_EQ(out.h, 1);
        ASSERT_EQ(out.c, 1000);
        for (std::int32_t q = 0; q < out.c; ++q) {
            if (!std::isfinite(out.channel(q)[0])) {
                ++bad;
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

TEST(Net, RunsYolo11nEndToEnd) {
    const char* dir_env = std::getenv("MNET_YOLO11N_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_YOLO11N_DIR to run the end-to-end model test";
    }
    const std::filesystem::path dir(dir_env);
    const std::string param = (dir / "yolo11n.ncnn.param").string();
    const std::string bin = (dir / "yolo11n.ncnn.bin").string();

    mnet::Net net;
    net.opt().num_threads = 4;
    ASSERT_EQ(net.load(param.c_str(), bin.c_str()), 0);

    mnet::Mat input(640, 640, 3);
    for (std::int32_t q = 0; q < input.c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < 640 * 640; ++i) {
            p[i] = std::sin(0.001F * static_cast<float>(i + q * 17));
        }
    }

    mnet::Net::Extractor ex = net.create_extractor();
    ASSERT_EQ(ex.input("in0", input), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract("out0", out), 0);
    ASSERT_FALSE(out.empty());

    std::int32_t bad = 0;
    for (std::int32_t q = 0; q < out.c; ++q) {
        const float* p = out.channel(q);
        for (std::int32_t i = 0; i < out.w * out.h * out.d; ++i) {
            if (!std::isfinite(p[i])) {
                ++bad;
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

TEST(Net, RunsQwen3DecoderStep) {
    const char* dir_env = std::getenv("MNET_QWEN3_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_QWEN3_DIR to run the decoder step test";
    }
    const std::filesystem::path dir(dir_env);
    const std::string param = (dir / "qwen3_decoder.ncnn.param").string();
    const std::string bin = (dir / "qwen3_decoder.ncnn.bin").string();

    mnet::Net net;
    net.opt().num_threads = 4;
    ASSERT_EQ(net.load(param.c_str(), bin.c_str()), 0);

    const std::int32_t hidden = 1024;
    const std::int32_t rope_half = 64;

    mnet::Mat embed(hidden, 1);
    for (std::int32_t i = 0; i < hidden; ++i) {
        embed.row(0)[i] = std::sin(0.01F * static_cast<float>(i));
    }

    mnet::Mat mask(1, 1);
    mask.fill(0.0F);

    mnet::Mat cos_cache(rope_half, 1);
    mnet::Mat sin_cache(rope_half, 1);
    for (std::int32_t i = 0; i < rope_half; ++i) {
        const float inv = 1.0F / std::pow(1000000.0F, static_cast<float>(i * 2) / 128.0F);
        cos_cache.row(0)[i] = std::cos(inv * 0.0F);
        sin_cache.row(0)[i] = std::sin(inv * 0.0F);
    }

    mnet::Net::Extractor ex = net.create_extractor();
    ASSERT_EQ(ex.input("in0", embed), 0);
    ASSERT_EQ(ex.input("in1", mask), 0);
    ASSERT_EQ(ex.input("in2", cos_cache), 0);
    ASSERT_EQ(ex.input("in3", sin_cache), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract("out0", out), 0);
    ASSERT_FALSE(out.empty());

    std::int32_t bad = 0;
    for (std::int32_t q = 0; q < out.c; ++q) {
        const float* p = out.channel(q);
        for (std::int32_t i = 0; i < out.w * out.h * out.d; ++i) {
            if (!std::isfinite(p[i])) {
                ++bad;
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

TEST(Net, RunsQwen3TokenStep) {
    const char* dir_env = std::getenv("MNET_QWEN3_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_QWEN3_DIR to run the token step test";
    }
    const std::filesystem::path dir(dir_env);
    const std::string embed_param = (dir / "qwen3_embed_token.ncnn.param").string();
    const std::string embed_bin = (dir / "qwen3_embed_token.ncnn.bin").string();
    const std::string decoder_param = (dir / "qwen3_decoder.ncnn.param").string();
    const std::string decoder_bin = (dir / "qwen3_decoder.ncnn.bin").string();
    const std::string proj_param = (dir / "qwen3_proj_out.ncnn.param").string();

    mnet::Net embed_net;
    embed_net.opt().num_threads = 4;
    ASSERT_EQ(embed_net.load(embed_param.c_str(), embed_bin.c_str()), 0);

    const std::int32_t token_id = 9707;
    mnet::Mat token(1, 1);
    *reinterpret_cast<std::int32_t*>(static_cast<void*>(token.data)) = token_id;

    mnet::Net::Extractor embed_ex = embed_net.create_extractor();
    ASSERT_EQ(embed_ex.input("in0", token), 0);
    mnet::Mat embedding;
    ASSERT_EQ(embed_ex.extract("out0", embedding), 0);
    ASSERT_FALSE(embedding.empty());

    mnet::Net decoder_net;
    decoder_net.opt().num_threads = 4;
    ASSERT_EQ(decoder_net.load(decoder_param.c_str(), decoder_bin.c_str()), 0);

    mnet::Mat mask(1, 1);
    mask.fill(0.0F);
    mnet::Mat cos_cache(64, 1);
    mnet::Mat sin_cache(64, 1);
    for (std::int32_t i = 0; i < 64; ++i) {
        cos_cache.row(0)[i] = 1.0F;
        sin_cache.row(0)[i] = 0.0F;
    }

    mnet::Net::Extractor decoder_ex = decoder_net.create_extractor();
    ASSERT_EQ(decoder_ex.input("in0", embedding), 0);
    ASSERT_EQ(decoder_ex.input("in1", mask), 0);
    ASSERT_EQ(decoder_ex.input("in2", cos_cache), 0);
    ASSERT_EQ(decoder_ex.input("in3", sin_cache), 0);
    mnet::Mat hidden;
    ASSERT_EQ(decoder_ex.extract("out0", hidden), 0);
    ASSERT_FALSE(hidden.empty());

    mnet::Net proj_net;
    proj_net.opt().num_threads = 4;
    ASSERT_EQ(proj_net.load(proj_param.c_str(), embed_bin.c_str()), 0);

    mnet::Net::Extractor proj_ex = proj_net.create_extractor();
    ASSERT_EQ(proj_ex.input("in0", hidden), 0);
    mnet::Mat logits;
    ASSERT_EQ(proj_ex.extract("out0", logits), 0);
    ASSERT_FALSE(logits.empty());

    const float* row = logits.dims == 1 ? static_cast<const float*>(logits) : logits.row(0);
    const std::int32_t vocab = logits.dims == 1 ? logits.w : logits.w;
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < vocab; ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    EXPECT_GE(best, 0);
    EXPECT_LT(best, vocab);
    EXPECT_TRUE(std::isfinite(row[best]));
}
