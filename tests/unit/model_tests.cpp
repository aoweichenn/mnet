// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <mnet/core/net.hpp>

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

TEST(Net, RunsQwen3TwoTokenDecode) {
    const char* dir_env = std::getenv("MNET_QWEN3_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_QWEN3_DIR to run the decode test";
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
    mnet::Net decoder_net;
    decoder_net.opt().num_threads = 4;
    ASSERT_EQ(decoder_net.load(decoder_param.c_str(), decoder_bin.c_str()), 0);
    mnet::Net proj_net;
    proj_net.opt().num_threads = 4;
    ASSERT_EQ(proj_net.load(proj_param.c_str(), embed_bin.c_str()), 0);
    std::vector<mnet::Mat> caches(56);
    std::int32_t token_id = 9707;

    auto embed_token = [&](std::int32_t id) {
        mnet::Mat token(1, 1);
        *reinterpret_cast<std::int32_t*>(static_cast<void*>(token.data)) = id;
        mnet::Net::Extractor ex = embed_net.create_extractor();
        EXPECT_EQ(ex.input("in0", token), 0);
        mnet::Mat out;
        EXPECT_EQ(ex.extract("out0", out), 0);
        return out;
    };

    auto rope = [](std::int32_t pos, mnet::Mat& cos_cache, mnet::Mat& sin_cache) {
        cos_cache.create(64, 1);
        sin_cache.create(64, 1);
        for (std::int32_t i = 0; i < 64; ++i) {
            const float inv = 1.0F / std::pow(1000000.0F, static_cast<float>(i * 2) / 128.0F);
            cos_cache.row(0)[i] = std::cos(inv * static_cast<float>(pos));
            sin_cache.row(0)[i] = std::sin(inv * static_cast<float>(pos));
        }
    };

    auto run_step = [&](std::int32_t pos, std::int32_t seqlen) {
        mnet::Mat embedding = embed_token(token_id);
        EXPECT_FALSE(embedding.empty());

        mnet::Mat mask(seqlen + 1, 1);
        mask.fill(0.0F);
        mnet::Mat cos_cache;
        mnet::Mat sin_cache;
        rope(pos, cos_cache, sin_cache);

        mnet::Net::Extractor ex = decoder_net.create_extractor();
        EXPECT_EQ(ex.input("in0", embedding), 0);
        EXPECT_EQ(ex.input("in1", mask), 0);
        EXPECT_EQ(ex.input("in2", cos_cache), 0);
        EXPECT_EQ(ex.input("in3", sin_cache), 0);
        for (std::int32_t i = 0; i < 28; ++i) {
            char kname[32];
            char vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            if (!caches[static_cast<std::size_t>(i * 2)].empty()) {
                EXPECT_EQ(ex.input(kname, caches[static_cast<std::size_t>(i * 2)]), 0);
                EXPECT_EQ(ex.input(vname, caches[static_cast<std::size_t>(i * 2 + 1)]), 0);
            }
        }

        mnet::Mat hidden_out;
        EXPECT_EQ(ex.extract("out0", hidden_out), 0);
        EXPECT_FALSE(hidden_out.empty());

        for (std::int32_t i = 0; i < 28; ++i) {
            char kname[32];
            char vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            EXPECT_EQ(ex.extract(kname, caches[static_cast<std::size_t>(i * 2)]), 0);
            EXPECT_EQ(ex.extract(vname, caches[static_cast<std::size_t>(i * 2 + 1)]), 0);
        }

        mnet::Net::Extractor pex = proj_net.create_extractor();
        EXPECT_EQ(pex.input("in0", hidden_out), 0);
        mnet::Mat logits;
        EXPECT_EQ(pex.extract("out0", logits), 0);
        EXPECT_FALSE(logits.empty());

        const float* row = logits.dims == 1 ? static_cast<const float*>(logits) : logits.row(0);
        const std::int32_t vocab = logits.w;
        std::int32_t best = 0;
        for (std::int32_t i = 1; i < vocab; ++i) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        EXPECT_TRUE(std::isfinite(row[best]));
        return best;
    };

    const std::int32_t next1 = run_step(0, 0);
    token_id = next1;
    const std::int32_t next2 = run_step(1, 1);
    EXPECT_GE(next1, 0);
    EXPECT_GE(next2, 0);
}

#include <mnet/tokenizer/tokenizer.hpp>

TEST(Net, RunsQwen3TextGeneration) {
    const char* dir_env = std::getenv("MNET_QWEN3_DIR");
    if (dir_env == nullptr) {
        GTEST_SKIP() << "set MNET_QWEN3_DIR to run the text generation test";
    }
    const std::filesystem::path dir(dir_env);

    mnet::Tokenizer tokenizer;
    ASSERT_EQ(tokenizer.load((dir / "vocab.txt").string().c_str(), (dir / "merges.txt").string().c_str()), 0);

    const std::string prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<std::int32_t> prompt_ids = tokenizer.encode(prompt);
    ASSERT_FALSE(prompt_ids.empty());
    std::fprintf(stderr, "roundtrip=%s ids=%d\n", tokenizer.decode(prompt_ids).c_str(),
                 prompt_ids.empty() ? -1 : prompt_ids[0]);

    const std::int32_t rope_half = 64;
    const std::int32_t prompt_len = static_cast<std::int32_t>(prompt_ids.size());

    const std::string embed_bin_text = (dir / "qwen3_embed_token.ncnn.bin").string();
    mnet::Net embed_net;
    embed_net.opt().num_threads = 4;
    ASSERT_EQ(embed_net.load((dir / "qwen3_embed_token.ncnn.param").string().c_str(), embed_bin_text.c_str()), 0);
    mnet::Net decoder_net;
    decoder_net.opt().num_threads = 4;
    ASSERT_EQ(decoder_net.load((dir / "qwen3_decoder.ncnn.param").string().c_str(),
                               (dir / "qwen3_decoder.ncnn.bin").string().c_str()),
              0);
    const bool proj_int8 = std::getenv("MNET_QWEN3_PROJ_INT8") != nullptr;
    const std::string proj_param =
        (dir / (proj_int8 ? "qwen3_proj_out_int8.ncnn.param" : "qwen3_proj_out.ncnn.param")).string();
    const std::string proj_bin = proj_int8 ? (dir / "qwen3_proj_out_int8.ncnn.bin").string() : embed_bin_text;

    mnet::Net proj_net;
    proj_net.opt().num_threads = 4;
    ASSERT_EQ(proj_net.load(proj_param.c_str(), proj_bin.c_str()), 0);

    std::vector<mnet::Mat> caches(56);

    auto rope = [&](std::int32_t seqlen, std::int32_t pos0, mnet::Mat& cos_cache, mnet::Mat& sin_cache) {
        cos_cache.create(rope_half, seqlen);
        sin_cache.create(rope_half, seqlen);
        for (std::int32_t i = 0; i < seqlen; ++i) {
            float* cptr = cos_cache.row(i);
            float* sptr = sin_cache.row(i);
            for (std::int32_t j = 0; j < rope_half; ++j) {
                const float inv = 1.0F / std::pow(1000000.0F, static_cast<float>(j * 2) / 128.0F);
                const float t = inv * static_cast<float>(pos0 + i);
                cptr[j] = std::cos(t);
                sptr[j] = std::sin(t);
            }
        }
    };

    auto run_decoder = [&](const mnet::Mat& embedding, const mnet::Mat& mask, const mnet::Mat& cos_cache,
                           const mnet::Mat& sin_cache) {
        mnet::Net::Extractor ex = decoder_net.create_extractor();
        EXPECT_EQ(ex.input("in0", embedding), 0);
        EXPECT_EQ(ex.input("in1", mask), 0);
        EXPECT_EQ(ex.input("in2", cos_cache), 0);
        EXPECT_EQ(ex.input("in3", sin_cache), 0);
        for (std::int32_t i = 0; i < 28; ++i) {
            char kname[32];
            char vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            if (!caches[static_cast<std::size_t>(i * 2)].empty()) {
                EXPECT_EQ(ex.input(kname, caches[static_cast<std::size_t>(i * 2)]), 0);
                EXPECT_EQ(ex.input(vname, caches[static_cast<std::size_t>(i * 2 + 1)]), 0);
            }
        }
        mnet::Mat hidden_out;
        EXPECT_EQ(ex.extract("out0", hidden_out), 0);
        for (std::int32_t i = 0; i < 28; ++i) {
            char kname[32];
            char vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            EXPECT_EQ(ex.extract(kname, caches[static_cast<std::size_t>(i * 2)]), 0);
            EXPECT_EQ(ex.extract(vname, caches[static_cast<std::size_t>(i * 2 + 1)]), 0);
        }
        return hidden_out;
    };

    auto argmax = [&](const mnet::Mat& hidden_row) {
        mnet::Net::Extractor pex = proj_net.create_extractor();
        EXPECT_EQ(pex.input("in0", hidden_row), 0);
        mnet::Mat logits;
        EXPECT_EQ(pex.extract("out0", logits), 0);
        const float* row = logits.dims == 1 ? static_cast<const float*>(logits) : logits.row(0);
        const std::int32_t vocab = logits.w;
        std::int32_t best = 0;
        for (std::int32_t i = 1; i < vocab; ++i) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        return best;
    };

    // prefill all prompt tokens
    mnet::Mat ids(prompt_len, 1);
    std::memcpy(ids.data, prompt_ids.data(), prompt_ids.size() * sizeof(std::int32_t));
    mnet::Net::Extractor embed_ex = embed_net.create_extractor();
    ASSERT_EQ(embed_ex.input("in0", ids), 0);
    mnet::Mat embedding;
    ASSERT_EQ(embed_ex.extract("out0", embedding), 0);

    mnet::Mat mask(prompt_len, prompt_len);
    for (std::int32_t i = 0; i < prompt_len; ++i) {
        float* row = mask.row(i);
        for (std::int32_t j = 0; j < prompt_len; ++j) {
            row[j] = j <= i ? 0.0F : -1e38F;
        }
    }
    mnet::Mat cos_cache;
    mnet::Mat sin_cache;
    rope(prompt_len, 0, cos_cache, sin_cache);

    const auto t_prefill0 = std::chrono::steady_clock::now();
    mnet::Mat hidden_states = run_decoder(embedding, mask, cos_cache, sin_cache);
    const std::int32_t next0 = argmax(hidden_states.row_range(prompt_len - 1, 1));
    const auto t_prefill1 = std::chrono::steady_clock::now();

    std::vector<std::int32_t> generated = {next0};
    std::int32_t token_id = next0;
    std::int32_t position = prompt_len;
    const auto t_decode0 = std::chrono::steady_clock::now();
    for (std::int32_t step = 0; step < 3; ++step) {
        mnet::Mat one_id(1, 1);
        *reinterpret_cast<std::int32_t*>(static_cast<void*>(one_id.data)) = token_id;
        mnet::Net::Extractor eex = embed_net.create_extractor();
        ASSERT_EQ(eex.input("in0", one_id), 0);
        mnet::Mat one_embed;
        ASSERT_EQ(eex.extract("out0", one_embed), 0);

        mnet::Mat decode_mask(position + 1, 1);
        decode_mask.fill(0.0F);
        mnet::Mat dcos;
        mnet::Mat dsin;
        rope(1, position, dcos, dsin);

        mnet::Mat one_hidden = run_decoder(one_embed, decode_mask, dcos, dsin);
        token_id = argmax(one_hidden);
        generated.push_back(token_id);
        ++position;
    }
    const auto t_decode1 = std::chrono::steady_clock::now();

    const std::string text = tokenizer.decode(generated);
    std::fprintf(stderr, "prompt=%s tokens=%zu prefill=%.1fms decode=%.1fms/3 text=%s\\n", prompt.c_str(),
                 prompt_ids.size(), std::chrono::duration<double, std::milli>(t_prefill1 - t_prefill0).count(),
                 std::chrono::duration<double, std::milli>(t_decode1 - t_decode0).count(), text.c_str());
    EXPECT_FALSE(generated.empty());
    EXPECT_GE(generated.front(), 0);
}
