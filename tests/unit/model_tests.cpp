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
