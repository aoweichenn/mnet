// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <string>

#include <gtest/gtest.h>
#include <mformat/data_reader.hpp>
#include <mformat/model_bin.hpp>
#include <mnet/core/layer.hpp>

TEST(ModelBinKernel, ModelBinReadsFp16WeightsAndBias) {
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

    ASSERT_EQ(conv1_w.w, 1728);
    ASSERT_EQ(conv1_b.w, 64);
    ASSERT_EQ(squeeze_w.w, 1024);
    ASSERT_EQ(squeeze_b.w, 16);

    for (std::int32_t i = 0; i < squeeze_w.w; ++i) {
        ASSERT_TRUE(std::isfinite(squeeze_w[static_cast<std::size_t>(i)])) << i;
    }
    for (std::int32_t i = 0; i < squeeze_b.w; ++i) {
        ASSERT_TRUE(std::isfinite(squeeze_b[static_cast<std::size_t>(i)])) << i;
    }
    EXPECT_NEAR(squeeze_w[64], -0.17248535F, 1e-5F);
}
