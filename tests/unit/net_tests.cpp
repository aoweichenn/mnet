// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <mnet/net.hpp>

namespace {

std::filesystem::path write_param(const std::filesystem::path& dir) {
    const std::filesystem::path path = dir / "relu.param";
    std::ofstream file(path);
    file << "7767517\n";
    file << "2 2\n";
    file << "Input in0 0 1 data\n";
    file << "ReLU relu 1 1 data out\n";
    return path;
}

std::filesystem::path write_empty_bin(const std::filesystem::path& dir) {
    const std::filesystem::path path = dir / "relu.bin";
    std::ofstream file(path, std::ios::binary);
    file.put('\0');
    return path;
}

}  // namespace

TEST(Net, LoadsParamAndRunsInplaceLayer) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = write_param(dir);
    const std::filesystem::path bin = write_empty_bin(dir);

    mnet::Net net;
    net.opt().num_threads = 2;
    ASSERT_EQ(net.load(param.string().c_str(), bin.string().c_str()), 0);

    mnet::Mat data(5);
    data[0] = -2.0F;
    data[1] = -1.0F;
    data[2] = 0.0F;
    data[3] = 1.0F;
    data[4] = 2.0F;

    mnet::Net::Extractor ex = net.create_extractor();
    ASSERT_EQ(ex.input("data", data), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract("out", out), 0);
    ASSERT_EQ(out.w, 5);
    EXPECT_FLOAT_EQ(out[0], 0.0F);
    EXPECT_FLOAT_EQ(out[1], 0.0F);
    EXPECT_FLOAT_EQ(out[2], 0.0F);
    EXPECT_FLOAT_EQ(out[3], 1.0F);
    EXPECT_FLOAT_EQ(out[4], 2.0F);
}

TEST(Net, RejectsUnknownLayerType) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = dir / "unknown.param";
    std::ofstream file(param);
    file << "7767517\n";
    file << "1 1\n";
    file << "DefinitelyNotALayer x 0 1 data\n";

    mnet::Net net;
    EXPECT_NE(net.load_param(param.string().c_str()), 0);
}
