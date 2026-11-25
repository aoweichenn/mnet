// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
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

TEST(Net, LoadsBinaryParamAndRuns) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = dir / "relu.binparam";
    const std::filesystem::path bin = write_empty_bin(dir);

    {
        std::ofstream file(param, std::ios::binary);
        auto write_i32 = [&file](std::int32_t v) { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
        write_i32(7767517);  // magic
        write_i32(2);        // layer count
        write_i32(2);        // blob count

        // Input, 0 bottoms, 1 top (blob 0)
        write_i32(16);
        write_i32(0);
        write_i32(1);
        write_i32(0);
        write_i32(-233);

        // ReLU, blob 0 -> blob 1
        write_i32(26);
        write_i32(1);
        write_i32(1);
        write_i32(0);
        write_i32(1);
        write_i32(-233);
    }

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
    ASSERT_EQ(ex.input(0, data), 0);

    mnet::Mat out;
    ASSERT_EQ(ex.extract(1, out), 0);
    EXPECT_EQ(out.w, 5);
    EXPECT_FLOAT_EQ(out[0], 0.0F);
    EXPECT_FLOAT_EQ(out[1], 0.0F);
    EXPECT_FLOAT_EQ(out[2], 0.0F);
    EXPECT_FLOAT_EQ(out[3], 1.0F);
    EXPECT_FLOAT_EQ(out[4], 2.0F);
}

TEST(Net, RunsZeroBottomMemoryData) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = dir / "memorydata.param";
    const std::filesystem::path bin = dir / "memorydata.bin";

    {
        std::ofstream file(param);
        file << "7767517\n";
        file << "1 1\n";
        file << "MemoryData md 0 1 data 0=2\n";
    }
    {
        std::ofstream file(bin, std::ios::binary);
        const float values[2] = {1.5F, -2.5F};
        file.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    mnet::Net net;
    net.opt().num_threads = 2;
    ASSERT_EQ(net.load(param.string().c_str(), bin.string().c_str()), 0);

    // MemoryData must not be treated as a user input layer.
    EXPECT_TRUE(net.input_names().empty());

    mnet::Net::Extractor ex = net.create_extractor();
    mnet::Mat out;
    ASSERT_EQ(ex.extract("data", out), 0);
    ASSERT_EQ(out.dims, 1);
    ASSERT_EQ(out.w, 2);
    EXPECT_FLOAT_EQ(out[0], 1.5F);
    EXPECT_FLOAT_EQ(out[1], -2.5F);
}
