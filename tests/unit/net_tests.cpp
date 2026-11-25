// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdint>

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <mnet/layer_registry.hpp>
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

TEST(Net, BinaryIndexTableCoversRegisteredLayers) {
    // Newly populated indices that models may reference from binary params.
    const std::int32_t indices[] = {0,  4,  8,  11, 13, 17, 23, 40, 46, 47, 54, 55,
                                    65, 67, 66, 69, 71, 75, 76, 77, 81, 84, 100};
    for (const std::int32_t index : indices) {
        mnet::Layer* layer = mnet::create_layer_by_index(index);
        ASSERT_NE(layer, nullptr) << "type index " << index;
        delete layer;
    }
    // Unsupported types must fail cleanly instead of creating a wrong layer.
    EXPECT_EQ(mnet::create_layer_by_index(1), nullptr);
    EXPECT_EQ(mnet::create_layer_by_index(9999), nullptr);
}

TEST(Net, LoadsFromMemoryLikeFiles) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = write_param(dir);
    const std::filesystem::path bin = write_empty_bin(dir);

    std::ifstream param_file(param, std::ios::binary);
    const std::string param_text((std::istreambuf_iterator<char>(param_file)), std::istreambuf_iterator<char>());
    std::ifstream bin_file(bin, std::ios::binary);
    const std::string bin_bytes((std::istreambuf_iterator<char>(bin_file)), std::istreambuf_iterator<char>());

    mnet::Net file_net;
    ASSERT_EQ(file_net.load(param.string().c_str(), bin.string().c_str()), 0);

    mnet::Net mem_net;
    ASSERT_EQ(mem_net.load_param_mem(param_text.c_str()), 0);
    ASSERT_EQ(mem_net.load_model_mem(bin_bytes.data(), bin_bytes.size()), 0);
    ASSERT_EQ(mem_net.input_names().size(), file_net.input_names().size());
    ASSERT_EQ(mem_net.output_names(), file_net.output_names());

    mnet::Mat data(4);
    data[0] = -1.0F;
    data[1] = 0.5F;
    data[2] = -3.0F;
    data[3] = 2.0F;

    mnet::Net::Extractor ex_file = file_net.create_extractor();
    ASSERT_EQ(ex_file.input("data", data), 0);
    mnet::Mat out_file;
    ASSERT_EQ(ex_file.extract("out", out_file), 0);

    mnet::Net::Extractor ex_mem = mem_net.create_extractor();
    ASSERT_EQ(ex_mem.input("data", data), 0);
    mnet::Mat out_mem;
    ASSERT_EQ(ex_mem.extract("out", out_mem), 0);

    ASSERT_EQ(out_file.w, out_mem.w);
    for (std::int32_t i = 0; i < out_file.w; ++i) {
        EXPECT_FLOAT_EQ(out_file[static_cast<std::size_t>(i)], out_mem[static_cast<std::size_t>(i)]);
    }
}

TEST(Net, ExtractorLightModeToggle) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mnet_net_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path param = dir / "relu2.param";
    {
        std::ofstream file(param);
        file << "7767517\n";
        file << "3 3\n";
        file << "Input in0 0 1 data\n";
        file << "ReLU relu 1 1 data mid\n";
        file << "ReLU relu2 1 1 mid out\n";
    }
    const std::filesystem::path bin = write_empty_bin(dir);

    mnet::Net net;
    ASSERT_EQ(net.load(param.string().c_str(), bin.string().c_str()), 0);

    mnet::Mat data(3);
    data[0] = -1.0F;
    data[1] = 2.0F;
    data[2] = -3.0F;

    float expected[3] = {0.0F, 2.0F, 0.0F};
    for (std::int32_t pass = 0; pass < 2; ++pass) {
        mnet::Net::Extractor ex = net.create_extractor();
        ex.set_light_mode(pass == 0);
        ex.set_num_threads(1);
        ASSERT_EQ(ex.input("data", data), 0);
        mnet::Mat mid;
        ASSERT_EQ(ex.extract("mid", mid), 0);
        for (std::int32_t i = 0; i < 3; ++i) {
            EXPECT_FLOAT_EQ(mid[static_cast<std::size_t>(i)], expected[i]);
        }
        mnet::Mat out;
        ASSERT_EQ(ex.extract("out", out), 0);
        for (std::int32_t i = 0; i < 3; ++i) {
            EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(i)], expected[i]);
        }
    }
}
