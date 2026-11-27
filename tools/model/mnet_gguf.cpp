// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <mnet/gguf.hpp>

namespace {

[[nodiscard]] std::vector<float> read_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("input is not float32: " + path);
    }
    file.seekg(0);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) {
        throw std::runtime_error("short read: " + path);
    }
    return values;
}

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void print_info(const mnet::gguf::Reader& reader, bool print_tensors) {
    std::printf("file       : %s\n", reader.path().c_str());
    std::printf("version    : %u\n", reader.version());
    const auto& strings = reader.metadata();
    const auto arch = strings.find("general.architecture");
    std::printf("arch       : %s\n", arch == strings.end() ? "?" : arch->second.c_str());
    std::printf("tensors    : %zu\n", reader.tensors().size());
    std::printf("metadata   : %zu strings, %zu integers\n", strings.size(), reader.integers().size());
    std::printf("data_start : %" PRIu64 "\n", reader.data_start());
    if (print_tensors) {
        for (const auto& tensor : reader.tensors()) {
            const auto info = mnet::gguf::type_info(tensor.type);
            std::printf("%-44s %-8.*s [", tensor.name.c_str(), static_cast<std::int32_t>(info.name.size()),
                        info.name.data());
            for (std::size_t i = 0; i < tensor.dims.size(); ++i) {
                std::printf("%s%" PRIu64 "", i == 0 ? "" : "x", tensor.dims[i]);
            }
            std::printf("]\n");
        }
    }
}

std::int32_t run_matmul(const mnet::gguf::Reader& reader, const std::string& tensor_name, const std::string& input_path,
                        const std::string& output_path, std::int32_t expert) {
    const auto& tensor = reader.tensor(tensor_name);
    const auto type = mnet::gguf::type_info(tensor.type);
    const std::size_t input_dim = tensor.dims[0];
    const std::size_t output_dim = tensor.dims[1];
    const auto start = std::chrono::steady_clock::now();
    const std::vector<float> weights = reader.dequantize(tensor, expert);
    if (weights.size() != input_dim * output_dim) {
        throw std::runtime_error("dequantized size mismatch");
    }
    const std::vector<float> input = read_floats(input_path);
    if (input.size() != input_dim) {
        std::fprintf(stderr, "input has %zu floats, tensor expects %" PRIu64 "\n", input.size(), input_dim);
        return 2;
    }
    std::vector<float> output(output_dim, 0.0F);
    for (std::size_t row = 0; row < output_dim; ++row) {
        double sum = 0.0;
        const std::size_t base = row * input_dim;
        for (std::size_t column = 0; column < input_dim; ++column) {
            sum += static_cast<double>(weights[base + column]) * static_cast<double>(input[column]);
        }
        output[row] = static_cast<float>(sum);
    }
    write_floats(output_path, output);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::printf("%s: %.*s [%" PRIu64 ", %" PRIu64 "] expert=%d -> %s (%.2f ms)\n", tensor.name.c_str(),
                static_cast<std::int32_t>(type.name.size()), type.name.data(), input_dim, output_dim, expert,
                output_path.c_str(), ms);
    return 0;
}

std::int32_t run_matmul_quantized(const mnet::gguf::Reader& reader, const std::string& tensor_name,
                                  const std::string& input_path, const std::string& output_path, std::int32_t expert) {
    const auto& info = reader.tensor(tensor_name);
    const auto type = mnet::gguf::type_info(info.type);
    const std::size_t input_dim = info.dims[0];
    const std::size_t output_dim = info.dims.size() >= 2 ? info.dims[1] : 1;
    const std::vector<float> input = read_floats(input_path);
    if (input.size() != input_dim) {
        std::fprintf(stderr, "input has %zu floats, tensor expects %" PRIu64 "\n", input.size(), input_dim);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto weights = reader.quantized(info, expert);
    std::vector<float> output(output_dim, 0.0F);
    mnet::gguf::Reader::matvec(weights, input.data(), output.data());
    write_floats(output_path, output);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    const auto bytes = weights.blocks.size();
    std::printf("%s: %.*s [%" PRIu64 ", %" PRIu64 "] expert=%d quantized=%zu bytes -> %s (%.2f ms)\n",
                info.name.c_str(), static_cast<std::int32_t>(type.name.size()), type.name.data(), input_dim, output_dim,
                expert, bytes, output_path.c_str(), ms);
    return 0;
}

[[nodiscard]] std::vector<float> tensor_floats(const mnet::gguf::Reader& reader, const std::string& name) {
    return reader.dequantize(reader.tensor(name));
}

void fwht(float* data, std::size_t size) {
    for (std::size_t length = 1; length < size; length <<= 1U) {
        for (std::size_t start = 0; start < size; start += 2 * length) {
            for (std::size_t i = 0; i < length; ++i) {
                const float a = data[start + i];
                const float b = data[start + i + length];
                data[start + i] = a + b;
                data[start + i + length] = a - b;
            }
        }
    }
}

void hadamard_sign(std::vector<float>& values, const std::vector<std::int32_t>& signs, std::uint32_t block_size) {
    if (!signs.empty()) {
        if (signs.size() != values.size()) {
            throw std::runtime_error("sign vector width mismatch");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] *= static_cast<float>(signs[i]);
        }
    }
    if (block_size == 0) {
        return;
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(block_size));
    for (std::size_t start = 0; start < values.size(); start += block_size) {
        if (start + block_size > values.size()) {
            throw std::runtime_error("Hadamard block size does not divide the activation width");
        }
        fwht(values.data() + start, block_size);
        for (std::size_t i = 0; i < block_size; ++i) {
            values[start + i] *= scale;
        }
    }
}

[[nodiscard]] std::vector<float> rms_norm(const std::vector<float>& input, const std::vector<float>& weight,
                                          float eps) {
    if (input.size() != weight.size()) {
        throw std::runtime_error("RMSNorm weight width mismatch");
    }
    double squares = 0.0;
    for (const float value : input) {
        squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto mean = squares / static_cast<double>(input.size());
    const auto scale = static_cast<float>(1.0 / std::sqrt(mean + static_cast<double>(eps)));
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * weight[i];
    }
    return output;
}

[[nodiscard]] float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

std::int32_t run_bonsai_ffn(const mnet::gguf::Reader& reader, std::int32_t layer, const std::string& input_path,
                            const std::string& output_path) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto& gate_info = reader.tensor(prefix + "ffn_gate.weight");
    const auto in_dim = static_cast<std::size_t>(gate_info.dims[0]);
    const auto hidden = static_cast<std::size_t>(gate_info.dims[1]);
    const std::vector<float> input = read_floats(input_path);
    if (input.size() != in_dim) {
        std::fprintf(stderr, "input has %zu floats, FFN expects %zu\n", input.size(), in_dim);
        return 2;
    }
    const std::uint32_t block_size = reader.integers().contains("prism.hadamard.block_size")
                                         ? static_cast<std::uint32_t>(reader.integers().at("prism.hadamard.block_size"))
                                         : 0U;
    const float eps = reader.floats().contains("qwen35.attention.layer_norm_rms_epsilon")
                          ? static_cast<float>(reader.floats().at("qwen35.attention.layer_norm_rms_epsilon"))
                          : 1.0e-6F;
    const auto signs_in = reader.signs_for_width(static_cast<std::uint32_t>(in_dim));
    const auto signs_hidden = reader.signs_for_width(static_cast<std::uint32_t>(hidden));
    std::printf("bonsai-ffn layer=%d in=%zu hidden=%zu block=%u eps=%.3g signs=%zu/%zu\n", layer, in_dim, hidden,
                block_size, static_cast<double>(eps), signs_in.size(), signs_hidden.size());

    const auto start = std::chrono::steady_clock::now();
    const auto norm = tensor_floats(reader, prefix + "post_attention_norm.weight");
    const auto gate = reader.quantized(reader.tensor(prefix + "ffn_gate.weight"));
    const auto up = reader.quantized(reader.tensor(prefix + "ffn_up.weight"));
    const auto down = reader.quantized(reader.tensor(prefix + "ffn_down.weight"));
    std::printf("quantized weights: gate=%zu up=%zu down=%zu bytes (fp32 would be %zu)\n", gate.blocks.size(),
                up.blocks.size(), down.blocks.size(), (hidden * in_dim * 3) * sizeof(float));

    std::vector<float> hidden_in = rms_norm(input, norm, eps);
    hadamard_sign(hidden_in, signs_in, block_size);
    std::vector<float> g(hidden);
    std::vector<float> u(hidden);
    mnet::gguf::Reader::matvec(gate, hidden_in.data(), g.data());
    mnet::gguf::Reader::matvec(up, hidden_in.data(), u.data());
    std::vector<float> act(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
        act[i] = silu(g[i]) * u[i];
    }
    hadamard_sign(act, signs_hidden, block_size);
    std::vector<float> ffn(in_dim);
    mnet::gguf::Reader::matvec(down, act.data(), ffn.data());
    std::vector<float> post(in_dim);
    for (std::size_t i = 0; i < in_dim; ++i) {
        post[i] = input[i] + ffn[i];
    }
    write_floats(output_path, post);
    write_floats(output_path + ".ffn.bin", ffn);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::printf("post_ffn -> %s, ffn_out -> %s (%.2f ms)\n", output_path.c_str(), (output_path + ".ffn.bin").c_str(),
                ms);
    return 0;
}
}  // namespace

int main(std::int32_t argc, char** argv) {
    try {
        if (argc < 3) {
            std::fprintf(stderr,
                         "usage:\n"
                         "  mnet_gguf info MODEL.gguf [--tensors]\n"
                         "  mnet_gguf matmul MODEL.gguf TENSOR INPUT.bin OUTPUT.bin [--expert N]\n"
                         "  mnet_gguf matmul-q MODEL.gguf TENSOR INPUT.bin OUTPUT.bin [--expert N]\n"
                         "  mnet_gguf dequant-raw TYPE NELEM RAW.bin OUT.bin\n");
            return 2;
        }
        const std::string command = argv[1];
        if (command == "dequant-raw") {
            if (argc < 6) {
                std::fprintf(stderr, "usage: mnet_gguf dequant-raw TYPE NELEM RAW.bin OUT.bin\n");
                return 2;
            }
            const auto type = static_cast<std::uint32_t>(std::stoul(argv[2]));
            const auto count = static_cast<std::uint64_t>(std::stoull(argv[3]));
            std::ifstream raw_file(argv[4], std::ios::binary | std::ios::ate);
            if (!raw_file) {
                std::fprintf(stderr, "cannot open %s\n", argv[4]);
                return 2;
            }
            const std::streamsize size = raw_file.tellg();
            raw_file.seekg(0);
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(size));
            raw_file.read(reinterpret_cast<char*>(raw.data()), size);
            const auto values = mnet::gguf::Reader::dequantize_block(type, raw, count);
            write_floats(argv[5], values);
            std::printf("dequant-raw type=%u n=%" PRIu64 " -> %s\n", type, count, argv[5]);
            return 0;
        }
        mnet::gguf::Reader reader(argv[2]);
        if (command == "info") {
            print_info(reader, argc > 3 && std::string(argv[3]) == "--tensors");
            return 0;
        }
        if (command == "matmul") {
            if (argc < 6) {
                std::fprintf(stderr, "usage: mnet_gguf matmul MODEL.gguf TENSOR INPUT.bin OUTPUT.bin [--expert N]\n");
                return 2;
            }
            std::int32_t expert = -1;
            if (argc > 7 && std::string(argv[6]) == "--expert") {
                expert = std::stoi(argv[7]);
            }
            return run_matmul(reader, argv[3], argv[4], argv[5], expert);
        }
        if (command == "matmul-q") {
            if (argc < 6) {
                std::fprintf(stderr, "usage: mnet_gguf matmul-q MODEL.gguf TENSOR INPUT.bin OUTPUT.bin [--expert N]\n");
                return 2;
            }
            std::int32_t expert = -1;
            if (argc > 7 && std::string(argv[6]) == "--expert") {
                expert = std::stoi(argv[7]);
            }
            return run_matmul_quantized(reader, argv[3], argv[4], argv[5], expert);
        }
        if (command == "bonsai-ffn") {
            if (argc < 6) {
                std::fprintf(stderr, "usage: mnet_gguf bonsai-ffn MODEL.gguf LAYER INPUT.bin OUTPUT.bin\n");
                return 2;
            }
            return run_bonsai_ffn(reader, std::stoi(argv[3]), argv[4], argv[5]);
        }
        std::fprintf(stderr, "unknown command: %s\n", command.c_str());
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_gguf: %s\n", error.what());
        return 1;
    }
}
