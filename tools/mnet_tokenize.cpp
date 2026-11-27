// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mnet/bpe.hpp>
#include <mnet/gguf.hpp>

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::fprintf(stderr, "usage: mnet_tokenize MODEL.gguf -p TEXT | -f FILE | --stdin [--decode IDS]\n");
            return 2;
        }
        mnet::gguf::Reader reader(argv[1]);
        const auto encoder = mnet::bpe::Encoder::from_reader(reader);
        std::string text;
        std::vector<std::int32_t> decode_ids;
        bool decode = false;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-p" && i + 1 < argc) {
                text = argv[++i];
            } else if (arg == "-f" && i + 1 < argc) {
                std::ifstream file(argv[++i], std::ios::binary);
                std::ostringstream buffer;
                buffer << file.rdbuf();
                text = buffer.str();
            } else if (arg == "--stdin") {
                std::ostringstream buffer;
                buffer << std::cin.rdbuf();
                text = buffer.str();
            } else if (arg == "--decode") {
                decode = true;
                while (i + 1 < argc) {
                    decode_ids.push_back(std::stoi(argv[++i]));
                }
            }
        }
        bool words = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--words") {
                words = true;
            }
        }
        if (words) {
            for (const auto& word : encoder.pre_tokenize(text)) {
                std::printf("[%s]", word.c_str());
            }
            std::printf("\n");
            return 0;
        }
        if (decode) {
            std::printf("%s\n", encoder.decode(decode_ids).c_str());
            return 0;
        }
        const auto ids = encoder.encode(text);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::printf("%s%d", i == 0 ? "" : " ", ids[i]);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_tokenize: %s\n", error.what());
        return 1;
    }
}
