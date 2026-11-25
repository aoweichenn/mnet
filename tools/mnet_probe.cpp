// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/net.hpp>

#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::int64_t read_vmhwm_kb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    std::int64_t value = 0;
    while (status >> key) {
        if (key == "VmHWM:") {
            status >> value;
            return value;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: mnet_probe model.param model.bin WxHxC [threads] [iters]\n");
        return 2;
    }
    const std::int32_t threads = argc > 4 ? std::atoi(argv[4]) : 4;
    const std::int32_t iters = argc > 5 ? std::atoi(argv[5]) : 5;

    std::int32_t w = 0;
    std::int32_t h = 0;
    std::int32_t c = 0;
    if (std::sscanf(argv[3], "%" SCNd32 "x%" SCNd32 "x%" SCNd32, &w, &h, &c) != 3 || w <= 0 || h <= 0 || c <= 0) {
        std::fprintf(stderr, "invalid shape %s\n", argv[3]);
        return 2;
    }

    mnet::Net net;
    net.opt().num_threads = threads;
    if (net.load(argv[1], argv[2]) != 0) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }

    const std::vector<std::string> inputs = net.input_names();
    const std::vector<std::string> outputs = net.output_names();
    if (inputs.empty() || outputs.empty()) {
        std::fprintf(stderr, "no inputs or outputs\n");
        return 1;
    }

    mnet::Mat input(w, h, c);
    for (std::int32_t q = 0; q < c; ++q) {
        float* p = input.channel(q);
        for (std::int32_t i = 0; i < w * h; ++i) {
            p[static_cast<std::size_t>(i)] = std::sin(0.01F * static_cast<float>(i + q * 17));
        }
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < iters; ++i) {
        mnet::Net::Extractor ex = net.create_extractor();
        for (const std::string& name : inputs) {
            if (ex.input(name.c_str(), input) != 0) {
                std::fprintf(stderr, "input %s failed\n", name.c_str());
                return 1;
            }
        }
        for (const std::string& name : outputs) {
            mnet::Mat out;
            if (ex.extract(name.c_str(), out) != 0) {
                std::fprintf(stderr, "extract %s failed\n", name.c_str());
                return 1;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    std::printf("model=%s threads=%" PRId32 " iters=%" PRId32 " avg_ms=%.3f peak_rss_mb=%.1f inputs=%zu outputs=%zu\n",
                argv[1], threads, iters, avg_ms, static_cast<double>(read_vmhwm_kb()) / 1024.0, inputs.size(),
                outputs.size());
    return 0;
}
