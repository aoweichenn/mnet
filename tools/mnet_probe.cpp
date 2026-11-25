// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/net.hpp>

#include <mplatform/cpu.hpp>

#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static std::int64_t process_cpu_us() {
    FILETIME created;
    FILETIME exited;
    FILETIME kernel;
    FILETIME user;
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) == 0) {
        return 0;
    }
    ULARGE_INTEGER k;
    ULARGE_INTEGER u;
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return static_cast<std::int64_t>((k.QuadPart + u.QuadPart) / 10);
}
#else
#include <sys/resource.h>

static std::int64_t process_cpu_us() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<std::int64_t>(usage.ru_utime.tv_sec) * 1000000 + usage.ru_utime.tv_usec +
           static_cast<std::int64_t>(usage.ru_stime.tv_sec) * 1000000 + usage.ru_stime.tv_usec;
}
#endif

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
    bool allow_empty = false;
    std::string dump_out;
    std::vector<const char*> positional;
    for (std::int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--allow-empty") == 0) {
            allow_empty = true;
        } else if (std::strcmp(argv[i], "--dump-out") == 0 && i + 1 < argc) {
            dump_out = argv[++i];
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() < 3) {
        std::fprintf(stderr,
                     "usage: mnet_probe [--allow-empty] [--dump-out FILE] model.param model.bin WxHxC "
                     "[threads] [iters]\n");
        return 2;
    }
    const char* const* arg_base = positional.data();
    const std::int32_t arg_count = static_cast<std::int32_t>(positional.size());
    const std::int32_t threads = arg_count > 3 ? std::atoi(arg_base[3]) : 4;
    const std::int32_t iters = arg_count > 4 ? std::atoi(arg_base[4]) : 5;

    std::int32_t w = 0;
    std::int32_t h = 0;
    std::int32_t c = 0;
    if (std::sscanf(arg_base[2], "%" SCNd32 "x%" SCNd32 "x%" SCNd32, &w, &h, &c) != 3 || w <= 0 || h <= 0 ||
        c <= 0) {
        std::fprintf(stderr, "invalid shape %s\n", arg_base[2]);
        return 2;
    }

    mnet::Net net;
    net.opt().num_threads = threads;
    if (net.load(arg_base[0], arg_base[1]) != 0) {
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

    mnet::Mat last_out;
    std::string last_out_name;

    const std::int64_t cpu_us_start = process_cpu_us();

    const auto start = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < iters; ++i) {
        mnet::Net::Extractor ex = net.create_extractor();
        for (const std::string& name : inputs) {
            if (name == "im_info") {
                mnet::Mat im_info(3);
                im_info[0] = static_cast<float>(h);
                im_info[1] = static_cast<float>(w);
                im_info[2] = 1.0F;
                if (ex.input(name.c_str(), im_info) != 0) {
                    std::fprintf(stderr, "input %s failed\n", name.c_str());
                    return 1;
                }
                continue;
            }
            if (ex.input(name.c_str(), input) != 0) {
                std::fprintf(stderr, "input %s failed\n", name.c_str());
                return 1;
            }
        }
        for (const std::string& name : outputs) {
            mnet::Mat out;
            const std::int32_t ret = ex.extract(name.c_str(), out);
            if (ret == -100 && allow_empty) {
                continue;
            }
            if (ret != 0) {
                std::fprintf(stderr, "extract %s failed\n", name.c_str());
                return 1;
            }
            if (!dump_out.empty()) {
                last_out = out;
                last_out_name = name;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    const double cpu_ms = static_cast<double>(process_cpu_us() - cpu_us_start) / 1000.0 / iters;
    if (!dump_out.empty() && !last_out.empty()) {
        std::ofstream dump(dump_out, std::ios::binary);
        const std::size_t count = static_cast<std::size_t>(last_out.w) * static_cast<std::size_t>(last_out.h) *
                                  static_cast<std::size_t>(last_out.c);
        dump.write(reinterpret_cast<const char*>(last_out.data), static_cast<std::streamsize>(count * sizeof(float)));
        std::printf("dump=%s blob=%s w=%" PRId32 " h=%" PRId32 " c=%" PRId32 " floats=%zu\n", dump_out.c_str(),
                    last_out_name.c_str(), last_out.w, last_out.h, last_out.c, count);
    }

    std::printf("model=%s threads=%" PRId32 " iters=%" PRId32
                " avg_ms=%.3f cpu_ms=%.3f peak_rss_mb=%.1f cpus=%" PRId32 " big=%" PRId32
                " inputs=%zu outputs=%zu\n",
                arg_base[0], threads, iters, avg_ms, cpu_ms,
                static_cast<double>(read_vmhwm_kb()) / 1024.0, mplatform::get_cpu_count(),
                mplatform::get_big_cpu_count(), inputs.size(), outputs.size());
    return 0;
}
