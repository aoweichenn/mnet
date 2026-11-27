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
#include <algorithm>
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
    long long max_mem_mb = 0;
    std::string dump_out;
    std::string extract_name;
    // --input NAME FILE WxHxC, repeatable: feed real data instead of the sine fill.
    struct InputOverride {
        std::string name;
        std::string path;
        mnet::Mat blob;
    };
    std::vector<InputOverride> overrides;
    std::vector<const char*> positional;
    for (std::int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--allow-empty") == 0) {
            allow_empty = true;
        } else if (std::strcmp(argv[i], "--max-mem-mb") == 0 && i + 1 < argc) {
            max_mem_mb = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--dump-out") == 0 && i + 1 < argc) {
            dump_out = argv[++i];
        } else if (std::strcmp(argv[i], "--extract") == 0 && i + 1 < argc) {
            extract_name = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 3 < argc) {
            InputOverride ovr;
            ovr.name = argv[i + 1];
            ovr.path = argv[i + 2];
            std::int32_t ow = 0;
            std::int32_t oh = 0;
            std::int32_t oc = 0;
            if (std::sscanf(argv[i + 3], "%" SCNd32 "x%" SCNd32 "x%" SCNd32, &ow, &oh, &oc) != 3 || ow <= 0 ||
                oh <= 0 || oc <= 0) {
                std::fprintf(stderr, "invalid --input shape %s\n", argv[i + 3]);
                return 2;
            }
            std::ifstream f(ovr.path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "cannot open %s\n", ovr.path.c_str());
                return 2;
            }
            ovr.blob.create(ow, oh, oc);
            // Rows are copied one at a time: ncnn aligns each channel to 16
            // bytes, so a flat copy of w*h*c floats lands at the wrong offsets
            // whenever w*h is not a multiple of 4.
            for (std::int32_t q = 0; q < oc; ++q) {
                mnet::Mat plane = ovr.blob.channel(q);
                for (std::int32_t y = 0; y < oh; ++y) {
                    f.read(reinterpret_cast<char*>(plane.row(y)),
                           static_cast<std::streamsize>(static_cast<std::size_t>(ow) * sizeof(float)));
                    if (f.gcount() != static_cast<std::streamsize>(static_cast<std::size_t>(ow) * sizeof(float))) {
                        std::fprintf(stderr, "%s: short read\n", ovr.path.c_str());
                        return 2;
                    }
                }
            }
            overrides.push_back(ovr);
            i += 3;
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
    // Dumping/extracting needs every blob alive: light mode frees a blob after
    // its recorded last use, which mis-handles inputs shared by later layers.
    net.opt().use_light_mode = false;
    if (max_mem_mb > 0) {
        net.set_memory_budget_bytes(static_cast<std::size_t>(max_mem_mb) * 1024U * 1024U);
    }
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
            const auto it = std::find_if(overrides.begin(), overrides.end(),
                                         [&](const InputOverride& o) { return o.name == name; });
            if (it != overrides.end()) {
                if (ex.input(name.c_str(), it->blob) != 0) {
                    std::fprintf(stderr, "input %s failed\n", name.c_str());
                    return 1;
                }
                continue;
            }
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
        std::vector<std::string> outputs_wanted = outputs;
        if (!extract_name.empty()) {
            outputs_wanted = {extract_name};
        }
        for (const std::string& name : outputs_wanted) {
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
        // Rows are written one at a time: ncnn aligns each channel to 16 bytes,
        // so a flat copy of `data` would include the padding.
        const std::int32_t depth = last_out.d > 0 ? last_out.d : 1;
        const std::size_t count = static_cast<std::size_t>(last_out.w) * static_cast<std::size_t>(last_out.h) *
                                  static_cast<std::size_t>(depth) * static_cast<std::size_t>(last_out.c);
        for (std::int32_t q = 0; q < last_out.c; ++q) {
            const mnet::Mat plane = last_out.channel(q);
            for (std::int32_t z = 0; z < depth; ++z) {
                const mnet::Mat slice = plane.depth(z);
                for (std::int32_t y = 0; y < last_out.h; ++y) {
                    dump.write(reinterpret_cast<const char*>(slice.row(y)),
                               static_cast<std::streamsize>(static_cast<std::size_t>(last_out.w) * sizeof(float)));
                }
            }
        }
        // A released or uninitialised blob can carry garbage dimensions; a
        // 14 GiB dump from a bogus w once filled the disk, so refuse them.
        constexpr std::int64_t kMaxDumpFloats = 1LL << 28;   // 1 GiB of floats
        if (last_out.w <= 0 || last_out.h < 0 || last_out.c <= 0 ||
            static_cast<std::int64_t>(last_out.w) * last_out.h * (last_out.d > 0 ? last_out.d : 1) * last_out.c > kMaxDumpFloats) {
            std::fprintf(stderr, "refusing to dump implausible blob %s w=%d h=%d c=%d\n", last_out_name.c_str(),
                         last_out.w, last_out.h, last_out.c);
            return 1;
        }
        std::printf("dump=%s blob=%s w=%" PRId32 " h=%" PRId32 " c=%" PRId32 " floats=%zu\n", dump_out.c_str(),
                    last_out_name.c_str(), last_out.w, last_out.h, last_out.c, count);
    }

    std::printf("model=%s threads=%" PRId32 " iters=%" PRId32
                " avg_ms=%.3f cpu_ms=%.3f peak_rss_mb=%.1f mem_peak_mb=%.1f mem_budget_mb=%zu budget_hit=%d"
                " cpus=%" PRId32 " big=%" PRId32 " inputs=%zu outputs=%zu\n",
                arg_base[0], threads, iters, avg_ms, cpu_ms,
                static_cast<double>(read_vmhwm_kb()) / 1024.0,
                static_cast<double>(net.memory_peak_bytes()) / (1024.0 * 1024.0),
                net.memory_budget_bytes() / (1024U * 1024U), net.memory_budget_exhausted() ? 1 : 0,
                mplatform::get_cpu_count(),
                mplatform::get_big_cpu_count(), inputs.size(), outputs.size());
    return 0;
}
