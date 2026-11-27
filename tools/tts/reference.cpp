// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <mnet/tts/tts_reference.hpp>
#include <mnet/tts/tts_tensor.hpp>

namespace mnet::tts {

void debug_dump(const std::string& name, const std::vector<float>& values) {
    if (g_debug_dir.empty()) {
        return;
    }
    std::ofstream file(g_debug_dir + "/" + name + ".bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void report(const Reference& ref, const std::string& label, const Act& act, const std::string& name, bool time_major) {
    const auto it = ref.tensors.find(label);
    if (it == ref.tensors.end()) {
        return;
    }
    if (it->second.size() != act.data.size()) {
        std::printf("  %-42s size %zu vs ref %zu\n", name.c_str(), act.data.size(), it->second.size());
        return;
    }
    double max_abs = 0.0;
    double sum = 0.0;
    std::size_t exact = 0;
    for (std::size_t i = 0; i < act.data.size(); ++i) {
        const std::size_t ref_index = time_major ? (i % act.length) * act.channels + (i / act.length) : i;
        const double d = std::fabs(static_cast<double>(act.data[i]) - static_cast<double>(it->second[ref_index]));
        max_abs = std::max(max_abs, d);
        sum += d;
        if (act.data[i] == it->second[ref_index]) {
            ++exact;
        }
    }
    std::printf("  %-42s max_abs %.3e  mean %.3e  exact %zu/%zu\n", name.c_str(), max_abs,
                sum / static_cast<double>(act.data.size()), exact, act.data.size());
}

void report_channels(const Reference& ref, const std::string& label, const Act& act, const std::string& name,
                     std::size_t channels) {
    const auto it = ref.tensors.find(label);
    if (it == ref.tensors.end()) {
        return;
    }
    const std::size_t count = std::min(act.data.size(), channels * act.length);
    double max_abs = 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double d = std::fabs(static_cast<double>(act.data[i]) - static_cast<double>(it->second[i]));
        max_abs = std::max(max_abs, d);
        sum += d;
    }
    std::printf("  %-42s max_abs %.3e  mean %.3e\n", name.c_str(), max_abs, sum / static_cast<double>(count));
}

bool write_wav(const std::string& path, const std::vector<float>& samples, std::uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot open %s for writing\n", path.c_str());
        return false;
    }
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
    const std::uint32_t byte_rate = sample_rate * 2;
    const std::uint16_t bits = 16;
    const std::uint16_t channels = 1;
    const auto put32 = [&out](std::uint32_t v) {
        out.put(static_cast<char>(v & 0xFFU));
        out.put(static_cast<char>((v >> 8U) & 0xFFU));
        out.put(static_cast<char>((v >> 16U) & 0xFFU));
        out.put(static_cast<char>((v >> 24U) & 0xFFU));
    };
    const auto put16 = [&out](std::uint16_t v) {
        out.put(static_cast<char>(v & 0xFFU));
        out.put(static_cast<char>((v >> 8U) & 0xFFU));
    };
    out.write("RIFF", 4);
    put32(36U + data_bytes);
    out.write("WAVEfmt ", 8);
    put32(16);
    put16(1);
    put16(channels);
    put32(sample_rate);
    put32(byte_rate);
    put16(2);
    put16(bits);
    out.write("data", 4);
    put32(data_bytes);
    for (const float value : samples) {
        const float clamped = std::min(std::max(value, -1.0F), 1.0F);
        const auto pcm = static_cast<std::int16_t>(std::lround(clamped * 32767.0F));
        put16(static_cast<std::uint16_t>(pcm));
    }
    return true;
}

}  // namespace mnet::tts
