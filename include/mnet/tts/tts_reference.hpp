#pragma once

// Reference-comparison helpers shared by the TTS runners: loading the pinned
// per-stage dumps, dumping intermediate tensors, and writing 16-bit PCM WAV.

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

#include <mnet/tts/tts_tensor.hpp>

namespace mnet::tts {

inline std::string g_debug_dir;

void debug_dump(const std::string& name, const std::vector<float>& values);

struct Reference {
    std::map<std::string, std::vector<float>> tensors;

    void load(const std::string& dir) {
        std::ifstream manifest(dir + "/manifest.txt");
        if (!manifest) {
            return;
        }
        std::string line;
        while (std::getline(manifest, line)) {
            std::size_t space = line.find(' ');
            if (space == std::string::npos) {
                continue;
            }
            const std::string label = line.substr(0, space);
            if (label == "ids" || label == "length") {
                continue;
            }
            std::ifstream file(dir + "/" + label + ".bin", std::ios::binary | std::ios::ate);
            if (!file) {
                continue;
            }
            const auto bytes = static_cast<std::size_t>(file.tellg());
            file.seekg(0);
            std::vector<float>& values = this->tensors[label];
            values.resize(bytes / sizeof(float));
            file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
        }
    }
};

// Some reference dumps are stored time-major ([1, T, C], the view the operator
// works on inside ONNX); the runner keeps activations channel-major.
void report(const Reference& ref, const std::string& label, const Act& act, const std::string& name,
            bool time_major = false);

// Like report(), but only the first `channels` channels (used for the harmonic
// spectrogram, whose phase in near-silent bins is defined by the reference
// FFT's rounding).
void report_channels(const Reference& ref, const std::string& label, const Act& act, const std::string& name,
                     std::size_t channels);

// 16-bit PCM mono WAV writer.
bool write_wav(const std::string& path, const std::vector<float>& samples, std::uint32_t sample_rate);

}  // namespace mnet::tts
