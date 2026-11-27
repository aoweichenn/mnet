// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Does the device layer preparation (RMSNorm -> signed Hadamard -> Q8_0)
// reproduce the host's Q8_0 codes? The host RMSNorm sums the squares in double
// and Apple GPUs have no fp64, so the float vectors differ by ~1e-7; the
// question is whether the requantisation absorbs it, which decides whether the
// device-resident residual stream can stay bit-identical or only
// tolerance-close.
//
// Measured on the M4 Pro, blk.0.post_attention_norm, 5120 values, block 1024:
//   q8 codes differing 0 of 5120, worst relative block-scale diff 0.000e+00
// so the requantisation absorbs the fp32 norm completely: the layer pipeline can
// move onto the device without changing a single code the matvecs consume.
//
// Build on the Mac from the repository root:
//   clang++ -std=c++20 -O2 -I include -isysroot \
//     /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/layer_prep_test.cpp -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o layer_prep_test
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <mnet/detail/compute.hpp>
#include <mnet/detail/ggml_kernels.hpp>
#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

namespace {
std::vector<float> rms_norm(const float* input, std::size_t size, const std::vector<float>& weight, float eps) {
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        squares += static_cast<double>(input[i] * input[i]);
    }
    const auto mean = static_cast<float>(squares / static_cast<double>(size));
    const float scale = 1.0F / std::sqrt(mean + eps);
    std::vector<float> output(size);
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
    return output;
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
}  // namespace

int main() {
    using namespace mnet;
    gguf::Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto& floats = reader.floats();
    const auto eps = static_cast<float>(floats.at("qwen35.attention.layer_norm_rms_epsilon"));
    const auto block = static_cast<std::uint32_t>(reader.integers().at("prism.hadamard.block_size"));
    const auto weight = reader.dequantize(reader.tensor("blk.0.post_attention_norm.weight"));
    const std::size_t size = weight.size();
    const auto signs = reader.signs_for_width(static_cast<std::uint32_t>(size));
    std::vector<float> residual(size);
    std::uint32_t state = 11U;
    for (auto& value : residual) {
        state = state * 1664525U + 1013904223U;
        value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 6.0F;
    }

    auto normed = rms_norm(residual.data(), size, weight, eps);
    for (std::size_t i = 0; i < size; ++i) {
        normed[i] *= static_cast<float>(signs[i]);
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(block));
    for (auto& value : normed) {
        value *= scale;
    }
    for (std::size_t start = 0; start < size; start += block) {
        fwht(normed.data() + start, block);
    }
    const auto host = gguf::make_q8_0_blocks(normed.data(), size);

    std::vector<std::int8_t> device_qs(size);
    std::vector<float> device_scales(size / 32U);
    const bool ok = detail::compute::layer_prep_f32(residual.data(), weight.data(), size, eps, signs.data(), block,
                                                    device_qs.data(), device_scales.data());
    std::printf("device layer_prep: %s\n", ok ? "ok" : "REJECTED");
    if (!ok) {
        return 1;
    }
    std::size_t code_diff = 0;
    double worst_scale = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        if (device_qs[i] != host.qs[i]) {
            ++code_diff;
        }
    }
    for (std::size_t i = 0; i < size / 32U; ++i) {
        worst_scale = std::max(worst_scale, std::abs(static_cast<double>(device_scales[i]) - host.d[i]) /
                                                std::max(1e-30, std::abs(static_cast<double>(host.d[i]))));
    }
    std::printf("q8 codes differing %zu of %zu, worst relative block scale diff %.3e\n", code_diff, size,
                worst_scale);
    return code_diff == 0 ? 0 : 3;
}
