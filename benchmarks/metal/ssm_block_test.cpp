// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Isolated test for the fused gated-delta (SSM) block: run blk.0 of a real
// model through both the device block (Reader::ssm_block) and a step-by-step
// reference for the same token sequence, then compare every stage of the last
// token plus the recurrent state both paths end up with.
//
// The block owns its state on the device and only seeds it from the caller, so
// the reference keeps the state on the host and the comparison of the two is
// part of the test: it is the only way to see that the delta-rule scan really
// tracks the reference over several tokens.
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/ssm_block_test.cpp -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o ssm_block_test

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <mnet/detail/ggml_kernels.hpp>
#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

namespace {

constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 48;
constexpr std::size_t kConvKernel = 4;
constexpr std::size_t kEmbd = 5120;
constexpr std::size_t kConvDim = 2 * kKeyHeads * kHeadDim + kValueHeads * kHeadDim;
constexpr std::size_t kHidden = kValueHeads * kHeadDim;
constexpr std::size_t kTokens = 6;  // several tokens so the recurrent state evolves

using mnet::gguf::Reader;

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

void hadamard_forward(std::vector<float>& values, const std::vector<std::int32_t>& signs, std::uint32_t block_size) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] *= static_cast<float>(signs[i]);
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(block_size));
    for (std::size_t start = 0; start < values.size(); start += block_size) {
        for (std::size_t i = 0; i < block_size; ++i) {
            values[start + i] *= scale;
        }
        fwht(values.data() + start, block_size);
    }
}

void l2_normalize(float* data, std::size_t size, float eps) {
    const float n = static_cast<float>(size);
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        squares += static_cast<double>(data[i] * data[i]);
    }
    const auto mean = static_cast<float>(squares / static_cast<double>(size));
    const float rms = 1.0F / std::sqrt(mean + eps / n);
    const float scale = 1.0F / std::sqrt(n);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = data[i] * rms * scale;
    }
}

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + mnet::detail::ggml::expf(-value));
}

[[nodiscard]] float softplus(float value) {
    return value > 20.0F ? value : mnet::detail::ggml::log1pf(mnet::detail::ggml::expf(value));
}

void write_stage(const std::string& dir, const char* name, const std::vector<float>& values) {
    if (std::FILE* file = std::fopen((dir + "/cpu-" + name).c_str(), "wb")) {
        std::fwrite(values.data(), sizeof(float), values.size(), file);
        std::fclose(file);
    }
}

struct Reference {
    std::vector<float> conv_state;  // (k - 1) * conv_dim
    std::vector<float> ssm_state;   // value_heads * head_dim^2
    std::vector<float> qkv;
    std::vector<float> z;
    std::vector<float> beta;
    std::vector<float> gate;
    std::vector<float> conv;
    std::vector<float> delta;
    std::vector<float> grouped;
    std::vector<float> projected;
    std::vector<float> window;  // the convolution window as it was used
};

// The step-by-step block, copied from the model so both sides compute the same
// quantities in the same order.
void reference_token(const Reader::Quantized& qkv_w, const Reader::Quantized& gate_w, const Reader::Quantized& out_w,
                     const std::vector<float>& beta_w, const std::vector<float>& alpha_w,
                     const std::vector<float>& dt, const std::vector<float>& ssm_a,
                     const std::vector<float>& conv_w, const std::vector<float>& ssm_norm,
                     const std::vector<std::int32_t>& signs, float eps, std::uint32_t block_size,
                     const std::vector<float>& xn, Reference& ref) {
    auto xh = xn;
    hadamard_forward(xh, signs, block_size);
    ref.qkv.assign(qkv_w.rows, 0.0F);
    ref.z.assign(gate_w.rows, 0.0F);
    Reader::matvec(qkv_w, xh.data(), ref.qkv.data());
    Reader::matvec(gate_w, xh.data(), ref.z.data());

    std::vector<float> beta_raw(kValueHeads);
    std::vector<float> alpha_raw(kValueHeads);
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        beta_raw[h] = mnet::detail::ggml::dot_f32(beta_w.data() + h * kEmbd, xn.data(), kEmbd);
        alpha_raw[h] = mnet::detail::ggml::dot_f32(alpha_w.data() + h * kEmbd, xn.data(), kEmbd);
    }
    ref.beta.resize(kValueHeads);
    ref.gate.resize(kValueHeads);
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        ref.beta[h] = sigmoid(beta_raw[h]);
        ref.gate[h] = softplus(alpha_raw[h] + dt[h]) * ssm_a[h];
    }

    std::vector<float> window((kConvKernel - 1) * kConvDim + kConvDim);
    std::copy(ref.conv_state.begin(), ref.conv_state.end(), window.begin());
    std::copy(ref.qkv.begin(), ref.qkv.end(), window.begin() + static_cast<std::ptrdiff_t>((kConvKernel - 1) * kConvDim));
    ref.window = window;
    ref.conv.assign(kConvDim, 0.0F);
    for (std::size_t c = 0; c < kConvDim; ++c) {
        float sum = 0.0F;
        for (std::size_t k = 0; k < kConvKernel; ++k) {
            const volatile float product = conv_w[c * kConvKernel + k] * window[k * kConvDim + c];
            sum += product;
        }
        ref.conv[c] = sum;
    }
    mnet::detail::ggml::silu_inplace(ref.conv.data(), ref.conv.size());
    std::copy(window.begin() + static_cast<std::ptrdiff_t>(kConvDim),
              window.begin() + static_cast<std::ptrdiff_t>((kConvKernel - 1) * kConvDim + kConvDim),
              ref.conv_state.begin());

    float* q = ref.conv.data();
    float* k = ref.conv.data() + kKeyHeads * kHeadDim;
    float* v = ref.conv.data() + 2 * kKeyHeads * kHeadDim;
    for (std::size_t h = 0; h < kKeyHeads; ++h) {
        l2_normalize(q + h * kHeadDim, kHeadDim, eps);
        l2_normalize(k + h * kHeadDim, kHeadDim, eps);
    }

    ref.delta.assign(kHidden, 0.0F);
    const float inv_dim = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        const std::size_t key_head = h % kKeyHeads;
        const float* qh = q + key_head * kHeadDim;
        const float* kh = k + key_head * kHeadDim;
        const float* vh = v + h * kHeadDim;
        float* sh = ref.ssm_state.data() + h * kHeadDim * kHeadDim;
        const float decay = mnet::detail::ggml::expf(ref.gate[h]);
        for (std::size_t i = 0; i < kHeadDim * kHeadDim; ++i) {
            sh[i] *= decay;
        }
        float d[kHeadDim] = {};
        for (std::size_t j = 0; j < kHeadDim; ++j) {
            const float* row = sh + j * kHeadDim;
            const float sum = mnet::detail::ggml::dot_f32(row, kh, kHeadDim);
            d[j] = (vh[j] - sum) * ref.beta[h];
        }
        for (std::size_t j = 0; j < kHeadDim; ++j) {
            float* row = sh + j * kHeadDim;
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                row[i] += d[j] * kh[i];
            }
        }
        for (std::size_t j = 0; j < kHeadDim; ++j) {
            const float* row = sh + j * kHeadDim;
            ref.delta[h * kHeadDim + j] = mnet::detail::ggml::dot_f32(row, qh, kHeadDim) * inv_dim;
        }
    }

    auto z_act = ref.z;
    mnet::detail::ggml::silu_inplace(z_act.data(), z_act.size());
    std::vector<float> normed(kHidden);
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        const float* dh = ref.delta.data() + h * kHeadDim;
        const float* zh = z_act.data() + h * kHeadDim;
        double squares = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            squares += static_cast<double>(dh[i] * dh[i]);
        }
        const auto mean = static_cast<float>(squares / static_cast<double>(kHeadDim));
        const float scale = 1.0F / std::sqrt(mean + eps);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            normed[h * kHeadDim + i] = (dh[i] * scale * ssm_norm[i]) * zh[i];
        }
    }
    ref.grouped.assign(kHidden, 0.0F);
    const std::size_t heads_per_key = kValueHeads / kKeyHeads;
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        const std::size_t key_head = h % kKeyHeads;
        const std::size_t rep = h / kKeyHeads;
        std::copy(normed.begin() + static_cast<std::ptrdiff_t>(h * kHeadDim),
                  normed.begin() + static_cast<std::ptrdiff_t>((h + 1) * kHeadDim),
                  ref.grouped.begin() + static_cast<std::ptrdiff_t>((rep + heads_per_key * key_head) * kHeadDim));
    }
    hadamard_forward(ref.grouped, signs, block_size);
    ref.projected.assign(kEmbd, 0.0F);
    Reader::matvec(out_w, ref.grouped.data(), ref.projected.data());
}

[[nodiscard]] float lcg(std::uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
}

[[nodiscard]] double normalized_diff(const std::vector<float>& a, const std::vector<float>& b) {
    const double scale = std::max(1e-30, static_cast<double>(*std::max_element(a.begin(), a.end())));
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) / scale);
    }
    return worst;
}

}  // namespace

int main() {
    const char* dir = std::getenv("MNET_SSM_DEBUG_DIR");
    if (dir == nullptr) {
        std::fprintf(stderr, "set MNET_SSM_DEBUG_DIR to a directory\n");
        return 2;
    }
    const std::string out_dir(dir);
    Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto& floats = reader.floats();
    const auto& ints = reader.integers();
    const auto eps = static_cast<float>(floats.at("qwen35.attention.layer_norm_rms_epsilon"));
    const auto block_size = static_cast<std::uint32_t>(ints.at("prism.hadamard.block_size"));

    const auto qkv_w = reader.quantized(reader.tensor("blk.0.attn_qkv.weight"));
    const auto gate_w = reader.quantized(reader.tensor("blk.0.attn_gate.weight"));
    const auto out_w = reader.quantized(reader.tensor("blk.0.ssm_out.weight"));
    const auto beta_w = reader.dequantize(reader.tensor("blk.0.ssm_beta.weight"));
    const auto alpha_w = reader.dequantize(reader.tensor("blk.0.ssm_alpha.weight"));
    const auto dt = reader.dequantize(reader.tensor("blk.0.ssm_dt.bias"));
    const auto ssm_a = reader.dequantize(reader.tensor("blk.0.ssm_a"));
    const auto conv_w = reader.dequantize(reader.tensor("blk.0.ssm_conv1d.weight"));
    const auto ssm_norm = reader.dequantize(reader.tensor("blk.0.ssm_norm.weight"));
    const auto signs = reader.signs_for_width(static_cast<std::uint32_t>(kHidden));

    Reference ref;
    ref.conv_state.assign((kConvKernel - 1) * kConvDim, 0.0F);
    ref.ssm_state.assign(kValueHeads * kHeadDim * kHeadDim, 0.0F);
    std::vector<float> device_state(kValueHeads * kHeadDim * kHeadDim, 0.0F);
    std::vector<float> device_conv_state((kConvKernel - 1) * kConvDim, 0.0F);

    std::uint32_t state = 11U;
    std::vector<float> xn(kEmbd);
    double worst_output = 0.0;
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (auto& value : xn) {
            value = lcg(state) * 2.0F;
        }
        reference_token(qkv_w, gate_w, out_w, beta_w, alpha_w, dt, ssm_a, conv_w, ssm_norm, signs, eps, block_size, xn,
                        ref);

        auto device_input = xn;
        hadamard_forward(device_input, signs, block_size);
        std::vector<float> device_out(kEmbd);
        Reader::SsmParams params;
        params.input = device_input.data();
        params.raw_input = xn.data();
        params.beta_w = beta_w.data();
        params.alpha_w = alpha_w.data();
        params.dt = dt.data();
        params.ssm_a = ssm_a.data();
        params.conv_w = conv_w.data();
        params.ssm_norm = ssm_norm.data();
        params.signs = signs.data();
        params.conv_state = device_conv_state.data();
        params.ssm_state = device_state.data();
        params.key_heads = static_cast<std::uint32_t>(kKeyHeads);
        params.value_heads = static_cast<std::uint32_t>(kValueHeads);
        params.head_dim = static_cast<std::uint32_t>(kHeadDim);
        params.conv_kernel = static_cast<std::uint32_t>(kConvKernel);
        params.block_size = block_size;
        params.eps = eps;
        if (!Reader::ssm_block(qkv_w, gate_w, out_w, params, device_out.data())) {
            std::fprintf(stderr, "device block REJECTED on token %zu\n", t);
            return 1;
        }
        const double diff = normalized_diff(ref.projected, device_out);
        worst_output = std::max(worst_output, diff);
        std::printf("token %zu: projected worst rel %.3e\n", t, diff);
    }

    write_stage(out_dir, "qkv.f32", ref.qkv);
    write_stage(out_dir, "z.f32", ref.z);
    write_stage(out_dir, "beta.f32", ref.beta);
    write_stage(out_dir, "gate.f32", ref.gate);
    write_stage(out_dir, "conv.f32", ref.conv);
    write_stage(out_dir, "delta.f32", ref.delta);
    write_stage(out_dir, "grouped.f32", ref.grouped);
    write_stage(out_dir, "out.f32", ref.projected);
    // The device keeps the state transposed within a head; undo that so the
    // two dumps have the same layout.
    std::vector<float> transposed(ref.ssm_state.size());
    for (std::size_t h = 0; h < kValueHeads; ++h) {
        for (std::size_t j = 0; j < kHeadDim; ++j) {
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                transposed[h * kHeadDim * kHeadDim + j * kHeadDim + i] =
                    ref.ssm_state[h * kHeadDim * kHeadDim + i * kHeadDim + j];
            }
        }
    }
    write_stage(out_dir, "state.f32", transposed);
    write_stage(out_dir, "window.f32", ref.window);
    std::printf("worst projected rel over %zu tokens: %.3e\n", kTokens, worst_output);
    std::printf("cpu reference written to %s\n", out_dir.c_str());
    return worst_output < 1.0e-2 ? 0 : 3;
}
