// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Isolated stage test for the fused attention block, mirroring the FFN
// harness: run blk.0's attention with a warm KV cache once through the device
// block (Reader::attention) and once through the step-by-step reference, and
// write both stage sets to $MNET_ATTN_DEBUG_DIR. compare_attn_stages.py then
// reports the worst relative difference per stage.
//
// The reference matvec goes through Reader::matvec, so run without
// MNET_METAL_FAST to isolate the new kernels (norm, RoPE, f16 append, softmax,
// gate) from the matvec precision: in that mode the matvecs are bit-identical
// and any difference is the attention code. Re-run with MNET_METAL_FAST=1 for
// the combined error.
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/attn_block_test.cpp -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o attn_block_test

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

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kHeads = 24;
constexpr std::size_t kHeadsKv = 4;
constexpr std::size_t kEmbd = 5120;
constexpr std::size_t kRopeDims = 64;
constexpr std::size_t kHidden = kHeads * kHeadDim;
constexpr std::size_t kCacheTokens = 7;  // warm cache: the new token is the 8th

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

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + mnet::detail::ggml::expf(-value));
}

void rope_tables(std::size_t position, float base, std::vector<float>& cos_table, std::vector<float>& sin_table) {
    cos_table.resize(kRopeDims / 2);
    sin_table.resize(kRopeDims / 2);
    const float theta_scale = mnet::detail::ggml::powf(base, -2.0F / static_cast<float>(kRopeDims));
    float theta = static_cast<float>(position);
    for (std::size_t i = 0; i < kRopeDims / 2; ++i) {
        cos_table[i] = std::cos(theta);
        sin_table[i] = std::sin(theta);
        theta *= theta_scale;
    }
}

void write_stage(const std::string& dir, const char* name, const std::vector<float>& values) {
    if (std::FILE* file = std::fopen((dir + "/cpu-" + name).c_str(), "wb")) {
        std::fwrite(values.data(), sizeof(float), values.size(), file);
        std::fclose(file);
    }
}

[[nodiscard]] float lcg(std::uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
}

}  // namespace

int main() {
    const char* dir = std::getenv("MNET_ATTN_DEBUG_DIR");
    if (dir == nullptr) {
        std::fprintf(stderr, "set MNET_ATTN_DEBUG_DIR to a directory\n");
        return 2;
    }
    const std::string out_dir(dir);
    Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto& floats = reader.floats();
    const auto& ints = reader.integers();
    const auto eps = static_cast<float>(floats.at("qwen35.attention.layer_norm_rms_epsilon"));
    const auto rope_base = static_cast<float>(floats.at("qwen35.rope.freq_base"));
    const auto block_size = static_cast<std::uint32_t>(ints.at("prism.hadamard.block_size"));

    const auto q = reader.quantized(reader.tensor("blk.3.attn_q.weight"));
    const auto k = reader.quantized(reader.tensor("blk.3.attn_k.weight"));
    const auto v = reader.quantized(reader.tensor("blk.3.attn_v.weight"));
    const auto out = reader.quantized(reader.tensor("blk.3.attn_output.weight"));
    const auto q_norm = reader.dequantize(reader.tensor("blk.3.attn_q_norm.weight"));
    const auto k_norm = reader.dequantize(reader.tensor("blk.3.attn_k_norm.weight"));
    const auto signs_embd = reader.signs_for_width(static_cast<std::uint32_t>(kEmbd));
    const auto signs_value = reader.signs_for_width(static_cast<std::uint32_t>(kHidden));

    std::uint32_t state = 7U;
    std::vector<float> xn(kEmbd);
    for (auto& value : xn) {
        value = lcg(state) * 3.0F;
    }
    auto xh = xn;
    hadamard_forward(xh, signs_embd, block_size);

    // Warm cache: the same f16 rounding the reference uses on append.
    const std::size_t kv_elems = kCacheTokens * kHeadsKv * kHeadDim;
    std::vector<float> k_cache(kv_elems);
    std::vector<float> v_cache(kv_elems);
    for (std::size_t i = 0; i < kv_elems; ++i) {
        k_cache[i] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(lcg(state) * 3.0F));
        v_cache[i] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(lcg(state) * 3.0F));
    }
    const std::size_t position = kCacheTokens;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    rope_tables(position, rope_base, cos_table, sin_table);

    // ---- reference, step by step -------------------------------------------
    std::vector<float> qfull(q.rows);
    std::vector<float> kraw(k.rows);
    std::vector<float> vraw(v.rows);
    Reader::matvec(q, xh.data(), qfull.data());
    Reader::matvec(k, xh.data(), kraw.data());
    Reader::matvec(v, xh.data(), vraw.data());
    write_stage(out_dir, "qfull.f32", qfull);

    std::vector<float> qprep(kHidden);
    std::vector<float> gates(kHidden);
    for (std::size_t h = 0; h < kHeads; ++h) {
        std::copy(qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim),
                  qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim + kHeadDim),
                  qprep.begin() + static_cast<std::ptrdiff_t>(h * kHeadDim));
        std::copy(qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim + kHeadDim),
                  qfull.begin() + static_cast<std::ptrdiff_t>((h + 1) * 2 * kHeadDim),
                  gates.begin() + static_cast<std::ptrdiff_t>(h * kHeadDim));
    }
    for (std::size_t h = 0; h < kHeads; ++h) {
        float* head = qprep.data() + h * kHeadDim;
        double squares = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            squares += static_cast<double>(head[i] * head[i]);
        }
        const auto mean = static_cast<float>(squares / static_cast<double>(kHeadDim));
        const float scale = 1.0F / std::sqrt(mean + eps);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            head[i] = head[i] * scale * q_norm[i];
        }
    }
    for (std::size_t h = 0; h < kHeadsKv; ++h) {
        float* head = kraw.data() + h * kHeadDim;
        double squares = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            squares += static_cast<double>(head[i] * head[i]);
        }
        const auto mean = static_cast<float>(squares / static_cast<double>(kHeadDim));
        const float scale = 1.0F / std::sqrt(mean + eps);
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            head[i] = head[i] * scale * k_norm[i];
        }
    }
    for (std::size_t h = 0; h < kHeads; ++h) {
        float* head = qprep.data() + h * kHeadDim;
        for (std::size_t i = 0; i < kRopeDims / 2; ++i) {
            const float x0 = head[i];
            const float x1 = head[i + kRopeDims / 2];
            head[i] = x0 * cos_table[i] - x1 * sin_table[i];
            head[i + kRopeDims / 2] = x0 * sin_table[i] + x1 * cos_table[i];
        }
    }
    for (std::size_t h = 0; h < kHeadsKv; ++h) {
        float* head = kraw.data() + h * kHeadDim;
        for (std::size_t i = 0; i < kRopeDims / 2; ++i) {
            const float x0 = head[i];
            const float x1 = head[i + kRopeDims / 2];
            head[i] = x0 * cos_table[i] - x1 * sin_table[i];
            head[i + kRopeDims / 2] = x0 * sin_table[i] + x1 * cos_table[i];
        }
    }
    write_stage(out_dir, "qprep.f32", qprep);
    for (auto& value : gates) {
        value = sigmoid(value);
    }
    write_stage(out_dir, "gatesig.f32", gates);

    k_cache.resize(kv_elems + kHeadsKv * kHeadDim);
    v_cache.resize(kv_elems + kHeadsKv * kHeadDim);
    for (std::size_t j = 0; j < kHeadsKv * kHeadDim; ++j) {
        k_cache[kv_elems + j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(kraw[j]));
        v_cache[kv_elems + j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(vraw[j]));
    }
    write_stage(out_dir, "kcache.f32", k_cache);
    write_stage(out_dir, "vcache.f32", v_cache);

    const std::size_t tokens = kCacheTokens + 1;
    const std::size_t group = kHeads / kHeadsKv;
    std::vector<float> attn(kHidden);
    std::vector<float> vkq(kHeadDim);
    const float attn_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    for (std::size_t h = 0; h < kHeads; ++h) {
        const std::size_t kv_head = h / group;
        const float* qh = qprep.data() + h * kHeadDim;
        float sum = 0.0F;
        float max_score = -std::numeric_limits<float>::infinity();
        std::fill(vkq.begin(), vkq.end(), 0.0F);
        for (std::size_t p = 0; p < tokens; ++p) {
            const float* kh = k_cache.data() + (p * kHeadsKv + kv_head) * kHeadDim;
            const float* vh = v_cache.data() + (p * kHeadsKv + kv_head) * kHeadDim;
            const float score = mnet::detail::ggml::dot_f16_f32(kh, qh, kHeadDim) * attn_scale;
            float ms = 1.0F;
            float vs = 1.0F;
            if (score > max_score) {
                const float old_max = max_score;
                max_score = score;
                ms = mnet::detail::ggml::expf(old_max - max_score);
                for (std::size_t i = 0; i < kHeadDim; ++i) {
                    vkq[i] *= ms;
                }
            } else {
                vs = mnet::detail::ggml::expf(score - max_score);
            }
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                vkq[i] = std::fma(vh[i], vs, vkq[i]);
            }
            sum = std::fma(sum, ms, vs);
        }
        const float inv = sum == 0.0F ? 0.0F : 1.0F / sum;
        float* head = attn.data() + h * kHeadDim;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            head[i] = vkq[i] * inv;
        }
    }
    for (std::size_t i = 0; i < attn.size(); ++i) {
        attn[i] *= gates[i];
    }
    hadamard_forward(attn, signs_value, block_size);
    write_stage(out_dir, "attn.f32", attn);
    std::vector<float> projected(kEmbd);
    Reader::matvec(out, attn.data(), projected.data());
    write_stage(out_dir, "out.f32", projected);

    // ---- device block ------------------------------------------------------
    std::vector<float> device_result(kEmbd);
    std::vector<float> device_k(k_cache.begin(), k_cache.begin() + static_cast<std::ptrdiff_t>(kv_elems));
    std::vector<float> device_v(v_cache.begin(), v_cache.begin() + static_cast<std::ptrdiff_t>(kv_elems));
    Reader::AttentionParams params;
    params.input = xh.data();
    params.q_norm = q_norm.data();
    params.k_norm = k_norm.data();
    params.rope_cos = cos_table.data();
    params.rope_sin = sin_table.data();
    params.signs = signs_value.data();
    params.rope_dims = static_cast<std::uint32_t>(kRopeDims);
    params.head_count = static_cast<std::uint32_t>(kHeads);
    params.head_count_kv = static_cast<std::uint32_t>(kHeadsKv);
    params.head_dim = static_cast<std::uint32_t>(kHeadDim);
    params.block_size = block_size;
    params.cache_tokens = static_cast<std::uint32_t>(kCacheTokens);
    params.eps = eps;
    params.attention_scale = attn_scale;
    const bool used = Reader::attention(q, k, v, out, params, device_k.data(), device_v.data(), device_result.data());
    std::printf("device block: %s\n", used ? "used" : "REJECTED");
    if (!used) {
        return 1;
    }
    std::printf("cpu reference written to %s\n", out_dir.c_str());
    return 0;
}
