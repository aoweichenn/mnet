// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Native Bonsai (qwen35) decoder layer for a single sequence with zero initial
// conv/SSM state: gated delta net + gated norm + SWIGLU FFN, all weights read
// directly from the GGUF. This is a pure-mnet primitive, not a llama.cpp call.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
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

[[nodiscard]] std::vector<float> tensor(const mnet::gguf::Reader& reader, const std::string& name) {
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
            throw std::runtime_error("Hadamard block size does not divide the width");
        }
        fwht(values.data() + start, block_size);
        for (std::size_t i = 0; i < block_size; ++i) {
            values[start + i] *= scale;
        }
    }
}

[[nodiscard]] std::vector<float> rms_norm(const std::vector<float>& input, const std::vector<float>& weight,
                                          float eps) {
    double squares = 0.0;
    for (const float value : input) {
        squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto scale =
        static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(input.size()) + static_cast<double>(eps)));
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * weight[i];
    }
    return output;
}

[[nodiscard]] float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

[[nodiscard]] float softplus(float value) {
    return value > 20.0F ? value : std::log1p(std::exp(value));
}

void l2_normalize_head(float* data, std::size_t size, float eps) {
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        squares += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(squares + static_cast<double>(eps)));
    for (std::size_t i = 0; i < size; ++i) {
        data[i] *= scale;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::fprintf(stderr, "usage: mnet_bonsai_block MODEL.gguf LAYER INPUT.bin OUTPUT.bin\n");
            return 2;
        }
        const std::string model_path = argv[1];
        const auto layer = std::stoi(argv[2]);
        const std::string input_path = argv[3];
        const std::string output_path = argv[4];
        mnet::gguf::Reader reader(model_path);
        const std::string prefix = "blk." + std::to_string(layer) + ".";

        const auto& qkv_info = reader.tensor(prefix + "attn_qkv.weight");
        const auto in_dim = static_cast<std::size_t>(qkv_info.dims[0]);
        const auto qkv_dim = static_cast<std::size_t>(qkv_info.dims[1]);
        const auto& gate_info = reader.tensor(prefix + "attn_gate.weight");
        const auto z_dim = static_cast<std::size_t>(gate_info.dims[1]);
        const auto& conv_info = reader.tensor(prefix + "ssm_conv1d.weight");
        const auto conv_kernel = static_cast<std::size_t>(conv_info.dims[0]);
        const auto conv_channels = static_cast<std::size_t>(conv_info.dims[1]);

        std::vector<float> input = read_floats(input_path);
        if (input.size() % in_dim != 0) {
            std::fprintf(stderr, "input size %zu is not a multiple of %zu\n", input.size(), in_dim);
            return 2;
        }
        const std::size_t tokens = input.size() / in_dim;

        const std::uint32_t block_size =
            reader.integers().contains("prism.hadamard.block_size")
                ? static_cast<std::uint32_t>(reader.integers().at("prism.hadamard.block_size"))
                : 0U;
        const float eps = reader.floats().contains("qwen35.attention.layer_norm_rms_epsilon")
                              ? static_cast<float>(reader.floats().at("qwen35.attention.layer_norm_rms_epsilon"))
                              : 1.0e-6F;
        const auto signs_in = reader.signs_for_width(static_cast<std::uint32_t>(in_dim));
        const auto signs_z = reader.signs_for_width(static_cast<std::uint32_t>(z_dim));
        std::printf("bonsai-block layer=%d tokens=%zu in=%zu qkv=%zu z=%zu block=%u eps=%.3g\n", layer, tokens, in_dim,
                    qkv_dim, z_dim, block_size, static_cast<double>(eps));

        const auto start = std::chrono::steady_clock::now();
        const auto attn_norm = tensor(reader, prefix + "attn_norm.weight");
        const auto post_norm = tensor(reader, prefix + "post_attention_norm.weight");
        const auto wqkv = reader.quantized(reader.tensor(prefix + "attn_qkv.weight"));
        const auto wgate_q = reader.quantized(reader.tensor(prefix + "attn_gate.weight"));
        const auto wbeta = reader.quantized(reader.tensor(prefix + "ssm_beta.weight"));
        const auto walpha = reader.quantized(reader.tensor(prefix + "ssm_alpha.weight"));
        const auto dt = tensor(reader, prefix + "ssm_dt.bias");
        const auto ssm_a = tensor(reader, prefix + "ssm_a");
        const auto conv_w = tensor(reader, prefix + "ssm_conv1d.weight");
        const auto ssm_norm = tensor(reader, prefix + "ssm_norm.weight");
        const auto wssm_out = reader.quantized(reader.tensor(prefix + "ssm_out.weight"));
        const auto ffn_gate = reader.quantized(reader.tensor(prefix + "ffn_gate.weight"));
        const auto ffn_up = reader.quantized(reader.tensor(prefix + "ffn_up.weight"));
        const auto ffn_down = reader.quantized(reader.tensor(prefix + "ffn_down.weight"));
        const auto& ffn_gate_info = reader.tensor(prefix + "ffn_gate.weight");
        const auto hidden = static_cast<std::size_t>(ffn_gate_info.dims[1]);
        const auto signs_hidden = reader.signs_for_width(static_cast<std::uint32_t>(hidden));
        const std::size_t quantized_bytes = wqkv.blocks.size() + wgate_q.blocks.size() + wssm_out.blocks.size() +
                                            ffn_gate.blocks.size() + ffn_up.blocks.size() + ffn_down.blocks.size();
        std::printf("quantized weights: %zu bytes (fp32 vector would be %zu bytes)\n", quantized_bytes,
                    (qkv_dim * in_dim + z_dim * in_dim + in_dim * z_dim + 3 * hidden * in_dim) * sizeof(float));

        const std::size_t key_dim = 128 * 16;
        const std::size_t value_dim = z_dim;
        const std::size_t head_dim = 128;
        const std::size_t k_heads = key_dim / head_dim;
        const std::size_t v_heads = value_dim / head_dim;

        // Normalized input and its Hadamard-rotated form (qkv and gate share it).
        std::vector<float> xn(tokens * in_dim);
        std::vector<float> rotated(tokens * in_dim);
        for (std::size_t t = 0; t < tokens; ++t) {
            std::vector<float> row(input.begin() + static_cast<std::ptrdiff_t>(t * in_dim),
                                   input.begin() + static_cast<std::ptrdiff_t>((t + 1) * in_dim));
            row = rms_norm(row, attn_norm, eps);
            std::copy(row.begin(), row.end(), xn.begin() + static_cast<std::ptrdiff_t>(t * in_dim));
            hadamard_sign(row, signs_in, block_size);
            std::copy(row.begin(), row.end(), rotated.begin() + static_cast<std::ptrdiff_t>(t * in_dim));
        }

        // qkv, z, beta, alpha, gate.
        std::vector<float> qkv(tokens * qkv_dim);
        std::vector<float> z(tokens * z_dim);
        std::vector<float> beta(tokens * v_heads);
        std::vector<float> gate(tokens * v_heads);
        for (std::size_t t = 0; t < tokens; ++t) {
            const std::vector<float> row(rotated.begin() + static_cast<std::ptrdiff_t>(t * in_dim),
                                         rotated.begin() + static_cast<std::ptrdiff_t>((t + 1) * in_dim));
            std::vector<float> qkv_row(qkv_dim);
            std::vector<float> z_row(z_dim);
            mnet::gguf::Reader::matvec(wqkv, row.data(), qkv_row.data());
            mnet::gguf::Reader::matvec(wgate_q, row.data(), z_row.data());
            std::copy(qkv_row.begin(), qkv_row.end(), qkv.begin() + static_cast<std::ptrdiff_t>(t * qkv_dim));
            std::copy(z_row.begin(), z_row.end(), z.begin() + static_cast<std::ptrdiff_t>(t * z_dim));
            const std::vector<float> x_row(xn.begin() + static_cast<std::ptrdiff_t>(t * in_dim),
                                           xn.begin() + static_cast<std::ptrdiff_t>((t + 1) * in_dim));
            std::vector<float> b(v_heads);
            std::vector<float> a(v_heads);
            mnet::gguf::Reader::matvec(wbeta, x_row.data(), b.data());
            mnet::gguf::Reader::matvec(walpha, x_row.data(), a.data());
            for (std::size_t h = 0; h < v_heads; ++h) {
                beta[t * v_heads + h] = sigmoid(b[h]);
                gate[t * v_heads + h] = softplus(a[h] + dt[h]) * ssm_a[h];
            }
        }

        const bool dump = std::getenv("MNET_BONSAI_DUMP") != nullptr;
        if (dump) {
            write_floats(output_path + ".qkv.bin", qkv);
            write_floats(output_path + ".z.bin", z);
            write_floats(output_path + ".beta.bin", beta);
            write_floats(output_path + ".gate.bin", gate);
        }

        // Causal depthwise conv (kernel 4), zero history, then SiLU.
        std::vector<float> conv_out(tokens * conv_channels);
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t c = 0; c < conv_channels; ++c) {
                float sum = 0.0F;
                for (std::size_t k = 0; k < conv_kernel; ++k) {
                    const auto source = static_cast<std::ptrdiff_t>(t) + static_cast<std::ptrdiff_t>(k) -
                                        static_cast<std::ptrdiff_t>(conv_kernel - 1);
                    if (source >= 0) {
                        sum += conv_w[c * conv_kernel + k] * qkv[static_cast<std::size_t>(source) * qkv_dim + c];
                    }
                }
                conv_out[t * conv_channels + c] = silu(sum);
            }
        }

        if (dump) {
            write_floats(output_path + ".conv.bin", conv_out);
        }

        // Split q/k/v and L2-normalize each q/k head.
        for (std::size_t t = 0; t < tokens; ++t) {
            float* row = conv_out.data() + t * conv_channels;
            for (std::size_t h = 0; h < k_heads; ++h) {
                l2_normalize_head(row + h * head_dim, head_dim, eps);
                l2_normalize_head(row + key_dim + h * head_dim, head_dim, eps);
            }
        }

        if (dump) {
            write_floats(output_path + ".qknorm.bin", conv_out);
        }

        // Gated delta rule per value head. q/k heads are shared by h % k_heads.
        std::vector<float> delta_out(tokens * value_dim);
        std::vector<float> state(head_dim * head_dim);
        std::vector<float> delta(head_dim);
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t h = 0; h < v_heads; ++h) {
            std::fill(state.begin(), state.end(), 0.0F);
            const std::size_t qk_head = h % k_heads;
            for (std::size_t t = 0; t < tokens; ++t) {
                const float* row = conv_out.data() + t * conv_channels;
                const float* q = row + qk_head * head_dim;
                const float* k = row + key_dim + qk_head * head_dim;
                const float* v = row + 2 * key_dim + h * head_dim;
                const float g = gate[t * v_heads + h];
                const float b = beta[t * v_heads + h];
                const float decay = std::exp(g);
                for (float& value : state) {
                    value *= decay;
                }
                for (std::size_t j = 0; j < head_dim; ++j) {
                    double sum = 0.0;
                    const float* state_row = state.data() + j * head_dim;
                    for (std::size_t i = 0; i < head_dim; ++i) {
                        sum += static_cast<double>(state_row[i]) * static_cast<double>(k[i]);
                    }
                    delta[j] = static_cast<float>((static_cast<double>(v[j]) - sum) * static_cast<double>(b));
                }
                for (std::size_t j = 0; j < head_dim; ++j) {
                    float* state_row = state.data() + j * head_dim;
                    for (std::size_t i = 0; i < head_dim; ++i) {
                        state_row[i] += delta[j] * k[i];
                    }
                }
                float* out_row = delta_out.data() + t * value_dim + h * head_dim;
                for (std::size_t j = 0; j < head_dim; ++j) {
                    const float* state_row = state.data() + j * head_dim;
                    double sum = 0.0;
                    for (std::size_t i = 0; i < head_dim; ++i) {
                        sum += static_cast<double>(state_row[i]) * static_cast<double>(q[i]);
                    }
                    out_row[j] = static_cast<float>(sum) * scale;
                }
            }
        }

        if (dump) {
            write_floats(output_path + ".delta.bin", delta_out);
        }

        // Gated RMS norm per value head, then the folded ssm_out projection.
        std::vector<float> final_state(tokens * in_dim);
        for (std::size_t t = 0; t < tokens; ++t) {
            std::vector<float> row(value_dim);
            for (std::size_t h = 0; h < v_heads; ++h) {
                std::vector<float> head(
                    delta_out.begin() + static_cast<std::ptrdiff_t>(t * value_dim + h * head_dim),
                    delta_out.begin() + static_cast<std::ptrdiff_t>(t * value_dim + (h + 1) * head_dim));
                head = rms_norm(head, ssm_norm, eps);
                for (std::size_t i = 0; i < head_dim; ++i) {
                    const float g = silu(z[t * z_dim + h * head_dim + i]);
                    row[h * head_dim + i] = head[i] * g;
                }
            }
            // The folded ssm_out weight was built with grouped value heads; the
            // delta output is in tiled head order [hd, key_head, rep].
            std::vector<float> grouped(value_dim);
            for (std::size_t h = 0; h < v_heads; ++h) {
                const std::size_t key_head = h % k_heads;
                const std::size_t rep = h / k_heads;
                std::copy(row.begin() + static_cast<std::ptrdiff_t>(h * head_dim),
                          row.begin() + static_cast<std::ptrdiff_t>((h + 1) * head_dim),
                          grouped.begin() + static_cast<std::ptrdiff_t>((rep + 3 * key_head) * head_dim));
            }
            hadamard_sign(grouped, signs_z, block_size);
            std::vector<float> projected(in_dim);
            mnet::gguf::Reader::matvec(wssm_out, grouped.data(), projected.data());
            for (std::size_t i = 0; i < in_dim; ++i) {
                final_state[t * in_dim + i] = input[t * in_dim + i] + projected[i];
            }
        }

        if (dump) {
            write_floats(output_path + ".final.bin", final_state);
        }

        // Post-attention norm + SWIGLU FFN + residual.
        std::vector<float> output(tokens * in_dim);
        for (std::size_t t = 0; t < tokens; ++t) {
            std::vector<float> residual(final_state.begin() + static_cast<std::ptrdiff_t>(t * in_dim),
                                        final_state.begin() + static_cast<std::ptrdiff_t>((t + 1) * in_dim));
            std::vector<float> row = rms_norm(residual, post_norm, eps);
            hadamard_sign(row, signs_in, block_size);
            std::vector<float> g(hidden);
            std::vector<float> u(hidden);
            mnet::gguf::Reader::matvec(ffn_gate, row.data(), g.data());
            mnet::gguf::Reader::matvec(ffn_up, row.data(), u.data());
            std::vector<float> act(hidden);
            for (std::size_t i = 0; i < hidden; ++i) {
                act[i] = silu(g[i]) * u[i];
            }
            hadamard_sign(act, signs_hidden, block_size);
            std::vector<float> down(in_dim);
            mnet::gguf::Reader::matvec(ffn_down, act.data(), down.data());
            for (std::size_t i = 0; i < in_dim; ++i) {
                output[t * in_dim + i] = residual[i] + down[i];
            }
        }

        write_floats(output_path, output);
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::printf("post_ffn -> %s (%.2f ms)\n", output_path.c_str(), ms);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_bonsai_block: %s\n", error.what());
        return 1;
    }
}
