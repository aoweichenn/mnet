// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Native Inkling2 stage-1 probes: token embedding, input norms, and the
// attention Q/K/V/relative projections of one SWA layer. Weights stay in their
// GGUF quantized form; no llama.cpp code is used.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mnet/gguf.hpp>

namespace {

using mnet::gguf::Reader;

struct Config {
    std::size_t n_embd = 2304;
    std::size_t n_head = 24;
    std::size_t n_head_kv = 8;
    std::size_t head_dim = 128;
    std::size_t n_relative = 16;
    std::size_t n_conv = 4;
    float eps = 1.0e-6F;
    float qk_clip = 22.6274166F;
    float relative_clip = 8.0F;
    float log_scale_alpha = 0.1F;
    std::size_t log_scale_floor = 8192;
};

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

[[nodiscard]] std::vector<float> rms_norm(const std::vector<float>& input, const std::vector<float>* weight,
                                          float eps) {
    double squares = 0.0;
    for (const float value : input) {
        squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto scale =
        static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(input.size()) + static_cast<double>(eps)));
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * (weight == nullptr ? 1.0F : (*weight)[i]);
    }
    return output;
}

// Row-wise L2 clip after an optional per-channel scale: t*x/max(|x|, t).
void clip_rows(std::vector<float>& values, const std::vector<float>& scale, std::size_t rows, std::size_t columns,
               float threshold, bool use_scale) {
    for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (std::size_t i = 0; i < columns; ++i) {
            const float s = use_scale ? scale[i + columns * row] : 1.0F;
            values[row * columns + i] *= s;
            squares += static_cast<double>(values[row * columns + i]) * static_cast<double>(values[row * columns + i]);
        }
        const auto norm = static_cast<float>(std::sqrt(squares));
        const float factor = threshold / std::max(norm, threshold);
        for (std::size_t i = 0; i < columns; ++i) {
            values[row * columns + i] *= factor;
        }
    }
}

[[nodiscard]] std::vector<float> read_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) {
        throw std::runtime_error("short read: " + path);
    }
    return values;
}

[[nodiscard]] std::vector<float> mix_states(const std::vector<std::vector<float>>& states,
                                            const std::vector<float>& query, float eps) {
    std::vector<std::vector<float>> normed;
    normed.reserve(states.size());
    for (const auto& state : states) {
        normed.push_back(rms_norm(state, nullptr, eps));
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(states.front().size()));
    std::vector<double> scores(normed.size());
    double max_score = -1e300;
    for (std::size_t i = 0; i < normed.size(); ++i) {
        double sum = 0.0;
        for (std::size_t d = 0; d < normed[i].size(); ++d) {
            sum += static_cast<double>(normed[i][d]) * static_cast<double>(query[d]);
        }
        scores[i] = sum * scale;
        max_score = std::max(max_score, scores[i]);
    }
    double total = 0.0;
    for (double& score : scores) {
        score = std::exp(score - max_score);
        total += score;
    }
    std::vector<float> out(states.front().size(), 0.0F);
    for (std::size_t i = 0; i < normed.size(); ++i) {
        const auto weight = static_cast<float>(scores[i] / total);
        for (std::size_t d = 0; d < out.size(); ++d) {
            out[d] += weight * normed[i][d];
        }
    }
    return out;
}

// Cross-pass control: (1 - forget) * cur + gain * message, gated on both.
[[nodiscard]] std::vector<float> control(const std::vector<float>& cur, const std::vector<float>& message,
                                         const Reader& reader, const std::string& prefix, const std::string& name,
                                         float eps) {
    const std::size_t n = cur.size();
    // The graph RMS-normalizes the [n_embd, 2] tensor over n_embd, so each row
    // is normalized on its own before the concatenation is read as 2*n_embd.
    const auto normed_cur = rms_norm(cur, nullptr, eps);
    const auto normed_message = rms_norm(message, nullptr, eps);
    std::vector<float> cat(2 * n);
    std::copy(normed_cur.begin(), normed_cur.end(), cat.begin());
    std::copy(normed_message.begin(), normed_message.end(), cat.begin() + static_cast<std::ptrdiff_t>(n));
    const auto down = reader.quantized(reader.tensor(prefix + "loop_" + name + "_ctrl_down.weight"));
    const auto gain_w = reader.quantized(reader.tensor(prefix + "loop_" + name + "_ctrl_gain.weight"));
    const auto forget_w = reader.quantized(reader.tensor(prefix + "loop_" + name + "_ctrl_forget.weight"));
    std::vector<float> gate(down.rows);
    Reader::matvec(down, cat.data(), gate.data());
    std::vector<float> gain(gain_w.rows);
    std::vector<float> forget(forget_w.rows);
    Reader::matvec(gain_w, gate.data(), gain.data());
    Reader::matvec(forget_w, gate.data(), forget.data());
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = cur[i] - std::tanh(forget[i]) * cur[i] + std::tanh(gain[i]) * message[i];
    }
    return out;
}

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

[[nodiscard]] float situ_gate(float value, float beta) {
    return beta * std::tanh(value / beta) * sigmoid(value);
}

[[nodiscard]] float situ_linear(float value, float beta) {
    return beta * std::tanh(value / beta);
}

[[nodiscard]] std::vector<float> convolve_row(const std::vector<float>& input, const std::vector<float>& weight,
                                              std::size_t channels) {
    std::vector<float> output(channels, 0.0F);
    for (std::size_t c = 0; c < channels; ++c) {
        // zero history for the first token: only the newest tap sees input
        output[c] = weight[c * 4 + 3] * input[c];
    }
    return output;
}

}  // namespace

int main(std::int32_t argc, char** argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr,
                         "usage: mnet_inkling2_attn MODEL.gguf TOKEN_ID DUMP_DIR [--layer N] [--slot N] [--pass N] "
                         "[--input F] [--bank F] [--bank-out F] [--position N]\n");
            return 2;
        }
        const std::string model_path = argv[1];
        const auto token = std::stoi(argv[2]);
        const std::string dump_dir = argv[3];
        std::size_t layer = 0;
        std::size_t slot = 0;
        std::size_t pass = 0;
        std::size_t position = 0;
        std::string input_path;
        std::string bank_path;
        std::string bank_out_path;
        for (std::int32_t i = 4; i + 1 < argc; i += 2) {
            if (std::string(argv[i]) == "--layer") {
                layer = std::stoul(argv[i + 1]);
            } else if (std::string(argv[i]) == "--input") {
                input_path = argv[i + 1];
            } else if (std::string(argv[i]) == "--position") {
                position = std::stoul(argv[i + 1]);
            } else if (std::string(argv[i]) == "--slot") {
                slot = std::stoul(argv[i + 1]);
            } else if (std::string(argv[i]) == "--pass") {
                pass = std::stoul(argv[i + 1]);
            } else if (std::string(argv[i]) == "--bank") {
                bank_path = argv[i + 1];
            } else if (std::string(argv[i]) == "--bank-out") {
                bank_out_path = argv[i + 1];
            }
        }
        if (slot == 0) {
            slot = layer;
        }
        const std::string layer_suffix = "-" + std::to_string(slot) + ".bin";
        std::filesystem::create_directories(dump_dir);

        Reader reader(model_path);
        Config cfg;
        cfg.eps = static_cast<float>(reader.floats().at("inkling2_loop.attention.layer_norm_rms_epsilon"));
        cfg.qk_clip = static_cast<float>(reader.floats().at("inkling2_loop.attention.qk_clip"));
        cfg.relative_clip = static_cast<float>(reader.floats().at("inkling2_loop.attention.relative_clip"));
        cfg.log_scale_alpha = static_cast<float>(reader.floats().at("inkling2_loop.attention.log_scale_alpha"));
        cfg.log_scale_floor = static_cast<std::size_t>(reader.integers().at("inkling2_loop.attention.log_scale_floor"));
        const std::string prefix = "blk." + std::to_string(layer) + ".";

        // Embedding row (IQ4_XS), then the input norm chain.
        const auto& info = reader.tensor("token_embd.weight");
        const auto type = mnet::gguf::type_info(info.type);
        const std::size_t row_bytes = cfg.n_embd / type.block_elements * type.block_bytes;
        const auto raw = reader.read_raw(info, static_cast<std::uint64_t>(token) * row_bytes, row_bytes);
        auto emb = Reader::dequantize_block(info.type, raw, cfg.n_embd);
        write_floats(dump_dir + "/embd.bin", emb);

        const auto tok_norm_w = reader.dequantize(reader.tensor("token_embd_norm.weight"));
        auto tok_pre = rms_norm(emb, nullptr, cfg.eps);
        write_floats(dump_dir + "/norm.bin", tok_pre);
        auto tok = tok_pre;
        for (std::size_t i = 0; i < tok.size(); ++i) {
            tok[i] *= tok_norm_w[i];
        }
        const auto& swa_pattern = reader.int_arrays().at("inkling2_loop.attention.sliding_window_pattern");
        const bool is_swa = swa_pattern[layer] != 0;
        std::vector<float> layer_input = tok;
        if (!input_path.empty()) {
            layer_input = read_floats(input_path);
        }
        std::vector<std::vector<float>> inputs(3);
        if (layer == 0) {
            inputs[0] = tok;
            inputs[1] = tok;
            inputs[2] = tok;
        } else {
            const auto attn_res_query = reader.dequantize(reader.tensor(prefix + "attn_res_query.weight"));
            if (is_swa) {
                inputs[0] = mix_states({tok, layer_input}, attn_res_query, cfg.eps);
                inputs[1] = inputs[0];
                inputs[2] = inputs[0];
            } else {
                for (std::size_t j = 0; j < 3; ++j) {
                    const std::vector<float> query(
                        attn_res_query.begin() + static_cast<std::ptrdiff_t>(j * cfg.n_embd),
                        attn_res_query.begin() + static_cast<std::ptrdiff_t>((j + 1) * cfg.n_embd));
                    inputs[j] = mix_states({tok, layer_input}, query, cfg.eps);
                }
            }
        }

        std::vector<std::vector<float>> messages;
        if (pass > 0) {
            if (bank_path.empty() || bank_out_path.empty()) {
                throw std::runtime_error("loop pass > 0 needs --bank and --bank-out");
            }
            const auto bank = read_floats(bank_path);
            const auto bank_out = read_floats(bank_out_path);
            const auto msg_query = reader.dequantize(reader.tensor(prefix + "loop_msg_query.weight"));
            for (std::size_t c = 0; c < 4; ++c) {
                const std::vector<float> query(msg_query.begin() + static_cast<std::ptrdiff_t>(c * cfg.n_embd),
                                               msg_query.begin() + static_cast<std::ptrdiff_t>((c + 1) * cfg.n_embd));
                messages.push_back(mix_states({bank, bank_out}, query, cfg.eps));
            }
            const char* names[3] = {"q", "k", "v"};
            for (std::size_t j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], messages[j], reader, prefix, names[j], cfg.eps);
            }
        }

        std::vector<float> nq;
        std::vector<float> nk;
        std::vector<float> nv;
        if (is_swa) {
            const auto attn_norm_w = reader.dequantize(reader.tensor(prefix + "attn_norm.weight"));
            auto pre = rms_norm(inputs[0], nullptr, cfg.eps);
            write_floats(dump_dir + "/norm" + layer_suffix, pre);
            for (std::size_t i = 0; i < pre.size(); ++i) {
                pre[i] *= attn_norm_w[i];
            }
            nq = pre;
            const auto pre_k = rms_norm(inputs[1], nullptr, cfg.eps);
            const auto pre_v = rms_norm(inputs[2], nullptr, cfg.eps);
            write_floats(dump_dir + "/norm_q" + layer_suffix, rms_norm(inputs[0], nullptr, cfg.eps));
            write_floats(dump_dir + "/norm_k" + layer_suffix, pre_k);
            write_floats(dump_dir + "/norm_v" + layer_suffix, pre_v);
            nk = rms_norm(inputs[1], &attn_norm_w, cfg.eps);
            nv = rms_norm(inputs[2], &attn_norm_w, cfg.eps);
        } else {
            const auto q_norm_w = reader.dequantize(reader.tensor(prefix + "attn_q_inp_norm.weight"));
            const auto k_norm_w = reader.dequantize(reader.tensor(prefix + "attn_k_inp_norm.weight"));
            const auto v_norm_w = reader.dequantize(reader.tensor(prefix + "attn_v_inp_norm.weight"));
            nq = rms_norm(inputs[0], &q_norm_w, cfg.eps);
            nk = rms_norm(inputs[1], &k_norm_w, cfg.eps);
            nv = rms_norm(inputs[2], &v_norm_w, cfg.eps);
        }
        const auto& norm = nq;

        // logarithmic position scale, 1 at position 0
        const float tau = static_cast<float>(
            1.0 +
            static_cast<double>(cfg.log_scale_alpha) *
                std::log(std::max(1.0, static_cast<double>(position + 1) / static_cast<double>(cfg.log_scale_floor))));

        // Q projection: per-head channel scale, then L2 clip.
        const auto wq = reader.quantized(reader.tensor(prefix + "attn_q.weight"));
        std::vector<float> qraw(cfg.n_head * cfg.head_dim);
        Reader::matvec(wq, nq.data(), qraw.data());
        write_floats(dump_dir + "/node_33" + layer_suffix, qraw);
        const auto q_clip = reader.dequantize(reader.tensor(prefix + "attn_q_clip.weight"));
        clip_rows(qraw, q_clip, cfg.n_head, cfg.head_dim, cfg.qk_clip, true);
        if (!is_swa) {
            for (float& value : qraw) {
                value *= tau;
            }
        }
        write_floats(dump_dir + "/Qcur" + layer_suffix, qraw);

        // K projection: static short conv, plus input-dependent taps on full attention.
        const auto wk = reader.quantized(reader.tensor(prefix + "attn_k.weight"));
        std::vector<float> kraw(cfg.n_head_kv * cfg.head_dim);
        Reader::matvec(wk, nk.data(), kraw.data());
        write_floats(dump_dir + "/node_15" + layer_suffix, kraw);
        const auto k_conv = reader.dequantize(reader.tensor(prefix + "attn_k_conv.weight"));
        auto kcur = convolve_row(kraw, k_conv, cfg.n_head_kv * cfg.head_dim);
        const float conv_scale = 1.0F / std::sqrt(static_cast<float>(cfg.n_conv + 1));
        const float dyn_scale = 1.0F / std::sqrt(static_cast<float>(2 * cfg.n_conv + 1));
        std::vector<float> kdyn;
        if (!is_swa) {
            const auto k_conv_proj = reader.quantized(reader.tensor(prefix + "attn_k_conv_proj.weight"));
            kdyn.resize(cfg.n_conv * cfg.n_head_kv);
            Reader::matvec(k_conv_proj, nk.data(), kdyn.data());
        }
        for (std::size_t c = 0; c < kcur.size(); ++c) {
            if (is_swa) {
                kcur[c] = (kraw[c] + kcur[c]) * conv_scale;
            } else {
                const std::size_t head = c / cfg.head_dim;
                kcur[c] = (kraw[c] + kcur[c] + kraw[c] * kdyn[head * cfg.n_conv]) * dyn_scale;
            }
        }
        clip_rows(kcur, {}, cfg.n_head_kv, cfg.head_dim, cfg.qk_clip, false);
        write_floats(dump_dir + "/Kcur" + layer_suffix, kcur);

        // V projection: static short conv plus dynamic taps, then per-head RMS norm.
        const auto wv = reader.quantized(reader.tensor(prefix + "attn_v.weight"));
        std::vector<float> vraw(cfg.n_head_kv * cfg.head_dim);
        Reader::matvec(wv, nv.data(), vraw.data());
        write_floats(dump_dir + "/node_36" + layer_suffix, vraw);
        const auto v_conv = reader.dequantize(reader.tensor(prefix + "attn_v_conv.weight"));
        auto vcur = convolve_row(vraw, v_conv, cfg.n_head_kv * cfg.head_dim);
        const float conv_scale_v = 1.0F / std::sqrt(static_cast<float>(cfg.n_conv + 1));
        std::vector<float> vdyn;
        if (!is_swa) {
            const auto v_conv_proj = reader.quantized(reader.tensor(prefix + "attn_v_conv_proj.weight"));
            vdyn.resize(cfg.n_conv * cfg.n_head_kv);
            Reader::matvec(v_conv_proj, nv.data(), vdyn.data());
        }
        for (std::size_t c = 0; c < vcur.size(); ++c) {
            if (is_swa) {
                vcur[c] = (vraw[c] + vcur[c]) * conv_scale_v;
            } else {
                const std::size_t head = c / cfg.head_dim;
                vcur[c] = (vraw[c] + vcur[c] + vraw[c] * vdyn[head * cfg.n_conv]) * dyn_scale;
            }
        }
        for (std::size_t h = 0; h < cfg.n_head_kv; ++h) {
            std::vector<float> head(vcur.begin() + static_cast<std::ptrdiff_t>(h * cfg.head_dim),
                                    vcur.begin() + static_cast<std::ptrdiff_t>((h + 1) * cfg.head_dim));
            head = rms_norm(head, nullptr, cfg.eps);
            std::copy(head.begin(), head.end(), vcur.begin() + static_cast<std::ptrdiff_t>(h * cfg.head_dim));
        }
        write_floats(dump_dir + "/Vcur" + layer_suffix, vcur);

        // Relative projection: per-(position, head) scale, then L2 clip to 8.
        const auto wrel = reader.quantized(reader.tensor(prefix + "attn_rel.weight"));
        std::vector<float> rcur(cfg.n_relative * cfg.n_head);
        Reader::matvec(wrel, nq.data(), rcur.data());
        write_floats(dump_dir + "/node_63" + layer_suffix, rcur);
        const auto rel_clip = reader.dequantize(reader.tensor(prefix + "attn_rel_clip.weight"));
        clip_rows(rcur, rel_clip, cfg.n_head, cfg.n_relative, cfg.relative_clip, true);
        if (!is_swa) {
            for (float& value : rcur) {
                value *= tau;
            }
        }
        write_floats(dump_dir + "/Rcur" + layer_suffix, rcur);

        // Attention for the single-token probe: relative bias for distance 0,
        // a zero sink logit, and the current token as the only visible key.
        const auto rel_bias = reader.dequantize(reader.tensor(prefix + "attn_rel_bias.weight"));
        const auto sink_w = reader.quantized(reader.tensor(prefix + "attn_sink_proj.weight"));
        std::vector<float> sink(cfg.n_head);
        Reader::matvec(sink_w, norm.data(), sink.data());
        std::vector<float> kqv(cfg.n_head * cfg.head_dim);
        const float attn_scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));
        for (std::size_t h = 0; h < cfg.n_head; ++h) {
            const std::size_t kv_head = h / (cfg.n_head / cfg.n_head_kv);
            double qk = 0.0;
            double bias = 0.0;
            for (std::size_t i = 0; i < cfg.head_dim; ++i) {
                qk += static_cast<double>(qraw[h * cfg.head_dim + i]) *
                      static_cast<double>(kcur[kv_head * cfg.head_dim + i]);
            }
            for (std::size_t rr = 0; rr < cfg.n_relative; ++rr) {
                bias += static_cast<double>(rel_bias[rr]) * static_cast<double>(rcur[h * cfg.n_relative + rr]);
            }
            const float score = static_cast<float>(qk * static_cast<double>(attn_scale) + bias) - sink[h];
            const float weight = sigmoid(score);
            for (std::size_t i = 0; i < cfg.head_dim; ++i) {
                kqv[h * cfg.head_dim + i] = weight * vcur[kv_head * cfg.head_dim + i];
            }
        }
        write_floats(dump_dir + "/kqv_out" + layer_suffix, kqv);

        const auto gate_w = reader.quantized(reader.tensor(prefix + "attn_gate.weight"));
        std::vector<float> gate(cfg.n_head * cfg.head_dim);
        Reader::matvec(gate_w, norm.data(), gate.data());
        for (std::size_t i = 0; i < gate.size(); ++i) {
            gate[i] = sigmoid(gate[i]);
        }
        for (std::size_t i = 0; i < kqv.size(); ++i) {
            kqv[i] *= gate[i];
        }
        const auto out_w = reader.quantized(reader.tensor(prefix + "attn_output.weight"));
        std::vector<float> projected(cfg.n_embd);
        Reader::matvec(out_w, kqv.data(), projected.data());
        const auto post_conv = reader.dequantize(reader.tensor(prefix + "attn_post_conv.weight"));
        auto post = convolve_row(projected, post_conv, cfg.n_embd);
        for (std::size_t i = 0; i < post.size(); ++i) {
            post[i] = (projected[i] + post[i]) * conv_scale;
        }
        const std::size_t block_size =
            static_cast<std::size_t>(reader.integers().at("inkling2_loop.attn_res.block_size"));
        const bool block_start = layer % block_size == 0;
        std::vector<float> residual = post;
        if (!block_start) {
            residual = layer_input;
            for (std::size_t i = 0; i < cfg.n_embd; ++i) {
                residual[i] += post[i];
            }
        }
        write_floats(dump_dir + "/ffn_inp" + layer_suffix, residual);

        // Depth mix of the block outputs with the current residual, then the MoE FFN.
        const auto ffn_res_query = reader.dequantize(reader.tensor(prefix + "ffn_res_query.weight"));
        std::vector<std::vector<float>> ffn_states;
        if (block_start && layer != 0) {
            ffn_states = {layer_input, tok, residual};
        } else {
            ffn_states = {tok, residual};
        }
        auto mixed = mix_states(ffn_states, ffn_res_query, cfg.eps);
        if (pass > 0) {
            mixed = control(mixed, messages[3], reader, prefix, "ffn", cfg.eps);
        }
        const auto ffn_norm_w = reader.dequantize(reader.tensor(prefix + "ffn_norm.weight"));
        const auto moe_in = rms_norm(mixed, &ffn_norm_w, cfg.eps);

        const std::size_t n_expert = static_cast<std::size_t>(reader.integers().at("inkling2_loop.expert_count"));
        const std::size_t n_used = static_cast<std::size_t>(reader.integers().at("inkling2_loop.expert_used_count"));
        const float situ_beta = static_cast<float>(reader.floats().at("inkling2_loop.activation.situ_beta"));
        const float situ_linear_beta =
            static_cast<float>(reader.floats().at("inkling2_loop.activation.situ_linear_beta"));
        const float expert_scale = static_cast<float>(reader.floats().at("inkling2_loop.expert_weights_scale"));
        const auto gate_inp = reader.quantized(reader.tensor(prefix + "ffn_gate_inp.weight"));
        const auto probs_bias = reader.dequantize(reader.tensor(prefix + "exp_probs_b.bias"));
        std::vector<float> logits_moe(n_expert);
        Reader::matvec(gate_inp, moe_in.data(), logits_moe.data());
        write_floats(dump_dir + "/ffn_moe_logits" + layer_suffix, logits_moe);
        std::vector<float> probs(n_expert);
        std::vector<float> selection(n_expert);
        for (std::size_t e = 0; e < n_expert; ++e) {
            probs[e] = sigmoid(logits_moe[e]);
            selection[e] = probs[e] + probs_bias[e];
        }
        write_floats(dump_dir + "/ffn_moe_probs" + layer_suffix, probs);
        write_floats(dump_dir + "/ffn_moe_probs_biased" + layer_suffix, selection);
        std::vector<std::size_t> order(n_expert);
        for (std::size_t e = 0; e < n_expert; ++e) {
            order[e] = e;
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return selection[a] > selection[b]; });
        std::vector<float> weights(n_used);
        double weight_sum = 0.0;
        for (std::size_t i = 0; i < n_used; ++i) {
            weights[i] = probs[order[i]];
            weight_sum += static_cast<double>(weights[i]);
        }
        write_floats(dump_dir + "/ffn_moe_weights" + layer_suffix, weights);
        const auto denom = static_cast<float>(weight_sum + 1e-20);
        for (float& weight : weights) {
            weight /= denom;
        }
        write_floats(dump_dir + "/ffn_moe_weights_norm" + layer_suffix, weights);
        for (float& weight : weights) {
            weight *= expert_scale;
        }
        write_floats(dump_dir + "/ffn_moe_weights_scaled" + layer_suffix, weights);

        const auto gate_up_info = reader.tensor(prefix + "ffn_gate_up_exps.weight");
        const auto down_info = reader.tensor(prefix + "ffn_down_exps.weight");
        const std::size_t n_ff = static_cast<std::size_t>(gate_up_info.dims[1]) / 2;
        std::vector<float> gate_up(n_used * 2 * n_ff);
        std::vector<float> expert_out(cfg.n_embd, 0.0F);
        std::vector<float> act(n_ff);
        for (std::size_t i = 0; i < n_used; ++i) {
            const auto expert = static_cast<std::int32_t>(order[i]);
            const auto gate_up_w = reader.quantized(gate_up_info, expert);
            std::vector<float> gu(2 * n_ff);
            Reader::matvec(gate_up_w, moe_in.data(), gu.data());
            std::copy(gu.begin(), gu.end(), gate_up.begin() + static_cast<std::ptrdiff_t>(i * 2 * n_ff));
            for (std::size_t v = 0; v < n_ff; ++v) {
                act[v] = situ_gate(gu[v], situ_beta) * situ_linear(gu[n_ff + v], situ_linear_beta);
            }
            const auto down_w = reader.quantized(down_info, expert);
            std::vector<float> down(cfg.n_embd);
            Reader::matvec(down_w, act.data(), down.data());
            for (std::size_t v = 0; v < cfg.n_embd; ++v) {
                expert_out[v] += weights[i] * down[v];
            }
            write_floats(dump_dir + "/ffn_moe_gate" + layer_suffix,
                         std::vector<float>(gu.begin(), gu.begin() + static_cast<std::ptrdiff_t>(n_ff)));
            write_floats(dump_dir + "/ffn_moe_up" + layer_suffix,
                         std::vector<float>(gu.begin() + static_cast<std::ptrdiff_t>(n_ff), gu.end()));
            write_floats(dump_dir + "/ffn_moe_situ" + layer_suffix, act);
            write_floats(dump_dir + "/ffn_moe_down" + layer_suffix, down);
        }
        write_floats(dump_dir + "/ffn_moe_gate_up" + layer_suffix, gate_up);
        write_floats(dump_dir + "/ffn_moe_out" + layer_suffix, expert_out);
        const auto ffn_post_conv = reader.dequantize(reader.tensor(prefix + "ffn_post_conv.weight"));
        auto ffn_out = convolve_row(expert_out, ffn_post_conv, cfg.n_embd);
        for (std::size_t i = 0; i < ffn_out.size(); ++i) {
            ffn_out[i] = (expert_out[i] + ffn_out[i]) * conv_scale;
        }
        write_floats(dump_dir + "/ffn_out" + layer_suffix, ffn_out);
        std::vector<float> l_out(cfg.n_embd);
        for (std::size_t i = 0; i < cfg.n_embd; ++i) {
            l_out[i] = residual[i] + ffn_out[i];
        }
        write_floats(dump_dir + "/l_out" + layer_suffix, l_out);

        std::printf("mnet_inkling2_attn: layer=%zu token=%d embd=%zu heads=%zu/%zu rel=%zu\n", layer, token, cfg.n_embd,
                    cfg.n_head, cfg.n_head_kv, cfg.n_relative);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_inkling2_attn: %s\n", error.what());
        return 1;
    }
}
