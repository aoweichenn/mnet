// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Native Inkling2 Mini full forward: prelude layers 0..5, the looped core
// 6..17 with three passes and cross-pass control, layers 18..23, the depth
// output mix, output norm and folded head with final logit softcapping.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mnet/detail/ggml_kernels.hpp>
#include <mnet/gguf.hpp>
#include <mnet/tokenizer/bpe.hpp>

namespace {

using mnet::gguf::Reader;

constexpr float kMixEps = 1.0e-6F;

struct Config {
    std::size_t n_layer = 24;
    std::size_t n_embd = 2304;
    std::size_t n_head = 24;
    std::size_t n_head_kv = 8;
    std::size_t head_dim = 128;
    std::size_t n_relative = 16;
    std::size_t n_conv = 4;
    std::size_t n_vocab_padded = 0;
    std::size_t n_vocab = 0;
    float eps = 1.0e-6F;
    float qk_clip = 22.6274166F;
    float relative_clip = 8.0F;
    float log_scale_alpha = 0.1F;
    std::size_t log_scale_floor = 8192;
    float situ_beta = 4.0F;
    float situ_linear_beta = 25.0F;
    float expert_scale = 8.0F;
    float softcap = 30.0F;
    std::size_t n_expert = 128;
    std::size_t n_used = 8;
    std::size_t block_size = 6;
    std::size_t n_swa = 1024;
    std::size_t n_relative_extent = 1024;
    std::size_t core_start = 6;
    std::size_t core_end = 18;
    std::size_t n_passes = 3;
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
    // llama.cpp: ggml_float (double) accumulator of float squares, float mean and scale.
    double squares = 0.0;
    for (const float value : input) {
        squares += static_cast<double>(value * value);
    }
    const auto mean = static_cast<float>(squares / static_cast<double>(input.size()));
    const float scale = 1.0F / std::sqrt(mean + eps);
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * (weight == nullptr ? 1.0F : (*weight)[i]);
    }
    return output;
}

// llama.cpp's clip(): ggml_mul(scale), ggml_l2_norm(threshold), ggml_scale(threshold).
// l2_norm accumulates float squares in a double, then y = x * (1/max(sqrtf(sum), eps)).
void clip_rows(std::vector<float>& values, const std::vector<float>& scale, std::size_t rows, std::size_t columns,
               float threshold, bool use_scale) {
    for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (std::size_t i = 0; i < columns; ++i) {
            const float s = use_scale ? scale[i + columns * row] : 1.0F;
            values[row * columns + i] *= s;
            squares += static_cast<double>(values[row * columns + i] * values[row * columns + i]);
        }
        const auto norm = static_cast<float>(squares);
        const float inv = 1.0F / std::fmax(std::sqrt(norm), threshold);
        for (std::size_t i = 0; i < columns; ++i) {
            values[row * columns + i] *= inv;
            values[row * columns + i] *= threshold;
        }
    }
}

// ggml_v_expf and friends live in <mnet/detail/ggml_kernels.hpp>.
// ggml_vec_dot_f32: 4 x 8-lane FMA accumulators, tree reduce, scalar tail.
// ggml_ssm_conv rounds every product before adding it (no FMA contraction), unlike
// a hand-written `sum += a*b`. Keep the product materialised so we round the same way.
[[nodiscard]] inline float rounded_product(float a, float b) {
    volatile float product = a * b;
    return product;
}

using mnet::detail::ggml::dot_f32;

// ggml_vec_soft_max_f32: exp(x - max) with the vector exp, double sum.
using mnet::detail::ggml::softmax_exp;

// A mix with several queries is one batched matmul in the reference graph, which
// takes the tinyBLAS path whenever the state count is a multiple of four: eight
// interleaved k-lanes folded with fused multiply-adds, reduced as
// (a0+a2)+(a1+a3) after the lane pairs are joined.
[[nodiscard]] float dot_batched(const std::vector<float>& state, const std::vector<float>& query) {
    const std::size_t n = query.size();
    float lanes[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] = std::fma(state[i + lane], query[i + lane], lanes[lane]);
        }
    }
    for (std::size_t lane = 0; lane + i < n; ++lane) {
        lanes[lane] = std::fma(state[i + lane], query[i + lane], lanes[lane]);
    }
    const float half[4] = {lanes[0] + lanes[4], lanes[1] + lanes[5], lanes[2] + lanes[6], lanes[3] + lanes[7]};
    return (half[0] + half[2]) + (half[1] + half[3]);
}

// ggml_vec_dot_f32 over the depth mix: 32-state tiles keep four eight-lane
// accumulators that fold with a fused multiply-add and reduce in the order the
// kernel emits (lane pairs, then the two halves of a 128-bit sum); the leftover
// path rounds full groups of four products and fuses only the last n % 4 terms.
// Verified bit-exact against probe_mix_* dumps (2..5 states) and against the
// exported llama kernel for the long output mix.
[[nodiscard]] float dot_state_column(const std::vector<float>& weights, const std::vector<std::vector<float>>& states,
                                     std::size_t column) {
    const std::size_t n = weights.size();
    float total = 0.0F;
    const std::size_t tiles = n & ~static_cast<std::size_t>(31);
    if (tiles != 0) {
        float acc[4][8] = {};
        for (std::size_t i = 0; i < tiles; i += 32) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const std::size_t index = i + j * 8 + lane;
                    acc[j][lane] = std::fma(weights[index], states[index][column], acc[j][lane]);
                }
            }
        }
        for (std::size_t lane = 0; lane < 8; ++lane) {
            acc[0][lane] += acc[2][lane];
            acc[1][lane] += acc[3][lane];
            acc[0][lane] += acc[1][lane];
        }
        const float half[4] = {acc[0][0] + acc[0][4], acc[0][1] + acc[0][5], acc[0][2] + acc[0][6],
                               acc[0][3] + acc[0][7]};
        total = (half[0] + half[1]) + (half[2] + half[3]);
    }
    const std::size_t remainder = n - tiles;
    const std::size_t fused = remainder % 4;
    const std::size_t rounded = remainder - fused;
    for (std::size_t k = 0; k < rounded; ++k) {
        total += rounded_product(weights[tiles + k], states[tiles + k][column]);
    }
    for (std::size_t k = rounded; k < remainder; ++k) {
        total = std::fma(weights[tiles + k], states[tiles + k][column], total);
    }
    return total;
}

[[nodiscard]] std::vector<float> mix_states(const std::vector<std::vector<float>>& states,
                                            const std::vector<float>& query, float eps, bool batched = false,
                                            bool round_all = false) {
    if (states.empty()) {
        throw std::runtime_error("depth mix without states");
    }
    // llama.cpp: ggml_mul_mat(states, query) -> soft_max_ext(scale = 1/sqrt(n_embd))
    // -> ggml_mul_mat(scores, states^T). f32 everywhere except the softmax sum.
    std::vector<std::vector<float>> normed;
    normed.reserve(states.size());
    for (const auto& state : states) {
        normed.push_back(rms_norm(state, nullptr, eps));
    }
    const auto width = normed.front().size();
    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    std::vector<float> scores(normed.size());
    float max_score = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < normed.size(); ++i) {
        // llama.cpp batches the queries into one matmul; with a state count that
        // is a multiple of four that matmul takes the tinyBLAS kernel.
        const float raw = batched && normed.size() % 4 == 0 ? dot_batched(normed[i], query)
                                                            : dot_f32(normed[i].data(), query.data(), width);
        scores[i] = raw * scale;
        max_score = std::max(max_score, scores[i]);
    }
    std::vector<float> probs(scores.size());
    const double total = softmax_exp(scores.data(), probs.data(), scores.size(), max_score);
    const auto inv = static_cast<float>(1.0 / total);
    for (float& value : probs) {
        value *= inv;
    }
    // The reference joins the states with ggml_vec_dot_f32, so the mix has to
    // follow that kernel shape exactly.
    std::vector<float> out(width, 0.0F);
    if (round_all) {
        // A multi-query mix goes through the batched matmul path, which this
        // model reproduces with rounded products throughout (verified against
        // the probe_msgs_* dumps).
        for (std::size_t i = 0; i < normed.size(); ++i) {
            const float weight = probs[i];
            for (std::size_t d = 0; d < width; ++d) {
                out[d] += rounded_product(weight, normed[i][d]);
            }
        }
        return out;
    }
    for (std::size_t d = 0; d < width; ++d) {
        out[d] = dot_state_column(probs, normed, d);
    }
    return out;
}

[[nodiscard]] std::vector<float> mix_queries(const std::vector<std::vector<float>>& states,
                                             const std::vector<float>& queries, std::size_t n_queries, float eps) {
    const std::size_t n = states.front().size();
    std::vector<float> out(n_queries * n);
    for (std::size_t q = 0; q < n_queries; ++q) {
        const std::vector<float> query(queries.begin() + static_cast<std::ptrdiff_t>(q * n),
                                       queries.begin() + static_cast<std::ptrdiff_t>((q + 1) * n));
        // A multi-query mix goes through the batched matmul path of the graph,
        // which this model always reproduces with rounded products throughout
        // (verified bit-exact against the probe_msgs_* dumps).
        const auto mixed = mix_states(states, query, eps, n_queries > 1, n_queries > 1);
        std::copy(mixed.begin(), mixed.end(), out.begin() + static_cast<std::ptrdiff_t>(q * n));
    }
    return out;
}

[[nodiscard]] std::vector<float> control(const std::vector<float>& cur, const std::vector<float>& message,
                                         const Reader& reader, const std::string& prefix, const std::string& name,
                                         float eps) {
    const std::size_t n = cur.size();
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
        // llama: add(sub(cur, mul(forget, cur)), mul(gain, message)) with each
        // product rounded separately.
        out[i] = (cur[i] - rounded_product(mnet::detail::ggml::tanhf(forget[i]), cur[i])) +
                 rounded_product(mnet::detail::ggml::tanhf(gain[i]), message[i]);
    }
    return out;
}

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + mnet::detail::ggml::expf(-value));
}
// llama.cpp: tanh(x * (1/beta)) scaled by beta, then * sigmoid(x) / up.
[[nodiscard]] float situ_gate(float value, float beta) {
    return mnet::detail::ggml::tanhf(value * (1.0F / beta)) * beta * sigmoid(value);
}
[[nodiscard]] float situ_linear(float value, float beta) {
    return mnet::detail::ggml::tanhf(value * (1.0F / beta)) * beta;
}

struct SlotState {
    std::vector<float> k_cache;  // tokens * n_head_kv * head_dim
    std::vector<float> v_cache;
    std::size_t tokens = 0;
    std::vector<float> k_hist;  // (n_conv - 1) * n_head_kv * head_dim
    std::vector<float> v_hist;
    std::vector<float> attn_hist;  // (n_conv - 1) * n_embd
    std::vector<float> ffn_hist;
};

// Causal short convolution over the slot history plus the current row.
// ggml_vec_dot_f16_f32: the KV cache holds f16, llama widens it with
// _mm256_cvtph_ps (exact), keeps four 8-lane FMA accumulators, pairs them as
// (acc0+acc1)+(acc2+acc3) and reduces with the hsum order. The remaining <8
// elements accumulate in double, which is also all that runs for a 1-2 token
// value matmul.
using mnet::detail::ggml::dot_f16_f32;

[[nodiscard]] std::vector<float> conv_step(const std::vector<float>& raw, const std::vector<float>& weight,
                                           std::vector<float>& history, std::size_t channels, std::size_t n_conv,
                                           const std::vector<float>* dynamic, std::size_t heads) {
    const std::size_t past = n_conv - 1;
    std::vector<float> out(channels, 0.0F);
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t k = 0; k < n_conv; ++k) {
            const float value = (k == past) ? raw[c] : history[k * channels + c];
            out[c] += rounded_product(weight[c * n_conv + k], value);
        }
    }
    if (dynamic != nullptr) {
        const std::size_t head_dim = channels / heads;
        for (std::size_t c = 0; c < channels; ++c) {
            const std::size_t head = c / head_dim;
            for (std::size_t lag = 0; lag < n_conv; ++lag) {
                const std::size_t row = past - lag;
                const float value = (row == past) ? raw[c] : history[row * channels + c];
                out[c] += rounded_product(value, (*dynamic)[lag + n_conv * head]);
            }
        }
    }
    for (std::size_t k = 0; k + 1 < past; ++k) {
        std::copy(history.begin() + static_cast<std::ptrdiff_t>((k + 1) * channels),
                  history.begin() + static_cast<std::ptrdiff_t>((k + 2) * channels),
                  history.begin() + static_cast<std::ptrdiff_t>(k * channels));
    }
    std::copy(raw.begin(), raw.end(), history.begin() + static_cast<std::ptrdiff_t>((past - 1) * channels));
    const float scale = dynamic != nullptr ? 1.0F / std::sqrt(static_cast<float>(2 * n_conv + 1))
                                           : 1.0F / std::sqrt(static_cast<float>(n_conv + 1));
    for (std::size_t c = 0; c < channels; ++c) {
        out[c] = (raw[c] + out[c]) * scale;
    }
    return out;
}

class Model {
public:
    explicit Model(std::string path) : reader_(std::move(path)) { this->load_config(); }

    [[nodiscard]] const Reader& reader() const { return this->reader_; }
    [[nodiscard]] const Config& config() const { return this->config_; }

    void set_kv_f16(bool enabled) { this->kv_f16_ = enabled; }

    [[nodiscard]] std::vector<std::vector<float>> embed(const std::vector<std::int32_t>& tokens) const {
        const auto& info = this->reader_.tensor("token_embd.weight");
        const auto type = mnet::gguf::type_info(info.type);
        const std::size_t row_bytes = this->config_.n_embd / type.block_elements * type.block_bytes;
        const auto tok_norm_w = this->reader_.dequantize(this->reader_.tensor("token_embd_norm.weight"));
        std::vector<std::vector<float>> out;
        out.reserve(tokens.size());
        for (const auto token : tokens) {
            const auto raw = this->reader_.read_raw(info, static_cast<std::uint64_t>(token) * row_bytes, row_bytes);
            auto emb = Reader::dequantize_block(info.type, raw, this->config_.n_embd);
            out.push_back(rms_norm(emb, &tok_norm_w, this->config_.eps));
        }
        return out;
    }

    [[nodiscard]] std::vector<float> logits(const std::vector<std::int32_t>& tokens, const std::string& dump_dir) {
        const auto embedded = this->embed(tokens);
        this->slots_.assign(48, SlotState{});
        for (auto& state : this->slots_) {
            state.k_hist.assign((this->config_.n_conv - 1) * this->config_.n_head_kv * this->config_.head_dim, 0.0F);
            state.v_hist.assign((this->config_.n_conv - 1) * this->config_.n_head_kv * this->config_.head_dim, 0.0F);
            state.attn_hist.assign((this->config_.n_conv - 1) * this->config_.n_embd, 0.0F);
            state.ffn_hist.assign((this->config_.n_conv - 1) * this->config_.n_embd, 0.0F);
        }

        std::vector<float> last;
        std::vector<std::vector<float>> last_blocks;
        for (std::size_t position = 0; position < tokens.size(); ++position) {
            std::vector<std::vector<float>> blocks;
            std::vector<float> cur = embedded[position];

            cur = this->run_layer(0, 0, cur, blocks, nullptr, position, dump_dir);
            for (std::size_t ip = 1; ip < this->config_.core_start; ++ip) {
                cur = this->run_layer(ip, ip, cur, blocks, nullptr, position, dump_dir);
            }
            const auto prelude = cur;
            const auto prelude_blocks = blocks;

            std::vector<float> bank_out;
            std::vector<std::vector<float>> bank;
            for (std::size_t pass = 0; pass < this->config_.n_passes; ++pass) {
                if (pass > 0) {
                    bank = std::vector<std::vector<float>>(
                        blocks.begin() + static_cast<std::ptrdiff_t>(prelude_blocks.size() + 1), blocks.end());
                    bank_out = cur;
                    cur = prelude;
                    blocks = prelude_blocks;
                }
                for (std::size_t ip = this->config_.core_start; ip < this->config_.core_end; ++ip) {
                    std::vector<std::vector<float>> messages;
                    const std::size_t slot =
                        pass == 0
                            ? ip
                            : this->config_.n_layer + (pass - 1) * (this->config_.core_end - this->config_.core_start) +
                                  (ip - this->config_.core_start);
                    if (pass > 0) {
                        std::vector<std::vector<float>> states;
                        for (auto it = bank.rbegin(); it != bank.rend(); ++it) {
                            states.push_back(*it);
                        }
                        states.push_back(bank_out);
                        const auto msg_query = this->reader_.dequantize(
                            this->reader_.tensor("blk." + std::to_string(ip) + ".loop_msg_query.weight"));
                        const auto mixed = mix_queries(states, msg_query, 4, kMixEps);
                        if (!dump_dir.empty()) {
                            write_floats(dump_dir + "/probe_msgs_" + std::to_string(slot) + ".bin", mixed);
                        }
                        for (std::size_t c = 0; c < 4; ++c) {
                            messages.emplace_back(
                                mixed.begin() + static_cast<std::ptrdiff_t>(c * this->config_.n_embd),
                                mixed.begin() + static_cast<std::ptrdiff_t>((c + 1) * this->config_.n_embd));
                        }
                    }
                    cur = this->run_layer(ip, slot, cur, blocks, pass > 0 ? &messages : nullptr, position, dump_dir);
                }
            }

            for (std::size_t ip = this->config_.core_end; ip < this->config_.n_layer; ++ip) {
                cur = this->run_layer(ip, ip, cur, blocks, nullptr, position, dump_dir);
            }
            last = cur;
            last_blocks = blocks;
        }

        std::vector<std::vector<float>> states_out;
        for (auto it = last_blocks.rbegin(); it != last_blocks.rend(); ++it) {
            states_out.push_back(*it);
        }
        states_out.push_back(last);
        const auto output_res_query = this->reader_.dequantize(this->reader_.tensor("output_res_query.weight"));
        auto mixed = mix_states(states_out, output_res_query, kMixEps);
        const auto output_norm_w = this->reader_.dequantize(this->reader_.tensor("output_norm.weight"));
        auto normed = rms_norm(mixed, &output_norm_w, this->config_.eps);
        if (!dump_dir.empty()) {
            write_floats(dump_dir + "/result_norm.bin", normed);
        }
        const auto output_w = this->reader_.quantized(this->reader_.tensor("output.weight"));
        std::vector<float> padded(output_w.rows);
        Reader::matvec(output_w, normed.data(), padded.data());
        // llama.cpp: scale(tanh(scale(logits, 1/cap)), cap), i.e. the reciprocal
        // is precomputed and multiplied instead of dividing.
        const float cap = this->config_.softcap;
        const float inv_cap = 1.0F / cap;
        std::vector<float> out(this->config_.n_vocab);
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = cap * mnet::detail::ggml::tanhf(inv_cap * padded[i]);
        }
        if (!dump_dir.empty()) {
            write_floats(dump_dir + "/result_output.bin", out);
        }
        return out;
    }

    bool kv_f16_ = false;

private:
    void load_config() {
        const auto& ints = this->reader_.integers();
        const auto& floats = this->reader_.floats();
        this->config_.n_layer = ints.at("inkling2_loop.block_count");
        this->config_.n_embd = ints.at("inkling2_loop.embedding_length");
        this->config_.n_vocab = ints.at("inkling2_loop.vocab_size_unpadded");
        this->config_.n_vocab_padded = ints.at("inkling2_loop.vocab_size");
        this->config_.eps = static_cast<float>(floats.at("inkling2_loop.attention.layer_norm_rms_epsilon"));
        this->config_.qk_clip = static_cast<float>(floats.at("inkling2_loop.attention.qk_clip"));
        this->config_.relative_clip = static_cast<float>(floats.at("inkling2_loop.attention.relative_clip"));
        this->config_.log_scale_alpha = static_cast<float>(floats.at("inkling2_loop.attention.log_scale_alpha"));
        this->config_.log_scale_floor = static_cast<std::size_t>(ints.at("inkling2_loop.attention.log_scale_floor"));
        this->config_.situ_beta = static_cast<float>(floats.at("inkling2_loop.activation.situ_beta"));
        this->config_.situ_linear_beta = static_cast<float>(floats.at("inkling2_loop.activation.situ_linear_beta"));
        this->config_.expert_scale = static_cast<float>(floats.at("inkling2_loop.expert_weights_scale"));
        this->config_.softcap = static_cast<float>(floats.at("inkling2_loop.final_logit_softcapping"));
        this->config_.n_expert = ints.at("inkling2_loop.expert_count");
        this->config_.n_used = ints.at("inkling2_loop.expert_used_count");
        this->config_.block_size = ints.at("inkling2_loop.attn_res.block_size");
        this->config_.n_swa = ints.at("inkling2_loop.attention.sliding_window");
        this->config_.n_relative_extent = ints.at("inkling2_loop.attention.relative_extent");
        this->config_.core_start = ints.at("inkling2_loop.loop.start");
        this->config_.core_end = ints.at("inkling2_loop.loop.end");
        this->config_.n_passes = ints.at("inkling2_loop.loop.count");
        const auto heads = this->reader_.int_arrays().at("inkling2_loop.attention.head_count");
        this->config_.n_head = static_cast<std::size_t>(heads.at(0));
        const auto heads_kv = this->reader_.int_arrays().at("inkling2_loop.attention.head_count_kv");
        this->config_.n_head_kv = static_cast<std::size_t>(heads_kv.at(0));
    }

    [[nodiscard]] std::vector<float> tensor_floats(const std::string& name) const {
        return this->reader_.dequantize(this->reader_.tensor(name));
    }

    [[nodiscard]] std::vector<float> run_layer(std::size_t ip, std::size_t slot, const std::vector<float>& input,
                                               std::vector<std::vector<float>>& blocks,
                                               const std::vector<std::vector<float>>* messages, std::size_t position,
                                               const std::string& dump_dir) {
        const std::string prefix = "blk." + std::to_string(ip) + ".";
        const std::string suffix = "-" + std::to_string(slot) + ".bin";
        const auto& swa_pattern = this->reader_.int_arrays().at("inkling2_loop.attention.sliding_window_pattern");
        const bool is_swa = swa_pattern[ip] != 0;

        auto dump = [&](const std::string& name, const std::vector<float>& values) {
            if (!dump_dir.empty()) {
                write_floats(dump_dir + "/" + name + suffix, values);
            }
        };
        auto dump_raw = [&](const std::string& name, const std::vector<float>& values) {
            if (!dump_dir.empty()) {
                write_floats(dump_dir + "/" + name + ".bin", values);
            }
        };

        std::vector<std::vector<float>> inputs(3);
        if (ip == 0) {
            inputs[0] = input;
            inputs[1] = input;
            inputs[2] = input;
        } else {
            const auto attn_res_query = this->tensor_floats(prefix + "attn_res_query.weight");
            std::vector<std::vector<float>> states;
            for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
                states.push_back(*it);
            }
            states.push_back(input);
            if (is_swa) {
                const auto mixed = mix_states(states, attn_res_query, kMixEps);
                inputs[0] = mixed;
                inputs[1] = mixed;
                inputs[2] = mixed;
            } else {
                for (std::size_t j = 0; j < 3; ++j) {
                    const std::vector<float> query(
                        attn_res_query.begin() + static_cast<std::ptrdiff_t>(j * this->config_.n_embd),
                        attn_res_query.begin() + static_cast<std::ptrdiff_t>((j + 1) * this->config_.n_embd));
                    inputs[j] = mix_states(states, query, kMixEps, true);
                }
            }
        }

        if (messages != nullptr) {
            const char* names[3] = {"q", "k", "v"};
            for (std::size_t j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], (*messages)[j], this->reader_, prefix, names[j], kMixEps);
                dump_raw("probe_ctrl_out_" + std::to_string(slot) + "_" + std::to_string(j), inputs[j]);
            }
        }

        std::vector<float> nq;
        std::vector<float> nk;
        std::vector<float> nv;
        if (is_swa) {
            const auto attn_norm_w = this->tensor_floats(prefix + "attn_norm.weight");
            nq = rms_norm(inputs[0], &attn_norm_w, this->config_.eps);
            nk = rms_norm(inputs[1], &attn_norm_w, this->config_.eps);
            nv = rms_norm(inputs[2], &attn_norm_w, this->config_.eps);
        } else {
            const auto q_norm_w = this->tensor_floats(prefix + "attn_q_inp_norm.weight");
            const auto k_norm_w = this->tensor_floats(prefix + "attn_k_inp_norm.weight");
            const auto v_norm_w = this->tensor_floats(prefix + "attn_v_inp_norm.weight");
            nq = rms_norm(inputs[0], &q_norm_w, this->config_.eps);
            nk = rms_norm(inputs[1], &k_norm_w, this->config_.eps);
            nv = rms_norm(inputs[2], &v_norm_w, this->config_.eps);
        }

        dump_raw("probe_nq_" + std::to_string(slot), nq);
        dump_raw("probe_nk_" + std::to_string(slot), nk);
        dump_raw("probe_nv_" + std::to_string(slot), nv);

        const float tau = static_cast<float>(
            1.0 + static_cast<double>(this->config_.log_scale_alpha) *
                      mnet::detail::ggml::dlog(std::max(1.0, static_cast<double>(position + 1) /
                                                                 static_cast<double>(this->config_.log_scale_floor))));

        const auto wq = this->reader_.quantized(this->reader_.tensor(prefix + "attn_q.weight"));
        std::vector<float> q(this->config_.n_head * this->config_.head_dim);
        Reader::matvec(wq, nq.data(), q.data());
        dump_raw("probe_qraw_" + std::to_string(slot), q);
        clip_rows(q, this->tensor_floats(prefix + "attn_q_clip.weight"), this->config_.n_head, this->config_.head_dim,
                  this->config_.qk_clip, true);
        if (!is_swa) {
            for (float& value : q) {
                value *= tau;
            }
        }
        dump("Qcur", q);

        const auto wk = this->reader_.quantized(this->reader_.tensor(prefix + "attn_k.weight"));
        std::vector<float> kraw(this->config_.n_head_kv * this->config_.head_dim);
        Reader::matvec(wk, nk.data(), kraw.data());
        const auto k_conv_w = this->tensor_floats(prefix + "attn_k_conv.weight");
        std::vector<float> kdyn;
        if (!is_swa) {
            const auto proj = this->reader_.quantized(this->reader_.tensor(prefix + "attn_k_conv_proj.weight"));
            kdyn.resize(this->config_.n_conv * this->config_.n_head_kv);
            Reader::matvec(proj, nk.data(), kdyn.data());
        }
        auto k = conv_step(kraw, k_conv_w, this->slots_[slot].k_hist, this->config_.n_head_kv * this->config_.head_dim,
                           this->config_.n_conv, is_swa ? nullptr : &kdyn, this->config_.n_head_kv);
        dump_raw("probe_kraw_" + std::to_string(slot), k);
        clip_rows(k, {}, this->config_.n_head_kv, this->config_.head_dim, this->config_.qk_clip, false);
        dump("Kcur", k);

        const auto wv = this->reader_.quantized(this->reader_.tensor(prefix + "attn_v.weight"));
        std::vector<float> vraw(this->config_.n_head_kv * this->config_.head_dim);
        Reader::matvec(wv, nv.data(), vraw.data());
        const auto v_conv_w = this->tensor_floats(prefix + "attn_v_conv.weight");
        std::vector<float> vdyn;
        if (!is_swa) {
            const auto proj = this->reader_.quantized(this->reader_.tensor(prefix + "attn_v_conv_proj.weight"));
            vdyn.resize(this->config_.n_conv * this->config_.n_head_kv);
            Reader::matvec(proj, nv.data(), vdyn.data());
        }
        auto v = conv_step(vraw, v_conv_w, this->slots_[slot].v_hist, this->config_.n_head_kv * this->config_.head_dim,
                           this->config_.n_conv, is_swa ? nullptr : &vdyn, this->config_.n_head_kv);
        dump_raw("probe_vraw_" + std::to_string(slot), v);
        for (std::size_t h = 0; h < this->config_.n_head_kv; ++h) {
            std::vector<float> head(v.begin() + static_cast<std::ptrdiff_t>(h * this->config_.head_dim),
                                    v.begin() + static_cast<std::ptrdiff_t>((h + 1) * this->config_.head_dim));
            head = rms_norm(head, nullptr, this->config_.eps);
            std::copy(head.begin(), head.end(), v.begin() + static_cast<std::ptrdiff_t>(h * this->config_.head_dim));
        }
        dump("Vcur", v);

        const auto wrel = this->reader_.quantized(this->reader_.tensor(prefix + "attn_rel.weight"));
        std::vector<float> rel(this->config_.n_relative * this->config_.n_head);
        Reader::matvec(wrel, nq.data(), rel.data());
        dump_raw("probe_rraw_" + std::to_string(slot), rel);
        clip_rows(rel, this->tensor_floats(prefix + "attn_rel_clip.weight"), this->config_.n_head,
                  this->config_.n_relative, this->config_.relative_clip, true);
        if (!is_swa) {
            for (float& value : rel) {
                value *= tau;
            }
        }
        dump("Rcur", rel);

        const auto rel_bias = this->tensor_floats(prefix + "attn_rel_bias.weight");
        const auto sink_w = this->reader_.quantized(this->reader_.tensor(prefix + "attn_sink_proj.weight"));
        std::vector<float> sink(this->config_.n_head);
        Reader::matvec(sink_w, nq.data(), sink.data());

        if (const char* fa_dir = std::getenv("MNET_FA_DUMP"); fa_dir != nullptr) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/msink_s%zu_q%zu.bin", fa_dir, slot, position);
            write_floats(path, sink);
        }
        auto& slot_state = this->slots_[slot];
        const std::size_t kv_dim = this->config_.n_head_kv * this->config_.head_dim;
        slot_state.k_cache.resize((slot_state.tokens + 1) * kv_dim);
        slot_state.v_cache.resize((slot_state.tokens + 1) * kv_dim);
        auto* k_dst = slot_state.k_cache.data() + slot_state.tokens * kv_dim;
        auto* v_dst = slot_state.v_cache.data() + slot_state.tokens * kv_dim;
        if (this->kv_f16_) {
            for (std::size_t j = 0; j < kv_dim; ++j) {
                k_dst[j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(k[j]));
                v_dst[j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(v[j]));
            }
        } else {
            std::copy(k.begin(), k.end(), k_dst);
            std::copy(v.begin(), v.end(), v_dst);
        }
        ++slot_state.tokens;

        std::vector<float> kqv(this->config_.n_head * this->config_.head_dim);
        const float attn_scale = 1.0F / std::sqrt(static_cast<float>(this->config_.head_dim));
        const std::size_t group = this->config_.n_head / this->config_.n_head_kv;
        const std::size_t n_extent = is_swa ? this->config_.n_swa : this->config_.n_relative_extent;
        // llama.cpp runs flash attention on CPU for this graph (the reference was
        // produced with -fa on). Replicate ggml_compute_forward_flash_attn_ext_f16:
        // online softmax with f16 K/V, kq_vec_dot = ggml_vec_dot_f16_f32, the bias
        // table (which already has -sink folded in) added to the scaled score, the
        // V mad as a single fma, S = S*ms + vs and a final 1/S rescale.
        std::vector<float> vkq(this->config_.head_dim);
        for (std::size_t h = 0; h < this->config_.n_head; ++h) {
            const std::size_t kv_head = h / group;
            float sum = 0.0F;
            float max_score = -std::numeric_limits<float>::infinity();
            std::fill(vkq.begin(), vkq.end(), 0.0F);
            std::FILE* score_file = nullptr;
            if (score_file != nullptr) {
                std::fclose(score_file);
            }
            if (const char* fa_dir = std::getenv("MNET_FA_DUMP"); fa_dir != nullptr) {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/msc_s%zu_q%zu_h%zu.bin", fa_dir, slot, position, h);
                score_file = std::fopen(path, "wb");
            }
            for (std::size_t p = 0; p < slot_state.tokens; ++p) {
                const float* kh = slot_state.k_cache.data() + p * kv_dim + kv_head * this->config_.head_dim;
                const float qk = this->kv_f16_
                                     ? dot_f16_f32(kh, q.data() + h * this->config_.head_dim, this->config_.head_dim)
                                     : dot_f32(q.data() + h * this->config_.head_dim, kh, this->config_.head_dim);
                const std::size_t distance = slot_state.tokens - 1 - p;
                const std::size_t bucket = std::min(distance, n_extent - 1);
                // n_relative (16) < 32, so this is ggml_vec_dot_f32 with its scalar
                // tail (verified bit-equal against the exported kernel).
                const float bias = dot_f32(rel_bias.data() + this->config_.n_relative * bucket,
                                           rel.data() + h * this->config_.n_relative, this->config_.n_relative);
                const float mask_value = bias - sink[h];
                // llama.cpp keeps the two roundings apart (s = s*scale; s += mask);
                // GCC would otherwise contract them into one fma.
                const float scaled_qk = rounded_product(qk, attn_scale);
                float score = scaled_qk;
                score += mask_value;
                if (score_file != nullptr) {
                    std::fwrite(&score, sizeof(float), 1, score_file);
                    std::fwrite(&mask_value, sizeof(float), 1, score_file);
                }
                const float* vh = slot_state.v_cache.data() + p * kv_dim + kv_head * this->config_.head_dim;
                float ms = 1.0F;
                float vs = 1.0F;
                if (score > max_score) {
                    const float old_max = max_score;
                    max_score = score;
                    ms = mnet::detail::ggml::expf(old_max - max_score);
                    for (std::size_t i = 0; i < this->config_.head_dim; ++i) {
                        vkq[i] *= ms;
                    }
                } else {
                    vs = mnet::detail::ggml::expf(score - max_score);
                }
                for (std::size_t i = 0; i < this->config_.head_dim; ++i) {
                    vkq[i] = std::fma(vh[i], vs, vkq[i]);
                }
                sum = sum * ms + vs;
            }
            // zero sink tensor: it still participates in the online softmax
            {
                const float sink_logit = 0.0F;
                float ms = 1.0F;
                float vs = 1.0F;
                if (sink_logit > max_score) {
                    ms = mnet::detail::ggml::expf(max_score - sink_logit);
                    max_score = sink_logit;
                    for (std::size_t i = 0; i < this->config_.head_dim; ++i) {
                        vkq[i] *= ms;
                    }
                } else {
                    vs = mnet::detail::ggml::expf(sink_logit - max_score);
                }
                sum = sum * ms + vs;
            }
            if (const char* fa_dir = std::getenv("MNET_FA_DUMP"); fa_dir != nullptr) {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/mfa_s%zu_q%zu_h%zu.bin", fa_dir, slot, position, h);
                if (std::FILE* fp = std::fopen(path, "wb")) {
                    std::fwrite(&max_score, sizeof(float), 1, fp);
                    std::fwrite(&sum, sizeof(float), 1, fp);
                    std::fwrite(vkq.data(), sizeof(float), vkq.size(), fp);
                    std::fclose(fp);
                }
            }
            const float inv_sum = sum == 0.0F ? 0.0F : 1.0F / sum;
            float* out = kqv.data() + h * this->config_.head_dim;
            for (std::size_t i = 0; i < this->config_.head_dim; ++i) {
                out[i] = vkq[i] * inv_sum;
            }
        }
        dump("kqv_out", kqv);

        const auto gate_w = this->reader_.quantized(this->reader_.tensor(prefix + "attn_gate.weight"));
        std::vector<float> gate(this->config_.n_head * this->config_.head_dim);
        Reader::matvec(gate_w, nq.data(), gate.data());
        for (std::size_t i = 0; i < gate.size(); ++i) {
            kqv[i] *= sigmoid(gate[i]);
        }
        const auto out_w = this->reader_.quantized(this->reader_.tensor(prefix + "attn_output.weight"));
        std::vector<float> projected(this->config_.n_embd);
        Reader::matvec(out_w, kqv.data(), projected.data());
        const auto post_conv = this->tensor_floats(prefix + "attn_post_conv.weight");
        auto post = conv_step(projected, post_conv, slot_state.attn_hist, this->config_.n_embd, this->config_.n_conv,
                              nullptr, 1);
        dump("attn_out", post);
        const bool block_start = ip % this->config_.block_size == 0;
        std::vector<float> residual = post;
        if (!block_start) {
            residual = input;
            for (std::size_t i = 0; i < this->config_.n_embd; ++i) {
                residual[i] += post[i];
            }
        }
        if (block_start) {
            blocks.push_back(input);
        }
        dump("ffn_inp", residual);

        const auto ffn_res_query = this->tensor_floats(prefix + "ffn_res_query.weight");
        std::vector<std::vector<float>> states;
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            states.push_back(*it);
        }
        states.push_back(residual);
        auto mixed = mix_states(states, ffn_res_query, kMixEps);
        if (messages != nullptr) {
            mixed = control(mixed, (*messages)[3], this->reader_, prefix, "ffn", kMixEps);
        }
        const auto ffn_norm_w = this->tensor_floats(prefix + "ffn_norm.weight");
        const auto moe_in = rms_norm(mixed, &ffn_norm_w, this->config_.eps);
        dump("moe_in", moe_in);

        const auto gate_inp = this->reader_.quantized(this->reader_.tensor(prefix + "ffn_gate_inp.weight"));
        const auto probs_bias = this->tensor_floats(prefix + "exp_probs_b.bias");
        std::vector<float> logits_moe(this->config_.n_expert);
        Reader::matvec(gate_inp, moe_in.data(), logits_moe.data());
        dump("ffn_moe_logits", logits_moe);
        std::vector<float> probs(this->config_.n_expert);
        std::vector<float> selection(this->config_.n_expert);
        for (std::size_t e = 0; e < this->config_.n_expert; ++e) {
            probs[e] = sigmoid(logits_moe[e]);
            selection[e] = probs[e] + probs_bias[e];
        }
        std::vector<std::size_t> order(this->config_.n_expert);
        for (std::size_t e = 0; e < this->config_.n_expert; ++e) {
            order[e] = e;
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return selection[a] > selection[b]; });
        double weight_sum = 0.0;
        for (std::size_t i = 0; i < this->config_.n_used; ++i) {
            weight_sum += static_cast<double>(probs[order[i]]);
        }
        std::vector<float> weights(this->config_.n_used);
        const auto denom = static_cast<float>(weight_sum + 1e-20);
        for (std::size_t i = 0; i < this->config_.n_used; ++i) {
            weights[i] = probs[order[i]] / denom * this->config_.expert_scale;
        }
        dump("ffn_moe_weights_scaled", weights);

        const auto gate_up_info = this->reader_.tensor(prefix + "ffn_gate_up_exps.weight");
        const auto down_info = this->reader_.tensor(prefix + "ffn_down_exps.weight");
        const std::size_t n_ff = static_cast<std::size_t>(gate_up_info.dims[1]) / 2;
        std::vector<float> expert_out(this->config_.n_embd, 0.0F);
        std::vector<float> act(n_ff);
        for (std::size_t i = 0; i < this->config_.n_used; ++i) {
            const auto expert = static_cast<std::int32_t>(order[i]);
            const auto gate_up_w = this->reader_.quantized(gate_up_info, expert);
            std::vector<float> gu(2 * n_ff);
            Reader::matvec(gate_up_w, moe_in.data(), gu.data());
            if (i == 0) {
                dump("ffn_moe_gate_up_first", gu);
            }
            for (std::size_t idx = 0; idx < n_ff; ++idx) {
                act[idx] = situ_gate(gu[idx], this->config_.situ_beta) *
                           situ_linear(gu[n_ff + idx], this->config_.situ_linear_beta);
            }
            if (i == 0) {
                dump("ffn_moe_situ_first", act);
            }
            const auto down_w = this->reader_.quantized(down_info, expert);
            std::vector<float> down(this->config_.n_embd);
            Reader::matvec(down_w, act.data(), down.data());
            for (std::size_t idx = 0; idx < this->config_.n_embd; ++idx) {
                // llama rounds the per-expert weighted product (ggml_mul) before the
                // expert sum; fusing it into the add is 1 ulp off.
                expert_out[idx] += rounded_product(weights[i], down[idx]);
            }
        }
        dump("ffn_moe_out", expert_out);
        const auto ffn_post_conv = this->tensor_floats(prefix + "ffn_post_conv.weight");
        auto ffn_out = conv_step(expert_out, ffn_post_conv, slot_state.ffn_hist, this->config_.n_embd,
                                 this->config_.n_conv, nullptr, 1);
        std::vector<float> out(this->config_.n_embd);
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = residual[i] + ffn_out[i];
        }
        dump("ffn_out", ffn_out);
        dump("l_out", out);
        return out;
    }

    Reader reader_;
    Config config_;
    std::vector<SlotState> slots_;
};

}  // namespace

int main(std::int32_t argc, char** argv) {
    try {
        if (argc < 3) {
            std::fprintf(stderr, "usage: mnet_inkling2 MODEL.gguf TOKEN_ID... [--dump DIR] [--text TEXT] [--kv-f32]\n");
            return 2;
        }
        const std::string model_path = argv[1];
        std::string dump_dir;
        std::string text;
        // llama.cpp's default KV cache type is F16; match it unless --kv-f32.
        bool kv_f16 = true;
        std::vector<std::int32_t> tokens;
        for (std::int32_t i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--dump" && i + 1 < argc) {
                dump_dir = argv[++i];
            } else if (arg == "--text" && i + 1 < argc) {
                text = argv[++i];
            } else if (arg == "--kv-f32") {
                kv_f16 = false;
            } else if (!arg.empty() && arg[0] != '-' && arg != "0") {
                tokens.push_back(std::stoi(arg));
            }
        }
        if (!dump_dir.empty()) {
            std::filesystem::create_directories(dump_dir);
        }
        const auto begin = std::chrono::steady_clock::now();
        Model model(model_path);
        model.set_kv_f16(kv_f16);
        if (!text.empty()) {
            const auto encoder = mnet::bpe::Encoder::from_reader(model.reader());
            tokens = encoder.encode(text);
        }
        if (tokens.empty()) {
            std::fprintf(stderr, "no tokens\n");
            return 2;
        }
        // >1 token: llama.cpp switches its matmuls to the tinyBLAS kernels.
        mnet::gguf::ggml_tinyblas_batch() = tokens.size() > 1;
        const auto logits = model.logits(tokens, dump_dir);
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        double sum = 0.0;
        for (const float value : logits) {
            sum += static_cast<double>(value);
        }
        std::printf("mnet_inkling2: tokens=%zu logits=%zu sum=%.4f time=%.0f ms\n", tokens.size(), logits.size(), sum,
                    ms);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_inkling2: %s\n", error.what());
        return 1;
    }
}
