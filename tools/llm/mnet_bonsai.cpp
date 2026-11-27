// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Native Bonsai (qwen35) decoder: 64 hybrid layers, weights streamed from the
// GGUF as quantized blocks, persistent gated-delta-net conv/SSM state and
// attention KV cache. Pure mnet, no llama.cpp calls.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <mnet/detail/compute.hpp>
#include <mnet/detail/ggml_kernels.hpp>
#include <mnet/gguf.hpp>
#include <mnet/tokenizer/bpe.hpp>

namespace {

using mnet::gguf::Reader;

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kSsmHeadDim = 128;
constexpr std::size_t kSsmKeyHeads = 16;
constexpr std::size_t kSsmValueHeads = 48;
constexpr std::size_t kSsmConvKernel = 4;
constexpr std::size_t kRopeDims = 64;

struct Config {
    std::size_t n_layer = 64;
    std::size_t n_embd = 5120;
    std::size_t n_ff = 17408;
    std::size_t n_head = 24;
    std::size_t n_head_kv = 4;
    std::size_t n_vocab = 0;
    std::size_t full_attention_interval = 4;
    float eps = 1.0e-6F;
    float rope_base = 1.0e7F;
    std::uint32_t hadamard_block = 1024;
};

void write_floats(const std::string& path, const float* data, std::size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

// llama.cpp folds the normalized Hadamard into a dense F32 rotation matrix and
// multiplies with ggml_vec_dot_f32 (4 x 8-lane FMA accumulators, then a tree
// reduction). The butterfly transform is mathematically identical but rounds
// differently, which flips a few Q8_0 quantization buckets per token, so the
// dense form is reproduced here bit for bit.
const std::vector<float>& rotation_matrix(std::uint32_t block_size) {
    static std::map<std::uint32_t, std::vector<float>> cache;
    auto it = cache.find(block_size);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<float> matrix(static_cast<std::size_t>(block_size) * block_size);
    const float scale = 1.0F / std::sqrt(static_cast<float>(block_size));
    for (std::uint32_t row = 0; row < block_size; ++row) {
        for (std::uint32_t col = 0; col < block_size; ++col) {
            const bool negative = (std::popcount(row & col) & 1) != 0;
            matrix[static_cast<std::size_t>(row) * block_size + col] = negative ? -scale : scale;
        }
    }
    return cache.emplace(block_size, std::move(matrix)).first->second;
}

// All float kernels live in <mnet/detail/ggml_kernels.hpp>; the model code
// below has no target branches of its own.
using mnet::detail::ggml::dot_f16_f32;
using mnet::detail::ggml::dot_f32;
using mnet::detail::ggml::rotate_block;

// ggml_ssm_conv rounds every product before adding it (no FMA contraction),
// so keep the product materialised. Only matters once the conv window holds
// more than one non-zero tap (i.e. from the second token on).
[[nodiscard]] inline float rounded_product(float a, float b) {
    volatile float product = a * b;
    return product;
}

bool g_use_fwht = true;

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

// Normalized block Hadamard, no sign vector (the inverse-lookup transform).
void hadamard_raw(std::vector<float>& values, std::uint32_t block_size) {
    if (g_use_fwht) {
        const float scale = 1.0F / std::sqrt(static_cast<float>(block_size));
        for (std::size_t start = 0; start < values.size(); start += block_size) {
            for (std::size_t i = 0; i < block_size; ++i) {
                values[start + i] *= scale;
            }
            fwht(values.data() + start, block_size);
        }
        return;
    }
    const auto& matrix = rotation_matrix(block_size);
    std::vector<float> rotated(values.size());
    for (std::size_t start = 0; start < values.size(); start += block_size) {
        rotate_block(matrix.data(), values.data() + start, rotated.data() + start, block_size);
    }
    values.swap(rotated);
}

// Forward activation transform for a folded weight: H(D x).
void hadamard_forward(std::vector<float>& values, const std::vector<std::int32_t>& signs, std::uint32_t block_size) {
    if (signs.size() != values.size()) {
        throw std::runtime_error("sign vector width mismatch");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] *= static_cast<float>(signs[i]);
    }
    hadamard_raw(values, block_size);
}

[[nodiscard]] std::vector<float> rms_norm(const float* input, std::size_t size, const std::vector<float>& weight,
                                          float eps) {
    // llama.cpp: ggml_float (double) accumulator of float squares, then float mean/scale.
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

// llama's build_gdn_l2_norm:
//   ggml_scale(ggml_rms_norm(x, eps/ne0), 1/sqrtf(ne0))
// i.e. two separate multiplies: ((x * 1/sqrt(mean(x^2) + eps/n)) * 1/sqrt(n)).
void l2_normalize(float* data, std::size_t size, float eps) {
    const float n = static_cast<float>(size);
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        squares += static_cast<double>(data[i] * data[i]);
    }
    const float mean = static_cast<float>(squares / static_cast<double>(size));
    const float rms = 1.0F / std::sqrt(mean + eps / n);
    const float scale = 1.0F / std::sqrt(n);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = data[i] * rms * scale;
    }
}

void linear_f32(const std::vector<float>& w, std::size_t out_dim, std::size_t in_dim, const float* x, float* y) {
    // ggml_mul_mat with F32 weights -> ggml_vec_dot_f32 per row.
    for (std::size_t row = 0; row < out_dim; ++row) {
        y[row] = dot_f32(w.data() + row * in_dim, x, in_dim);
    }
}

[[nodiscard]] float sigmoid(float value) {
    return 1.0F / (1.0F + mnet::detail::ggml::expf(-value));
}
[[nodiscard]] float softplus(float value) {
    return value > 20.0F ? value : mnet::detail::ggml::log1pf(mnet::detail::ggml::expf(value));
}

using mnet::detail::ggml::silu_inplace;
using mnet::detail::ggml::swiglu;

void rope_neox(float* head, std::size_t head_dim, std::size_t position, float base) {
    const float theta_scale = mnet::detail::ggml::powf(base, -2.0F / static_cast<float>(kRopeDims));
    float theta = static_cast<float>(position);
    for (std::size_t i = 0; i < kRopeDims / 2; ++i) {
        const float cos_theta = std::cos(theta);
        const float sin_theta = std::sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + kRopeDims / 2];
        head[i] = x0 * cos_theta - x1 * sin_theta;
        head[i + kRopeDims / 2] = x0 * sin_theta + x1 * cos_theta;
        theta *= theta_scale;
    }
    (void)head_dim;
}

// Same rotation factors rope_neox uses, tabulated for the device block. The
// iterative theta *= theta_scale keeps the values identical to the scalar path.
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

struct LayerWeights {
    // attention layers
    Reader::Quantized q;
    Reader::Quantized k;
    Reader::Quantized v;
    Reader::Quantized output;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    // recurrent layers
    Reader::Quantized qkv;
    Reader::Quantized gate;
    Reader::Quantized ssm_out;
    std::vector<float> beta_w;
    std::vector<float> alpha_w;
    std::vector<float> dt;
    std::vector<float> ssm_a;
    std::vector<float> conv_w;
    std::vector<float> ssm_norm;
    // shared
    std::vector<float> attn_norm;
    std::vector<float> post_norm;
    Reader::Quantized ffn_gate;
    Reader::Quantized ffn_up;
    Reader::Quantized ffn_down;
};

struct LayerState {
    std::vector<float> k_cache;  // [kv_heads * head_dim * tokens]
    std::vector<float> v_cache;
    std::size_t cache_tokens = 0;
    std::vector<float> conv_state;  // [(kernel-1) * conv_dim]
    std::vector<float> ssm_state;   // [value_heads * head_dim * head_dim]
};

// TEMP profiling: MNET_PROFILE=1 prints the CPU split accumulated per section.
namespace {
struct TimeTable {
    static constexpr std::size_t kSlots = 24;
    const char* names[kSlots] = {};
    double seconds[kSlots] = {};
    std::size_t count = 0;
    ~TimeTable() {
        if (std::getenv("MNET_PROFILE") == nullptr) {
            return;
        }
        for (std::size_t i = 0; i < this->count; ++i) {
            std::fprintf(stderr, "mnet: profile %-14s %8.3f s\n", this->names[i], this->seconds[i]);
        }
    }
};
TimeTable& time_table() {
    static TimeTable table;
    return table;
}
}  // namespace

void add_time(const char* name, double seconds) {
    TimeTable& table = time_table();
    for (std::size_t i = 0; i < table.count; ++i) {
        if (std::string_view(table.names[i]) == name) {
            table.seconds[i] += seconds;
            return;
        }
    }
    if (table.count < TimeTable::kSlots) {
        table.names[table.count] = name;
        table.seconds[table.count] = seconds;
        ++table.count;
    }
}

[[nodiscard]] inline double since(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

class Model {
public:
    Model(std::string path) : reader_(std::move(path)) { this->load_config(); }

    [[nodiscard]] const Config& config() const { return this->config_; }

    // Set when the run saves or loads the recurrent state: the device keeps
    // that state resident, so the step-by-step path must own it instead.
    void set_state_io(bool enabled) { this->state_io_ = enabled; }
    [[nodiscard]] const Reader& reader() const { return this->reader_; }

    [[nodiscard]] bool is_recurrent(std::size_t layer) const {
        return (layer + 1) % this->config_.full_attention_interval != 0;
    }

    void init_state(std::size_t layers) {
        this->state_.resize(layers);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            if (this->is_recurrent(layer)) {
                this->state_[layer].conv_state.assign((kSsmConvKernel - 1) * 10240, 0.0F);
                this->state_[layer].ssm_state.assign(kSsmValueHeads * kSsmHeadDim * kSsmHeadDim, 0.0F);
            }
        }
    }

    void save_state(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot write state " + path);
        }
        file.write("MNETBST1", 8);
        const auto write_u64 = [&file](std::uint64_t value) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        const auto write_vector = [&file, &write_u64](const std::vector<float>& values) {
            write_u64(values.size());
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(float)));
        };
        write_u64(this->state_.size());
        for (const auto& state : this->state_) {
            write_u64(state.cache_tokens);
            write_vector(state.k_cache);
            write_vector(state.v_cache);
            write_vector(state.conv_state);
            write_vector(state.ssm_state);
        }
        if (!file) {
            throw std::runtime_error("short write of state " + path);
        }
    }

    void load_state(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open state " + path);
        }
        char magic[8] = {};
        file.read(magic, 8);
        if (std::memcmp(magic, "MNETBST1", 8) != 0) {
            throw std::runtime_error("bad state file " + path);
        }
        const auto read_u64 = [&file]() {
            std::uint64_t value = 0;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            return value;
        };
        const auto read_vector = [&file, &read_u64]() {
            std::vector<float> values(read_u64());
            file.read(reinterpret_cast<char*>(values.data()),
                      static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!file) {
                throw std::runtime_error("short state read");
            }
            return values;
        };
        const auto layers = read_u64();
        if (layers != this->state_.size()) {
            throw std::runtime_error("state layer count mismatch");
        }
        for (auto& state : this->state_) {
            state.cache_tokens = read_u64();
            state.k_cache = read_vector();
            state.v_cache = read_vector();
            state.conv_state = read_vector();
            state.ssm_state = read_vector();
        }
    }

    // Embedding rows are stored in the folded basis: h = D H z.
    [[nodiscard]] std::vector<float> embed(std::int32_t token) const {
        using mnet::gguf::Reader;
        const auto& info = this->reader_.tensor("token_embd.weight");
        const auto rows = static_cast<std::size_t>(info.dims[1]);
        if (token < 0 || static_cast<std::size_t>(token) >= rows) {
            throw std::runtime_error("token id out of range");
        }
        const std::size_t row_bytes = this->config_.n_embd / 64 * 18;
        const auto raw = this->reader_.read_raw(info, static_cast<std::uint64_t>(token) * row_bytes, row_bytes);
        std::vector<float> row = Reader::dequantize_block(info.type, raw, this->config_.n_embd);
        hadamard_raw(row, this->config_.hadamard_block);
        const auto signs = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_embd));
        if (signs.size() != row.size()) {
            throw std::runtime_error("embedding sign width mismatch");
        }
        for (std::size_t i = 0; i < row.size(); ++i) {
            row[i] *= static_cast<float>(signs[i]);
        }
        return row;
    }

    // Run `tokens` through every layer starting at `start_position`, updating
    // hidden states in place. Returns the final hidden states.
    std::vector<float> forward(std::vector<float> hidden, const std::vector<std::int32_t>& positions,
                               const std::string& dump_dir, bool dump) {
        const std::size_t tokens = positions.size();
        for (std::size_t layer = 0; layer < this->config_.n_layer; ++layer) {
            LayerWeights weights = this->load_layer(layer);
            LayerState& state = this->state_[layer];
            for (std::size_t t = 0; t < tokens; ++t) {
                float* x = hidden.data() + t * this->config_.n_embd;
                this->layer_forward(layer, weights, state, x, static_cast<std::size_t>(positions[t]),
                                    dump ? &dump_dir : nullptr, t);
            }
        }
        return hidden;
    }

    [[nodiscard]] std::vector<float> logits(const float* hidden) const {
        auto normed = rms_norm(hidden, this->config_.n_embd, this->output_norm_, this->config_.eps);
        const auto signs = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_embd));
        hadamard_forward(normed, signs, this->config_.hadamard_block);
        std::vector<float> out(this->config_.n_vocab);
        Reader::matvec(this->output_, normed.data(), out.data());
        return out;
    }

    void load_shared() {
        this->output_norm_ = this->reader_.dequantize(this->reader_.tensor("output_norm.weight"));
        this->output_ = this->reader_.quantized(this->reader_.tensor("output.weight"));
    }

private:
    void load_config() {
        const auto& ints = this->reader_.integers();
        const auto& floats = this->reader_.floats();
        this->config_.n_layer = ints.at("qwen35.block_count");
        this->config_.n_embd = ints.at("qwen35.embedding_length");
        this->config_.n_ff = ints.at("qwen35.feed_forward_length");
        this->config_.n_head = ints.at("qwen35.attention.head_count");
        this->config_.n_head_kv = ints.at("qwen35.attention.head_count_kv");
        this->config_.full_attention_interval = ints.at("qwen35.full_attention_interval");
        this->config_.eps = static_cast<float>(floats.at("qwen35.attention.layer_norm_rms_epsilon"));
        this->config_.rope_base = static_cast<float>(floats.at("qwen35.rope.freq_base"));
        this->config_.hadamard_block = static_cast<std::uint32_t>(ints.at("prism.hadamard.block_size"));
        this->config_.n_vocab = this->reader_.tensor("token_embd.weight").dims[1];
    }

    [[nodiscard]] std::vector<float> tensor_floats(const std::string& name) const {
        return this->reader_.dequantize(this->reader_.tensor(name));
    }

    [[nodiscard]] LayerWeights load_layer(std::size_t layer) const {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        LayerWeights weights;
        weights.attn_norm = this->tensor_floats(prefix + "attn_norm.weight");
        weights.post_norm = this->tensor_floats(prefix + "post_attention_norm.weight");
        weights.ffn_gate = this->reader_.quantized(this->reader_.tensor(prefix + "ffn_gate.weight"));
        weights.ffn_up = this->reader_.quantized(this->reader_.tensor(prefix + "ffn_up.weight"));
        weights.ffn_down = this->reader_.quantized(this->reader_.tensor(prefix + "ffn_down.weight"));
        if (this->is_recurrent(layer)) {
            weights.qkv = this->reader_.quantized(this->reader_.tensor(prefix + "attn_qkv.weight"));
            weights.gate = this->reader_.quantized(this->reader_.tensor(prefix + "attn_gate.weight"));
            weights.ssm_out = this->reader_.quantized(this->reader_.tensor(prefix + "ssm_out.weight"));
            weights.beta_w = this->tensor_floats(prefix + "ssm_beta.weight");
            weights.alpha_w = this->tensor_floats(prefix + "ssm_alpha.weight");
            weights.dt = this->tensor_floats(prefix + "ssm_dt.bias");
            weights.ssm_a = this->tensor_floats(prefix + "ssm_a");
            weights.conv_w = this->tensor_floats(prefix + "ssm_conv1d.weight");
            weights.ssm_norm = this->tensor_floats(prefix + "ssm_norm.weight");
        } else {
            weights.q = this->reader_.quantized(this->reader_.tensor(prefix + "attn_q.weight"));
            weights.k = this->reader_.quantized(this->reader_.tensor(prefix + "attn_k.weight"));
            weights.v = this->reader_.quantized(this->reader_.tensor(prefix + "attn_v.weight"));
            weights.output = this->reader_.quantized(this->reader_.tensor(prefix + "attn_output.weight"));
            weights.q_norm = this->tensor_floats(prefix + "attn_q_norm.weight");
            weights.k_norm = this->tensor_floats(prefix + "attn_k_norm.weight");
        }
        return weights;
    }

    void dump_tensor(const std::string& dir, const std::string& name, std::size_t layer,
                     const std::vector<float>& values, std::size_t token) const {
        const std::string stem = dir + "/" + name + "-" + std::to_string(layer);
        write_floats(stem + "-t" + std::to_string(token) + ".bin", values.data(), values.size());
        if (token == 0) {
            write_floats(stem + ".bin", values.data(), values.size());
        }
    }

    void layer_forward(std::size_t layer, const LayerWeights& w, LayerState& state, float* x, std::size_t position,
                       const std::string* dump_dir, std::size_t token) {
        const auto signs_embd = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_embd));
        const auto t_norm = std::chrono::steady_clock::now();
        auto xn = rms_norm(x, this->config_.n_embd, w.attn_norm, this->config_.eps);
        add_time("rms_norm_in", since(t_norm));
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_norm", layer, xn, token);
        }

        const auto t_branch = std::chrono::steady_clock::now();
        std::vector<float> branch;
        if (this->is_recurrent(layer)) {
            branch = this->recurrent_branch(w, state, xn, dump_dir, layer, token);
        } else {
            branch = this->attention_branch(w, state, xn, position, dump_dir, layer, token);
        }
        add_time("branch", since(t_branch));
        const auto t_tail = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < this->config_.n_embd; ++i) {
            x[i] += branch[i];
        }
        std::vector<float> residual(x, x + this->config_.n_embd);
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_residual", layer, residual, token);
        }

        auto pn = rms_norm(residual.data(), this->config_.n_embd, w.post_norm, this->config_.eps);
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_post_norm", layer, pn, token);
        }
        auto ph = pn;
        hadamard_forward(ph, signs_embd, this->config_.hadamard_block);
        const auto signs_hidden = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_ff));
        std::vector<float> down(this->config_.n_embd);
        bool fused = false;
        if (mnet::detail::compute::fast_path_enabled() && mnet::detail::compute::fused_ffn_enabled()) {
            fused = Reader::ffn_swiglu(w.ffn_gate, w.ffn_up, w.ffn_down, ph.data(), signs_hidden.data(),
                                       this->config_.hadamard_block, down.data());
        }
        if (!fused) {
            std::vector<float> gate(w.ffn_gate.rows);
            std::vector<float> up(w.ffn_up.rows);
            Reader::matvec_batch({{&w.ffn_gate, gate.data()}, {&w.ffn_up, up.data()}}, ph.data());
            std::vector<float> act(w.ffn_gate.rows);
            swiglu(act.data(), gate.data(), up.data(), act.size());
            if (dump_dir != nullptr) {
                this->dump_tensor(*dump_dir, "ffn_gate", layer, gate, token);
                this->dump_tensor(*dump_dir, "ffn_up", layer, up, token);
                this->dump_tensor(*dump_dir, "ffn_swiglu", layer, act, token);
            }
            hadamard_forward(act, signs_hidden, this->config_.hadamard_block);
            Reader::matvec(w.ffn_down, act.data(), down.data());
        }
        for (std::size_t i = 0; i < this->config_.n_embd; ++i) {
            x[i] += down[i];
        }
        add_time("layer_tail", since(t_tail));
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "ffn_out", layer, down, token);
            this->dump_tensor(*dump_dir, "l_out", layer, std::vector<float>(x, x + this->config_.n_embd), token);
        }
    }

    [[nodiscard]] std::vector<float> recurrent_branch(const LayerWeights& w, LayerState& state,
                                                      const std::vector<float>& xn, const std::string* dump_dir,
                                                      std::size_t layer, std::size_t token) {
        const auto signs_embd = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_embd));
        auto xh = xn;
        hadamard_forward(xh, signs_embd, this->config_.hadamard_block);

        // One submission for the whole block. --dump and the state save/load
        // tools keep the step-by-step path: the device keeps the recurrent
        // state resident and does not hand it back.
        if (dump_dir == nullptr && !this->state_io_ && mnet::detail::compute::fast_path_enabled() &&
            mnet::detail::compute::fused_ssm_enabled()) {
            const auto signs_value =
                this->reader_.signs_for_width(static_cast<std::uint32_t>(kSsmValueHeads * kSsmHeadDim));
            std::vector<float> projected(this->config_.n_embd);
            Reader::SsmParams params;
            params.input = xh.data();
            params.raw_input = xn.data();
            params.beta_w = w.beta_w.data();
            params.alpha_w = w.alpha_w.data();
            params.dt = w.dt.data();
            params.ssm_a = w.ssm_a.data();
            params.conv_w = w.conv_w.data();
            params.ssm_norm = w.ssm_norm.data();
            params.signs = signs_value.data();
            params.conv_state = state.conv_state.data();
            params.ssm_state = state.ssm_state.data();
            params.key_heads = static_cast<std::uint32_t>(kSsmKeyHeads);
            params.value_heads = static_cast<std::uint32_t>(kSsmValueHeads);
            params.head_dim = static_cast<std::uint32_t>(kSsmHeadDim);
            params.conv_kernel = static_cast<std::uint32_t>(kSsmConvKernel);
            params.block_size = this->config_.hadamard_block;
            params.eps = this->config_.eps;
            if (Reader::ssm_block(w.qkv, w.gate, w.ssm_out, params, projected.data())) {
                return projected;
            }
        }

        const auto t_matvec = std::chrono::steady_clock::now();
        std::vector<float> qkv(w.qkv.rows);
        std::vector<float> z(w.gate.rows);
        Reader::matvec(w.qkv, xh.data(), qkv.data());
        Reader::matvec(w.gate, xh.data(), z.data());
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "linear_attn_qkv_mixed", layer, qkv, token);
            this->dump_tensor(*dump_dir, "z", layer, z, token);
        }

        add_time("ssm_matvec", since(t_matvec));
        const auto t_gates = std::chrono::steady_clock::now();
        std::vector<float> beta_raw(kSsmValueHeads);
        std::vector<float> alpha_raw(kSsmValueHeads);
        linear_f32(w.beta_w, kSsmValueHeads, this->config_.n_embd, xn.data(), beta_raw.data());
        linear_f32(w.alpha_w, kSsmValueHeads, this->config_.n_embd, xn.data(), alpha_raw.data());
        std::vector<float> beta(kSsmValueHeads);
        std::vector<float> gate(kSsmValueHeads);
        for (std::size_t h = 0; h < kSsmValueHeads; ++h) {
            beta[h] = sigmoid(beta_raw[h]);
            gate[h] = softplus(alpha_raw[h] + w.dt[h]) * w.ssm_a[h];
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "beta_sigmoid", layer, beta, token);
            this->dump_tensor(*dump_dir, "gate", layer, gate, token);
        }

        add_time("ssm_gates", since(t_gates));
        const auto t_conv = std::chrono::steady_clock::now();
        // Causal depthwise convolution over the last kernel rows of qkv.
        std::vector<float> window((kSsmConvKernel - 1) * 10240 + 10240);
        std::copy(state.conv_state.begin(), state.conv_state.end(), window.begin());
        std::copy(qkv.begin(), qkv.end(), window.begin() + static_cast<std::ptrdiff_t>((kSsmConvKernel - 1) * 10240));
        std::vector<float> conv(10240);
        for (std::size_t c = 0; c < conv.size(); ++c) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < kSsmConvKernel; ++k) {
                sum += rounded_product(w.conv_w[c * kSsmConvKernel + k], window[k * 10240 + c]);
            }
            conv[c] = sum;
        }
        silu_inplace(conv.data(), conv.size());
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "conv_output_silu", layer, conv, token);
        }
        std::copy(window.begin() + static_cast<std::ptrdiff_t>(10240),
                  window.begin() + static_cast<std::ptrdiff_t>((kSsmConvKernel - 1) * 10240 + 10240),
                  state.conv_state.begin());

        add_time("ssm_conv", since(t_conv));
        const auto t_scan = std::chrono::steady_clock::now();
        float* q = conv.data();
        float* k = conv.data() + 2048;
        float* v = conv.data() + 4096;
        for (std::size_t h = 0; h < kSsmKeyHeads; ++h) {
            l2_normalize(q + h * kSsmHeadDim, kSsmHeadDim, this->config_.eps);
            l2_normalize(k + h * kSsmHeadDim, kSsmHeadDim, this->config_.eps);
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "q_conv_predelta", layer, std::vector<float>(q, q + 2048), token);
            this->dump_tensor(*dump_dir, "k_conv_predelta", layer, std::vector<float>(k, k + 2048), token);
            this->dump_tensor(*dump_dir, "v_conv_predelta", layer, std::vector<float>(v, v + 6144), token);
        }

        std::vector<float> delta(6144);
        for (std::size_t h = 0; h < kSsmValueHeads; ++h) {
            const std::size_t key_head = h % kSsmKeyHeads;
            const float* qh = q + key_head * kSsmHeadDim;
            const float* kh = k + key_head * kSsmHeadDim;
            const float* vh = v + h * kSsmHeadDim;
            float* sh = state.ssm_state.data() + h * kSsmHeadDim * kSsmHeadDim;
            const float decay = mnet::detail::ggml::expf(gate[h]);
            for (std::size_t i = 0; i < kSsmHeadDim * kSsmHeadDim; ++i) {
                sh[i] *= decay;
            }
            float d[kSsmHeadDim];
            const float inv_dim = 1.0F / std::sqrt(static_cast<float>(kSsmHeadDim));
            for (std::size_t j = 0; j < kSsmHeadDim; ++j) {
                const float* row = sh + j * kSsmHeadDim;
                const float sum = dot_f32(row, kh, kSsmHeadDim);
                d[j] = (vh[j] - sum) * beta[h];
            }
            for (std::size_t j = 0; j < kSsmHeadDim; ++j) {
                float* row = sh + j * kSsmHeadDim;
                for (std::size_t i = 0; i < kSsmHeadDim; ++i) {
                    row[i] += d[j] * kh[i];
                }
            }
            for (std::size_t j = 0; j < kSsmHeadDim; ++j) {
                const float* row = sh + j * kSsmHeadDim;
                delta[h * kSsmHeadDim + j] = dot_f32(row, qh, kSsmHeadDim) * inv_dim;
            }
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_output", layer, delta, token);
        }

        add_time("ssm_scan", since(t_scan));
        const auto t_post = std::chrono::steady_clock::now();
        // Gated RMS norm per value head, then the tiled -> grouped head permute
        // that the folded ssm_out weight was built with.
        std::vector<float> normed(6144);
        auto z_act = z;
        silu_inplace(z_act.data(), z_act.size());
        // llama's build_norm_gated: ggml_mul(build_norm(input, ssm_norm, RMS), silu(z))
        for (std::size_t h = 0; h < kSsmValueHeads; ++h) {
            const float* dh = delta.data() + h * kSsmHeadDim;
            const float* zh = z_act.data() + h * kSsmHeadDim;
            double squares = 0.0;
            for (std::size_t i = 0; i < kSsmHeadDim; ++i) {
                squares += static_cast<double>(dh[i] * dh[i]);
            }
            const float mean = static_cast<float>(squares / static_cast<double>(kSsmHeadDim));
            const float scale = 1.0F / std::sqrt(mean + this->config_.eps);
            for (std::size_t i = 0; i < kSsmHeadDim; ++i) {
                normed[h * kSsmHeadDim + i] = (dh[i] * scale * w.ssm_norm[i]) * zh[i];
            }
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "final_output", layer, normed, token);
        }
        std::vector<float> grouped(6144);
        for (std::size_t h = 0; h < kSsmValueHeads; ++h) {
            const std::size_t key_head = h % kSsmKeyHeads;
            const std::size_t rep = h / kSsmKeyHeads;
            std::copy(normed.begin() + static_cast<std::ptrdiff_t>(h * kSsmHeadDim),
                      normed.begin() + static_cast<std::ptrdiff_t>((h + 1) * kSsmHeadDim),
                      grouped.begin() + static_cast<std::ptrdiff_t>((rep + 3 * key_head) * kSsmHeadDim));
        }
        const auto signs_value = this->reader_.signs_for_width(static_cast<std::uint32_t>(6144));
        hadamard_forward(grouped, signs_value, this->config_.hadamard_block);
        std::vector<float> projected(this->config_.n_embd);
        Reader::matvec(w.ssm_out, grouped.data(), projected.data());
        add_time("ssm_post", since(t_post));
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "linear_attn_out", layer, projected, token);
        }
        return projected;
    }

    [[nodiscard]] std::vector<float> attention_branch(const LayerWeights& w, LayerState& state,
                                                      const std::vector<float>& xn, std::size_t position,
                                                      const std::string* dump_dir, std::size_t layer,
                                                      std::size_t token) {
        const auto signs_embd = this->reader_.signs_for_width(static_cast<std::uint32_t>(this->config_.n_embd));
        auto xh = xn;
        hadamard_forward(xh, signs_embd, this->config_.hadamard_block);

        // One submission for the whole block. `--dump` keeps the step-by-step
        // path so the reference dump stays complete; it is also what
        // MNET_NO_FUSED_ATTN selects for A/B runs.
        if (dump_dir == nullptr && mnet::detail::compute::fast_path_enabled() &&
            mnet::detail::compute::fused_attention_enabled()) {
            const auto signs_value = this->reader_.signs_for_width(
                static_cast<std::uint32_t>(kHeadDim * this->config_.n_head));
            this->rope_table(position);
            const std::size_t kv_elems = (state.cache_tokens + 1) * this->config_.n_head_kv * kHeadDim;
            state.k_cache.resize(kv_elems);
            state.v_cache.resize(kv_elems);
            std::vector<float> projected(this->config_.n_embd);
            Reader::AttentionParams params;
            params.input = xh.data();
            params.q_norm = w.q_norm.data();
            params.k_norm = w.k_norm.data();
            params.rope_cos = this->rope_cos_.data();
            params.rope_sin = this->rope_sin_.data();
            params.signs = signs_value.data();
            params.rope_dims = static_cast<std::uint32_t>(kRopeDims);
            params.head_count = static_cast<std::uint32_t>(this->config_.n_head);
            params.head_count_kv = static_cast<std::uint32_t>(this->config_.n_head_kv);
            params.head_dim = static_cast<std::uint32_t>(kHeadDim);
            params.block_size = this->config_.hadamard_block;
            params.cache_tokens = static_cast<std::uint32_t>(state.cache_tokens);
            params.eps = this->config_.eps;
            params.attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
            if (Reader::attention(w.q, w.k, w.v, w.output, params, state.k_cache.data(), state.v_cache.data(),
                                  projected.data())) {
                ++state.cache_tokens;
                return projected;
            }
        }

        std::vector<float> qfull(w.q.rows);
        std::vector<float> k(w.k.rows);
        std::vector<float> v(w.v.rows);
        Reader::matvec(w.q, xh.data(), qfull.data());
        Reader::matvec(w.k, xh.data(), k.data());
        Reader::matvec(w.v, xh.data(), v.data());
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "Qcur_full", layer, qfull, token);
        }

        std::vector<float> q(kHeadDim * this->config_.n_head);
        std::vector<float> gate(kHeadDim * this->config_.n_head);
        for (std::size_t h = 0; h < this->config_.n_head; ++h) {
            std::copy(qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim),
                      qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim + kHeadDim),
                      q.begin() + static_cast<std::ptrdiff_t>(h * kHeadDim));
            std::copy(qfull.begin() + static_cast<std::ptrdiff_t>(h * 2 * kHeadDim + kHeadDim),
                      qfull.begin() + static_cast<std::ptrdiff_t>((h + 1) * 2 * kHeadDim),
                      gate.begin() + static_cast<std::ptrdiff_t>(h * kHeadDim));
        }
        for (std::size_t h = 0; h < this->config_.n_head; ++h) {
            float* head = q.data() + h * kHeadDim;
            double squares = 0.0;
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                squares += static_cast<double>(head[i] * head[i]);
            }
            const auto mean = static_cast<float>(squares / static_cast<double>(kHeadDim));
            const float scale = 1.0F / std::sqrt(mean + this->config_.eps);
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                head[i] = head[i] * scale * w.q_norm[i];
            }
        }
        for (std::size_t h = 0; h < this->config_.n_head_kv; ++h) {
            float* head = k.data() + h * kHeadDim;
            double squares = 0.0;
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                squares += static_cast<double>(head[i] * head[i]);
            }
            const auto mean = static_cast<float>(squares / static_cast<double>(kHeadDim));
            const float scale = 1.0F / std::sqrt(mean + this->config_.eps);
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                head[i] = head[i] * scale * w.k_norm[i];
            }
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "Qcur_normed", layer, q, token);
            this->dump_tensor(*dump_dir, "Kcur_normed", layer, k, token);
        }
        for (std::size_t h = 0; h < this->config_.n_head; ++h) {
            rope_neox(q.data() + h * kHeadDim, kHeadDim, position, this->config_.rope_base);
        }
        for (std::size_t h = 0; h < this->config_.n_head_kv; ++h) {
            rope_neox(k.data() + h * kHeadDim, kHeadDim, position, this->config_.rope_base);
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "Qcur", layer, q, token);
            this->dump_tensor(*dump_dir, "Kcur", layer, k, token);
            this->dump_tensor(*dump_dir, "Vcur", layer, v, token);
        }

        // Append to the persistent KV cache. llama keeps it in f16 (the reference
        // attention output equals float(f16(V))), so round on store and run the
        // CPU flash-attention update over the cached keys.
        const std::size_t cache_elems = state.cache_tokens * this->config_.n_head_kv * kHeadDim;
        state.k_cache.resize(cache_elems + this->config_.n_head_kv * kHeadDim);
        state.v_cache.resize(cache_elems + this->config_.n_head_kv * kHeadDim);
        for (std::size_t j = 0; j < this->config_.n_head_kv * kHeadDim; ++j) {
            state.k_cache[cache_elems + j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(k[j]));
            state.v_cache[cache_elems + j] = mnet::gguf::half_to_float(mnet::gguf::float_to_half(v[j]));
        }
        ++state.cache_tokens;

        const std::size_t group = this->config_.n_head / this->config_.n_head_kv;
        std::vector<float> attn(this->config_.n_head * kHeadDim);
        std::vector<float> vkq(kHeadDim);
        const float attn_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
        for (std::size_t h = 0; h < this->config_.n_head; ++h) {
            const std::size_t kv_head = h / group;
            const float* qh = q.data() + h * kHeadDim;
            float sum = 0.0F;
            float max_score = -std::numeric_limits<float>::infinity();
            std::fill(vkq.begin(), vkq.end(), 0.0F);
            for (std::size_t p = 0; p < state.cache_tokens; ++p) {
                const float* kh = state.k_cache.data() + (p * this->config_.n_head_kv + kv_head) * kHeadDim;
                const float* vh = state.v_cache.data() + (p * this->config_.n_head_kv + kv_head) * kHeadDim;
                float score = dot_f16_f32(kh, qh, kHeadDim) * attn_scale;
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
            float* out = attn.data() + h * kHeadDim;
            for (std::size_t i = 0; i < kHeadDim; ++i) {
                out[i] = vkq[i] * inv;
            }
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_pregate", layer, attn, token);
        }
        for (std::size_t i = 0; i < attn.size(); ++i) {
            attn[i] *= sigmoid(gate[i]);
        }
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_gated", layer, attn, token);
        }
        const auto signs_value =
            this->reader_.signs_for_width(static_cast<std::uint32_t>(kHeadDim * this->config_.n_head));
        hadamard_forward(attn, signs_value, this->config_.hadamard_block);
        std::vector<float> projected(this->config_.n_embd);
        Reader::matvec(w.output, attn.data(), projected.data());
        if (dump_dir != nullptr) {
            this->dump_tensor(*dump_dir, "attn_output", layer, projected, token);
        }
        return projected;
    }

    Reader reader_;
    Config config_;
    bool state_io_ = false;
    std::vector<LayerState> state_;
    std::vector<float> output_norm_;
    Reader::Quantized output_;
    std::vector<float> rope_cos_;
    std::vector<float> rope_sin_;
    std::size_t rope_position_ = static_cast<std::size_t>(-1);

    // Tabulated RoPE factors for `position`, cached because every attention
    // layer of one token uses the same rotation.
    void rope_table(std::size_t position) {
        if (this->rope_position_ == position) {
            return;
        }
        rope_tables(position, this->config_.rope_base, this->rope_cos_, this->rope_sin_);
        this->rope_position_ = position;
    }
};

}  // namespace

int main(std::int32_t argc, char** argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr,
                         "usage: mnet_bonsai MODEL.gguf TOKENS.txt OUT_LOGITS.bin [--start-pos N] [--generate N]"
                         " [--dump DIR] [--save-state F] [--load-state F] [--dense-hadamard]\n"
                         "       mnet_bonsai MODEL.gguf - OUT_LOGITS.bin --text \"Hello world\" [...]\n"
                         "       TOKENS.txt: whitespace-separated token ids; '-' with --text for native BPE\n");
            return 2;
        }
        const std::string model_path = argv[1];
        const std::string tokens_path = argv[2];
        const std::string output_path = argv[3];
        std::size_t start_pos = 0;
        std::size_t generate = 0;
        std::string dump_dir;
        std::string text;
        std::string save_state_path;
        std::string load_state_path;
        for (std::int32_t i = 4; i + 1 < argc; i += 2) {
            const std::string flag = argv[i];
            if (flag == "--start-pos") {
                start_pos = std::stoul(argv[i + 1]);
            } else if (flag == "--generate") {
                generate = std::stoul(argv[i + 1]);
            } else if (flag == "--dump") {
                dump_dir = argv[i + 1];
            } else if (flag == "--text") {
                text = argv[i + 1];
            } else if (flag == "--dense-hadamard") {
                g_use_fwht = false;
            } else if (flag == "--save-state") {
                save_state_path = argv[i + 1];
            } else if (flag == "--load-state") {
                load_state_path = argv[i + 1];
            } else {
                std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
                return 2;
            }
        }

        const auto begin = std::chrono::steady_clock::now();
        Model model(model_path);
        std::optional<mnet::bpe::Encoder> encoder;
        std::vector<std::int32_t> tokens;
        if (!text.empty()) {
            encoder.emplace(mnet::bpe::Encoder::from_reader(model.reader()));
            tokens = encoder->encode(text);
        } else {
            std::ifstream token_file(tokens_path);
            if (!token_file) {
                std::fprintf(stderr, "cannot open %s\n", tokens_path.c_str());
                return 2;
            }
            for (std::int32_t value = 0; token_file >> value;) {
                tokens.push_back(value);
            }
        }
        if (tokens.empty()) {
            std::fprintf(stderr, "no tokens\n");
            return 2;
        }
        model.load_shared();
        model.set_state_io(!load_state_path.empty() || !save_state_path.empty());
        model.init_state(model.config().n_layer);
        if (!load_state_path.empty()) {
            model.load_state(load_state_path);
        }

        const auto& cfg = model.config();
        std::vector<float> hidden(tokens.size() * cfg.n_embd);
        std::vector<std::int32_t> positions(tokens.size());
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            const auto row = model.embed(tokens[t]);
            std::copy(row.begin(), row.end(), hidden.begin() + static_cast<std::ptrdiff_t>(t * cfg.n_embd));
            positions[t] = static_cast<std::int32_t>(start_pos + t);
        }
        if (!dump_dir.empty()) {
            std::filesystem::create_directories(dump_dir);
            write_floats(dump_dir + "/model.input_embed.bin", hidden.data(), cfg.n_embd);
        } else {
            dump_dir.clear();
        }

        const auto logits_start = std::chrono::steady_clock::now();
        hidden = model.forward(std::move(hidden), positions, dump_dir, !dump_dir.empty());
        const auto logits_begin = std::chrono::steady_clock::now();
        std::vector<float> last(hidden.begin() + static_cast<std::ptrdiff_t>((tokens.size() - 1) * cfg.n_embd),
                                hidden.end());
        auto normed = rms_norm(last.data(), cfg.n_embd,
                               model.reader().dequantize(model.reader().tensor("output_norm.weight")), cfg.eps);
        if (!dump_dir.empty()) {
            write_floats(dump_dir + "/result_norm.bin", normed.data(), normed.size());
        }
        auto logits = model.logits(last.data());
        write_floats(output_path, logits.data(), logits.size());
        if (!dump_dir.empty()) {
            write_floats(dump_dir + "/result_output.bin", logits.data(), logits.size());
        }
        const auto end = std::chrono::steady_clock::now();
        const double forward_ms = std::chrono::duration<double, std::milli>(logits_begin - logits_start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        std::printf("mnet_bonsai: tokens=%zu layers=%zu forward=%.1f ms total=%.1f ms -> %s\n", tokens.size(),
                    cfg.n_layer, forward_ms, total_ms, output_path.c_str());

        if (generate > 0) {
            const auto& vocab = model.reader().string_arrays().at("tokenizer.ggml.tokens");
            std::vector<std::int32_t> generated;
            generated.reserve(generate);
            std::string output_text;
            const auto gen_begin = std::chrono::steady_clock::now();
            for (std::size_t step = 0; step < generate; ++step) {
                const auto best = std::max_element(logits.begin(), logits.end());
                const auto next = static_cast<std::int32_t>(std::distance(logits.begin(), best));
                generated.push_back(next);
                if (encoder.has_value()) {
                    output_text += encoder->decode({next});
                } else if (static_cast<std::size_t>(next) < vocab.size()) {
                    output_text += vocab[static_cast<std::size_t>(next)];
                }
                std::printf("gen %zu: token %d\n", step, next);
                if (step + 1 == generate) {
                    break;
                }
                auto row = model.embed(next);
                const auto position = start_pos + tokens.size() + step;
                last = model.forward(std::move(row), {static_cast<std::int32_t>(position)}, "", false);
                logits = model.logits(last.data());
            }
            const auto gen_end = std::chrono::steady_clock::now();
            const double gen_ms = std::chrono::duration<double, std::milli>(gen_end - gen_begin).count();
            std::ofstream token_out(output_path + ".tokens.txt");
            for (const auto value : generated) {
                token_out << value << '\n';
            }
            std::printf("mnet_bonsai: generated %zu tokens in %.1f ms (%.2f tok/s)\n", generated.size(), gen_ms,
                        static_cast<double>(generated.size()) / (gen_ms / 1000.0));
            std::printf("text: %s\n", output_text.c_str());
        }
        if (!save_state_path.empty()) {
            model.save_state(save_state_path);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_bonsai: %s\n", error.what());
        return 1;
    }
}
