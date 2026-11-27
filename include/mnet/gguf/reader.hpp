// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_GGUF_READER_HPP
#define MNET_GGUF_READER_HPP

// GGUF container reader: parses the header, hands out tensor bytes and runs
// the quantised matvec through <mnet/gguf/kernels.hpp>. The heavy methods live
// in src/gguf/reader.cpp; the small accessors stay inline.

#include <cstdint>

#include <fstream>
#include <istream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mnet/gguf/format.hpp>

namespace mnet::gguf {

class Reader {
public:
    explicit Reader(std::string path);

    [[nodiscard]] const std::string& path() const { return this->path_; }
    [[nodiscard]] std::uint32_t version() const { return this->version_; }
    [[nodiscard]] std::uint64_t data_start() const { return this->data_start_; }
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return this->tensors_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const { return this->strings_; }
    [[nodiscard]] const std::unordered_map<std::string, std::uint64_t>& integers() const { return this->integers_; }
    [[nodiscard]] const std::unordered_map<std::string, double>& floats() const { return this->floats_; }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::int64_t>>& int_arrays() const {
        return this->int_arrays_;
    }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::string>>& string_arrays() const {
        return this->string_arrays_;
    }

    // Sign vector for a Hadamard-folded weight width, concatenated as stored in
    // `prism.hadamard.sign_widths` / `sign_values`.
    [[nodiscard]] std::vector<std::int32_t> signs_for_width(std::uint32_t width) const;

    [[nodiscard]] const TensorInfo& tensor(std::string_view name) const;

    [[nodiscard]] std::vector<std::uint8_t> read_raw(const TensorInfo& info, std::uint64_t byte_offset = 0,
                                                     std::uint64_t byte_count = 0) const;

    // Dequantize one tensor to fp32. `expert` slices a 3-D MoE weight and
    // applies the per-expert `<name>.scale` when present.
    [[nodiscard]] std::vector<float> dequantize(const TensorInfo& info, std::int32_t expert = -1) const;

    [[nodiscard]] static std::vector<float> dequantize_block(std::uint32_t type, const std::vector<std::uint8_t>& raw,
                                                             std::uint64_t count);
    [[nodiscard]] static float dot_row(std::uint32_t type, const std::uint8_t* data, const float* input,
                                       std::uint64_t columns);

    // A tensor kept in its GGUF block-quantized form. `matvec` dequantizes each
    // block directly into the dot product, so the fp32 copy of the weight is
    // never materialized (a 16x saving for Q2_0 versus fp32).
    struct Quantized {
        std::uint32_t type = kGgmlF32;
        std::uint64_t rows = 0;      // output features
        std::uint64_t columns = 0;   // input features
        float scale = 1.0F;          // per-expert MoE scale, folded at the output
        std::uint64_t identity = 0;  // unique per loaded tensor; device memory caches key on it
        std::vector<std::uint8_t> blocks;

        [[nodiscard]] bool empty() const { return this->blocks.empty(); }
    };

    [[nodiscard]] Quantized quantized(const TensorInfo& info, std::int32_t expert = -1) const;

    // y = W x with W kept quantized. Deterministic: each output element is
    // accumulated left to right in double, independent of thread count.
    static void matvec(const Quantized& matrix, const float* input, float* output);

    // Several independent matrices applied to the same activation. The compute
    // layer may submit them as one batch (one GPU wait instead of N); the
    // result is identical to calling matvec for each pair.
    static void matvec_batch(std::initializer_list<std::pair<const Quantized*, float*>> jobs, const float* input);

    // Fused FFN: gate/up matvec, SwiGLU, signed normalized Hadamard and the
    // down matvec in one device submission. Returns false when the backend
    // cannot take the whole block; the caller then runs the steps itself.
    [[nodiscard]] static bool ffn_swiglu(const Quantized& gate, const Quantized& up, const Quantized& down,
                                         const float* input, const std::int32_t* signs, std::uint32_t block_size,
                                         float* output);

    // Geometry and side inputs of the fused attention block; the weights come
    // as q/k/v/out matrices. rope_cos/rope_sin hold rope_dims/2 rotations for
    // the current position. k_cache/v_cache are the f16-valued f32 caches and
    // grow by one token on success.
    struct AttentionParams {
        const float* input = nullptr;   // n_embd, already Hadamard'ed
        const float* q_norm = nullptr;  // head_dim
        const float* k_norm = nullptr;  // head_dim
        const float* rope_cos = nullptr;
        const float* rope_sin = nullptr;
        const std::int32_t* signs = nullptr;  // n_head * head_dim
        std::uint32_t rope_dims = 0;
        std::uint32_t head_count = 0;
        std::uint32_t head_count_kv = 0;
        std::uint32_t head_dim = 0;
        std::uint32_t block_size = 0;
        std::uint32_t cache_tokens = 0;
        float eps = 0.0F;
        float attention_scale = 0.0F;
    };

    // Fused attention: q/k/v matvec, head RMSNorm, RoPE, f16 KV append,
    // online-softmax attention, gate multiply, signed Hadamard and the output
    // projection in one device submission. False means "run the steps".
    [[nodiscard]] static bool attention(const Quantized& q, const Quantized& k, const Quantized& v,
                                        const Quantized& out, const AttentionParams& params, float* k_cache,
                                        float* v_cache, float* result);

    // Geometry and side data of the fused gated-delta block. The head
    // projections, convolution taps and norm weights are fp32 (llama keeps
    // them unquantized); conv_state/ssm_state seed the device-resident state.
    struct SsmParams {
        const float* input = nullptr;      // n_embd, Hadamard'ed
        const float* raw_input = nullptr;  // n_embd, before the Hadamard
        const float* beta_w = nullptr;     // value_heads * n_embd
        const float* alpha_w = nullptr;    // value_heads * n_embd
        const float* dt = nullptr;
        const float* ssm_a = nullptr;
        const float* conv_w = nullptr;
        const float* ssm_norm = nullptr;
        const std::int32_t* signs = nullptr;  // value_heads * head_dim
        const float* conv_state = nullptr;
        const float* ssm_state = nullptr;
        std::uint32_t key_heads = 0;
        std::uint32_t value_heads = 0;
        std::uint32_t head_dim = 0;
        std::uint32_t conv_kernel = 0;
        std::uint32_t block_size = 0;
        float eps = 0.0F;
    };

    // Fused gated-delta block: qkv/gate matvec, head gates, causal depthwise
    // convolution, the delta-rule scan, the gated RMS norm with the grouped
    // permute, signed Hadamard and the ssm_out projection in one submission.
    [[nodiscard]] static bool ssm_block(const Quantized& qkv, const Quantized& gate, const Quantized& out,
                                        const SsmParams& params, float* result);

    static void matmul(const Quantized& matrix, const float* input, float* output, std::size_t tokens,
                       std::size_t input_stride, std::size_t output_stride);

private:
    static std::uint16_t read_u16(std::istream& in);
    static std::uint32_t read_u32(std::istream& in);
    static std::uint64_t read_u64(std::istream& in);
    static std::string read_string(std::istream& in);
    static std::int64_t read_scalar_int(std::istream& in, std::uint32_t type);

    static void skip_value(std::istream& in, std::uint32_t type);

    void parse_header();

    std::string path_;
    mutable std::ifstream file_;
    std::uint32_t version_ = 0;
    std::uint64_t data_start_ = 0;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, std::uint64_t> integers_;
    std::unordered_map<std::string, double> floats_;
    std::unordered_map<std::string, std::vector<std::int64_t>> int_arrays_;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays_;
};

}  // namespace mnet::gguf

#endif  // MNET_GGUF_READER_HPP
