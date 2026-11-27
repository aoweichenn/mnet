// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Kokoro (StyleTTS2) runner: PLBERT text encoder, prosody predictor, duration
// predictor, decoder and the iSTFT generator, reproducing the pinned ONNX
// reference stage by stage.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <bit>
#include <complex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tts/common.hpp>

namespace {

using namespace mnet_tts;

// Kokoro's exported LayerNormalization nodes carry epsilon = 1e-12.
constexpr float kKokoroLayerNormEps = 1.0e-12F;

// activations.gelu with the tanh approximation the Kokoro export uses:
// 0.5 x (1 + tanh(0.7978846 (x + 0.044715 x^3)))
// ONNX Runtime's fused SkipLayerNormalization (contrib_ops/cpu/
// skip_layer_norm.cc): Welford mean/variance in float over input + skip,
// then (x - mean) / std * gamma + beta. The fused kernel is NOT the MLAS
// LayerNorm, so the rounding differs from an Add followed by layer_norm.
[[nodiscard]] Act skip_layer_norm(const Act& input, const Act& skip, const Weight& gamma, const Weight& beta,
                                  const Weight* bias, float eps) {
    Act out(input.channels, input.length);
    std::vector<float> val(input.channels);
    for (std::size_t t = 0; t < input.length; ++t) {
        float mean = 0.0F;
        float m2 = 0.0F;
        for (std::size_t h = 0; h < input.channels; ++h) {
            // The fused kernel adds input + skip first and the bias last
            // (skip_layer_norm.cc ComputeJob), which rounds differently from
            // a MatMul + Add followed by the residual Add.
            val[h] = input.at(h, t) + skip.at(h, t);
            if (bias != nullptr) {
                val[h] += bias->values[h];
            }
            const float delta = val[h] - mean;
            mean += delta / static_cast<float>(h + 1);
            const float delta2 = val[h] - mean;
            m2 += delta * delta2;
        }
        const float std_dev = std::sqrt(m2 / static_cast<float>(input.channels) + eps);
        for (std::size_t h = 0; h < input.channels; ++h) {
            out.at(h, t) = (val[h] - mean) / std_dev * gamma.values[h] + beta.values[h];
        }
    }
    return out;
}

[[nodiscard]] Act gelu_tanh(const Act& input) {
    // ONNX Runtime's FastGelu (contrib_ops/cpu/bert/bias_gelu.cc):
    //   y = 0.5 * x * (1 + tanh(x * (C * x * x + B)))
    // with B = sqrt(2/pi) and C = 0.044715 * sqrt(2/pi), tanh from
    // MlasComputeTanh (the FMA3 rational approximation).
    constexpr float kGeluB = 0.7978845608028654F;
    constexpr float kGeluC = 0.035677408136300125F;
    Act out = input;
    for (float& value : out.data) {
        const float x = value;
        const float arg = x * (kGeluC * x * x + kGeluB);
        value = 0.5F * x * (ort_tanh(arg) + 1.0F);
    }
    return out;
}

// One ALBERT layer (PLBERT): multi-head self attention + LayerNorm, then the
// 2048-wide gelu-tanh FFN + LayerNorm. The graph unrolls the same group twelve
// times.
[[nodiscard]] Act kokoro_albert_layer(const Act& x, const Weights& weights, std::size_t heads, std::size_t head_dim,
                                      const std::string& layer) {
    const std::size_t hidden = x.channels;
    const std::size_t t_len = x.length;
    const Act q = linear(x, weights.at(layer + "attention.query.weight"), &weights.at(layer + "attention.query.bias"));
    const Act k = linear(x, weights.at(layer + "attention.key.weight"), &weights.at(layer + "attention.key.bias"));
    const Act v = linear(x, weights.at(layer + "attention.value.weight"), &weights.at(layer + "attention.value.bias"));
    // The graph scales Q and K elementwise by sqrt(1 / sqrt(head_dim)) before
    // the product (attention/Div, Sqrt_1, Mul, Mul_1), so the scaling rounds
    // per element instead of after the dot product.
    const float attn_scale = std::sqrt(1.0F / std::sqrt(static_cast<float>(head_dim)));
    std::vector<float> q_scaled(q.data.size());
    std::vector<float> k_scaled(k.data.size());
    for (std::size_t i = 0; i < q.data.size(); ++i) {
        q_scaled[i] = q.data[i] * attn_scale;
    }
    for (std::size_t i = 0; i < k.data.size(); ++i) {
        k_scaled[i] = k.data[i] * attn_scale;
    }
    Act context(hidden, t_len);
    const std::size_t kblock_qk = mlas_gemm_kblock(t_len, head_dim);
    const std::size_t kblock_ctx = mlas_gemm_kblock(head_dim, t_len);
    std::vector<float> scores(t_len * t_len);
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t base = h * head_dim;
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t j = 0; j < t_len; ++j) {
                float total = 0.0F;
                for (std::size_t d0 = 0; d0 < head_dim; d0 += kblock_qk) {
                    const std::size_t d1 = std::min(d0 + kblock_qk, head_dim);
                    float partial = 0.0F;
                    for (std::size_t d = d0; d < d1; ++d) {
                        partial = std::fma(q_scaled[(base + d) * t_len + i], k_scaled[(base + d) * t_len + j], partial);
                    }
                    total += partial;
                }
                // attention/Add adds the (zero) attention mask.
                scores[i * t_len + j] = total + 0.0F;
            }
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            mlas_softmax_row(scores.data() + i * t_len, scores.data() + i * t_len, t_len);
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                float total = 0.0F;
                for (std::size_t j0 = 0; j0 < t_len; j0 += kblock_ctx) {
                    const std::size_t j1 = std::min(j0 + kblock_ctx, t_len);
                    float partial = 0.0F;
                    for (std::size_t j = j0; j < j1; ++j) {
                        partial = std::fma(scores[i * t_len + j], v.at(base + d, j), partial);
                    }
                    total += partial;
                }
                context.at(base + d, i) = total;
            }
        }
    }
    const Act dense =
        linear(context, weights.at(layer + "attention.dense.weight"), &weights.at(layer + "attention.dense.bias"));
    Act residual = x;
    for (std::size_t i = 0; i < residual.data.size(); ++i) {
        residual.data[i] += dense.data[i];
    }
    Act normed = layer_norm(residual, weights.at(layer + "attention.LayerNorm.weight"),
                            weights.at(layer + "attention.LayerNorm.bias"), kKokoroLayerNormEps);
    Act hidden_act = gelu_tanh(linear(normed, weights.at(layer + "ffn.weight"), &weights.at(layer + "ffn.bias")));
    const Act ffn_out = linear(hidden_act, weights.at(layer + "ffn_output.weight"), nullptr);
    return skip_layer_norm(ffn_out, normed, weights.at(layer + "full_layer_layer_norm.weight"),
                           weights.at(layer + "full_layer_layer_norm.bias"), &weights.at(layer + "ffn_output.bias"),
                           kKokoroLayerNormEps);
}

// ONNX LSTM (bidirectional): gates (i, o, f, c) with the exported W (dir, 4H,
// in), R (dir, 4H, H) and B (dir, 8H). The output concatenates both directions
// per timestep.
[[nodiscard]] Act lstm_bidirectional(const Act& input, const Weight& w_ih, const Weight& w_hh, const Weight& bias) {
    const ProfileScope profile(kProfileLstm);
    const std::size_t dirs = w_ih.size(0);
    const std::size_t gates = w_ih.size(1);
    const std::size_t hidden = gates / 4;
    if (dirs != 2 || input.channels != w_ih.size(2) || w_hh.size(2) != hidden) {
        throw std::runtime_error("lstm shape mismatch");
    }
    // Mirrors ONNX Runtime: the input and recurrent products are separate
    // MlasGemm calls (X*W with beta=0, then H*R accumulating with beta=1)
    // whose K dimension is walked in blocks of kMlasGemmPackedStrideK, then
    // the fused bias (Wb + Rb, precomputed in float) is added and clipping is
    // a no-op because clip defaults to FLT_MAX.
    const auto gemm_dot = [](const float* row, const float* value, std::size_t size, float accum) {
        for (std::size_t k0 = 0; k0 < size; k0 += kMlasGemmPackedStrideK) {
            const std::size_t k1 = std::min(k0 + kMlasGemmPackedStrideK, size);
            float partial = 0.0F;
            for (std::size_t k = k0; k < k1; ++k) {
                partial = std::fma(row[k], value[k], partial);
            }
            accum += partial;
        }
        return accum;
    };
    Act out(dirs * hidden, input.length);
    for (std::size_t d = 0; d < dirs; ++d) {
        const float* dir_bias = bias.values.data() + d * 2 * gates;
        std::vector<float> gate_bias(gates);
        for (std::size_t g = 0; g < gates; ++g) {
            gate_bias[g] = dir_bias[g] + dir_bias[gates + g];  // Wb[g] + Rb[g]
        }
        std::vector<float> h(hidden, 0.0F);
        std::vector<float> c(hidden, 0.0F);
        std::vector<float> x(input.channels);
        std::vector<float> gate(gates, 0.0F);
        for (std::size_t step = 0; step < input.length; ++step) {
            const std::size_t t = d == 0 ? step : input.length - 1 - step;
            for (std::size_t i = 0; i < input.channels; ++i) {
                x[i] = input.at(i, t);
            }
            for (std::size_t g = 0; g < gates; ++g) {
                const float* row_ih = w_ih.values.data() + (d * gates + g) * input.channels;
                const float* row_hh = w_hh.values.data() + (d * gates + g) * hidden;
                float accum = gemm_dot(row_ih, x.data(), input.channels, 0.0F);
                accum = gemm_dot(row_hh, h.data(), hidden, accum);
                gate[g] = accum + gate_bias[g];
            }
            for (std::size_t j = 0; j < hidden; ++j) {
                const float i_gate = ort_logistic(gate[j]);
                const float o_gate = ort_logistic(gate[hidden + j]);
                const float f_gate = ort_logistic(gate[2 * hidden + j]);
                const float c_gate = ort_tanh(gate[3 * hidden + j]);
                // deepcpu::merge_lstm_gates_to_memory: C = C_prev*f + i*g with
                // separate multiplies (no FMA contraction in the reference).
                c[j] = f_gate * c[j] + i_gate * c_gate;
                // deepcpu::tanh_m: H = tanh(C) * o.
                h[j] = ort_tanh(c[j]) * o_gate;
            }
            for (std::size_t j = 0; j < hidden; ++j) {
                out.at(d * hidden + j, t) = h[j];
            }
        }
    }
    return out;
}

// StyleTTS2 AdainResBlk1d: AdaIN -> LeakyReLU -> [depthwise upsample] -> conv1
// -> AdaIN -> LeakyReLU -> conv2, plus the residual (optionally upsampled and
// 1x1 projected), then / sqrt(2).

[[nodiscard]] Act adain_resblock(const Act& x, const Weights& weights, const std::string& prefix,
                                 const std::string& norm_prefix, const Act& style_act, bool upsample, bool learned_sc,
                                 bool strided_input) {
    Act out = adain_norm(x, weights.at(norm_prefix + "norm1.fc.weight"), weights.at(norm_prefix + "norm1.fc.bias"),
                         style_act, strided_input);
    leaky_relu(out, 0.2F);
    if (upsample) {
        out = depthwise_transpose(out, weight_or(weights, prefix + "pool.weight", norm_prefix + "pool.weight"),
                                  weight_or(weights, prefix + "pool.bias", norm_prefix + "pool.bias"), 2, 1);
    }
    out = conv1d(out, weight_or(weights, prefix + "conv1.weight", norm_prefix + "conv1.weight"),
                 weight_or(weights, prefix + "conv1.bias", norm_prefix + "conv1.bias"), 3, 1);
    out = adain_norm(out, weights.at(norm_prefix + "norm2.fc.weight"), weights.at(norm_prefix + "norm2.fc.bias"),
                     style_act, false);
    leaky_relu(out, 0.2F);
    out = conv1d(out, weight_or(weights, prefix + "conv2.weight", norm_prefix + "conv2.weight"),
                 weight_or(weights, prefix + "conv2.bias", norm_prefix + "conv2.bias"), 3, 1);
    Act residual = upsample ? nearest_upsample(x, 2) : x;
    if (learned_sc) {
        residual = conv1d(residual, weight_or(weights, prefix + "conv1x1.weight", norm_prefix + "conv1x1.weight"),
                          Weight{}, 1, 0);
    }
    for (std::size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = (out.data[i] + residual.data[i]) / 1.4142135F;
    }
    return out;
}

void snake_inplace(Act& act, const Weight& alpha) {
    // The graph computes x + Reciprocal(alpha) * Pow(Sin(alpha * x), 2); the
    // Sin kernel is Eigen's vectorised psin over the flattened tensor.
    for (std::size_t c = 0; c < act.channels; ++c) {
        const float a = alpha.values[c];
        const float inverse = 1.0F / a;
        for (std::size_t t = 0; t < act.length; ++t) {
            const float s = eigen_sin_element(a * act.at(c, t), c * act.length + t, act.data.size());
            act.at(c, t) += s * s * inverse;
        }
    }
}

// StyleTTS2's AdainResBlk1d for the generator: AdaIN -> snake -> dilated conv ->
// AdaIN -> snake -> conv, plus the residual.
[[nodiscard]] Act generator_resblock(const Act& input, const Weights& weights, const std::string& prefix,
                                     const Act& style_act, std::size_t kernel) {
    const std::string tag = prefix + ".adain";
    const std::string dbg_tag = prefix + ".";
    const std::size_t dilations[3] = {1, 3, 5};
    Act out = input;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string index = std::to_string(i);
        const std::size_t dilation = dilations[i];
        const std::size_t pad = dilation * (kernel - 1) / 2;
        Act xt = adain_norm(out, weights.at(tag + "1." + index + ".fc.weight"),
                            weights.at(tag + "1." + index + ".fc.bias"), style_act, false);
        if (!g_debug_dir.empty() &&
            (prefix == "decoder.decoder.generator.noise_res.0" || prefix == "decoder.decoder.generator.noise_res.1") &&
            i == 0) {
            debug_dump(dbg_tag + "adain1_0", xt.data);
        }
        snake_inplace(xt, weights.at(prefix + ".alpha1." + index));
        if (!g_debug_dir.empty() &&
            (prefix == "decoder.decoder.generator.noise_res.0" || prefix == "decoder.decoder.generator.noise_res.1") &&
            i == 0) {
            debug_dump(dbg_tag + "snake1_0", xt.data);
        }
        xt = conv1d(xt, weights.at(prefix + ".convs1." + index + ".weight"),
                    weights.at(prefix + ".convs1." + index + ".bias"), kernel, pad, dilation);
        if (!g_debug_dir.empty() &&
            (prefix == "decoder.decoder.generator.noise_res.0" || prefix == "decoder.decoder.generator.noise_res.1") &&
            i == 0) {
            debug_dump(dbg_tag + "conv1_0", xt.data);
        }
        xt = adain_norm(xt, weights.at(tag + "2." + index + ".fc.weight"), weights.at(tag + "2." + index + ".fc.bias"),
                        style_act, false);
        if (!g_debug_dir.empty() &&
            (prefix == "decoder.decoder.generator.noise_res.0" || prefix == "decoder.decoder.generator.noise_res.1") &&
            i == 0) {
            debug_dump(dbg_tag + "adain2_0", xt.data);
        }
        snake_inplace(xt, weights.at(prefix + ".alpha2." + index));
        xt = conv1d(xt, weights.at(prefix + ".convs2." + index + ".weight"),
                    weights.at(prefix + ".convs2." + index + ".bias"), kernel, (kernel - 1) / 2, 1);
        if (!g_debug_dir.empty() &&
            (prefix == "decoder.decoder.generator.noise_res.0" || prefix == "decoder.decoder.generator.noise_res.1") &&
            i == 0) {
            debug_dump(dbg_tag + "conv2_0", xt.data);
        }
        for (std::size_t k = 0; k < out.data.size(); ++k) {
            out.data[k] += xt.data[k];
        }
    }
    return out;
}

// ONNX Runtime's UpsampleTrilinear along the width axis of a [1, C, L]
// tensor (core/providers/cpu/tensor/upsample.cc). The channel and depth axes
// keep scale 1, so their weights degenerate: an interior row uses
// dy1 = 0, dy2 = 1 and the last row (y1 == y2 after clamping) uses
// dy1 = dy2 = 0.5; the depth axis always uses dz1 = dz2 = 0.5. The width uses
// x = (i + 0.5) / scale - 0.5 with half_pixel coordinates, and the eight
// terms are summed left to right.
[[nodiscard]] std::vector<float> ort_resize_width(const float* in, std::size_t channels, std::size_t in_len,
                                                  std::size_t out_len, float scale) {
    std::vector<float> out(channels * out_len);
    for (std::size_t c = 0; c < channels; ++c) {
        const float* row = in + c * in_len;
        const bool degenerate_y = channels == 1 || c + 1 == channels;
        const float dy1 = degenerate_y ? 0.5F : 0.0F;
        const float dy2 = degenerate_y ? 0.5F : 1.0F;
        constexpr float kDz1 = 0.5F;
        constexpr float kDz2 = 0.5F;
        for (std::size_t i = 0; i < out_len; ++i) {
            const float x = (static_cast<float>(i) + 0.5F) / scale - 0.5F;
            const float clamped = std::min(std::max(x, 0.0F), static_cast<float>(in_len - 1));
            const std::size_t left = std::min(static_cast<std::size_t>(clamped), in_len - 1);
            const std::size_t right = std::min(left + 1, in_len - 1);
            float dx1 = std::fabs(clamped - static_cast<float>(left));
            float dx2 = std::fabs(clamped - static_cast<float>(right));
            if (left == right) {
                dx1 = 0.5F;
                dx2 = 0.5F;
            }
            const float t1 = ((dx2 * dy2) * kDz2) * row[left];
            const float t2 = ((dx1 * dy2) * kDz2) * row[right];
            const float t3 = ((dx2 * dy1) * kDz2) * row[left];
            const float t4 = ((dx1 * dy1) * kDz2) * row[right];
            const float t5 = ((dx2 * dy2) * kDz1) * row[left];
            const float t6 = ((dx1 * dy2) * kDz1) * row[right];
            const float t7 = ((dx2 * dy1) * kDz1) * row[left];
            const float t8 = ((dx1 * dy1) * kDz1) * row[right];
            // The reference build evaluates the eight-term sum left to right.
            out[c * out_len + i] = ((((((t1 + t2) + t3) + t4) + t5) + t6) + t7) + t8;
        }
    }
    return out;
}

// SourceModuleHnNSF: a harmonic sine source driven by F0, then a learned 9 -> 1
// projection and tanh. The graph upsamples F0 by 300 (linear half_pixel),
// takes the fractional phase of (f0_up * harmonic) / 24000 at the *sample*
// rate, downsamples that phase by 300 back to 104 frames (linear half_pixel),
// cumsums it, scales by 2*pi*300 and upsamples again before the sine:
//   f0_upsamp -> Mul(harmonics) -> Div(24000) -> Sub(floor) -> Resize(1/300)
//   -> CumSum -> *2 -> *pi -> *300 -> Resize(300) -> Sin -> *0.1 -> *gate
[[nodiscard]] Act harmonic_source(const Act& f0, const Weights& weights, std::size_t upsample) {
    const ProfileScope profile(kProfileHarmonic);
    constexpr float kSampleRate = 24000.0F;
    constexpr std::size_t kHarmonics = 9;
    constexpr float kPi = 3.1415927F;
    const std::size_t frames = f0.length;
    const std::size_t samples = frames * upsample;
    const float up_scale = static_cast<float>(upsample);
    const float down_scale = 1.0F / static_cast<float>(upsample);

    // f0_upsamp/Resize is mode=nearest with coordinate_transformation_mode
    // asymmetric and nearest_mode floor: index = floor(i / 300).
    std::vector<float> f0_up(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        f0_up[i] = f0.at(0, i / upsample);
    }
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_f0_upsamp", f0_up);
    }

    // The graph keeps the nine harmonics as the "height" axis of a
    // [1, 9, L] tensor, so the resize rows are not interchangeable: every row
    // but the last uses dy1 = 0, dy2 = 1, the last one dy1 = dy2 = 0.5.
    std::vector<float> frac(kHarmonics * samples);
    for (std::size_t h = 0; h < kHarmonics; ++h) {
        for (std::size_t i = 0; i < samples; ++i) {
            const float product = f0_up[i] * static_cast<float>(h + 1);
            const float div = product / kSampleRate;
            frac[h * samples + i] = div - std::floor(div);
        }
    }
    const auto downsampled = ort_resize_width(frac.data(), kHarmonics, samples, frames, down_scale);

    std::vector<float> sines(samples * kHarmonics);
    std::vector<float> resized(frames * kHarmonics);
    std::vector<float> cumsum_dump(frames * kHarmonics);
    std::vector<float> phase(kHarmonics * frames);
    for (std::size_t h = 0; h < kHarmonics; ++h) {
        float accumulated = 0.0F;
        for (std::size_t j = 0; j < frames; ++j) {
            accumulated += downsampled[h * frames + j];
            phase[h * frames + j] = ((accumulated * 2.0F) * kPi) * up_scale;
            cumsum_dump[j * kHarmonics + h] = accumulated;
            resized[h * frames + j] = phase[h * frames + j];
        }
    }
    const auto phase_up = ort_resize_width(phase.data(), kHarmonics, frames, samples, up_scale);
    for (std::size_t h = 0; h < kHarmonics; ++h) {
        for (std::size_t i = 0; i < samples; ++i) {
            sines[i * kHarmonics + h] = eigen_sin(phase_up[h * samples + i]);
        }
    }

    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_har_cumsum", cumsum_dump);
        debug_dump("kokoro_har_resized", resized);
        debug_dump("kokoro_har_sin", sines);
        std::vector<float> scaled(sines.size());
        for (std::size_t i = 0; i < samples; ++i) {
            const float gate = f0_up[i] > 10.0F ? 1.0F : 0.0F;
            for (std::size_t h = 0; h < kHarmonics; ++h) {
                scaled[i * kHarmonics + h] = sines[i * kHarmonics + h] * 0.1F * gate;
            }
        }
        debug_dump("kokoro_har_mul6", scaled);
    }

    Act projected(1, samples);
    const Weight& linear_w = weights.at("decoder.decoder.generator.m_source.l_linear.weight");
    const Weight& linear_b = weights.at("decoder.decoder.generator.m_source.l_linear.bias");
    for (std::size_t i = 0; i < samples; ++i) {
        const float gate = f0_up[i] > 10.0F ? 1.0F : 0.0F;
        // MatMul accumulates the dot product first, then the bias is added.
        // ONNX Runtime's SGEMM kernel uses vfmadd231ps per k element, so
        // accumulate with fma.
        float sum = 0.0F;
        for (std::size_t h = 0; h < kHarmonics; ++h) {
            const float scaled = sines[i * kHarmonics + h] * 0.1F * gate;
            sum = std::fma(scaled, linear_w.values[h], sum);
        }
        projected.at(0, i) = eigen_fast_tanh(sum + linear_b.values[0]);
    }
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_har_tanh", projected.data);
    }
    return projected;
}

// ONNX Runtime's radix-2 FFT (core/providers/cpu/signal/dft.cc, fft_radix2):
// bit-reversed input, `V` holds the bit-reversed twiddles, and the inverse
// transform reuses the forward table and only scales by 1/N.
std::size_t ort_bit_reverse(std::size_t index, std::uint32_t bits) {
    std::size_t result = 0;
    for (std::uint32_t b = 0; b < bits; ++b) {
        result = (result << 1U) | (index & 1U);
        index >>= 1U;
    }
    return result;
}

void ort_fft(std::complex<float>* out, const std::complex<float>* in, std::size_t length,
             const std::vector<std::complex<float>>& v, bool inverse) {
    std::uint32_t bits = 0;
    while ((static_cast<std::size_t>(1) << bits) < length) {
        ++bits;
    }
    for (std::size_t i = 0; i < length; ++i) {
        out[i] = in[ort_bit_reverse(i, bits)];
    }
    std::uint32_t current_bits = 0;
    for (std::size_t span = 2; span <= length; span <<= 1U) {
        const std::size_t mid = span >> 1U;
        ++current_bits;
        for (std::size_t k = 0; k < mid; ++k) {
            const std::size_t first_idx = ort_bit_reverse(k, current_bits);
            const std::size_t second_idx = ort_bit_reverse(mid + k, current_bits);
            for (std::size_t j = 0; j < length; j += span) {
                const std::size_t even_index = k + j;
                const std::size_t odd_index = k + j + mid;
                const std::complex<float> even = out[even_index];
                const std::complex<float> odd = out[odd_index];
                out[even_index] = even + v[first_idx] * odd;
                out[odd_index] = even + v[second_idx] * odd;
            }
        }
    }
    if (inverse) {
        const float divisor = static_cast<float>(length);
        for (std::size_t i = 0; i < length; ++i) {
            out[i] /= divisor;
        }
    }
}

// ONNX Runtime's dft_bluestein_z_chirp for the 20-point real DFT: chirp-z with
// M = next_power_of_2(2N - 1) = 64. Reproduces the reference bit for bit,
// including the phase of bins whose magnitude is numerical noise.
class OrtDft20 {
public:
    OrtDft20() : v_(kM), chirp_(kN, {0.0F, 0.0F}), b_fft_(kM), a_(kM, {0.0F, 0.0F}), a_fft_(kM), c_(kM) {
        constexpr float kPi = 3.14159265358979323846F;
        const float tau = 2.0F * kPi;
        const float velocity = -1.0F * tau / static_cast<float>(kM);
        for (std::size_t i = 0; i < kM; ++i) {
            const float angle = static_cast<float>(i) * velocity;
            v_[ort_bit_reverse(i, 6)] = {std::cos(angle), std::sin(angle)};
        }
        std::array<std::complex<float>, kM> b{};
        for (std::size_t n = 0; n < kN; ++n) {
            const float exponent = -1.0F * kPi * static_cast<float>(n) * static_cast<float>(n) / static_cast<float>(kN);
            chirp_[n] = {std::cos(exponent), std::sin(exponent)};
            b[n] = std::conj(chirp_[n]);
        }
        for (std::size_t n = kM - kN + 1; n < kM; ++n) {
            b[n] = b[kM - n];
        }
        ort_fft(b_fft_.data(), b.data(), kM, v_, false);
    }

    void forward(const float* frame, const float* window, std::complex<float>* out) {
        a_.assign(kM, {0.0F, 0.0F});
        for (std::size_t n = 0; n < kN; ++n) {
            const float scaled = frame[n] * window[n];
            a_[n] = std::complex<float>(scaled * chirp_[n].real(), scaled * chirp_[n].imag());
        }
        ort_fft(a_fft_.data(), a_.data(), kM, v_, false);
        for (std::size_t i = 0; i < kM; ++i) {
            a_fft_[i] *= b_fft_[i];
        }
        ort_fft(c_.data(), a_fft_.data(), kM, v_, true);
        for (std::size_t k = 0; k < kN; ++k) {
            out[k] = (k == 0 ? c_[0] : c_[kM - k]) * chirp_[k];
        }
    }

private:
    static constexpr std::size_t kN = 20;
    static constexpr std::size_t kM = 64;
    std::vector<std::complex<float>> v_;
    std::vector<std::complex<float>> chirp_;
    std::vector<std::complex<float>> b_fft_;
    std::vector<std::complex<float>> a_;
    std::vector<std::complex<float>> a_fft_;
    std::vector<std::complex<float>> c_;
};

// TorchSTFT.transform: hann window, n_fft 20, hop 5, reflect padding of 10.
[[nodiscard]] Act harmonic_spectrogram(const Act& source, const Weight& window) {
    constexpr std::size_t kNfft = 20;
    constexpr std::size_t kHop = 5;
    constexpr std::size_t kPad = kNfft / 2;
    constexpr std::size_t kBins = kNfft / 2 + 1;
    constexpr float kPi = 3.1415927F;
    const std::size_t length = source.length;
    std::vector<float> padded(length + 2 * kPad);
    for (std::size_t i = 0; i < kPad; ++i) {
        padded[i] = source.at(0, kPad - i);
        padded[kPad + length + i] = source.at(0, length - 2 - i);
    }
    for (std::size_t i = 0; i < length; ++i) {
        padded[kPad + i] = source.at(0, i);
    }
    const std::size_t frames = 1 + (padded.size() - kNfft) / kHop;
    Act har(2 * kBins, frames);
    OrtDft20 dft;
    std::array<std::complex<float>, kNfft> bins{};
    float frame[kNfft];
    for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t t = 0; t < kNfft; ++t) {
            frame[t] = padded[f * kHop + t];
        }
        dft.forward(frame, window.values.data(), bins.data());
        for (std::size_t b = 0; b < kBins; ++b) {
            const float re = bins[b].real();
            const float im = bins[b].imag();
            har.at(b, f) = std::sqrt(re * re + im * im);
            // The graph computes atan(im / re) and fixes the quadrant with
            // Where(im > 0, atan + pi, atan - pi) when re < 0. ONNX Runtime's
            // Atan is Eigen's .atan() over the flattened [1, 11, frames]
            // tensor: the packet path (Eigen's ptan) handles the bulk and the
            // count % 4 tail uses std::atan.
            const std::size_t flat = b * frames + f;
            const float arg = im / re;
            const float atan = eigen_scalar_tail(flat, kBins * frames) ? std::atan(arg) : eigen_atan(arg);
            har.at(kBins + b, f) = re < 0.0F ? (im > 0.0F ? atan + kPi : atan - kPi) : atan;
        }
    }
    return har;
}

// Inverse STFT as exported: ConvTranspose with the inverse basis, the first
// window divided by window_sum, scaled by 4 and trimmed by n_fft / 2.
[[nodiscard]] Act istft_waveform(const Act& spec, const Weight& basis, const Weight& window_sum) {
    const ProfileScope profile(kProfileIstft);
    constexpr std::size_t kNfft = 20;
    constexpr std::size_t kHop = 5;
    constexpr std::size_t kTrim = kNfft / 2;
    const std::size_t out_len = (spec.length - 1) * kHop + kNfft;
    // The graph runs ConvTranspose(inverse_basis, stride=5) and then ScatterND:
    // first a Gemm(CblasTrans, CblasNoTrans, kernel_dim=20, frames, channels)
    // writes col[m][n] = sum_c W[m][c] * spec[c][n] (one fma chain, K < 128 so a
    // single block), then Col2imNd adds the col rows in increasing m (tap) order.
    std::vector<float> col(kNfft * spec.length);
    for (std::size_t m = 0; m < kNfft; ++m) {
        for (std::size_t n = 0; n < spec.length; ++n) {
            float total = 0.0F;
            for (std::size_t c = 0; c < spec.channels; ++c) {
                total = std::fma(basis.values[c * kNfft + m], spec.at(c, n), total);
            }
            col[m * spec.length + n] = total;
        }
    }
    std::vector<float> accum(out_len, 0.0F);
    for (std::size_t m = 0; m < kNfft; ++m) {
        for (std::size_t n = 0; n < spec.length; ++n) {
            accum[n * kHop + m] += col[m * spec.length + n];
        }
    }
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_istft_conv", accum);
    }
    for (std::size_t k = 1; k < kNfft; ++k) {
        if (window_sum.values[k] > 1.1754944e-38F) {
            accum[k] /= window_sum.values[k];
        }
    }
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_istft_scatter", accum);
    }
    std::vector<float> mul4(accum.size());
    for (std::size_t i = 0; i < accum.size(); ++i) {
        mul4[i] = accum[i] * 4.0F;
    }
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_istft_mul4", mul4);
    }
    Act wave(1, out_len - 2 * kTrim);
    for (std::size_t i = 0; i < wave.length; ++i) {
        wave.at(0, i) = accum[i + kTrim] * 4.0F;
    }
    return wave;
}

}  // namespace

namespace mnet_tts {

[[nodiscard]] std::int32_t run_kokoro(const TtsOptions& opt, const Weights& weights) {
    Reference kokoro_ref;
    if (!opt.ref_dir.empty()) {
        kokoro_ref.load(opt.ref_dir);
    }
    const std::vector<std::int64_t> ids = parse_id_list(opt.ids_arg);
    if (ids.empty()) {
        std::fprintf(stderr, "kokoro needs --ids a,b,c\n");
        return 2;
    }
    // PLBERT: embeddings -> LayerNorm -> hidden mapping -> 12 shared layers
    constexpr std::size_t kEmbd = 128;
    constexpr std::size_t kHeads = 12;
    constexpr std::size_t kHeadDim = 64;
    constexpr std::size_t kLayers = 12;
    const std::size_t tokens = ids.size();
    Act x(kEmbd, tokens);
    const Weight& wemb = weights.at("encoder.bert.embeddings.word_embeddings.weight");
    const Weight& pemb = weights.at("encoder.bert.embeddings.position_embeddings.weight");
    const Weight& temb = weights.at("encoder.bert.embeddings.token_type_embeddings.weight");
    for (std::size_t t = 0; t < tokens; ++t) {
        const std::size_t row = static_cast<std::size_t>(ids[t]);
        for (std::size_t c = 0; c < kEmbd; ++c) {
            // The graph sums (word_embeddings + token_type_embeddings)
            // first and then adds position_embeddings; the grouping changes
            // the float rounding and must match Add + Add_1.
            x.at(c, t) = (wemb.values[row * kEmbd + c] + temb.values[c]) + pemb.values[t * kEmbd + c];
        }
    }
    x = layer_norm(x, weights.at("encoder.bert.embeddings.LayerNorm.weight"),
                   weights.at("encoder.bert.embeddings.LayerNorm.bias"), kKokoroLayerNormEps);
    if (!g_debug_dir.empty()) {
        debug_dump("kokoro_embed", x.data);
    }
    report(kokoro_ref, "encoder_bert_embeddings_LayerNorm_LayerNormalization", x, "plbert embeddings", true);
    Act h = linear(x, weights.at("encoder.bert.encoder.embedding_hidden_mapping_in.weight"),
                   &weights.at("encoder.bert.encoder.embedding_hidden_mapping_in.bias"));
    report(kokoro_ref, "encoder_bert_encoder_embedding_hidden_mapping_in_Add", h, "plbert hidden mapping", true);
    const std::string layer = "encoder.bert.encoder.albert_layer_groups.0.albert_layers.0.";
    for (std::size_t i = 0; i < kLayers; ++i) {
        h = kokoro_albert_layer(h, weights, kHeads, kHeadDim, layer);
        const std::string suffix = i == 0 ? "" : "_" + std::to_string(i);
        report(kokoro_ref,
               "encoder_bert_encoder_albert_layer_groups.0_albert_layers.0_full_layer_layer_norm" + suffix +
                   "_LayerNormalization",
               h, "plbert layer " + std::to_string(i), true);
    }
    const Act text = linear(h, weights.at("encoder.bert_encoder.weight"), &weights.at("encoder.bert_encoder.bias"));
    report(kokoro_ref, "encoder_bert_encoder_Add", text, "plbert text features", true);
    std::printf("  %-42s %zu tokens\n", "plbert text features", tokens);

    // encoder.text_encoder: embedding -> 3 x [conv5, LayerNorm, LeakyReLU]
    // -> bidirectional LSTM(256), the decoder's "d_en" input.
    constexpr std::size_t kText = 512;
    Act te(kText, tokens);
    const Weight& text_emb = weights.at("encoder.text_encoder.embedding.weight");
    for (std::size_t t = 0; t < tokens; ++t) {
        const std::size_t row = static_cast<std::size_t>(ids[t]);
        for (std::size_t c = 0; c < kText; ++c) {
            te.at(c, t) = text_emb.values[row * kText + c];
        }
    }
    report(kokoro_ref, "encoder_text_encoder_embedding_Gather", te, "text encoder embedding", true);
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string tag = "encoder.text_encoder.cnn." + std::to_string(i) + ".cnn." + std::to_string(i);
        const std::string conv_bias = "encoder.text_encoder.cnn." + std::to_string(i) + ".0.bias";
        Act conv = conv1d(te, weights.at(tag + ".0.weight"), weights.at(conv_bias), 5, 2);
        report(kokoro_ref, "encoder_text_encoder_cnn." + std::to_string(i) + "_cnn." + std::to_string(i) + ".0_Conv",
               conv, "text encoder conv " + std::to_string(i));
        const std::string ln_tag = "encoder.text_encoder.cnn." + std::to_string(i) + ".1";
        conv = layer_norm(conv, weights.at(ln_tag + ".gamma"), weights.at(ln_tag + ".beta"));
        report(kokoro_ref,
               "encoder_text_encoder_cnn." + std::to_string(i) + "_cnn." + std::to_string(i) + ".1_LayerNormalization",
               conv, "text encoder norm " + std::to_string(i), true);
        leaky_relu(conv, 0.2F);
        te = conv;
    }
    const Act d_en = lstm_bidirectional(te, weights.at("encoder.text_encoder.lstm.weight_ih"),
                                        weights.at("encoder.text_encoder.lstm.weight_hh"),
                                        weights.at("encoder.text_encoder.lstm.bias"));
    report(kokoro_ref, "encoder_text_encoder_lstm_LSTM", d_en, "text encoder BiLSTM", true);
    std::printf("  %-42s %zu x %zu\n", "text encoder output (d_en)", d_en.channels, d_en.length);

    // Prosody predictor, duration branch: StyleTTS2 DurationEncoder =
    // 3 x [concat(text, style) -> BiLSTM -> AdaLayerNorm(style)], then a
    // BiLSTM over cat(t_en, style) and a 50-bin linear head whose sigmoid
    // sum (divided by opt.speed) is the frame duration.
    // the predictor consumes style[:, 128:] (the decoder uses the first half)
    constexpr std::size_t kStyle = 128;
    if (opt.style_path.empty()) {
        std::fprintf(stderr, "kokoro needs --style FILE (256 float32 values)\n");
        return 2;
    }
    std::ifstream style_file(opt.style_path, std::ios::binary);
    std::vector<float> style_row(2 * kStyle);
    style_file.read(reinterpret_cast<char*>(style_row.data()),
                    static_cast<std::streamsize>(style_row.size() * sizeof(float)));
    if (!style_file) {
        std::fprintf(stderr, "cannot read %s\n", opt.style_path.c_str());
        return 2;
    }
    const std::vector<float> style_values(style_row.begin() + static_cast<std::ptrdiff_t>(kStyle), style_row.end());
    // the predictor starts from the PLBERT text features (512), not from the
    // text encoder's BiLSTM output (that one feeds the decoder).
    Act t_en = text;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string tag = "encoder.predictor.text_encoder.lstms." + std::to_string(i * 2);
        Act cat(512 + kStyle, tokens);
        for (std::size_t c = 0; c < 512; ++c) {
            for (std::size_t t = 0; t < tokens; ++t) {
                cat.at(c, t) = t_en.at(c, t);
            }
        }
        for (std::size_t c = 0; c < kStyle; ++c) {
            for (std::size_t t = 0; t < tokens; ++t) {
                cat.at(512 + c, t) = style_values[c];
            }
        }
        const Act enc_lstm = lstm_bidirectional(cat, weights.at(tag + ".weight_ih"), weights.at(tag + ".weight_hh"),
                                                weights.at(tag + ".bias"));
        report(kokoro_ref, "encoder_predictor_text_encoder_lstms." + std::to_string(i * 2) + "_LSTM", enc_lstm,
               "predictor lstm " + std::to_string(i), true);
        // AdaLayerNorm: fc(style) -> gamma/beta, x = LN(x) * (1+gamma) + beta
        Act style_act(kStyle, 1);
        for (std::size_t c = 0; c < kStyle; ++c) {
            style_act.at(c, 0) = style_values[c];
        }
        const std::string adaln = "encoder.predictor.text_encoder.lstms." + std::to_string(i * 2 + 1) + ".";
        const Act hb = linear(style_act, weights.at(adaln + "fc.weight"), &weights.at(adaln + "fc.bias"));
        Act normed = layer_norm_unit(enc_lstm, 1.0e-5F);
        for (std::size_t c = 0; c < normed.channels; ++c) {
            const float gamma = hb.at(c, 0);
            const float beta = hb.at(normed.channels + c, 0);
            for (std::size_t t = 0; t < normed.length; ++t) {
                normed.at(c, t) = normed.at(c, t) * (1.0F + gamma) + beta;
            }
        }
        report(kokoro_ref, "encoder_predictor_text_encoder_lstms." + std::to_string(i * 2 + 1) + "_Add_2", normed,
               "predictor adaln " + std::to_string(i), true);
        t_en = normed;
    }
    // predictor.lstm over cat(t_en, style)
    Act plstm_in(512 + kStyle, tokens);
    for (std::size_t c = 0; c < 512; ++c) {
        for (std::size_t t = 0; t < tokens; ++t) {
            plstm_in.at(c, t) = t_en.at(c, t);
        }
    }
    for (std::size_t c = 0; c < kStyle; ++c) {
        for (std::size_t t = 0; t < tokens; ++t) {
            plstm_in.at(512 + c, t) = style_values[c];
        }
    }
    const Act d_lstm =
        lstm_bidirectional(plstm_in, weights.at("encoder.predictor.lstm.weight_ih"),
                           weights.at("encoder.predictor.lstm.weight_hh"), weights.at("encoder.predictor.lstm.bias"));
    report(kokoro_ref, "encoder_predictor_lstm_LSTM", d_lstm, "predictor lstm", true);
    const Act logits = linear(d_lstm, weights.at("encoder.predictor.duration_proj.linear_layer.weight"),
                              &weights.at("encoder.predictor.duration_proj.linear_layer.bias"));
    report(kokoro_ref, "encoder_predictor_duration_proj_linear_layer_Add", logits, "duration logits", true);
    Act duration_logit(50, tokens);
    Act sig(50, tokens);
    for (std::size_t t = 0; t < tokens; ++t) {
        double sum = 0.0;
        for (std::size_t b = 0; b < 50; ++b) {
            // ONNX Runtime's Sigmoid<float> calls MlasComputeLogistic
            // (the Eigen rational approximation), not expf.
            const float v = ort_logistic(logits.at(b, t));
            sig.at(b, t) = v;
            sum += static_cast<double>(v);
        }
        duration_logit.at(0, t) = static_cast<float>(sum);
    }
    report(kokoro_ref, "encoder_predictor_Sigmoid", sig, "duration sigmoid", true);
    std::printf("  %-42s", "duration (frames per token)");
    std::int32_t total_frames = 0;
    for (std::size_t t = 0; t < tokens; ++t) {
        const auto frames = static_cast<std::int32_t>(std::llround(duration_logit.at(0, t) / opt.speed));
        total_frames += frames;
        std::printf(" %d", frames);
    }
    std::printf("  (sum %d)\n", total_frames);
    {
        const auto it = kokoro_ref.tensors.find("encoder_Round");
        if (it != kokoro_ref.tensors.end() && it->second.size() == tokens) {
            std::size_t same = 0;
            for (std::size_t t = 0; t < tokens; ++t) {
                const auto expected = static_cast<std::int32_t>(std::llround(it->second[t]));
                const auto ours = static_cast<std::int32_t>(
                    std::llround(static_cast<double>(duration_logit.at(0, t)) / static_cast<double>(opt.speed)));
                if (ours == expected) {
                    ++same;
                }
            }
            std::printf("  %-42s %zu/%zu tokens\n", "duration vs encoder_Round", same, tokens);
        }
    }
    // Prosody heads: align the duration encoder output to the frame grid,
    // run the shared BiLSTM, then the F0 / N AdainResBlk stacks.
    std::vector<std::int32_t> durations(tokens, 0);
    std::int32_t frame_total = 0;
    for (std::size_t t = 0; t < tokens; ++t) {
        durations[t] = static_cast<std::int32_t>(
            std::llround(static_cast<double>(duration_logit.at(0, t)) / static_cast<double>(opt.speed)));
        frame_total += durations[t];
    }
    std::vector<std::int32_t> frame_token(static_cast<std::size_t>(frame_total), 0);
    {
        std::int32_t frame = 0;
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::int32_t k = 0; k < durations[t] && frame < frame_total; ++k) {
                frame_token[static_cast<std::size_t>(frame)] = static_cast<std::int32_t>(t);
                ++frame;
            }
        }
    }
    Act aligned(512 + kStyle, static_cast<std::size_t>(frame_total));
    for (std::size_t c = 0; c < 512 + kStyle; ++c) {
        for (std::size_t f = 0; f < aligned.length; ++f) {
            const std::size_t token = static_cast<std::size_t>(frame_token[f]);
            aligned.at(c, f) = c < 512 ? t_en.at(c, token) : style_values[c - 512];
        }
    }
    const Act shared = lstm_bidirectional(aligned, weights.at("encoder.shared.weight_ih"),
                                          weights.at("encoder.shared.weight_hh"), weights.at("encoder.shared.bias"));
    report(kokoro_ref, "encoder_shared_LSTM", shared, "shared lstm", true);
    std::vector<float> frame_mask(static_cast<std::size_t>(frame_total), 1.0F);
    Act f0_values;
    Act n_values;
    for (std::int32_t head = 0; head < 2; ++head) {
        const std::string name = head == 0 ? "F0" : "N";
        Act head_act = shared;
        const std::string divs[3] = {"Div_2", "Div_4", "Div_2"};
        for (std::size_t i = 0; i < 3; ++i) {
            const std::string tag = "encoder." + name + "." + std::to_string(i) + ".";
            const std::string norm_tag = "encoder.predictor." + name + "." + std::to_string(i) + ".";
            const Act style_act = style_act_of(style_values);
            head_act = adain_resblock(head_act, weights, tag, norm_tag, style_act, i == 1, i == 1, i == 0);
            report(kokoro_ref, "encoder_" + name + "." + std::to_string(i) + "_" + divs[i], head_act,
                   name + " block " + std::to_string(i));
        }
        const std::string proj = "encoder.predictor." + name + "_proj.";
        Act v = conv1d(head_act, weights.at(proj + "weight"), weights.at(proj + "bias"), 1, 0);
        report(kokoro_ref, "encoder_" + name + "_proj_Conv", v, name + " proj");
        if (head == 0) {
            f0_values = v;
        } else {
            n_values = v;
        }
    }
    // Decoder: concat(d_en expanded to frames, F0, N) = 514 -> encode ->
    // 4 x decode (each block sees features + asr_res(64) + F0 + N = 1090).
    {
        const std::size_t dec_frames = static_cast<std::size_t>(frame_total);
        const std::vector<float> style_first_vec(style_row.begin(),
                                                 style_row.begin() + static_cast<std::ptrdiff_t>(kStyle));
        const Act style_first = style_act_of(style_first_vec);
        Act d_en_exp(512, dec_frames);
        for (std::size_t c = 0; c < 512; ++c) {
            for (std::size_t f = 0; f < dec_frames; ++f) {
                d_en_exp.at(c, f) = d_en.at(c, static_cast<std::size_t>(frame_token[f]));
            }
        }
        const Act asr = conv1d(d_en_exp, weights.at("decoder.decoder.asr_res.asr_res.0.weight"),
                               weights.at("decoder.decoder.asr_res.0.bias"), 1, 0);
        const Act f0 = conv1d(f0_values, weights.at("decoder.decoder.F0_conv.weight"),
                              weights.at("decoder.decoder.F0_conv.bias"), 3, 1, 1, 1, 2);
        const Act nn = conv1d(n_values, weights.at("decoder.decoder.N_conv.weight"),
                              weights.at("decoder.decoder.N_conv.bias"), 3, 1, 1, 1, 2);
        report(kokoro_ref, "decoder_decoder_F0_conv_Conv", f0, "decoder F0 conv");
        report(kokoro_ref, "decoder_decoder_N_conv_Conv", nn, "decoder N conv");
        const auto make_block_in = [&](const Act& features) {
            Act block_in(1090, dec_frames);
            for (std::size_t c = 0; c < features.channels; ++c) {
                for (std::size_t f = 0; f < dec_frames; ++f) {
                    block_in.at(c, f) = features.at(c, f);
                }
            }
            for (std::size_t c = 0; c < 64; ++c) {
                for (std::size_t f = 0; f < dec_frames; ++f) {
                    block_in.at(1024 + c, f) = asr.at(c, f);
                }
            }
            for (std::size_t f = 0; f < dec_frames; ++f) {
                block_in.at(1088, f) = f0.at(0, f);
                block_in.at(1089, f) = nn.at(0, f);
            }
            return block_in;
        };
        Act dec_in(514, dec_frames);
        for (std::size_t c = 0; c < 512; ++c) {
            for (std::size_t f = 0; f < dec_frames; ++f) {
                dec_in.at(c, f) = d_en_exp.at(c, f);
            }
        }
        for (std::size_t f = 0; f < dec_frames; ++f) {
            dec_in.at(512, f) = f0.at(0, f);
            dec_in.at(513, f) = nn.at(0, f);
        }
        report(kokoro_ref, "decoder_decoder_Concat", dec_in, "decoder input concat");
        Act enc = adain_resblock(dec_in, weights, "decoder.decoder.encode.", "decoder.decoder.encode.", style_first,
                                 false, true, false);
        report(kokoro_ref, "decoder_decoder_encode_Div_3", enc, "decoder encode block");
        for (std::size_t b = 0; b < 3; ++b) {
            const std::string tag = "decoder.decoder.decode." + std::to_string(b) + ".";
            const Act block_in = make_block_in(enc);
            enc = adain_resblock(block_in, weights, tag, tag, style_first, false, true, false);
            report(kokoro_ref, "decoder_decoder_decode." + std::to_string(b) + "_Div_3", enc,
                   "decoder block " + std::to_string(b));
        }
        const std::string tag3 = "decoder.decoder.decode.3.";
        const Act block_in3 = make_block_in(enc);
        const Act z_dec = adain_resblock(block_in3, weights, tag3, tag3, style_first, true, true, false);
        report(kokoro_ref, "decoder_decoder_decode.3_Div_4", z_dec, "decoder block 3 (x2 upsample)");
        // Generator: harmonic source -> noise_res, two upsample stages with
        // a 3-block receptive-field fusion each, conv_post (22 bins) and the
        // inverse STFT.
        Act source = harmonic_source(f0_values, weights, 300);
        if (opt.ref_f0) {
            const auto& rf = kokoro_ref.tensors.at("encoder_F0_proj_Conv");
            Act f0_ref(1, rf.size());
            std::copy(rf.begin(), rf.end(), f0_ref.data.begin());
            source = harmonic_source(f0_ref, weights, 300);
        }
        report(kokoro_ref, "decoder_decoder_generator_m_source_l_tanh_Tanh", source, "harmonic source");

        Act har = harmonic_spectrogram(source, weights.at("/decoder/decoder/generator/Constant_17_output_0"));
        if (opt.ref_har) {
            const auto& rh = kokoro_ref.tensors.at("decoder_decoder_generator_Concat_3");
            std::copy(rh.begin(), rh.end(), har.data.begin());
            std::printf("  %-42s %s\n", "harmonic spectrogram", "using the reference (diagnostic)");
        }
        report_channels(kokoro_ref, "decoder_decoder_generator_Concat_3", har, "generator har magnitude", 11);
        if (!g_debug_dir.empty()) {
            debug_dump("kokoro_har_full", har.data);
        }
        Act noise0 = conv1d(har, weights.at("decoder.decoder.generator.noise_convs.0.weight"),
                            weights.at("decoder.decoder.generator.noise_convs.0.bias"), 12, 3, 1, 1, 6);
        report(kokoro_ref, "decoder_decoder_generator_noise_convs.0_Conv", noise0, "generator noise conv 0");
        if (!g_debug_dir.empty()) {
            debug_dump("kokoro_noise0", noise0.data);
        }
        noise0 = generator_resblock(noise0, weights, "decoder.decoder.generator.noise_res.0", style_first, 7);
        report(kokoro_ref, "decoder_decoder_generator_noise_res.0_Add_8", noise0, "generator noise res 0");
        Act gen = z_dec;
        leaky_relu(gen, 0.1F);
        gen = conv_transpose1d(gen, weights.at("decoder.decoder.generator.ups.0.weight"),
                               weights.at("decoder.decoder.generator.ups.0.bias"), 10, 20, 5);
        report(kokoro_ref, "decoder_decoder_generator_ups.0_ConvTranspose", gen, "generator ups.0");
        for (std::size_t i = 0; i < gen.data.size(); ++i) {
            gen.data[i] += noise0.data[i];
        }
        report(kokoro_ref, "decoder_decoder_generator_Add_2", gen, "generator add noise 0");
        std::vector<Act> stage_low;
        for (std::size_t b = 0; b < 3; ++b) {
            const std::string tag = "decoder.decoder.generator.resblocks." + std::to_string(b);
            Act rb = generator_resblock(gen, weights, tag, style_first, 3 + 4 * b);
            report(kokoro_ref, "decoder_decoder_generator_resblocks." + std::to_string(b) + "_Add_8", rb,
                   "generator resblock " + std::to_string(b));
            stage_low.push_back(std::move(rb));
        }
        const Act sum0 = [&] {
            Act acc = stage_low[0];
            for (std::size_t k = 0; k < acc.data.size(); ++k) {
                acc.data[k] += stage_low[1].data[k];
            }
            return acc;
        }();
        report(kokoro_ref, "decoder_decoder_generator_Add_3", sum0, "generator sum 0a");
        Act sum0b = sum0;
        for (std::size_t k = 0; k < sum0b.data.size(); ++k) {
            sum0b.data[k] += stage_low[2].data[k];
        }
        report(kokoro_ref, "decoder_decoder_generator_Add_4", sum0b, "generator sum 0b");
        Act blended(gen.channels, gen.length);
        for (std::size_t k = 0; k < blended.data.size(); ++k) {
            blended.data[k] = sum0b.data[k] / 3.0F;
        }
        report(kokoro_ref, "decoder_decoder_generator_Div_2", blended, "generator blend 0");
        leaky_relu(blended, 0.1F);
        report(kokoro_ref, "decoder_decoder_generator_LeakyRelu_1", blended, "generator act 1");
        Act gen2 = conv_transpose1d(blended, weights.at("decoder.decoder.generator.ups.1.weight"),
                                    weights.at("decoder.decoder.generator.ups.1.bias"), 6, 12, 3);
        report(kokoro_ref, "decoder_decoder_generator_ups.1_ConvTranspose", gen2, "generator ups.1");
        Act padded(gen2.channels, gen2.length + 1);
        for (std::size_t c = 0; c < gen2.channels; ++c) {
            padded.at(c, 0) = gen2.at(c, 1);
            for (std::size_t t = 0; t < gen2.length; ++t) {
                padded.at(c, t + 1) = gen2.at(c, t);
            }
        }
        report(kokoro_ref, "decoder_decoder_generator_reflection_pad_Pad", padded, "generator reflect pad");
        Act noise1 = conv1d(har, weights.at("decoder.decoder.generator.noise_convs.1.weight"),
                            weights.at("decoder.decoder.generator.noise_convs.1.bias"), 1, 0);
        report(kokoro_ref, "decoder_decoder_generator_noise_convs.1_Conv", noise1, "generator noise conv 1");
        noise1 = generator_resblock(noise1, weights, "decoder.decoder.generator.noise_res.1", style_first, 11);
        report(kokoro_ref, "decoder_decoder_generator_noise_res.1_Add_8", noise1, "generator noise res 1");
        for (std::size_t k = 0; k < padded.data.size(); ++k) {
            padded.data[k] += noise1.data[k];
        }
        report(kokoro_ref, "decoder_decoder_generator_Add_5", padded, "generator add noise 1");
        std::vector<Act> stage2;
        for (std::size_t b = 3; b < 6; ++b) {
            const std::string tag = "decoder.decoder.generator.resblocks." + std::to_string(b);
            Act rb = generator_resblock(padded, weights, tag, style_first, 3 + 4 * (b - 3));
            report(kokoro_ref, "decoder_decoder_generator_resblocks." + std::to_string(b) + "_Add_8", rb,
                   "generator resblock " + std::to_string(b));
            stage2.push_back(std::move(rb));
        }
        Act sum1 = stage2[0];
        for (std::size_t k = 0; k < sum1.data.size(); ++k) {
            sum1.data[k] += stage2[1].data[k];
        }
        report(kokoro_ref, "decoder_decoder_generator_Add_6", sum1, "generator sum 1a");
        for (std::size_t k = 0; k < sum1.data.size(); ++k) {
            sum1.data[k] += stage2[2].data[k];
        }
        report(kokoro_ref, "decoder_decoder_generator_Add_7", sum1, "generator sum 1b");
        Act blended2(padded.channels, padded.length);
        for (std::size_t k = 0; k < blended2.data.size(); ++k) {
            blended2.data[k] = sum1.data[k] / 3.0F;
        }
        report(kokoro_ref, "decoder_decoder_generator_Div_4", blended2, "generator blend 1");
        leaky_relu(blended2, 0.01F);
        report(kokoro_ref, "decoder_decoder_generator_LeakyRelu_2", blended2, "generator act 2");
        const Act post = conv1d(blended2, weights.at("decoder.decoder.generator.conv_post.weight"),
                                weights.at("decoder.decoder.generator.conv_post.bias"), 7, 3);
        report(kokoro_ref, "decoder_decoder_generator_conv_post_Conv", post, "generator conv_post");
        const std::size_t bins = post.channels / 2;
        const std::size_t slice_len = bins * post.length;
        Act spec(post.channels, post.length);
        for (std::size_t b = 0; b < bins; ++b) {
            for (std::size_t t = 0; t < post.length; ++t) {
                // The graph runs Exp over the magnitude slice and then
                // Sin -> (Sin, Cos) over the phase slice, each as a flat
                // Eigen array of bins * T elements (so the packet tail is
                // the last three entries of the last bin).
                const std::size_t index = b * post.length + t;
                const float magnitude = mlas_exp(post.at(b, t));
                const float phase = eigen_sin_element(post.at(bins + b, t), index, slice_len);
                spec.at(b, t) = magnitude * eigen_cos_element(phase, index, slice_len);
                spec.at(bins + b, t) = magnitude * eigen_sin_element(phase, index, slice_len);
            }
        }
        if (!g_debug_dir.empty()) {
            debug_dump("kokoro_istft_concat", spec.data);
        }
        const Act wave = istft_waveform(spec, weights.at("decoder.decoder.generator.stft.istft.stft.inverse_basis"),
                                        weights.at("decoder.decoder.generator.stft.istft.stft.window_sum"));
        report(kokoro_ref, "output", wave, "waveform");
        const auto out_it = kokoro_ref.tensors.find("output");
        if (out_it != kokoro_ref.tensors.end() && out_it->second.size() == wave.data.size()) {
            double dot = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;
            for (std::size_t i = 0; i < wave.data.size(); ++i) {
                const double a = static_cast<double>(wave.data[i]);
                const double b = static_cast<double>(out_it->second[i]);
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            const double cos = norm_a > 0.0 && norm_b > 0.0 ? dot / std::sqrt(norm_a * norm_b) : 0.0;
            std::printf("  %-42s cos %.9f  rms %.4f\n", "waveform similarity", cos,
                        std::sqrt(norm_a / static_cast<double>(wave.data.size())));
        }
        if (!opt.wav_out.empty()) {
            write_wav(opt.wav_out, wave.data, 24000U);
        }
    }
    return 0;
}

}  // namespace mnet_tts
