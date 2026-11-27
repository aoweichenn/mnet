// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Piper/VITS runner: text encoder, stochastic duration predictor, flow and the
// HiFiGAN decoder, reproducing the pinned ONNX reference stage by stage.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tts/common.hpp>

namespace {

using namespace mnet_tts;

constexpr std::size_t kSplineBins = 10;
// constant = log(exp(1 - min_derivative) - 1) for the linear tails

// MlasComputeLogisticF32KernelFma3 coefficients (mlas/lib/logistic.cpp,
// "same polynomial coefficients and algorithm as found in Eigen").

// MlasComputeTanhF32KernelFma3 coefficients (mlas/lib/tanh.cpp).

// attentions.MultiHeadAttention._get_relative_embeddings
[[nodiscard]] std::vector<float> relative_embeddings(const Weight& table, std::size_t window, std::size_t length) {
    const std::size_t table_len = table.size(1);  // 2*window + 1
    const std::size_t channels = table.size(2);
    const std::size_t pad_length = length > window + 1 ? length - (window + 1) : 0;
    const std::size_t slice_start = length < window + 1 ? (window + 1) - length : 0;
    const std::size_t slice_end = slice_start + 2 * length - 1;
    std::vector<float> out((slice_end - slice_start) * channels, 0.0F);
    for (std::size_t row = slice_start; row < slice_end; ++row) {
        const std::ptrdiff_t src = static_cast<std::ptrdiff_t>(row) - static_cast<std::ptrdiff_t>(pad_length);
        if (src < 0 || src >= static_cast<std::ptrdiff_t>(table_len)) {
            continue;
        }
        for (std::size_t c = 0; c < channels; ++c) {
            out[(row - slice_start) * channels + c] = table.values[static_cast<std::size_t>(src) * channels + c];
        }
    }
    return out;
}

// attentions.MultiHeadAttention._relative_position_to_absolute_position:
// pad one column, flatten, append length-1 zeros, view (length+1, 2*length-1)
// and keep rows [0, length) from column length-1.
[[nodiscard]] std::vector<float> relative_to_absolute(const std::vector<float>& input, std::size_t length) {
    const std::size_t width = 2 * length - 1;
    std::vector<float> padded(length * (width + 1), 0.0F);
    for (std::size_t i = 0; i < length; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            padded[i * (width + 1) + j] = input[i * width + j];
        }
    }
    padded.resize(padded.size() + (length - 1), 0.0F);
    std::vector<float> out(length * length, 0.0F);
    for (std::size_t i = 0; i < length; ++i) {
        for (std::size_t j = 0; j < length; ++j) {
            out[i * length + j] = padded[i * width + (length - 1) + j];
        }
    }
    return out;
}

// attentions.MultiHeadAttention._absolute_position_to_relative_position:
// pad length-1 columns (stride 2*length-1), flatten, prepend length zeros, view
// (length, 2*length) and drop column 0.
[[nodiscard]] std::vector<float> absolute_to_relative(const std::vector<float>& input, std::size_t length) {
    const std::size_t stride = 2 * length - 1;
    std::vector<float> flat(length * stride, 0.0F);
    for (std::size_t i = 0; i < length; ++i) {
        for (std::size_t j = 0; j < length; ++j) {
            flat[i * stride + j] = input[i * length + j];
        }
    }
    std::vector<float> shifted(flat.size() + length, 0.0F);
    std::copy(flat.begin(), flat.end(), shifted.begin() + static_cast<std::ptrdiff_t>(length));
    std::vector<float> out(length * stride, 0.0F);
    for (std::size_t i = 0; i < length; ++i) {
        for (std::size_t j = 0; j < stride; ++j) {
            out[i * stride + j] = shifted[i * (stride + 1) + j + 1];
        }
    }
    return out;
}

// modules.Flip: torch.flip(x, [1]) reverses the whole channel axis.
void flip_channels(Act& act) {
    for (std::size_t c = 0; c < act.channels / 2; ++c) {
        for (std::size_t t = 0; t < act.length; ++t) {
            std::swap(act.at(c, t), act.at(act.channels - 1 - c, t));
        }
    }
}

// Standard normal samples (Box-Muller over a fixed-seed mt19937) so a run is
// reproducible from --seed.
class Gaussian {
public:
    explicit Gaussian(std::uint32_t seed) : engine_(seed) {}

    [[nodiscard]] float next() {
        if (this->has_spare_) {
            this->has_spare_ = false;
            return this->spare_;
        }
        std::uniform_real_distribution<float> uniform(1.0e-7F, 1.0F);
        const float u1 = uniform(this->engine_);
        const float u2 = uniform(this->engine_);
        const float radius = std::sqrt(-2.0F * std::log(u1));
        const float angle = 6.28318530718F * u2;
        this->spare_ = radius * std::sin(angle);
        this->has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    std::mt19937 engine_;
    float spare_ = 0.0F;
    bool has_spare_ = false;
};

// modules.ElementwiseAffine(reverse=True): (x - m) * exp(-logs). The exporter
// folds exp(-logs) into a constant tensor, so the caller passes it directly.
void elementwise_affine(Act& act, const Weight& mean, const Weight& inverse_logs, const std::vector<float>& mask) {
    for (std::size_t c = 0; c < act.channels; ++c) {
        const float shift = mean.values[c];
        const float scale = inverse_logs.values[c];
        for (std::size_t t = 0; t < act.length; ++t) {
            act.at(c, t) = (act.at(c, t) - shift) * scale * mask[t];
        }
    }
}

// Eigen 5.0.1's float kernels as ONNX Runtime compiles them for its CPU EP:
// 4-lane SSE packets without FMA contraction, so every pmadd is a multiply
// followed by an add. pexp_float / plog_float / the Kahan log1p are evaluated
// lane-wise here; the Softplus op is (x > 0) ? x + log1p(exp(-x)) : log1p(exp(x)).

// transforms.rational_quadratic_spline with inverse=True, tails="linear",
// tail_bound=5: one element of the inverse rational-quadratic spline.
[[nodiscard]] float spline_inverse(float input, const float* unnormalized_widths, const float* unnormalized_heights,
                                   const float* unnormalized_derivatives, std::size_t num_bins, float tail_bound,
                                   std::size_t frame_index = 0, std::size_t frame_count = 0) {
    constexpr float kMinBinWidth = 1.0e-3F;
    constexpr float kMinBinHeight = 1.0e-3F;
    constexpr float kMinDerivative = 1.0e-3F;
    constexpr float kSearchEps = 1.0e-6F;
    const float left = -tail_bound;
    const float right = tail_bound;
    if (input < left || input > right) {
        return input;
    }
    // The exported spline softmaxes through ONNX Runtime's Softmax, i.e.
    // MlasComputeSoftmax (the SumExpF32 kernel + exp * (1 / sum)), not std::exp.
    const auto softmax_into = [](const float* in, float* out, std::size_t n) { mlas_softmax_row(in, out, n); };
    float widths[kSplineBins];
    float heights[kSplineBins];
    float cumwidths[kSplineBins + 1];
    float cumheights[kSplineBins + 1];
    float delta[kSplineBins];
    float derivatives[kSplineBins + 1];
    softmax_into(unnormalized_widths, widths, num_bins);
    for (std::size_t i = 0; i < num_bins; ++i) {
        widths[i] = kMinBinWidth + (1.0F - kMinBinWidth * static_cast<float>(num_bins)) * widths[i];
    }
    cumwidths[0] = 0.0F;
    for (std::size_t i = 0; i < num_bins; ++i) {
        cumwidths[i + 1] = cumwidths[i] + widths[i];
    }
    for (std::size_t i = 0; i <= num_bins; ++i) {
        cumwidths[i] = (right - left) * cumwidths[i] + left;
    }
    for (std::size_t i = 0; i < num_bins; ++i) {
        widths[i] = cumwidths[i + 1] - cumwidths[i];
    }
    for (std::size_t i = 0; i <= num_bins; ++i) {
        // F.pad(unnormalized_derivatives, (1, 1)) pads with zeros, and the
        // Softplus/Add(1e-3) chain then runs on the padded [1, T, bins+1]
        // tensor, so the Eigen element tail follows the frame-major index.
        const float raw = i == 0 || i == num_bins ? 0.0F : unnormalized_derivatives[i - 1];
        derivatives[i] = eigen_softplus_element(raw, frame_index * (num_bins + 1) + i, frame_count * (num_bins + 1)) +
                         kMinDerivative;
    }
    softmax_into(unnormalized_heights, heights, num_bins);
    for (std::size_t i = 0; i < num_bins; ++i) {
        heights[i] = kMinBinHeight + (1.0F - kMinBinHeight * static_cast<float>(num_bins)) * heights[i];
    }
    cumheights[0] = 0.0F;
    for (std::size_t i = 0; i < num_bins; ++i) {
        cumheights[i + 1] = cumheights[i] + heights[i];
    }
    for (std::size_t i = 0; i <= num_bins; ++i) {
        cumheights[i] = (right - left) * cumheights[i] + left;
    }
    for (std::size_t i = 0; i < num_bins; ++i) {
        heights[i] = cumheights[i + 1] - cumheights[i];
    }
    for (std::size_t i = 0; i < num_bins; ++i) {
        delta[i] = heights[i] / widths[i];
    }
    // searchsorted(cumheights, input) - 1, with the last location nudged by eps
    float locations[kSplineBins + 1];
    for (std::size_t i = 0; i <= num_bins; ++i) {
        locations[i] = cumheights[i];
    }
    locations[num_bins] += kSearchEps;
    std::size_t bin = 0;
    for (std::size_t i = 0; i <= num_bins; ++i) {
        if (input >= locations[i]) {
            bin += 1;
        }
    }
    bin -= 1;
    if (bin >= num_bins) {
        bin = num_bins - 1;
    }
    const float input_cumwidths = cumwidths[bin];
    const float input_bin_widths = widths[bin];
    const float input_cumheights = cumheights[bin];
    const float input_delta = delta[bin];
    const float input_derivatives = derivatives[bin];
    const float input_derivatives_plus_one = derivatives[bin + 1];
    const float input_heights = heights[bin];
    const float a = (input - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2.0F * input_delta) +
                    input_heights * (input_delta - input_derivatives);
    const float b = input_heights * input_derivatives -
                    (input - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2.0F * input_delta);
    const float c = -input_delta * (input - input_cumheights);
    const float discriminant = std::max(b * b - 4.0F * a * c, 0.0F);
    const float root = (2.0F * c) / (-b - std::sqrt(discriminant));
    return root * input_bin_widths + input_cumwidths;
}

// modules.ConvFlow with reverse=True: the second half of the channels goes
// through the inverse rational-quadratic spline.
[[nodiscard]] Act conv_flow(const Act& z, const Weights& weights, const std::string& prefix, const Act* cond,
                            const std::vector<float>& mask, const Reference* ref = nullptr,
                            const std::string& label = {}) {
    const std::size_t half = z.channels / 2;
    Act x0(half, z.length);
    Act x1(half, z.length);
    for (std::size_t c = 0; c < half; ++c) {
        for (std::size_t t = 0; t < z.length; ++t) {
            x0.at(c, t) = z.at(c, t);
            x1.at(c, t) = z.at(c + half, t);
        }
    }
    Act h = conv1d(x0, weights.at(prefix + "pre.weight"), weights.at(prefix + "pre.bias"), 1, 0);
    if (ref != nullptr && !label.empty()) {
        report(*ref, label + "_pre_Conv", h, label + " pre");
    }
    h = dds_conv(h, weights, prefix + "convs.", cond, mask, 3);
    h = conv1d(h, weights.at(prefix + "proj.weight"), weights.at(prefix + "proj.bias"), 1, 0);
    if (ref != nullptr && !label.empty()) {
        report(*ref, label + "_proj_Conv", h, label + " proj");
    }
    apply_mask(h, mask);
    const std::size_t proj_channels = h.channels;  // half * (num_bins * 3 - 1)
    const std::size_t per = proj_channels / half;
    const std::size_t num_bins = (per + 1) / 3;
    // The graph divides the raw widths/heights by the exported sqrt(192)
    // constant (a Div node), which does not round like a reciprocal multiply.
    constexpr float kSplineScaleDivisor = 13.856406211853027F;
    for (std::size_t c = 0; c < half; ++c) {
        for (std::size_t t = 0; t < z.length; ++t) {
            float widths[kSplineBins];
            float heights[kSplineBins];
            float derivs[kSplineBins - 1];
            const std::size_t base = c * per;
            for (std::size_t i = 0; i < num_bins; ++i) {
                widths[i] = h.at(base + i, t) / kSplineScaleDivisor;
                heights[i] = h.at(base + num_bins + i, t) / kSplineScaleDivisor;
            }
            for (std::size_t i = 0; i < num_bins - 1; ++i) {
                derivs[i] = h.at(base + 2 * num_bins + i, t);
            }
            x1.at(c, t) = spline_inverse(x1.at(c, t), widths, heights, derivs, num_bins, 5.0F, t, z.length);
        }
    }
    Act out(z.channels, z.length);
    for (std::size_t c = 0; c < half; ++c) {
        for (std::size_t t = 0; t < z.length; ++t) {
            out.at(c, t) = x0.at(c, t) * mask[t];
            out.at(c + half, t) = x1.at(c, t) * mask[t];
        }
    }
    return out;
}

// modules.ResidualCouplingLayer(mean_only=True, reverse=True)
[[nodiscard]] Act residual_coupling(const Act& z, const Weights& weights, const std::string& prefix,
                                    const std::vector<float>& mask, const Reference* ref, const std::string& label) {
    const std::size_t half = z.channels / 2;
    Act x0(half, z.length);
    Act x1(half, z.length);
    for (std::size_t c = 0; c < half; ++c) {
        for (std::size_t t = 0; t < z.length; ++t) {
            x0.at(c, t) = z.at(c, t);
            x1.at(c, t) = z.at(c + half, t);
        }
    }
    Act h = conv1d(x0, weights.at(prefix + "pre.weight"), weights.at(prefix + "pre.bias"), 1, 0);
    if (ref != nullptr && !label.empty()) {
        report(*ref, label + "_pre_Conv", h, label + " pre");
    }
    apply_mask(h, mask);
    h = wave_net(h, weights, prefix + "enc.", mask, 4, 5);
    Act stats = conv1d(h, weights.at(prefix + "post.weight"), weights.at(prefix + "post.bias"), 1, 0);
    if (ref != nullptr && !label.empty()) {
        report(*ref, label + "_post_Conv", stats, label + " post");
    }
    apply_mask(stats, mask);
    Act out(z.channels, z.length);
    for (std::size_t c = 0; c < half; ++c) {
        for (std::size_t t = 0; t < z.length; ++t) {
            out.at(c, t) = x0.at(c, t) * mask[t];
            out.at(c + half, t) = (x1.at(c, t) - stats.at(c, t)) * mask[t];
        }
    }
    return out;
}

// modules.Generator.forward: conv_pre, three (leaky, ConvTranspose, 3 x ResBlock
// averaged) stages, then leaky(0.01), conv_post and tanh.
[[nodiscard]] Act decode(const Act& z, const Weights& weights, const Reference& ref) {
    Act x = conv1d(z, weights.at("dec.conv_pre.weight"), weights.at("dec.conv_pre.bias"), 7, 3);
    report(ref, "dec_conv_pre_Conv", x, "dec conv_pre");
    const std::size_t strides[3] = {8, 8, 4};
    const std::size_t kernels[3] = {16, 16, 8};
    const std::size_t pads[3] = {4, 4, 2};
    const std::size_t res_kernels[3] = {3, 5, 7};
    const std::size_t res_d0[3] = {1, 2, 3};
    const std::size_t res_d1[3] = {2, 6, 12};
    for (std::size_t i = 0; i < 3; ++i) {
        leaky_relu(x, 0.1F);
        const std::string up = "dec.ups." + std::to_string(i);
        x = conv_transpose1d(x, weights.at(up + ".weight"), weights.at(up + ".bias"), strides[i], kernels[i], pads[i]);
        report(ref, "dec_ups." + std::to_string(i) + "_ConvTranspose", x, "dec ups " + std::to_string(i));
        Act sum(x.channels, x.length);
        for (std::size_t j = 0; j < 3; ++j) {
            const std::string rb = "dec.resblocks." + std::to_string(i * 3 + j);
            const std::string label = "dec_resblocks." + std::to_string(i * 3 + j);
            const Act block = res_block2(x, weights, rb + ".", res_kernels[j], res_d0[j], res_d1[j], &ref, label);
            for (std::size_t c = 0; c < x.data.size(); ++c) {
                sum.data[c] += block.data[c];
            }
        }
        for (float& value : sum.data) {
            value /= 3.0F;
        }
        x = sum;
    }
    leaky_relu(x, 0.01F);
    Act post = conv1d(x, weights.at("dec.conv_post.weight"), Weight{}, 7, 3);
    report(ref, "dec_conv_post_Conv", post, "dec conv_post");
    Act out = post;
    for (float& value : out.data) {
        value = ort_tanh(value);
    }
    return out;
}

struct Config {
    std::size_t hidden = 192;
    std::size_t filter = 768;
    std::size_t heads = 2;
    std::size_t layers = 6;
    std::size_t window = 4;
    std::size_t out_channels = 192;
};

[[nodiscard]] Act attention(const Act& x, const Weights& weights, const std::string& prefix, const Config& cfg) {
    const std::size_t t_len = x.length;
    const std::size_t heads = cfg.heads;
    const std::size_t head_dim = cfg.hidden / heads;
    const Weight& wq = weights.at(prefix + "conv_q.weight");
    const Weight& bq = weights.at(prefix + "conv_q.bias");
    const Weight& wk = weights.at(prefix + "conv_k.weight");
    const Weight& bk = weights.at(prefix + "conv_k.bias");
    const Weight& wv = weights.at(prefix + "conv_v.weight");
    const Weight& bv = weights.at(prefix + "conv_v.bias");
    const Act q = conv1d(x, wq, bq, 1, 0);
    const Act k = conv1d(x, wk, bk, 1, 0);
    const Act v = conv1d(x, wv, bv, 1, 0);

    const std::vector<float> rel_k = relative_embeddings(weights.at(prefix + "emb_rel_k"), cfg.window, t_len);
    const std::vector<float> rel_v = relative_embeddings(weights.at(prefix + "emb_rel_v"), cfg.window, t_len);
    // The graph scales Q with a Div node (Q / sqrt(head_dim)); the graph
    // optimiser hoists the constant out of the MatMuls, so the products use the
    // unscaled Q and the result is multiplied by the float reciprocal
    // (verified against MatMul_1: 276/276 with the post-multiply, 181/276 with
    // a pre-divided Q).
    const float div_scale = std::sqrt(static_cast<float>(head_dim));
    const float inv_scale = 1.0F / div_scale;
    // All four MatMuls here have K below the MLAS stride, so each output is one
    // sequential fma chain; the blocking helper keeps that explicit.
    const std::size_t kblock_scores = mlas_gemm_kblock(t_len, head_dim);
    const std::size_t kblock_rel = mlas_gemm_kblock(2 * t_len - 1, head_dim);
    const std::size_t kblock_v = mlas_gemm_kblock(head_dim, t_len);
    const std::size_t kblock_relv = mlas_gemm_kblock(head_dim, 2 * t_len - 1);

    const std::string tag = "L" + prefix.substr(prefix.find("attn_layers.") + 12, 2);
    debug_dump(tag + "_rel_k", rel_k);
    debug_dump(tag + "_rel_v", rel_v);
    Act out(cfg.hidden, t_len);
    std::vector<float> scores(t_len * t_len);
    std::vector<float> rel_logits(t_len * (2 * t_len - 1));
    std::vector<float> probs(t_len * t_len);
    std::vector<float> rel_weights(t_len * (2 * t_len - 1));
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t base = h * head_dim;
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t j = 0; j < t_len; ++j) {
                float total = 0.0F;
                for (std::size_t d0 = 0; d0 < head_dim; d0 += kblock_scores) {
                    const std::size_t d1 = std::min(d0 + kblock_scores, head_dim);
                    float partial = 0.0F;
                    for (std::size_t d = d0; d < d1; ++d) {
                        partial = std::fma(q.at(base + d, i), k.at(base + d, j), partial);
                    }
                    total += partial;
                }
                scores[i * t_len + j] = total * inv_scale;
            }
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t m = 0; m < 2 * t_len - 1; ++m) {
                float total = 0.0F;
                for (std::size_t d0 = 0; d0 < head_dim; d0 += kblock_rel) {
                    const std::size_t d1 = std::min(d0 + kblock_rel, head_dim);
                    float partial = 0.0F;
                    for (std::size_t d = d0; d < d1; ++d) {
                        partial = std::fma(q.at(base + d, i), rel_k[m * head_dim + d], partial);
                    }
                    total += partial;
                }
                rel_logits[i * (2 * t_len - 1) + m] = total * inv_scale;
            }
        }
        const std::vector<float> local = relative_to_absolute(rel_logits, t_len);
        if (h == 0) {
            debug_dump(tag + "_rel_logits", rel_logits);
            debug_dump(tag + "_rel_abs", local);
        }
        for (std::size_t i = 0; i < t_len * t_len; ++i) {
            scores[i] += local[i];
        }
        if (h == 0) {
            debug_dump(tag + "_scores", scores);
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            mlas_softmax_row(scores.data() + i * t_len, probs.data() + i * t_len, t_len);
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                float total = 0.0F;
                for (std::size_t j0 = 0; j0 < t_len; j0 += kblock_v) {
                    const std::size_t j1 = std::min(j0 + kblock_v, t_len);
                    float partial = 0.0F;
                    for (std::size_t j = j0; j < j1; ++j) {
                        partial = std::fma(probs[i * t_len + j], v.at(base + d, j), partial);
                    }
                    total += partial;
                }
                out.at(base + d, i) = total;
            }
        }
        if (h == 0) {
            debug_dump(tag + "_probs", probs);
        }
        const std::vector<float> rels = absolute_to_relative(probs, t_len);
        if (h == 0) {
            debug_dump(tag + "_probs_rel", rels);
        }
        for (std::size_t i = 0; i < t_len; ++i) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                float total = 0.0F;
                for (std::size_t m0 = 0; m0 < 2 * t_len - 1; m0 += kblock_relv) {
                    const std::size_t m1 = std::min(m0 + kblock_relv, 2 * t_len - 1);
                    float partial = 0.0F;
                    for (std::size_t m = m0; m < m1; ++m) {
                        partial = std::fma(rels[i * (2 * t_len - 1) + m], rel_v[m * head_dim + d], partial);
                    }
                    total += partial;
                }
                out.at(base + d, i) += total;
            }
        }
    }
    return conv1d(out, weights.at(prefix + "conv_o.weight"), weights.at(prefix + "conv_o.bias"), 1, 0);
}

[[nodiscard]] Act ffn(const Act& x, const Weights& weights, const std::string& prefix, const std::vector<float>& mask) {
    Act h = conv1d(x, weights.at(prefix + "conv_1.weight"), weights.at(prefix + "conv_1.bias"), 3, 1);
    relu_in_place(h);
    apply_mask(h, mask);
    h = conv1d(h, weights.at(prefix + "conv_2.weight"), weights.at(prefix + "conv_2.bias"), 3, 1);
    apply_mask(h, mask);
    return h;
}

[[nodiscard]] std::vector<std::int64_t> load_ids(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto bytes = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<float> values(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    std::vector<std::int64_t> ids(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        ids[i] = static_cast<std::int64_t>(values[i]);
    }
    return ids;
}

}  // namespace

namespace mnet_tts {

[[nodiscard]] std::int32_t run_piper(const TtsOptions& opt, const Weights& weights) {
    Reference ref;
    if (!opt.ref_dir.empty()) {
        ref.load(opt.ref_dir);
    }
    const std::vector<std::int64_t> ids =
        opt.ids_arg.empty() ? load_ids(opt.model_dir + "/piper_ids.bin") : parse_id_list(opt.ids_arg);
    const std::size_t tokens = ids.size();
    const Config cfg;

    // TextEncoder.forward: x = emb(ids) * sqrt(hidden), [T, C] -> [C, T], mask.
    std::vector<float> mask(tokens, 1.0F);
    Act x(cfg.hidden, tokens);
    const Weight& emb = weights.at("sid");
    for (std::size_t t = 0; t < tokens; ++t) {
        const std::size_t row = static_cast<std::size_t>(ids[t]);
        for (std::size_t c = 0; c < cfg.hidden; ++c) {
            x.at(c, t) = emb.values[row * cfg.hidden + c] * std::sqrt(static_cast<float>(cfg.hidden));
        }
    }
    {
        // the reference dump is the raw Gather, before the sqrt(hidden) scale
        Act gathered(cfg.hidden, tokens);
        for (std::size_t t = 0; t < tokens; ++t) {
            const std::size_t row = static_cast<std::size_t>(ids[t]);
            for (std::size_t c = 0; c < cfg.hidden; ++c) {
                gathered.at(c, t) = emb.values[row * cfg.hidden + c];
            }
        }
        report(ref, "enc_p_emb_Gather", gathered, "embedding", true);
    }
    apply_mask(x, mask);

    for (std::size_t layer = 0; layer < cfg.layers; ++layer) {
        const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(layer) + ".";
        const Act attn = attention(x, weights, prefix, cfg);
        // the reference dumps conv_q/k/v/o separately
        {
            const Act q = conv1d(x, weights.at(prefix + "conv_q.weight"), weights.at(prefix + "conv_q.bias"), 1, 0);
            const Act k = conv1d(x, weights.at(prefix + "conv_k.weight"), weights.at(prefix + "conv_k.bias"), 1, 0);
            const Act v = conv1d(x, weights.at(prefix + "conv_v.weight"), weights.at(prefix + "conv_v.bias"), 1, 0);
            const std::string base = "enc_p_encoder_attn_layers." + std::to_string(layer) + "_conv_";
            report(ref, base + "q_Conv", q, "layer " + std::to_string(layer) + " conv_q");
            report(ref, base + "k_Conv", k, "layer " + std::to_string(layer) + " conv_k");
            report(ref, base + "v_Conv", v, "layer " + std::to_string(layer) + " conv_v");
            report(ref, base + "o_Conv", attn, "layer " + std::to_string(layer) + " conv_o");
        }
        Act residual = x;
        for (std::size_t i = 0; i < x.data.size(); ++i) {
            residual.data[i] += attn.data[i];
        }
        const std::string ln1 = "enc_p.encoder.norm_layers_1." + std::to_string(layer) + ".";
        x = piper_norm(residual, weights.at(ln1 + "gamma"), weights.at(ln1 + "beta"));

        const std::string ffn_prefix = "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".";
        const std::string ffn_tag = "enc_p_encoder_ffn_layers." + std::to_string(layer) + "_conv_";
        {
            const Act h =
                conv1d(x, weights.at(ffn_prefix + "conv_1.weight"), weights.at(ffn_prefix + "conv_1.bias"), 3, 1);
            report(ref, ffn_tag + "1_Conv", h, "layer " + std::to_string(layer) + " ffn conv_1");
            Act hr = h;
            relu_in_place(hr);
            apply_mask(hr, mask);
            const Act h2 =
                conv1d(hr, weights.at(ffn_prefix + "conv_2.weight"), weights.at(ffn_prefix + "conv_2.bias"), 3, 1);
            report(ref, ffn_tag + "2_Conv", h2, "layer " + std::to_string(layer) + " ffn conv_2");
        }
        const Act f = ffn(x, weights, ffn_prefix, mask);
        Act residual2 = x;
        for (std::size_t i = 0; i < x.data.size(); ++i) {
            residual2.data[i] += f.data[i];
        }
        const std::string ln2 = "enc_p.encoder.norm_layers_2." + std::to_string(layer) + ".";
        x = piper_norm(residual2, weights.at(ln2 + "gamma"), weights.at(ln2 + "beta"));
        report(ref, "enc_p_encoder_norm_layers_2." + std::to_string(layer) + "_Mul",
               piper_norm(residual2, weights.at(ln2 + "gamma"), weights.at(ln2 + "beta"), false),
               "layer " + std::to_string(layer) + " norm_2", true);
    }
    apply_mask(x, mask);

    // TextEncoder.proj -> (m, logs), 384 channels.
    Act stats = conv1d(x, weights.at("enc_p.proj.weight"), weights.at("enc_p.proj.bias"), 1, 0);
    apply_mask(stats, mask);
    report(ref, "enc_p_proj_Conv", stats, "proj (m||logs)");

    // StochasticDurationPredictor.forward(reverse=True): pre -> DDSConv -> proj,
    // then the reversed flow list (Flip, ConvFlow, ..., ElementwiseAffine) on a
    // zero noise vector, and logw is the first channel.
    Act dp = conv1d(x, weights.at("dp.pre.weight"), weights.at("dp.pre.bias"), 1, 0);
    report(ref, "dp_pre_Conv", dp, "dp pre");
    {
        Act h = dds_conv(dp, weights, "dp.convs.", nullptr, mask, 3, &ref);
        dp = conv1d(h, weights.at("dp.proj.weight"), weights.at("dp.proj.bias"), 1, 0);
        apply_mask(dp, mask);
    }
    report(ref, "dp_proj_Conv", dp, "dp proj");
    Act z(2, tokens);
    if (opt.noise_w != 0.0F) {
        Gaussian gaussian(opt.seed);
        for (std::size_t c = 0; c < z.channels; ++c) {
            for (std::size_t t = 0; t < z.length; ++t) {
                z.at(c, t) = gaussian.next() * opt.noise_w;
            }
        }
    }
    flip_channels(z);
    z = conv_flow(z, weights, "dp.flows.7.", &dp, mask, &ref, "dp_flows.7");
    flip_channels(z);
    z = conv_flow(z, weights, "dp.flows.5.", &dp, mask, &ref, "dp_flows.5");
    flip_channels(z);
    z = conv_flow(z, weights, "dp.flows.3.", &dp, mask, &ref, "dp_flows.3");
    flip_channels(z);
    elementwise_affine(z, weights.at("dp.flows.0.m"), weights.at("/dp/flows.0/Exp_output_0"), mask);
    Act logw(1, tokens);
    for (std::size_t t = 0; t < tokens; ++t) {
        logw.at(0, t) = z.at(0, t);
    }
    report(ref, "dp_Split", logw, "dp logw");

    // duration = ceil(exp(logw) * length_scale), summed over tokens
    constexpr float kLengthScale = 1.0F;
    std::vector<std::int32_t> duration(tokens, 0);
    std::int32_t frames = 0;
    for (std::size_t t = 0; t < tokens; ++t) {
        const float w = std::exp(logw.at(0, t)) * mask[t] * kLengthScale;
        duration[t] = static_cast<std::int32_t>(std::ceil(w));
        frames += duration[t];
    }
    std::printf("  %-42s %d frames\n", "duration (sum of ceil(exp(logw)))", frames);

    // generate_path: every text token owns duration[t] frames.
    std::vector<std::int32_t> frame_token(static_cast<std::size_t>(frames), 0);
    {
        std::int32_t frame = 0;
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::int32_t k = 0; k < duration[t] && frame < frames; ++k) {
                frame_token[static_cast<std::size_t>(frame)] = static_cast<std::int32_t>(t);
                ++frame;
            }
        }
    }
    // z_p = m_p + randn * exp(logs_p) * opt.noise_scale, expanded to the frame grid
    Gaussian z_gaussian(opt.seed ^ 0x9E3779B9U);
    const bool add_z_noise = opt.noise_scale != 0.0F;
    Act zf(cfg.hidden, static_cast<std::size_t>(frames));
    for (std::size_t c = 0; c < cfg.hidden; ++c) {
        for (std::size_t f = 0; f < zf.length; ++f) {
            const std::size_t token = static_cast<std::size_t>(frame_token[f]);
            float value = stats.at(c, token);
            if (add_z_noise) {
                value += z_gaussian.next() * std::exp(stats.at(cfg.hidden + c, token)) * opt.noise_scale;
            }
            zf.at(c, f) = value;
        }
    }
    std::vector<float> frame_mask(zf.length, 1.0F);
    flip_channels(zf);
    zf = residual_coupling(zf, weights, "flow.flows.6.", frame_mask, &ref, "flow_flows.6");
    flip_channels(zf);
    zf = residual_coupling(zf, weights, "flow.flows.4.", frame_mask, &ref, "flow_flows.4");
    flip_channels(zf);
    zf = residual_coupling(zf, weights, "flow.flows.2.", frame_mask, &ref, "flow_flows.2");
    flip_channels(zf);
    zf = residual_coupling(zf, weights, "flow.flows.0.", frame_mask, &ref, "flow_flows.0");
    const Act audio = decode(zf, weights, ref);
    if (!opt.dump_dir.empty()) {
        std::ofstream out(opt.dump_dir + "/audio.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(audio.data.data()),
                  static_cast<std::streamsize>(audio.data.size() * sizeof(float)));
        out.close();
    }
    std::printf("  %-42s %zu samples\n", "decoded waveform", audio.length);
    if (!opt.wav_out.empty()) {
        write_wav(opt.wav_out, audio.data, 22050U);
    }
    if (!ref.tensors.empty()) {
        const auto it = ref.tensors.find("output");
        if (it != ref.tensors.end() && it->second.size() == audio.data.size()) {
            double max_abs = 0.0;
            double sum = 0.0;
            double dot = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;
            for (std::size_t i = 0; i < audio.data.size(); ++i) {
                const double a = static_cast<double>(audio.data[i]);
                const double b = static_cast<double>(it->second[i]);
                const double d = std::fabs(a - b);
                max_abs = std::max(max_abs, d);
                sum += d;
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            const double corr = norm_a > 0.0 && norm_b > 0.0 ? dot / std::sqrt(norm_a * norm_b) : 0.0;
            std::printf("  %-42s max_abs %.3e  mean %.3e  cos %.9f\n", "waveform vs ref/output", max_abs,
                        sum / static_cast<double>(audio.data.size()), corr);
        }
    }

    if (!opt.dump_dir.empty()) {
        std::ofstream out(opt.dump_dir + "/encoder.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(x.data.data()),
                  static_cast<std::streamsize>(x.data.size() * sizeof(float)));
        out.close();
        std::ofstream stats_out(opt.dump_dir + "/stats.bin", std::ios::binary);
        stats_out.write(reinterpret_cast<const char*>(stats.data.data()),
                        static_cast<std::streamsize>(stats.data.size() * sizeof(float)));
    }
    return 0;
}

}  // namespace mnet_tts
