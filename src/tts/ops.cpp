// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mnet/tts/tts_exact.hpp>
#include <mnet/tts/tts_ops.hpp>
#include <mnet/tts/tts_reference.hpp>
#include <mnet/tts/tts_tensor.hpp>

namespace mnet::tts {

[[nodiscard]] Act conv1d(const Act& input, const Weight& weight, const Weight& bias, std::size_t kernel,
                         std::size_t pad, std::size_t dilation, std::size_t groups, std::size_t stride) {
    const ProfileScope profile(kProfileConv);
    const std::size_t out_channels = weight.size(0);
    const std::size_t in_per_group = weight.size(1);
    const std::size_t taps = weight.size(2);
    if (taps != kernel || in_per_group * groups != input.channels) {
        throw std::runtime_error("conv1d shape mismatch: kernel " + std::to_string(taps) + " vs " +
                                 std::to_string(kernel) + ", in " + std::to_string(input.channels) + " vs " +
                                 std::to_string(in_per_group * groups) + ", groups " + std::to_string(groups));
    }
    const std::size_t out_per_group = out_channels / groups;
    const std::size_t effective_kernel = dilation * (kernel - 1) + 1;
    const std::size_t out_len = (input.length + 2 * pad - effective_kernel) / stride + 1;
    // ONNX Runtime's Conv (MlasConvAlgorithmExpandThenGemm) expands the input
    // to im2col columns and runs SGEMM over K in blocks of StrideK: StrideK
    // starts at MLAS_SGEMM_STRIDEK = 128, halves while the output segment is
    // at least K, and doubles (with the N stride halving) otherwise. Each
    // block accumulates sequentially with vfmadd and the blocks are then
    // added, which is not the same rounding as one long sum.
    const std::size_t total_k = in_per_group * kernel;
    const bool pointwise = taps == 1 && stride == 1 && pad == 0 && groups == 1;
    const std::vector<std::pair<std::size_t, std::size_t>> ranges =
        mlas_conv_ranges(out_channels, out_len, total_k, pointwise, g_ort_threads);
    Act out(out_channels, out_len);
    if (groups == input.channels && in_per_group == 1 && out_per_group == 1) {
        // Depthwise convolution (convolve.cpp: GroupCount == InputChannels) is
        // dispatched to MlasConvDepthwiseFloat_CHW, which accumulates the taps
        // with separate multiplies and adds instead of the GEMM path's FMA.
        parallel_for(out_channels, out_len * kernel, [&](std::size_t c) {
            const float* tap_weight = weight.values.data() + c * kernel;
            const float* src = input.data.data() + c * input.length;
            const float b = bias.values.empty() ? 0.0F : bias.values[c];
            for (std::size_t t = 0; t < out_len; ++t) {
                float acc = 0.0F;
                for (std::size_t k = 0; k < kernel; ++k) {
                    const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(t * stride) +
                                              static_cast<std::ptrdiff_t>(k * dilation) -
                                              static_cast<std::ptrdiff_t>(pad);
                    if (at >= 0 && at < static_cast<std::ptrdiff_t>(input.length)) {
                        acc += tap_weight[k] * src[at];
                    }
                }
                out.at(c, t) = acc + b;
            }
        });
        return out;
    }
    if (out_channels == 1 && groups == 1) {
        // A single filter makes M == 1, so MlasSgemmOperation takes its small-M
        // path before any StrideK blocking: MlasSgemmKernelM1Avx walks the
        // whole K in groups of four rows, forming ((p0 + p1) + p2) + p3 with
        // separate multiplies (no FMA) and accumulating each group into C.
        const float* row = weight.values.data();
        const float b = bias.values.empty() ? 0.0F : bias.values[0];
        parallel_for(out_len, total_k, [&](std::size_t t) {
            std::vector<float> image(total_k);
            for (std::size_t kidx = 0; kidx < total_k; ++kidx) {
                const std::size_t ic = kidx / kernel;
                const std::size_t tap = kidx % kernel;
                const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(t * stride) +
                                          static_cast<std::ptrdiff_t>(tap * dilation) -
                                          static_cast<std::ptrdiff_t>(pad);
                image[kidx] = (at >= 0 && at < static_cast<std::ptrdiff_t>(input.length))
                                  ? input.at(ic, static_cast<std::size_t>(at))
                                  : 0.0F;
            }
            float total = 0.0F;
            std::size_t k = 0;
            for (; k + 4 <= total_k; k += 4) {
                const float p0 = row[k] * image[k];
                const float p1 = row[k + 1] * image[k + 1];
                const float p2 = row[k + 2] * image[k + 2];
                const float p3 = row[k + 3] * image[k + 3];
                total += ((p0 + p1) + p2) + p3;
            }
            if (k + 2 <= total_k) {
                total += row[k] * image[k] + row[k + 1] * image[k + 1];
                k += 2;
            }
            if (k < total_k) {
                total += row[k] * image[k];
            }
            out.at(0, t) = total + b;
        });
        return out;
    }
    const std::size_t jobs = groups * out_per_group;
    parallel_for(jobs, out_len * total_k, [&](std::size_t job) {
        {
            const std::size_t g = job / out_per_group;
            const std::size_t o = job % out_per_group;
            const std::size_t oc = g * out_per_group + o;
            const float* row = weight.values.data() + oc * in_per_group * kernel;
            std::vector<float> partial(out_len);
            std::vector<float> total(out_len);
            std::fill(total.begin(), total.end(), 0.0F);
            for (const auto& range : ranges) {
                const std::size_t chunk = mlas_gemm_kblock(range.second, total_k);
                for (std::size_t k0 = 0; k0 < total_k; k0 += chunk) {
                    const std::size_t k1 = std::min(k0 + chunk, total_k);
                    std::fill(partial.begin(), partial.end(), 0.0F);
                    for (std::size_t kidx = k0; kidx < k1; ++kidx) {
                        const std::size_t ic = kidx / kernel;
                        const std::size_t tap = kidx % kernel;
                        const float w = row[kidx];
                        const float* src = input.data.data() + (g * in_per_group + ic) * input.length;
                        for (std::size_t t = range.first; t < range.first + range.second; ++t) {
                            const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(t * stride) +
                                                      static_cast<std::ptrdiff_t>(tap * dilation) -
                                                      static_cast<std::ptrdiff_t>(pad);
                            if (at >= 0 && at < static_cast<std::ptrdiff_t>(input.length)) {
                                partial[t] = std::fma(w, src[at], partial[t]);
                            }
                        }
                    }
                    for (std::size_t t = range.first; t < range.first + range.second; ++t) {
                        total[t] += partial[t];
                    }
                }
            }
            const float b = bias.values.empty() ? 0.0F : bias.values[oc];
            for (std::size_t t = 0; t < out_len; ++t) {
                out.at(oc, t) = total[t] + b;
            }
        }
    });
    return out;
}

[[nodiscard]] Act layer_norm(const Act& input, const Weight& gamma, const Weight& beta, float eps) {
    // ONNX Runtime's MlasLayerNormKernelAvx2:
    //   mean: double accumulators in four strided lanes, reduced (d0+d2)+(d1+d3)
    //         then the n % 4 tail; cast to float;
    //   var : eight float lanes of fma((x-mean),(x-mean)) reduced
    //         ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)) plus the n % 8 tail;
    //   out : fma((x - mean) * inv, gamma, beta).
    Act out(input.channels, input.length);
    const std::size_t n = input.channels;
    for (std::size_t t = 0; t < input.length; ++t) {
        double lanes_d[4] = {0.0, 0.0, 0.0, 0.0};
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            for (std::size_t j = 0; j < 4; ++j) {
                lanes_d[j] += static_cast<double>(input.at(i + j, t));
            }
        }
        double dsum = (lanes_d[0] + lanes_d[2]) + (lanes_d[1] + lanes_d[3]);
        for (; i < n; ++i) {
            dsum += static_cast<double>(input.at(i, t));
        }
        const float mean = static_cast<float>(dsum / static_cast<double>(n));
        float lanes_v[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        i = 0;
        for (; i + 8 <= n; i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                const float v = input.at(i + j, t) - mean;
                lanes_v[j] = std::fma(v, v, lanes_v[j]);
            }
        }
        float var = ((lanes_v[0] + lanes_v[4]) + (lanes_v[2] + lanes_v[6])) +
                    ((lanes_v[1] + lanes_v[5]) + (lanes_v[3] + lanes_v[7]));
        for (; i < n; ++i) {
            const float v = input.at(i, t) - mean;
            var += v * v;
        }
        const float inv = 1.0F / std::sqrt(var / static_cast<float>(n) + eps);
        for (std::size_t c = 0; c < n; ++c) {
            out.at(c, t) = std::fma((input.at(c, t) - mean) * inv, gamma.values[c], beta.values[c]);
        }
    }
    return out;
}

[[nodiscard]] Act layer_norm_unit(const Act& input, float eps) {
    // The graph feeds ONNX LayerNormalization scale=1, bias=0, so the MLAS
    // kernel runs its affine path with gamma 1, beta 0:
    //   mean/var exactly as layer_norm, output = fma((x-mean)*inv, 1, 0).
    Act out(input.channels, input.length);
    const std::size_t n = input.channels;
    for (std::size_t t = 0; t < input.length; ++t) {
        double lanes_d[4] = {0.0, 0.0, 0.0, 0.0};
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            for (std::size_t j = 0; j < 4; ++j) {
                lanes_d[j] += static_cast<double>(input.at(i + j, t));
            }
        }
        double dsum = (lanes_d[0] + lanes_d[2]) + (lanes_d[1] + lanes_d[3]);
        for (; i < n; ++i) {
            dsum += static_cast<double>(input.at(i, t));
        }
        const float mean = static_cast<float>(dsum / static_cast<double>(n));
        float lanes_v[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        i = 0;
        for (; i + 8 <= n; i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                const float v = input.at(i + j, t) - mean;
                lanes_v[j] = std::fma(v, v, lanes_v[j]);
            }
        }
        float var = ((lanes_v[0] + lanes_v[4]) + (lanes_v[2] + lanes_v[6])) +
                    ((lanes_v[1] + lanes_v[5]) + (lanes_v[3] + lanes_v[7]));
        for (; i < n; ++i) {
            const float v = input.at(i, t) - mean;
            var += v * v;
        }
        const float inv = 1.0F / std::sqrt(var / static_cast<float>(n) + eps);
        for (std::size_t c = 0; c < n; ++c) {
            out.at(c, t) = std::fma((input.at(c, t) - mean) * inv, 1.0F, 0.0F);
        }
    }
    return out;
}

void apply_mask(Act& act, const std::vector<float>& mask) {
    for (std::size_t c = 0; c < act.channels; ++c) {
        for (std::size_t t = 0; t < act.length; ++t) {
            act.at(c, t) *= mask[t];
        }
    }
}

void relu_in_place(Act& act) {
    for (float& value : act.data) {
        value = std::max(value, 0.0F);
    }
}

[[nodiscard]] Act piper_norm(const Act& input, const Weight& gamma, const Weight& beta, bool with_beta) {
    Act out(input.channels, input.length);
    const float n = static_cast<float>(input.channels);
    for (std::size_t t = 0; t < input.length; ++t) {
        float sum = 0.0F;
        for (std::size_t c = 0; c < input.channels; ++c) {
            sum += input.at(c, t);
        }
        const float mean = sum / n;
        float sum_squared = 0.0F;
        for (std::size_t c = 0; c < input.channels; ++c) {
            const float delta = input.at(c, t) - mean;
            sum_squared += delta * delta;
        }
        const float variance = sum_squared / n;
        const float denom = std::sqrt(variance + 1.0e-5F);
        for (std::size_t c = 0; c < input.channels; ++c) {
            const float scaled = ((input.at(c, t) - mean) / denom) * gamma.values[c];
            out.at(c, t) = with_beta ? scaled + beta.values[c] : scaled;
        }
    }
    return out;
}

[[nodiscard]] Act gelu(const Act& input) {
    constexpr float kInverseSqrt2 = 0.70710678118654752440F;
    Act out = input;
    for (float& value : out.data) {
        const float x = value;
        const float scaled = x * kInverseSqrt2;
        value = 0.5F * x * (mlas_erf(scaled) + 1.0F);
    }
    return out;
}

[[nodiscard]] Act dds_conv(const Act& input, const Weights& weights, const std::string& prefix, const Act* cond,
                           const std::vector<float>& mask, std::size_t kernel, const Reference* ref) {
    Act x = input;
    if (cond != nullptr) {
        for (std::size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] += cond->data[i];
        }
    }
    const std::size_t layers = 3;
    for (std::size_t i = 0; i < layers; ++i) {
        const std::size_t dilation = static_cast<std::size_t>(std::pow(kernel, static_cast<double>(i)));
        const std::size_t pad = (kernel * dilation - dilation) / 2;
        const std::string tag = prefix + "convs_sep." + std::to_string(i);
        Act y = conv1d(x, weights.at(tag + ".weight"), weights.at(tag + ".bias"), kernel, pad, dilation, x.channels);
        if (ref != nullptr && prefix == "dp.convs.") {
            report(*ref, "dp_convs_convs_sep." + std::to_string(i) + "_Conv", y, "dp convs_sep " + std::to_string(i));
        }
        // Both norms in DDSConv are the same channel-axis LayerNorm
        // decomposition as the encoder norms, so they take the sequential
        // ReduceMean path (see piper_norm).
        y = piper_norm(y, weights.at(prefix + "norms_1." + std::to_string(i) + ".gamma"),
                       weights.at(prefix + "norms_1." + std::to_string(i) + ".beta"));
        y = gelu(y);
        const std::string tag2 = prefix + "convs_1x1." + std::to_string(i);
        y = conv1d(y, weights.at(tag2 + ".weight"), weights.at(tag2 + ".bias"), 1, 0);
        if (ref != nullptr && prefix == "dp.convs.") {
            report(*ref, "dp_convs_convs_1x1." + std::to_string(i) + "_Conv", y, "dp convs_1x1 " + std::to_string(i));
        }
        y = piper_norm(y, weights.at(prefix + "norms_2." + std::to_string(i) + ".gamma"),
                       weights.at(prefix + "norms_2." + std::to_string(i) + ".beta"));
        y = gelu(y);
        for (std::size_t j = 0; j < x.data.size(); ++j) {
            x.data[j] += y.data[j];
        }
    }
    apply_mask(x, mask);
    return x;
}

[[nodiscard]] Act conv_transpose1d(const Act& input, const Weight& weight, const Weight& bias, std::size_t stride,
                                   std::size_t kernel, std::size_t pad) {
    const ProfileScope profile(kProfileConvTranspose);
    const std::size_t in_channels = weight.size(0);
    const std::size_t out_channels = weight.size(1);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv_transpose1d shape mismatch");
    }
    // ONNX Runtime's ConvTranspose (providers/cpu/nn/conv_transpose.cc):
    // first Gemm(CblasTrans, CblasNoTrans, kernel_dim, input_image_size,
    // in_channels) writes col = W^T * X (one fma chain per element over the
    // input channels, K blocks of StrideK; TransA != NoTrans keeps StrideK at
    // 128), then Col2imNd scatters the col rows into the zeroed output in
    // increasing col index (i.e. increasing tap) order, and the bias is added
    // last.
    const std::size_t cols = out_channels * kernel;
    const std::size_t frames = input.length;
    std::vector<float> col(cols * frames);
    const std::size_t chunk = mlas_gemm_kblock(frames, in_channels);
    parallel_for(cols, frames * in_channels, [&](std::size_t c) {
        const std::size_t oc = c / kernel;
        const std::size_t tap = c % kernel;
        for (std::size_t n = 0; n < frames; ++n) {
            float total = 0.0F;
            for (std::size_t k0 = 0; k0 < in_channels; k0 += chunk) {
                const std::size_t k1 = std::min(k0 + chunk, in_channels);
                float partial = 0.0F;
                for (std::size_t ic = k0; ic < k1; ++ic) {
                    partial =
                        std::fma(weight.values[(ic * out_channels + oc) * kernel + tap], input.at(ic, n), partial);
                }
                total += partial;
            }
            col[c * frames + n] = total;
        }
    });
    const std::size_t out_len = (frames - 1) * stride + kernel - 2 * pad;
    Act out(out_channels, out_len);
    // Col2imNd adds the col rows in increasing tap order; parallelising over
    // output channels keeps that per-element order untouched.
    parallel_for(out_channels, frames * kernel, [&](std::size_t oc) {
        for (std::size_t tap = 0; tap < kernel; ++tap) {
            const float* row = col.data() + (oc * kernel + tap) * frames;
            for (std::size_t n = 0; n < frames; ++n) {
                const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(n * stride) + static_cast<std::ptrdiff_t>(tap) -
                                          static_cast<std::ptrdiff_t>(pad);
                if (at >= 0 && at < static_cast<std::ptrdiff_t>(out_len)) {
                    out.at(oc, static_cast<std::size_t>(at)) += row[n];
                }
            }
        }
        const float b = bias.values[oc];
        for (std::size_t t = 0; t < out_len; ++t) {
            out.at(oc, t) += b;
        }
    });
    return out;
}

void leaky_relu(Act& act, float slope) {
    for (float& value : act.data) {
        value = value >= 0.0F ? value : value * slope;
    }
}

[[nodiscard]] Act wave_net(const Act& input, const Weights& weights, const std::string& prefix,
                           const std::vector<float>& mask, std::size_t layers, std::size_t kernel) {
    const std::size_t hidden = input.channels;
    Act x = input;
    Act output(hidden, x.length);
    for (std::size_t i = 0; i < layers; ++i) {
        const std::string tag = prefix + "in_layers." + std::to_string(i);
        const Act x_in = conv1d(x, weights.at(tag + ".weight"), weights.at(tag + ".bias"), kernel, (kernel - 1) / 2);
        // fused_add_tanh_sigmoid_multiply: tanh(low half) * sigmoid(high half),
        // through ONNX Runtime's Tanh/Sigmoid kernels (MlasComputeTanh /
        // MlasComputeLogistic), not std::tanh/std::exp.
        Act acts(hidden, x.length);
        for (std::size_t c = 0; c < hidden; ++c) {
            for (std::size_t t = 0; t < x.length; ++t) {
                acts.at(c, t) = ort_tanh(x_in.at(c, t)) * ort_logistic(x_in.at(c + hidden, t));
            }
        }
        const std::string skip = prefix + "res_skip_layers." + std::to_string(i);
        Act res_skip = conv1d(acts, weights.at(skip + ".weight"), weights.at(skip + ".bias"), 1, 0);
        const std::size_t skip_channels = res_skip.channels;
        if (i + 1 < layers) {
            for (std::size_t c = 0; c < hidden; ++c) {
                for (std::size_t t = 0; t < x.length; ++t) {
                    x.at(c, t) = (x.at(c, t) + res_skip.at(c, t)) * mask[t];
                }
            }
            for (std::size_t c = hidden; c < skip_channels; ++c) {
                for (std::size_t t = 0; t < x.length; ++t) {
                    output.at(c - hidden, t) += res_skip.at(c, t);
                }
            }
        } else {
            for (std::size_t c = 0; c < skip_channels; ++c) {
                for (std::size_t t = 0; t < x.length; ++t) {
                    output.at(c, t) += res_skip.at(c, t);
                }
            }
        }
    }
    apply_mask(output, mask);
    return output;
}

[[nodiscard]] Act res_block2(const Act& input, const Weights& weights, const std::string& prefix, std::size_t kernel,
                             std::size_t d0, std::size_t d1, const Reference* ref, const std::string& label) {
    Act x = input;
    const std::size_t dilations[2] = {d0, d1};
    for (std::size_t i = 0; i < 2; ++i) {
        Act xt = x;
        leaky_relu(xt, 0.1F);
        const std::string tag = prefix + "convs." + std::to_string(i);
        xt = conv1d(xt, weights.at(tag + ".weight"), weights.at(tag + ".bias"), kernel, dilations[i] * (kernel - 1) / 2,
                    dilations[i]);
        if (i == 0 && ref != nullptr && !label.empty()) {
            report(*ref, label + "_convs.0_Conv", xt, label + " conv 0");
        }
        for (std::size_t j = 0; j < x.data.size(); ++j) {
            x.data[j] += xt.data[j];
        }
    }
    return x;
}

[[nodiscard]] Act linear(const Act& input, const Weight& weight, const Weight* bias) {
    const ProfileScope profile(kProfileLinear);
    // MatMul(x, W) exports (in, out); torch Linear exports the Gemm weight as
    // (out, in). Both layouts occur, so pick by which axis matches the input.
    const std::size_t in = input.channels;
    const bool transposed = weight.size(0) != in;
    const std::size_t out = transposed ? weight.size(0) : weight.size(1);
    if ((transposed ? weight.size(1) : weight.size(0)) != in) {
        throw std::runtime_error("linear shape mismatch");
    }
    // ONNX Runtime prepacks constant MatMul/Gemm weights (MlasGemm with
    // BIsPacked), so K runs in blocks of MLAS_SGEMM_PACKED_STRIDEK = 256 and
    // the partial sums of the blocks are added (sgemm.cpp,
    // MlasSgemmPackedOperation).
    constexpr std::size_t kblock = kMlasGemmPackedStrideK;
    Act result(out, input.length);
    parallel_for(input.length, out * in, [&](std::size_t t) {
        for (std::size_t o = 0; o < out; ++o) {
            const float* row = transposed ? weight.values.data() + o * in : nullptr;
            float total = 0.0F;
            for (std::size_t k0 = 0; k0 < in; k0 += kblock) {
                const std::size_t k1 = std::min(k0 + kblock, in);
                float partial = 0.0F;
                if (transposed) {
                    for (std::size_t i = k0; i < k1; ++i) {
                        partial = std::fma(row[i], input.at(i, t), partial);
                    }
                } else {
                    for (std::size_t i = k0; i < k1; ++i) {
                        partial = std::fma(weight.values[i * out + o], input.at(i, t), partial);
                    }
                }
                total += partial;
            }
            result.at(o, t) = total + (bias != nullptr ? bias->values[o] : 0.0F);
        }
    });
    return result;
}

[[nodiscard]] const Weight& weight_or(const Weights& weights, const std::string& first, const std::string& second) {
    return weights.has(first) ? weights.at(first) : weights.at(second);
}

[[nodiscard]] Act style_act_of(const std::vector<float>& style_values) {
    Act act(style_values.size(), 1);
    for (std::size_t i = 0; i < style_values.size(); ++i) {
        act.at(i, 0) = style_values[i];
    }
    return act;
}

[[nodiscard]] float eigen_redux(const float* data, std::size_t n) {
    constexpr std::size_t kPacket = 4;
    // Eigen's redux starts the packet loop at first_default_aligned(): rows
    // whose byte offset is not 16-byte aligned get a scalar head that is added
    // *after* predux (Eigen's Redux.h, LinearVectorizedTraversal).
    const std::size_t misalign = (reinterpret_cast<std::uintptr_t>(data) & 15U) / sizeof(float);
    std::size_t start = misalign == 0 ? 0 : kPacket - misalign;
    if (start > n) {
        start = n;
    }
    const std::size_t aligned = ((n - start) / kPacket) * kPacket;
    const std::size_t aligned2 = ((n - start) / (2 * kPacket)) * (2 * kPacket);
    float res = 0.0F;
    if (aligned != 0) {
        float r0[kPacket];
        float r1[kPacket];
        for (std::size_t j = 0; j < kPacket; ++j) {
            r0[j] = data[start + j];
        }
        if (aligned > kPacket) {
            for (std::size_t j = 0; j < kPacket; ++j) {
                r1[j] = data[start + kPacket + j];
            }
            std::size_t index = start + 2 * kPacket;
            for (; index < start + aligned2; index += 2 * kPacket) {
                for (std::size_t j = 0; j < kPacket; ++j) {
                    r0[j] += data[index + j];
                    r1[j] += data[index + kPacket + j];
                }
            }
            for (std::size_t j = 0; j < kPacket; ++j) {
                r0[j] += r1[j];
            }
            if (aligned > aligned2) {
                for (std::size_t j = 0; j < kPacket; ++j) {
                    r0[j] += data[start + aligned2 + j];
                }
            }
        }
        res = (r0[0] + r0[2]) + (r0[1] + r0[3]);
        for (std::size_t j = 0; j < start; ++j) {
            res += data[j];
        }
        for (std::size_t j = start + aligned; j < n; ++j) {
            res += data[j];
        }
    } else {
        res = data[0];
        for (std::size_t j = 1; j < n; ++j) {
            res += data[j];
        }
    }
    return res;
}

[[nodiscard]] float reduce_mean_scalar(const float* data, std::size_t n) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum / static_cast<float>(n);
}

[[nodiscard]] Act adain_norm(const Act& x, const Weight& fc_w, const Weight& fc_b, const Act& style_act,
                             bool strided_input) {
    const ProfileScope profile(kProfileNorm);
    // Kokoro exports AdaIN as an explicit decomposition (not
    // InstanceNormalization):
    //   mean = ReduceMean(x); sub = x - mean; var = ReduceMean(sub*sub);
    //   gamma' = hb + 1; out = gamma' * (sub / sqrt(var + eps)) + beta
    // The division and the separate Mul/Add round differently from an
    // affine InstanceNorm.
    const Act hb = linear(style_act, fc_w, &fc_b);
    Act normed(x.channels, x.length);
    // The reference keeps Sub and Sub*Sub as full [C, T] tensors, so row c of
    // the variance reduction starts at c*T floats and inherits that row's
    // 16-byte alignment (Eigen's redux starts its packet loop at the first
    // aligned element). A per-channel scratch row would always look aligned.
    std::vector<float> sub(x.data.size());
    std::vector<float> squared(x.data.size());
    for (std::size_t c = 0; c < x.channels; ++c) {
        const float* row = x.data.data() + c * x.length;
        float* sub_row = sub.data() + c * x.length;
        float* squared_row = squared.data() + c * x.length;
        const float mean = strided_input ? reduce_mean_scalar(row, x.length)
                                         : eigen_redux(row, x.length) / static_cast<float>(x.length);
        for (std::size_t t = 0; t < x.length; ++t) {
            sub_row[t] = row[t] - mean;
            squared_row[t] = sub_row[t] * sub_row[t];
        }
        const float variance = strided_input ? reduce_mean_scalar(squared_row, x.length)
                                             : eigen_redux(squared_row, x.length) / static_cast<float>(x.length);
        const float denom = std::sqrt(variance + 1.0e-5F);
        const float gamma = 1.0F + hb.at(c, 0);
        const float beta = hb.at(x.channels + c, 0);
        for (std::size_t t = 0; t < x.length; ++t) {
            normed.at(c, t) = gamma * (sub_row[t] / denom) + beta;
        }
    }
    return normed;
}

[[nodiscard]] Act nearest_upsample(const Act& x, std::size_t factor) {
    Act out(x.channels, x.length * factor);
    for (std::size_t c = 0; c < x.channels; ++c) {
        for (std::size_t t = 0; t < x.length; ++t) {
            for (std::size_t k = 0; k < factor; ++k) {
                out.at(c, t * factor + k) = x.at(c, t);
            }
        }
    }
    return out;
}

[[nodiscard]] Act depthwise_transpose(const Act& x, const Weight& weight, const Weight& bias, std::size_t stride,
                                      std::size_t pad) {
    const std::size_t channels = x.channels;
    const std::size_t kernel = weight.size(2);
    const std::size_t out_len = (x.length - 1) * stride + kernel + 1 - 2 * pad;
    Act out(channels, out_len);
    for (std::size_t c = 0; c < channels; ++c) {
        const float* taps = weight.values.data() + c * kernel;
        for (std::size_t t = 0; t < x.length; ++t) {
            const float v = x.at(c, t);
            const std::ptrdiff_t base = static_cast<std::ptrdiff_t>(t * stride) - static_cast<std::ptrdiff_t>(pad);
            for (std::size_t k = 0; k < kernel; ++k) {
                const std::ptrdiff_t at = base + static_cast<std::ptrdiff_t>(k);
                if (at >= 0 && at < static_cast<std::ptrdiff_t>(out_len)) {
                    out.at(c, static_cast<std::size_t>(at)) += v * taps[k];
                }
            }
        }
        const float b = bias.values[c];
        for (std::size_t t = 0; t < out_len; ++t) {
            out.at(c, t) += b;
        }
    }
    return out;
}

}  // namespace mnet::tts
