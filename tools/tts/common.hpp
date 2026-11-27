#pragma once

// Shared declarations for the pure-mnet Piper and Kokoro runners: the command
// line options, the phoneme-id parser, and the two model entry points.

#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>

#include <mnet/tts/tts_ops.hpp>

// The exact-arithmetic kernels live in <mnet/tts/tts_exact.hpp>; pull the ones the
// model code calls into this translation unit's anonymous namespace.
using mnet::tts::eigen_atan;
using mnet::tts::eigen_cos;
using mnet::tts::eigen_cos_element;
using mnet::tts::eigen_exp;
using mnet::tts::eigen_fast_tanh;
using mnet::tts::eigen_frexp;
using mnet::tts::eigen_log1p;
using mnet::tts::eigen_log_positive;
using mnet::tts::eigen_scalar_tail;
using mnet::tts::eigen_sin;
using mnet::tts::eigen_sin_element;
using mnet::tts::eigen_softplus;
using mnet::tts::eigen_softplus_element;
using mnet::tts::eigen_trig_reduce_huge;
using mnet::tts::EigenFrexp;
using mnet::tts::mlas_conv_ranges;
using mnet::tts::mlas_erf;
using mnet::tts::mlas_exp;
using mnet::tts::mlas_gemm_kblock;
using mnet::tts::mlas_softmax_row;
using mnet::tts::mlas_sumexp_term;
using mnet::tts::ort_logistic;
using mnet::tts::ort_tanh;

// Operator layer and shared tensor/reference helpers (<mnet/tts/tts_ops.hpp>).
using mnet::tts::Act;
using mnet::tts::adain_norm;
using mnet::tts::apply_mask;
using mnet::tts::conv1d;
using mnet::tts::conv_transpose1d;
using mnet::tts::dds_conv;
using mnet::tts::debug_dump;
using mnet::tts::depthwise_transpose;
using mnet::tts::eigen_redux;
using mnet::tts::g_debug_dir;
using mnet::tts::g_mem_budget_bytes;
using mnet::tts::g_ort_threads;
using mnet::tts::g_peak_bytes;
using mnet::tts::g_profile;
using mnet::tts::g_profile_calls;
using mnet::tts::g_profile_ns;
using mnet::tts::gelu;
using mnet::tts::kLayerNormEps;
using mnet::tts::kMlasGemmPackedStrideK;
using mnet::tts::kProfileHarmonic;
using mnet::tts::kProfileIstft;
using mnet::tts::kProfileLstm;
using mnet::tts::kProfileNorm;
using mnet::tts::layer_norm;
using mnet::tts::layer_norm_unit;
using mnet::tts::leaky_relu;
using mnet::tts::linear;
using mnet::tts::live_bytes;
using mnet::tts::nearest_upsample;
using mnet::tts::piper_norm;
using mnet::tts::ProfileScope;
using mnet::tts::reduce_mean_scalar;
using mnet::tts::Reference;
using mnet::tts::relu_in_place;
using mnet::tts::report;
using mnet::tts::report_channels;
using mnet::tts::res_block2;
using mnet::tts::style_act_of;
using mnet::tts::wave_net;
using mnet::tts::Weight;
using mnet::tts::weight_or;
using mnet::tts::Weights;
using mnet::tts::write_wav;

namespace mnet_tts {

// Command line options shared by both models (see tools/tts/mnet_tts.cpp).
struct TtsOptions {
    std::string model_dir;
    std::string ref_dir;
    std::string dump_dir;
    std::string model_type = "piper";
    std::string ids_arg;
    std::string style_path;
    std::string wav_out;
    std::string stage = "encoder";
    bool ref_har = false;
    bool ref_f0 = false;
    float speed = 1.0F;
    float noise_scale = 0.667F;  // VITS noise_scale (flow / z_p sampling)
    float noise_w = 0.0F;        // VITS noise_w (duration predictor)
    std::uint32_t seed = 0;
};

// Comma-separated phoneme ids ("43,60,66"); both models accept them.
[[nodiscard]] inline std::vector<std::int64_t> parse_id_list(const std::string& list) {
    std::vector<std::int64_t> ids;
    std::size_t start = 0;
    while (start <= list.size()) {
        const std::size_t next = list.find(',', start);
        const std::string part = list.substr(start, next == std::string::npos ? std::string::npos : next - start);
        if (!part.empty()) {
            ids.push_back(std::stoll(part));
        }
        if (next == std::string::npos) {
            break;
        }
        start = next + 1;
    }
    return ids;
}

// Piper/VITS path (tools/tts/mnet_tts_piper.cpp).
[[nodiscard]] std::int32_t run_piper(const TtsOptions& opt, const mnet::tts::Weights& weights);

// Kokoro/StyleTTS2 path (tools/tts/mnet_tts_kokoro.cpp).
[[nodiscard]] std::int32_t run_kokoro(const TtsOptions& opt, const mnet::tts::Weights& weights);

}  // namespace mnet_tts
