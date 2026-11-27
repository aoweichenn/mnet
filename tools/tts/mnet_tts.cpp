// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Pure mnet Piper/VITS text-to-speech: loads the flattened parameter blob that
// tools/export_piper_weights.py writes and reproduces the ONNX reference stage
// by stage (text encoder -> duration predictor -> flow -> HiFiGAN decoder).
//
//   mnet_tts --model DIR [--ref DIR] [--stage encoder|all] [--dump DIR]
//
// Reference dumps come from tools/dump_piper_reference.py (deterministic noise).

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <new>
#include <string>

#include <mnet/tts/tts_ops.hpp>
#include <tts/common.hpp>

namespace {

[[nodiscard]] static std::int32_t run_tts(std::int32_t argc, char** argv) {
    mnet_tts::TtsOptions opt;
    for (std::int32_t i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            opt.model_dir = argv[++i];
        } else if (arg == "--ref" && i + 1 < argc) {
            opt.ref_dir = argv[++i];
        } else if (arg == "--stage" && i + 1 < argc) {
            opt.stage = argv[++i];
        } else if (arg == "--dump" && i + 1 < argc) {
            opt.dump_dir = argv[++i];
            mnet::tts::g_debug_dir = opt.dump_dir;
            std::error_code ec;
            std::filesystem::create_directories(opt.dump_dir, ec);
        } else if (arg == "--noise-scale" && i + 1 < argc) {
            opt.noise_scale = std::strtof(argv[++i], nullptr);
        } else if (arg == "--noise-w" && i + 1 < argc) {
            opt.noise_w = std::strtof(argv[++i], nullptr);
        } else if (arg == "--model-type" && i + 1 < argc) {
            opt.model_type = argv[++i];
        } else if (arg == "--profile") {
            mnet::tts::g_profile = true;
        } else if (arg == "--ort-threads" && i + 1 < argc) {
            mnet::tts::g_ort_threads = std::max<std::size_t>(1, std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--ids" && i + 1 < argc) {
            opt.ids_arg = argv[++i];
        } else if (arg == "--style" && i + 1 < argc) {
            opt.style_path = argv[++i];
        } else if (arg == "--speed" && i + 1 < argc) {
            opt.speed = std::strtof(argv[++i], nullptr);
        } else if (arg == "--ref-f0") {
            opt.ref_f0 = true;
        } else if (arg == "--ref-har") {
            opt.ref_har = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            opt.seed = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--wav-out" && i + 1 < argc) {
            opt.wav_out = argv[++i];
        } else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 2;
        }
    }
    if (mnet::tts::g_profile) {
        std::atexit([] {
            static const char* kNames[] = {"conv",       "conv_transpose", "linear", "lstm",
                                           "norm/adain", "attention",      "istft",  "harmonic"};
            std::printf("\nprofile (kernel totals; peak tracked footprint %.1f MiB)\n",
                        static_cast<double>(mnet::tts::g_peak_bytes.load()) / (1024.0 * 1024.0));
            for (std::size_t i = 0; i < 8; ++i) {
                if (mnet::tts::g_profile_calls[i] == 0) {
                    continue;
                }
                std::printf("  %-15s %8.3f s  %6" PRIu64 " calls  %7.3f ms/call\n", kNames[i],
                            static_cast<double>(mnet::tts::g_profile_ns[i]) * 1.0e-9,
                            static_cast<std::uint64_t>(mnet::tts::g_profile_calls[i]),
                            static_cast<double>(mnet::tts::g_profile_ns[i]) * 1.0e-6 /
                                static_cast<double>(mnet::tts::g_profile_calls[i]));
            }
        });
    }
    if (opt.model_dir.empty()) {
        std::fprintf(stderr,
                     "usage: mnet_tts --model DIR [--ref DIR] [--stage encoder|all] [--dump DIR] "
                     "[--noise-scale F] [--noise-w F] [--seed N] [--wav-out FILE] [--ref-har] [--ref-f0] "
                     "[--ort-threads N] [--profile]\n");
        return 2;
    }

    mnet::tts::Weights weights;
    const std::string blob = opt.model_type == "kokoro" ? "kokoro_weights" : "piper_weights";
    if (!weights.load(opt.model_dir + "/" + blob + ".txt", opt.model_dir + "/" + blob + ".bin")) {
        return 1;
    }
    if (opt.model_type == "kokoro") {
        return mnet_tts::run_kokoro(opt, weights);
    }
    return mnet_tts::run_piper(opt, weights);
}

}  // namespace

int main(std::int32_t argc, char** argv) {
    // MNET_MAX_MEM_MB budgets the parameter blob plus the live activations;
    // an over-budget run fails cleanly instead of thrashing.
    if (const char* env = std::getenv("MNET_MAX_MEM_MB")) {
        const std::uint64_t megabytes = std::strtoul(env, nullptr, 10);
        if (megabytes > 0) {
            mnet::tts::g_mem_budget_bytes = static_cast<std::size_t>(megabytes) * 1024ULL * 1024ULL;
        }
    }
    try {
        return run_tts(argc, argv);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "mnet_tts: out of memory (MNET_MAX_MEM_MB=%s)\n",
                     std::getenv("MNET_MAX_MEM_MB") != nullptr ? std::getenv("MNET_MAX_MEM_MB") : "unset");
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mnet_tts: %s\n", error.what());
        return 1;
    }
}
