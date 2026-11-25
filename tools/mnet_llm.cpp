// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Minimal generic LLM runner: reads model.json, tokenizes a prompt, runs the
// decoder with KV cache reuse and greedily decodes tokens.

#include <mnet/net.hpp>
#include <mnet/tokenizer.hpp>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ModelConfig {
    std::string dir;
    std::string embed_param;
    std::string embed_bin;
    std::string decoder_param;
    std::string decoder_bin;
    std::string proj_param;
    std::string proj_bin;
    std::string vocab_file = "vocab.txt";
    std::string merges_file = "merges.txt";
    std::int32_t attn_cnt = 0;
    std::int32_t rope_head_dim = 0;
    float rope_theta = 10000.0F;
};

std::int64_t read_vmhwm_kb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    std::int64_t value = 0;
    while (status >> key) {
        if (key == "VmHWM:") {
            status >> value;
            return value;
        }
    }
    return 0;
}

std::string read_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

bool find_string(const std::string& text, const std::string& key, std::string& out) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(text, match, re)) {
        out = match[1].str();
        return true;
    }
    return false;
}

bool find_int(const std::string& text, const std::string& key, std::int32_t& out) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(text, match, re)) {
        out = std::atoi(match[1].str().c_str());
        return true;
    }
    return false;
}

bool find_float(const std::string& text, const std::string& key, float& out) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9.eE+]+)");
    std::smatch match;
    if (std::regex_search(text, match, re)) {
        out = std::strtof(match[1].str().c_str(), nullptr);
        return true;
    }
    return false;
}

bool load_config(const std::string& dir, ModelConfig& cfg) {
    cfg.dir = dir;
    const std::string text = read_text(dir + "/model.json");
    if (text.empty()) {
        std::fprintf(stderr, "cannot read %s/model.json\n", dir.c_str());
        return false;
    }
    if (!find_string(text, "embed_token_param", cfg.embed_param) ||
        !find_string(text, "decoder_param", cfg.decoder_param)) {
        std::fprintf(stderr, "model.json missing embed/decoder params\n");
        return false;
    }
    cfg.embed_bin = cfg.embed_param;
    cfg.decoder_bin = cfg.decoder_param;
    cfg.proj_param = cfg.embed_param;
    cfg.proj_bin = cfg.embed_bin;
    find_string(text, "embed_token_bin", cfg.embed_bin);
    find_string(text, "decoder_bin", cfg.decoder_bin);
    find_string(text, "proj_out_param", cfg.proj_param);
    find_string(text, "proj_out_bin", cfg.proj_bin);
    find_string(text, "vocab_file", cfg.vocab_file);
    find_string(text, "merges_file", cfg.merges_file);
    find_int(text, "attn_cnt", cfg.attn_cnt);
    find_int(text, "rope_head_dim", cfg.rope_head_dim);
    find_float(text, "rope_theta", cfg.rope_theta);
    if (cfg.attn_cnt <= 0 || cfg.rope_head_dim <= 0) {
        std::fprintf(stderr, "model.json missing attention/rope settings\n");
        return false;
    }
    return true;
}

void make_rope(std::int32_t seqlen, std::int32_t pos0, std::int32_t rope_half, float theta, mnet::Mat& cos_cache,
               mnet::Mat& sin_cache) {
    cos_cache.create(rope_half, seqlen);
    sin_cache.create(rope_half, seqlen);
    for (std::int32_t i = 0; i < seqlen; ++i) {
        float* cptr = cos_cache.row(i);
        float* sptr = sin_cache.row(i);
        for (std::int32_t j = 0; j < rope_half; ++j) {
            const float inv = 1.0F / std::pow(theta, static_cast<float>(j * 2) / static_cast<float>(rope_half * 2));
            const float t = inv * static_cast<float>(pos0 + i);
            cptr[j] = std::cos(t);
            sptr[j] = std::sin(t);
        }
    }
}

std::int32_t argmax(const mnet::Net& proj_net, const mnet::Mat& hidden_row) {
    mnet::Net::Extractor ex = proj_net.create_extractor();
    if (ex.input("in0", hidden_row) != 0) {
        return -1;
    }
    mnet::Mat logits;
    if (ex.extract("out0", logits) != 0) {
        return -1;
    }
    const float* row = logits.dims == 1 ? static_cast<const float*>(logits) : logits.row(0);
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < logits.w; ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt = "Hello";
    std::int32_t threads = 4;
    std::int32_t max_new_tokens = 8;
    bool load_only = false;
    for (std::int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--load-only") == 0) {
            load_only = true;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr,
                     "usage: mnet_llm --model DIR [--prompt TEXT] [--threads N] [--max-new-tokens N]\n");
        return 2;
    }

    ModelConfig cfg;
    if (!load_config(model_dir, cfg)) {
        return 1;
    }

    mnet::Tokenizer tokenizer;
    if (tokenizer.load((model_dir + "/" + cfg.vocab_file).c_str(), (model_dir + "/" + cfg.merges_file).c_str()) != 0) {
        std::fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }

    mnet::Net embed_net;
    embed_net.opt().num_threads = threads;
    if (embed_net.load((model_dir + "/" + cfg.embed_param).c_str(), (model_dir + "/" + cfg.embed_bin).c_str()) != 0) {
        std::fprintf(stderr, "embed load failed\n");
        return 1;
    }
    mnet::Net decoder_net;
    decoder_net.opt().num_threads = threads;
    if (decoder_net.load((model_dir + "/" + cfg.decoder_param).c_str(),
                         (model_dir + "/" + cfg.decoder_bin).c_str()) != 0) {
        std::fprintf(stderr, "decoder load failed\n");
        return 1;
    }
    mnet::Net proj_net;
    proj_net.opt().num_threads = threads;
    if (proj_net.load((model_dir + "/" + cfg.proj_param).c_str(), (model_dir + "/" + cfg.proj_bin).c_str()) != 0) {
        std::fprintf(stderr, "proj load failed\n");
        return 1;
    }

    const std::vector<std::int32_t> prompt_ids = tokenizer.encode(prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "tokenizer produced no ids\n");
        return 1;
    }
    const std::int32_t prompt_len = static_cast<std::int32_t>(prompt_ids.size());

    // Every decoder input except the four activations/position tensors is a
    // recurrent state (KV cache, conv state, gated-delta-rule state). The
    // matching output is named "out_" + input name.
    std::vector<std::string> cache_names;
    for (const std::string& name : decoder_net.input_names()) {
        if (name != "in0" && name != "in1" && name != "in2" && name != "in3") {
            cache_names.push_back(name);
        }
    }
    std::vector<mnet::Mat> caches(cache_names.size());

    auto run_decoder = [&](const mnet::Mat& embedding, const mnet::Mat& mask, const mnet::Mat& cos_cache,
                           const mnet::Mat& sin_cache) -> mnet::Mat {
        mnet::Net::Extractor ex = decoder_net.create_extractor();
        ex.input("in0", embedding);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);
        for (std::size_t i = 0; i < cache_names.size(); ++i) {
            if (!caches[i].empty()) {
                ex.input(cache_names[i].c_str(), caches[i]);
            }
        }
        mnet::Mat hidden_out;
        if (ex.extract("out0", hidden_out) != 0) {
            std::fprintf(stderr, "decoder extract out0 failed\n");
        }
        for (std::size_t i = 0; i < cache_names.size(); ++i) {
            const std::string out_name = "out_" + cache_names[i];
            if (ex.extract(out_name.c_str(), caches[i]) != 0) {
                std::fprintf(stderr, "decoder cache %s extract failed\n", out_name.c_str());
            }
        }
        return hidden_out;
    };

    mnet::Mat ids(static_cast<std::int32_t>(prompt_ids.size()), 1);
    std::memcpy(ids.data, prompt_ids.data(), prompt_ids.size() * sizeof(std::int32_t));
    mnet::Net::Extractor embed_ex = embed_net.create_extractor();
    if (embed_ex.input("in0", ids) != 0) {
        std::fprintf(stderr, "embed input failed\n");
        return 1;
    }
    mnet::Mat embedding;
    if (embed_ex.extract("out0", embedding) != 0) {
        std::fprintf(stderr, "embed extract failed\n");
        return 1;
    }

    mnet::Mat mask(prompt_len, prompt_len);
    for (std::int32_t i = 0; i < prompt_len; ++i) {
        float* row = mask.row(i);
        for (std::int32_t j = 0; j < prompt_len; ++j) {
            row[j] = j <= i ? 0.0F : -1e38F;
        }
    }

    const std::int32_t rope_half = cfg.rope_head_dim / 2;
    mnet::Mat cos_cache;
    mnet::Mat sin_cache;
    make_rope(prompt_len, 0, rope_half, cfg.rope_theta, cos_cache, sin_cache);

    std::fprintf(stderr, "rss_after_load_kb=%" PRId64 "\n", read_vmhwm_kb());
    if (load_only) {
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();
    mnet::Mat hidden = run_decoder(embedding, mask, cos_cache, sin_cache);
    if (hidden.empty()) {
        std::fprintf(stderr, "prefill failed\n");
        return 1;
    }
    std::int32_t token = argmax(proj_net, hidden.row_range(prompt_len - 1, 1));
    const auto t1 = std::chrono::steady_clock::now();
    if (token < 0) {
        std::fprintf(stderr, "proj argmax failed\n");
        return 1;
    }

    std::vector<std::int32_t> generated = {token};
    std::int32_t position = prompt_len;
    for (std::int32_t step = 0; step < max_new_tokens; ++step) {
        mnet::Mat one_id(1, 1);
        *reinterpret_cast<std::int32_t*>(static_cast<void*>(one_id.data)) = token;
        mnet::Net::Extractor eex = embed_net.create_extractor();
        eex.input("in0", one_id);
        mnet::Mat one_embed;
        if (eex.extract("out0", one_embed) != 0) {
            break;
        }

        mnet::Mat decode_mask(position + 1, 1);
        decode_mask.fill(0.0F);
        mnet::Mat dcos;
        mnet::Mat dsin;
        make_rope(1, position, rope_half, cfg.rope_theta, dcos, dsin);

        mnet::Mat one_hidden = run_decoder(one_embed, decode_mask, dcos, dsin);
        if (one_hidden.empty()) {
            break;
        }
        token = argmax(proj_net, one_hidden);
        if (token < 0) {
            break;
        }
        generated.push_back(token);
        ++position;
    }
    const auto t2 = std::chrono::steady_clock::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const std::size_t decode_steps = generated.size() - 1;
    std::printf("model=%s threads=%d tokens=%zu prefill=%.1fms decode=%.2fms/tok text=%s\n", model_dir.c_str(),
                threads, generated.size(), prefill_ms,
                decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0,
                tokenizer.decode(generated).c_str());
    return 0;
}
