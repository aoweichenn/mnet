// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Minimal generic LLM runner: reads model.json, tokenizes a prompt, runs the
// decoder with KV cache reuse and greedily decodes tokens.

#include <mnet/net.hpp>
#include <mnet/tokenizer.hpp>

#include <algorithm>
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
    if (std::getenv("MNET_LLM_DEBUG") != nullptr) {
        double sum = 0.0;
        double abs_sum = 0.0;
        float maxv = row[0];
        for (std::int32_t i = 0; i < logits.w; ++i) {
            sum += static_cast<double>(row[i]);
            abs_sum += std::fabs(static_cast<double>(row[i]));
            maxv = std::max(maxv, row[i]);
        }
        std::fprintf(stderr, "logits n=%d sum=%.6f abs_sum=%.6f max=%.6f first=%.6f\n", logits.w, sum, abs_sum,
                     static_cast<double>(maxv), static_cast<double>(row[0]));
    }
    for (std::int32_t i = 1; i < logits.w; ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}

}  // namespace

#if !defined(_WIN32)
static std::int64_t read_status_kb(const char* key) {
    std::ifstream status("/proc/self/status");
    std::string name;
    std::int64_t value = 0;
    while (status >> name) {
        if (name == key) {
            status >> value;
            return value;
        }
    }
    return 0;
}
#endif

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt = "Hello";
    std::int32_t threads = 4;
    std::int32_t max_new_tokens = 8;
    bool load_only = false;
    bool fast_int4 = false;
    bool int8_kv = false;
    std::int32_t kv_capacity = 0;
    long long max_mem_mb = 0;
    std::int32_t prefill_chunk = 512;
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
        } else if (std::strcmp(argv[i], "--fast-int4") == 0) {
            fast_int4 = true;
        } else if (std::strcmp(argv[i], "--int8-kv") == 0) {
            int8_kv = true;
        } else if (std::strcmp(argv[i], "--max-mem-mb") == 0 && i + 1 < argc) {
            max_mem_mb = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-capacity") == 0 && i + 1 < argc) {
            kv_capacity = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefill-chunk") == 0 && i + 1 < argc) {
            prefill_chunk = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr,
                     "usage: mnet_llm --model DIR [--prompt TEXT] [--threads N] [--max-new-tokens N] "
                     "[--fast-int4] [--int8-kv] [--kv-capacity N] [--prefill-chunk N] [--max-mem-mb N]\n");
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

    // One process budget, shared by the three sub-graphs: embeddings and the
    // LM head are loaded outside the budgeted allocator, so subtract their file
    // sizes and give the rest to the decoder (where KV cache and activations
    // live). Without this a big --kv-capacity would be OOM-killed on WSL.
    std::size_t mem_budget = 0;
    if (max_mem_mb > 0) {
        mem_budget = static_cast<std::size_t>(max_mem_mb) * 1024U * 1024U;
    } else {
        // Same default as Net (50% of MemTotal on Linux), so the three
        // sub-graphs share one cap instead of each taking the full amount.
        mem_budget = mnet::default_memory_budget_bytes();
    }
    const auto file_bytes = [](const std::string& path) -> std::size_t {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        return f ? static_cast<std::size_t>(f.tellg()) : 0;
    };

    mnet::Net embed_net;
    embed_net.opt().num_threads = threads;
    if (embed_net.load((model_dir + "/" + cfg.embed_param).c_str(), (model_dir + "/" + cfg.embed_bin).c_str()) != 0) {
        std::fprintf(stderr, "embed load failed\n");
        return 1;
    }
    mnet::Net decoder_net;
    decoder_net.opt().num_threads = threads;
    decoder_net.opt().use_fast_int4_gemm = fast_int4;
    decoder_net.opt().use_int8_kv_cache = int8_kv;
    decoder_net.opt().kvcache_max_seqlen_hint = kv_capacity;
    if (decoder_net.load((model_dir + "/" + cfg.decoder_param).c_str(),
                         (model_dir + "/" + cfg.decoder_bin).c_str()) != 0) {
        std::fprintf(stderr, "decoder load failed\n");
        return 1;
    }
    if (mem_budget != 0) {
        const std::size_t embed_bytes = file_bytes(model_dir + "/" + cfg.embed_bin);
        const std::size_t proj_bytes = file_bytes(model_dir + "/" + cfg.proj_bin);
        const std::size_t others = embed_bytes + proj_bytes;
        const std::size_t decoder_budget = mem_budget > others ? mem_budget - others : 1;
        decoder_net.set_memory_budget_bytes(decoder_budget);
        std::printf("mem_budget_mb=%zu decoder_budget_mb=%zu\n", mem_budget / (1024U * 1024U),
                    decoder_budget / (1024U * 1024U));
    }

    mnet::Net proj_net;
    proj_net.opt().num_threads = threads;
    proj_net.opt().use_fast_int4_gemm = fast_int4;
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
                if (decoder_net.memory_budget_exhausted()) {
                    std::fprintf(stderr, "mnet: decoder memory budget exhausted (budget=%zu MiB, peak=%zu MiB)\n",
                                 decoder_net.memory_budget_bytes() / (1024U * 1024U),
                                 decoder_net.memory_peak_bytes() / (1024U * 1024U));
                }
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

    const std::int32_t rope_half = cfg.rope_head_dim / 2;
    if (prefill_chunk < 1) {
        prefill_chunk = prompt_len;
    }

    std::fprintf(stderr, "rss_after_load_kb=%" PRId64 "\n", read_vmhwm_kb());
    if (load_only) {
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();
    // Chunked prefill: the attention mask is (dst x src) for one chunk only, so
    // its size stays O(chunk * seqlen) instead of O(seqlen^2). Every chunk
    // appends to the KV cache, exactly like a decode step.
    mnet::Mat hidden;
    std::int32_t last_chunk_len = 0;
    for (std::int32_t done = 0; done < prompt_len;) {
        const std::int32_t n = std::min(prefill_chunk, prompt_len - done);
        const std::int32_t dst = done + n;

        mnet::Mat mask(dst, n);
        for (std::int32_t i = 0; i < n; ++i) {
            float* row = mask.row(i);
            const std::int32_t visible = done + i + 1;
            for (std::int32_t j = 0; j < dst; ++j) {
                row[j] = j < visible ? 0.0F : -1e38F;
            }
        }

        mnet::Mat cos_cache;
        mnet::Mat sin_cache;
        make_rope(n, done, rope_half, cfg.rope_theta, cos_cache, sin_cache);

        hidden = run_decoder(embedding.row_range(done, n), mask, cos_cache, sin_cache);
        if (hidden.empty()) {
            std::fprintf(stderr, "prefill failed at %d/%d\n", done, prompt_len);
            return 1;
        }
        last_chunk_len = n;
        done += n;
    }
    std::int32_t token = argmax(proj_net, hidden.row_range(last_chunk_len - 1, 1));
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
#if !defined(_WIN32)
    const double peak_rss_mb = static_cast<double>(read_status_kb("VmHWM:")) / 1024.0;
    const double peak_vm_mb = static_cast<double>(read_status_kb("VmPeak:")) / 1024.0;
#else
    const double peak_rss_mb = 0.0;
    const double peak_vm_mb = 0.0;
#endif
    std::printf("model=%s threads=%d prompt_tokens=%d tokens=%zu prefill=%.1fms decode=%.2fms/tok peak_rss_mb=%.1f "
                "peak_vm_mb=%.1f mem_peak_mb=%.1f mem_budget_mb=%zu budget_hit=%d text=%s\n",
                model_dir.c_str(), threads, prompt_len, generated.size(), prefill_ms,
                decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0, peak_rss_mb, peak_vm_mb,
                static_cast<double>(decoder_net.memory_peak_bytes()) / (1024.0 * 1024.0),
                decoder_net.memory_budget_bytes() / (1024U * 1024U), decoder_net.memory_budget_exhausted() ? 1 : 0,
                tokenizer.decode(generated).c_str());
    return 0;
}
