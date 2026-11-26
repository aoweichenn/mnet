// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/net.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mformat/data_reader.hpp>
#include <mformat/model_bin.hpp>
#include <mgraph/executor.hpp>
#include <mgraph/extractor.hpp>
#include <mgraph/graph.hpp>
#include <mmemory/allocator.hpp>
#include <mnet/detail/txn_log.hpp>
#include <mnet/layer_registry.hpp>

namespace mnet {

struct NetPrivate final : public mgraph::ExecutionBackend<Layer, Mat, Option> {
    std::vector<Layer*> layers;
    std::vector<mgraph::Blob> blobs;
    std::vector<std::string> layer_types;
    std::vector<std::int32_t> blob_last_use;
    std::vector<std::int32_t> input_blobs;
    std::vector<std::int32_t> output_blobs;
    std::unordered_map<const Layer*, std::int32_t> layer_index;
    mutable std::int32_t profile_index = 0;
    mmemory::UnlockedPoolAllocator blob_allocator;
    mmemory::UnlockedPoolAllocator workspace_allocator;
    mmemory::BudgetAllocator budget_blob{this->blob_allocator, 0};
    mmemory::BudgetAllocator budget_workspace{this->workspace_allocator, 0};
    // Total budget and the bytes already taken by weights.
    std::size_t budget_total = 0;
    std::size_t weight_bytes = 0;

    // Effective budget for activations: total minus weights.
    [[nodiscard]] std::size_t activation_budget() const {
        if (this->budget_total == 0) {
            return 0;
        }
        return this->weight_bytes < this->budget_total ? this->budget_total - this->weight_bytes : 1;
    }

    mutable bool budget_reported = false;

    // One-line, once-per-model explanation when an allocation was refused.
    void report_budget() const {
        if (this->budget_reported) {
            return;
        }
        if (!this->budget_blob.exhausted() && !this->budget_workspace.exhausted()) {
            return;
        }
        this->budget_reported = true;
        detail::write_diagnostic(
            "mnet: memory budget exhausted; lower the context length/--kv-capacity or raise MNET_MAX_MEM_MB\n");
    }

    void apply_budget() {
        const std::size_t budget = this->activation_budget();
        this->budget_blob.set_budget(budget);
        this->budget_workspace.set_budget(budget);
    }

    ~NetPrivate() {
        for (Layer* layer : this->layers) {
            delete layer;
        }
        this->layers.clear();
    }

    [[nodiscard]] bool is_input_layer(const Layer& layer) const override {
        if (!layer.bottoms.empty()) {
            return false;
        }
        // Only Input produces a blob the user feeds. MemoryData and other
        // zero-bottom layers still have to run to materialise their weights.
        if (layer.typeindex == 16) {
            return true;
        }
        const auto it = this->layer_index.find(&layer);
        if (it == this->layer_index.end()) {
            return true;
        }
        const auto index = static_cast<std::size_t>(it->second);
        return index < this->layer_types.size() && this->layer_types[index] == "Input";
    }

    [[nodiscard]] Option mask_option(const Option& opt, std::int32_t) const override { return opt; }

    [[nodiscard]] bool should_release_bottom(const Mat&, const Option&) const override { return false; }

    std::int32_t invoke(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const override {
        const auto t0 = std::chrono::steady_clock::now();
        const std::int32_t result = this->invoke_impl(layer, blob_mats, opt);
        if (result == 0 && opt.use_light_mode) {
            const auto it = this->layer_index.find(layer);
            if (it != this->layer_index.end()) {
                this->release_dead_bottoms(it->second, blob_mats);
            }
        }
        if (std::getenv("MNET_PROFILE") != nullptr) {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const auto index_it = this->layer_index.find(layer);
            const std::int32_t index = index_it != this->layer_index.end() ? index_it->second : -1;
            const char* type = index >= 0 && static_cast<std::size_t>(index) < this->layer_types.size()
                                   ? this->layer_types[static_cast<std::size_t>(index)].c_str()
                                   : "?";
            const std::int32_t top_index = layer->tops.empty() ? -1 : layer->tops[0];
            const Mat* top = top_index >= 0 && static_cast<std::size_t>(top_index) < blob_mats.size()
                                 ? &blob_mats[static_cast<std::size_t>(top_index)]
                                 : nullptr;
            char line[160];
            if (top != nullptr) {
                std::snprintf(line, sizeof(line), "PROFILE %d %s %.3f %dx%dx%dx%d\n", index, type, ms, top->w, top->h,
                              top->d, top->c);
            } else {
                std::snprintf(line, sizeof(line), "PROFILE %d %s %.3f -\n", index, type, ms);
            }
            detail::write_diagnostic(line);
        }
        return result;
    }

    void release_dead_bottoms(std::int32_t index, std::vector<Mat>& blob_mats) const {
        if (index < 0 || static_cast<std::size_t>(index) >= this->layers.size()) {
            return;
        }
        const Layer* layer = this->layers[static_cast<std::size_t>(index)];
        for (const std::int32_t bottom : layer->bottoms) {
            if (bottom < 0 || static_cast<std::size_t>(bottom) >= blob_mats.size()) {
                continue;
            }
            bool is_top = false;
            for (const std::int32_t top : layer->tops) {
                if (top == bottom) {
                    is_top = true;
                }
            }
            if (is_top) {
                continue;
            }
            if (static_cast<std::size_t>(bottom) < this->blob_last_use.size() &&
                this->blob_last_use[static_cast<std::size_t>(bottom)] == index) {
                blob_mats[static_cast<std::size_t>(bottom)].release();
            }
        }
    }

    std::int32_t invoke_impl(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const {
        if (layer->one_blob_only) {
            if (layer->bottoms.size() != 1 || layer->tops.size() != 1) {
                return -1;
            }
            const auto bottom_index = static_cast<std::size_t>(layer->bottoms[0]);
            const auto top_index = static_cast<std::size_t>(layer->tops[0]);
            if (bottom_index >= blob_mats.size() || top_index >= blob_mats.size()) {
                return -1;
            }
            const Mat& bottom = blob_mats[bottom_index];
            Mat& top = blob_mats[top_index];
            if (layer->support_inplace) {
                const bool exclusive = bottom.refcount != nullptr && *bottom.refcount == 1;
                if (top_index != bottom_index) {
                    if (exclusive) {
                        top = bottom;
                    } else {
                        top = bottom.clone(opt.blob_allocator);
                        if (top.empty()) {
                            return -100;
                        }
                    }
                }
                const std::int32_t ret = layer->forward_inplace(blob_mats[top_index], opt);
                if (ret != 0) {
                    return ret;
                }
                return 0;
            }
            return layer->forward(bottom, top, opt);
        }

        std::vector<Mat> bottoms;
        bottoms.reserve(layer->bottoms.size());
        for (std::int32_t index : layer->bottoms) {
            if (index < 0 || static_cast<std::size_t>(index) >= blob_mats.size()) {
                return -1;
            }
            bottoms.push_back(blob_mats[static_cast<std::size_t>(index)]);
        }
        std::vector<Mat> tops(layer->tops.size());
        const std::int32_t ret = layer->forward(bottoms, tops, opt);
        if (ret != 0) {
            return ret;
        }
        for (std::size_t i = 0; i < layer->tops.size(); ++i) {
            const auto top_index = static_cast<std::size_t>(layer->tops[i]);
            if (top_index >= blob_mats.size()) {
                return -1;
            }
            blob_mats[top_index] = tops[i];
        }
        return 0;
    }

    // Graph-level optimization: absorb activation layers that directly follow a
    // convolution into the convolution epilogue. This removes one full
    // read+write pass over every feature map and the intermediate blob. The
    // pass is conservative: the convolution output must have exactly one
    // consumer and the activation output must not be a requested model output.
    void fuse_convolution_activation() {
        const auto blob_count = static_cast<std::int32_t>(this->blobs.size());
        std::vector<std::vector<std::int32_t>> consumers(static_cast<std::size_t>(blob_count));
        for (std::size_t i = 0; i < this->layers.size(); ++i) {
            for (const std::int32_t bottom : this->layers[i]->bottoms) {
                if (bottom >= 0 && bottom < blob_count) {
                    consumers[static_cast<std::size_t>(bottom)].push_back(static_cast<std::int32_t>(i));
                }
            }
        }

        std::vector<bool> removed(this->layers.size(), false);
        std::size_t fused = 0;
        for (std::size_t i = 0; i < this->layers.size(); ++i) {
            Layer* act = this->layers[i];
            if (act->bottoms.size() != 1 || act->tops.size() != 1) {
                continue;
            }
            std::int32_t type = 0;
            Mat params;
            if (act->get_fused_activation(type, params) != 0) {
                continue;
            }
            const std::int32_t bottom = act->bottoms[0];
            const std::int32_t top = act->tops[0];
            if (bottom < 0 || top < 0 || bottom >= blob_count || top >= blob_count) {
                continue;
            }
            bool is_output = false;
            for (const std::int32_t output : this->output_blobs) {
                if (output == top) {
                    is_output = true;
                }
            }
            if (is_output) {
                continue;
            }
            const std::int32_t producer = this->blobs[static_cast<std::size_t>(bottom)].producer;
            if (producer < 0 || static_cast<std::size_t>(producer) >= this->layers.size()) {
                continue;
            }
            Layer* conv = this->layers[static_cast<std::size_t>(producer)];
            if (conv->tops.size() != 1 || conv->tops[0] != bottom) {
                continue;
            }
            if (!conv->can_fuse_activation(type)) {
                continue;
            }
            if (consumers[static_cast<std::size_t>(bottom)].size() != 1) {
                continue;
            }

            conv->set_fused_activation(type, params);
            for (const std::int32_t consumer : consumers[static_cast<std::size_t>(top)]) {
                for (std::int32_t& consumer_bottom : this->layers[static_cast<std::size_t>(consumer)]->bottoms) {
                    if (consumer_bottom == top) {
                        consumer_bottom = bottom;
                    }
                }
            }
            removed[i] = true;
            ++fused;
        }

        if (fused == 0) {
            return;
        }

        std::vector<Layer*> kept_layers;
        std::vector<std::string> kept_types;
        kept_layers.reserve(this->layers.size() - fused);
        kept_types.reserve(this->layers.size() - fused);
        std::vector<std::int32_t> remap(this->layers.size(), -1);
        this->layer_index.clear();
        for (std::size_t i = 0; i < this->layers.size(); ++i) {
            if (removed[i]) {
                delete this->layers[i];
                continue;
            }
            remap[i] = static_cast<std::int32_t>(kept_layers.size());
            this->layer_index[this->layers[i]] = static_cast<std::int32_t>(kept_layers.size());
            kept_layers.push_back(this->layers[i]);
            kept_types.push_back(i < this->layer_types.size() ? this->layer_types[i] : std::string());
        }
        this->layers = std::move(kept_layers);
        this->layer_types = std::move(kept_types);

        // Blob producer/consumer indices refer to layer positions, so they have
        // to follow the compaction. Blobs produced by a removed activation are
        // no longer produced at all.
        for (mgraph::Blob& blob : this->blobs) {
            blob.producer = blob.producer >= 0 && static_cast<std::size_t>(blob.producer) < remap.size()
                                ? remap[static_cast<std::size_t>(blob.producer)]
                                : -1;
            blob.consumer = blob.consumer >= 0 && static_cast<std::size_t>(blob.consumer) < remap.size()
                                ? remap[static_cast<std::size_t>(blob.consumer)]
                                : -1;
        }

        this->blob_last_use.assign(this->blobs.size(), -1);
        for (std::size_t i = 0; i < this->layers.size(); ++i) {
            for (const std::int32_t bottom : this->layers[i]->bottoms) {
                if (bottom >= 0 && static_cast<std::size_t>(bottom) < this->blob_last_use.size()) {
                    this->blob_last_use[static_cast<std::size_t>(bottom)] = static_cast<std::int32_t>(i);
                }
            }
        }
    }

    [[nodiscard]] std::int32_t find_blob(const char* name) const {
        if (name == nullptr) {
            return -1;
        }
        for (std::size_t i = 0; i < this->blobs.size(); ++i) {
            if (this->blobs[i].name == name) {
                return static_cast<std::int32_t>(i);
            }
        }
        return -1;
    }
};

std::size_t default_memory_budget_bytes() {
    if (const char* env = std::getenv("MNET_MAX_MEM_MB")) {
        const long long mb = std::atoll(env);
        return mb > 0 ? static_cast<std::size_t>(mb) * 1024U * 1024U : 0;
    }
#if defined(__linux__)
    // /proc/meminfo: MemTotal in kB. Half of it leaves room for the weight
    // buffers (allocated outside the budget) and the rest of the session.
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file != nullptr) {
        char line[256];
        while (std::fgets(line, sizeof(line), file) != nullptr) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1 && kb > 0) {
                std::fclose(file);
                return static_cast<std::size_t>(kb / 2U) * 1024U;
            }
        }
        std::fclose(file);
    }
#endif
    return 0;
}

Net::Net() : d(new NetPrivate()) {
    this->opt_.num_threads = 1;
    static_cast<void>(this->d->blob_allocator.set_size_compare_ratio(0.0F));
    static_cast<void>(this->d->workspace_allocator.set_size_compare_ratio(0.0F));
    this->opt_.blob_allocator = &this->d->budget_blob;
    this->opt_.workspace_allocator = &this->d->budget_workspace;
    // The KV cache is the dominant allocation for long contexts; it must be
    // under the same budget or a large --kv-capacity still gets OOM-killed.
    this->opt_.kvcache_allocator = &this->d->budget_blob;

    // WSL runs are frequently killed by the kernel OOM killer, which also
    // takes down the calling shell. The budget is on by default; setting
    // MNET_MAX_MEM_MB=0 disables it explicitly.
    this->set_memory_budget_bytes(default_memory_budget_bytes());
}

void Net::set_memory_budget_bytes(std::size_t bytes) {
    this->d->budget_total = bytes;
    this->d->apply_budget();
}

std::size_t Net::memory_budget_bytes() const {
    return this->d->budget_total;
}

std::size_t Net::memory_peak_bytes() const {
    return std::max(this->d->budget_blob.peak_bytes(), this->d->budget_workspace.peak_bytes());
}

std::size_t Net::memory_current_bytes() const {
    return this->d->budget_blob.current_bytes() + this->d->budget_workspace.current_bytes();
}

bool Net::memory_budget_exhausted() const {
    return this->d->budget_blob.exhausted() || this->d->budget_workspace.exhausted();
}

Net::~Net() = default;

std::int32_t Net::load_param_bytes(const std::uint8_t* data, std::size_t size, bool binary) {
    if (data == nullptr || size == 0) {
        return -1;
    }
    std::vector<std::uint8_t> bytes(data, data + size);
    if (!binary) {
        bytes.push_back(0);
    }

    const auto* memory = bytes.data();
    mformat::DataReaderFromMemory reader(memory, bytes.size());
    mgraph::Graph graph;
    const std::int32_t load_ret = binary ? graph.load_param_bin(reader) : graph.load_param(reader);
    if (load_ret != 0) {
        detail::write_diagnostic("mnet: graph load_param failed\n");
        return -1;
    }

    std::vector<Layer*> layers;
    std::vector<std::string> layer_types;
    layers.reserve(graph.layers.size());
    std::vector<std::int32_t> input_blobs;
    const mgraph::ExecutionPlan plan = graph.build_execution_plan();
    for (const mgraph::LayerSpec& spec : graph.layers) {
        Layer* layer = spec.type.empty() ? create_layer_by_index(spec.type_index) : create_layer(spec.type.c_str());
        if (layer == nullptr) {
            const std::string type = spec.type.empty() ? std::to_string(spec.type_index) : spec.type;
            detail::write_diagnostic(std::string("mnet: create_layer failed: ") + type + "\n");
            for (Layer* created : layers) {
                delete created;
            }
            return -1;
        }
        if (layer->load_param(spec.params) != 0) {
            const std::string type = spec.type.empty() ? std::to_string(spec.type_index) : spec.type;
            detail::write_diagnostic(std::string("mnet: layer load_param failed: ") + type + "\n");
            delete layer;
            for (Layer* created : layers) {
                delete created;
            }
            return -1;
        }
        layer->bottoms = spec.bottoms;
        layer->tops = spec.tops;
        layer->typeindex = spec.type_index;
        this->d->layer_index[layer] = static_cast<std::int32_t>(layers.size());
        // Only real Input layers are user-feedable. MemoryData and other
        // weight-initialised layers produce their blobs from the model bin.
        if (spec.type_index == 16 || spec.type == "Input") {
            for (const std::int32_t top : spec.tops) {
                input_blobs.push_back(top);
            }
        }
        layers.push_back(layer);
        layer_types.push_back(spec.type);
    }

    for (Layer* layer : this->d->layers) {
        delete layer;
    }
    if (input_blobs.empty()) {
        for (const std::int32_t blob_index : plan.input_blobs) {
            if (blob_index < 0 || static_cast<std::size_t>(blob_index) >= graph.blobs.size()) {
                continue;
            }
            const std::int32_t producer = graph.blobs[static_cast<std::size_t>(blob_index)].producer;
            if (producer < 0 || static_cast<std::size_t>(producer) >= graph.layers.size()) {
                continue;
            }
            const mgraph::LayerSpec& producer_spec = graph.layers[static_cast<std::size_t>(producer)];
            if (producer_spec.type_index == 16 || producer_spec.type == "Input") {
                input_blobs.push_back(blob_index);
            }
        }
    }
    if (input_blobs.empty()) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            Layer* layer = layers[i];
            const bool zero_bottom_data = layer->typeindex == 19 || layer_types[i] == "MemoryData";
            if (layer->bottoms.empty() && !zero_bottom_data) {
                for (const std::int32_t top : layer->tops) {
                    input_blobs.push_back(top);
                }
            }
        }
    }
    this->d->layers = std::move(layers);
    this->d->blobs = graph.blobs;
    this->d->layer_types = std::move(layer_types);
    this->d->blob_last_use = plan.blob_last_use;
    this->d->input_blobs = std::move(input_blobs);
    this->d->output_blobs = plan.output_blobs;
    if (std::getenv("MNET_NO_FUSE_ACTIVATION") == nullptr) {
        this->d->fuse_convolution_activation();
    }
    return 0;
}

std::int32_t Net::load_param(const char* param_path) {
    if (param_path == nullptr) {
        return -1;
    }
    std::FILE* file = std::fopen(param_path, "rb");
    if (file == nullptr) {
        return -1;
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buffer[4096];
    std::size_t nread = 0;
    while ((nread = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        bytes.insert(bytes.end(), buffer, buffer + nread);
    }
    std::fclose(file);
    if (bytes.empty()) {
        return -1;
    }
    const bool binary = bytes.size() >= sizeof(std::int32_t) && [&bytes]() {
        std::int32_t magic = 0;
        std::memcpy(&magic, bytes.data(), sizeof(magic));
        return magic == 7767517;
    }();
    return this->load_param_bytes(bytes.data(), bytes.size(), binary);
}

std::int32_t Net::load_param_mem(const char* mem) {
    if (mem == nullptr) {
        return -1;
    }
    return this->load_param_bytes(reinterpret_cast<const std::uint8_t*>(mem), std::strlen(mem), false);
}

std::int32_t Net::load_param_bin_mem(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        return -1;
    }
    return this->load_param_bytes(static_cast<const std::uint8_t*>(data), size, true);
}

std::vector<std::string> Net::input_names() const {
    std::vector<std::string> names;
    names.reserve(this->d->input_blobs.size());
    for (const std::int32_t index : this->d->input_blobs) {
        if (index >= 0 && static_cast<std::size_t>(index) < this->d->blobs.size()) {
            names.push_back(this->d->blobs[static_cast<std::size_t>(index)].name);
        }
    }
    return names;
}

std::vector<std::string> Net::output_names() const {
    std::vector<std::string> names;
    names.reserve(this->d->output_blobs.size());
    for (const std::int32_t index : this->d->output_blobs) {
        if (index >= 0 && static_cast<std::size_t>(index) < this->d->blobs.size()) {
            names.push_back(this->d->blobs[static_cast<std::size_t>(index)].name);
        }
    }
    return names;
}

std::int32_t Net::load_model_mem(const void* data, std::size_t size) {
    if (this->d->budget_total != 0 && size >= this->d->budget_total) {
        detail::write_diagnostic("mnet: model weights exceed the memory budget\n");
        return -1;
    }
    this->d->weight_bytes = size;
    this->d->apply_budget();
    if (data == nullptr || size == 0) {
        return -1;
    }
    const auto* memory = static_cast<const std::uint8_t*>(data);
    mformat::DataReaderFromMemory reader(memory, size);
    mformat::ModelBinFromDataReader model_bin(reader);
    for (std::size_t i = 0; i < this->d->layers.size(); ++i) {
        const std::int32_t ret = this->d->layers[i]->load_model(model_bin);
        if (ret != 0) {
            const std::string& type = this->d->layer_types[i];
            detail::write_diagnostic(std::string("mnet: layer load_model failed: ") + type + "\n");
            return ret;
        }
    }
    return 0;
}

std::int32_t Net::load_model(const char* bin_path) {
    if (bin_path == nullptr) {
        return -1;
    }
    std::FILE* file = std::fopen(bin_path, "rb");
    if (file == nullptr) {
        return -1;
    }

    // Weights are allocated by ModelBin outside the budgeted allocators, so
    // check the file size up front: a model that cannot fit must fail here,
    // not after the kernel has already started swapping/killing.
    if (std::fseek(file, 0, SEEK_END) == 0) {
        const long size = std::ftell(file);
        std::rewind(file);
        if (size > 0) {
            const auto weights = static_cast<std::size_t>(size);
            if (this->d->budget_total != 0 && weights >= this->d->budget_total) {
                detail::write_diagnostic("mnet: model weights (" + std::to_string(weights / (1024U * 1024U)) +
                                         " MiB) exceed the memory budget (" +
                                         std::to_string(this->d->budget_total / (1024U * 1024U)) + " MiB)\n");
                std::fclose(file);
                return -1;
            }
            this->d->weight_bytes = weights;
            this->d->apply_budget();
        }
    }

    std::int32_t ret = 0;
    {
        mformat::DataReaderFromStdio reader(file);
        mformat::ModelBinFromDataReader model_bin(reader);
        for (std::size_t i = 0; i < this->d->layers.size(); ++i) {
            ret = this->d->layers[i]->load_model(model_bin);
            if (std::getenv("MNET_TRACE_BIN") != nullptr) {
                const long pos = std::ftell(file);
                std::fprintf(stderr, "[binload] %zu %s %ld\n", i, this->d->layer_types[i].c_str(), pos);
            }
            if (ret != 0) {
                const std::string& type = this->d->layer_types[i];
                detail::write_diagnostic(std::string("mnet: layer load_model failed: ") + type + "\n");
                break;
            }
        }
    }
    std::fclose(file);
    return ret;
}

std::int32_t Net::load(const char* param_path, const char* bin_path) {
    const std::int32_t ret = this->load_param(param_path);
    if (ret != 0) {
        return ret;
    }
    return this->load_model(bin_path);
}

Net::Extractor::Extractor(const Net* net) : net_(net), opt_(net->opt_) {
    this->blob_mats_.resize(net_->d->blobs.size());
}

std::int32_t Net::Extractor::input(const char* blob_name, const Mat& in) {
    return this->input(this->net_->d->find_blob(blob_name), in);
}

std::int32_t Net::Extractor::input(std::int32_t blob_index, const Mat& in) {
    if (blob_index < 0 || static_cast<std::size_t>(blob_index) >= this->blob_mats_.size()) {
        return -1;
    }
    this->blob_mats_[static_cast<std::size_t>(blob_index)] = in;
    return 0;
}

std::int32_t Net::Extractor::extract(const char* blob_name, Mat& out) {
    return this->extract(this->net_->d->find_blob(blob_name), out);
}

std::int32_t Net::Extractor::extract(std::int32_t blob_index, Mat& out) {
    if (blob_index < 0 || static_cast<std::size_t>(blob_index) >= this->blob_mats_.size()) {
        return -1;
    }

    mgraph::Extractor<Layer, Mat, Option> extractor(this->net_->d->layers, this->net_->d->blobs, *this->net_->d,
                                                    this->blob_mats_, this->opt_);
    const std::int32_t ret = extractor.ensure(blob_index);
    if (ret != 0) {
        this->net_->d->report_budget();
        return ret;
    }
    out = this->blob_mats_[static_cast<std::size_t>(blob_index)];
    if (out.empty()) {
        this->net_->d->report_budget();
        return -100;
    }
    return 0;
}

}  // namespace mnet
