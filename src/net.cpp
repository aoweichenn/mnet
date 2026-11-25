// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/net.hpp>

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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
#include <mnet/layer_registry.hpp>

#include <mmemory/allocator.hpp>

namespace mnet {

struct NetPrivate final : public mgraph::ExecutionBackend<Layer, Mat, Option> {
    std::vector<Layer*> layers;
    std::vector<mgraph::Blob> blobs;
    std::vector<std::string> layer_types;
    std::vector<std::int32_t> blob_last_use;
    std::unordered_map<const Layer*, std::int32_t> layer_index;
    mutable std::int32_t profile_index = 0;
    mmemory::UnlockedPoolAllocator blob_allocator;
    mmemory::UnlockedPoolAllocator workspace_allocator;

    ~NetPrivate() {
        for (Layer* layer : this->layers) {
            delete layer;
        }
        this->layers.clear();
    }

    [[nodiscard]] bool is_input_layer(const Layer& layer) const override { return layer.bottoms.empty(); }

    [[nodiscard]] Option mask_option(const Option& opt, std::int32_t) const override { return opt; }

    [[nodiscard]] bool should_release_bottom(const Mat&, const Option&) const override { return false; }

    std::int32_t invoke(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const override {
        const auto t0 = std::chrono::steady_clock::now();
        const std::int32_t result = this->invoke_impl(layer, blob_mats, opt);
        if (result == 0) {
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
            if (top != nullptr) {
                std::fprintf(stderr, "PROFILE %d %s %.3f %dx%dx%dx%d\n", index, type, ms, top->w, top->h, top->d,
                             top->c);
            } else {
                std::fprintf(stderr, "PROFILE %d %s %.3f -\n", index, type, ms);
            }
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

Net::Net() : d(new NetPrivate()) {
    this->opt_.num_threads = 1;
    this->d->blob_allocator.set_size_compare_ratio(0.0F);
    this->d->workspace_allocator.set_size_compare_ratio(0.0F);
    this->opt_.blob_allocator = &this->d->blob_allocator;
    this->opt_.workspace_allocator = &this->d->workspace_allocator;
}

Net::~Net() = default;

std::int32_t Net::load_param(const char* param_path) {
    if (param_path == nullptr) {
        return -1;
    }
    std::ifstream file(param_path, std::ios::binary);
    if (!file) {
        return -1;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    const std::string text = stream.str();

    const auto* memory = reinterpret_cast<const std::uint8_t*>(text.c_str());
    mformat::DataReaderFromMemory reader(memory, text.size());
    mgraph::Graph graph;
    if (graph.load_param(reader) != 0) {
        return -1;
    }

    std::vector<Layer*> layers;
    std::vector<std::string> layer_types;
    layers.reserve(graph.layers.size());
    const mgraph::ExecutionPlan plan = graph.build_execution_plan();
    for (const mgraph::LayerSpec& spec : graph.layers) {
        Layer* layer = create_layer(spec.type.c_str());
        if (layer == nullptr) {
            for (Layer* created : layers) {
                delete created;
            }
            return -1;
        }
        if (layer->load_param(spec.params) != 0) {
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
        layers.push_back(layer);
        layer_types.push_back(spec.type);
    }

    for (Layer* layer : this->d->layers) {
        delete layer;
    }
    this->d->layers = std::move(layers);
    this->d->blobs = graph.blobs;
    this->d->layer_types = std::move(layer_types);
    this->d->blob_last_use = plan.blob_last_use;
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
    std::int32_t ret = 0;
    {
        mformat::DataReaderFromStdio reader(file);
        mformat::ModelBinFromDataReader model_bin(reader);
        for (Layer* layer : this->d->layers) {
            ret = layer->load_model(model_bin);
            if (ret != 0) {
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

Net::Extractor::Extractor(const Net* net) : net_(net) {
    this->blob_mats_.resize(net_->d->blobs.size());
}

std::int32_t Net::Extractor::input(const char* blob_name, const Mat& in) {
    const std::int32_t index = this->net_->d->find_blob(blob_name);
    if (index < 0) {
        return -1;
    }
    this->blob_mats_[static_cast<std::size_t>(index)] = in;
    return 0;
}

std::int32_t Net::Extractor::extract(const char* blob_name, Mat& out) {
    const std::int32_t index = this->net_->d->find_blob(blob_name);
    if (index < 0) {
        return -1;
    }

    Option opt = this->net_->opt_;
    mgraph::Extractor<Layer, Mat, Option> extractor(this->net_->d->layers, this->net_->d->blobs, *this->net_->d,
                                                    this->blob_mats_, opt);
    const std::int32_t ret = extractor.ensure(index);
    if (ret != 0) {
        return ret;
    }
    out = this->blob_mats_[static_cast<std::size_t>(index)];
    return out.empty() ? -100 : 0;
}

}  // namespace mnet
