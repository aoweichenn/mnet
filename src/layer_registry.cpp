// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/layer_registry.hpp>

#include <cstdint>

#include <string>
#include <vector>

#include <mformat/model_bin.hpp>
#include <mlayer/bias.hpp>
#include <mlayer/dropout.hpp>
#include <mlayer/flatten.hpp>
#include <mlayer/gelu.hpp>
#include <mlayer/power.hpp>
#include <mlayer/relu.hpp>
#include <mlayer/scale.hpp>
#include <mlayer/sigmoid.hpp>
#include <mlayer/softmax.hpp>
#include <mlayer/swish.hpp>
#include <mlayer/tanh.hpp>
#include <mlayer/threshold.hpp>

namespace mnet {
namespace {

using MatT = mnet::Mat;
using OptionT = mnet::Option;
using ParamDictT = mformat::ParamDict;
using ModelBinT = mformat::ModelBin;

class InputLayer final : public Layer {
public:
    InputLayer() { this->one_blob_only = true; }
};

template<typename L>
Layer* make_layer() {
    return new L;
}

struct Entry {
    std::string type;
    LayerCreator creator;
};

std::vector<Entry>& entries() {
    static std::vector<Entry> table;
    return table;
}

void add(const char* type, LayerCreator creator) {
    entries().push_back(Entry{type, creator});
}

void register_builtins() {
    if (!entries().empty()) {
        return;
    }

    add("Input", &make_layer<InputLayer>);
    add("ReLU", &make_layer<mlayer::ReLU<Layer, MatT, OptionT, ParamDictT>>);
    add("Sigmoid", &make_layer<mlayer::Sigmoid<Layer, MatT, OptionT>>);
    add("TanH", &make_layer<mlayer::TanH<Layer, MatT, OptionT>>);
    add("Swish", &make_layer<mlayer::Swish<Layer, MatT, OptionT>>);
    add("Flatten", &make_layer<mlayer::Flatten<Layer, MatT, OptionT>>);
    add("Dropout", &make_layer<mlayer::Dropout<Layer, MatT, OptionT, ParamDictT>>);
    add("Threshold", &make_layer<mlayer::Threshold<Layer, MatT, OptionT, ParamDictT>>);
    add("Power", &make_layer<mlayer::Power<Layer, MatT, OptionT, ParamDictT>>);
    add("GELU", &make_layer<mlayer::GELU<Layer, MatT, OptionT, ParamDictT>>);
    add("Softmax", &make_layer<mlayer::Softmax<Layer, MatT, OptionT, ParamDictT>>);
    add("Bias", &make_layer<mlayer::Bias<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Scale", &make_layer<mlayer::Scale<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
}

}  // namespace

Layer* create_layer(const char* type) {
    register_builtins();
    if (type == nullptr) {
        return nullptr;
    }
    for (const Entry& entry : entries()) {
        if (entry.type == type && entry.creator != nullptr) {
            return entry.creator();
        }
    }
    return nullptr;
}

std::int32_t register_layer(const char* type, LayerCreator creator) {
    register_builtins();
    if (type == nullptr || creator == nullptr) {
        return -1;
    }
    for (Entry& entry : entries()) {
        if (entry.type == type) {
            entry.creator = creator;
            return 0;
        }
    }
    entries().push_back(Entry{type, creator});
    return 0;
}

}  // namespace mnet
