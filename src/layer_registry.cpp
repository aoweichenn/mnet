// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/layer_registry.hpp>

#include <cstdint>

#include <string>
#include <vector>

#include <mformat/model_bin.hpp>
#include <mlayer/batchnorm.hpp>
#include <mlayer/bias.hpp>
#include <mlayer/binaryop.hpp>
#include <mlayer/concat.hpp>
#include <mlayer/convolution.hpp>
#include <mlayer/convolutiondepthwise.hpp>
#include <mlayer/dropout.hpp>
#include <mlayer/flatten.hpp>
#include <mlayer/gelu.hpp>
#include <mlayer/innerproduct.hpp>
#include <mlayer/interp.hpp>
#include <mlayer/matmul.hpp>
#include <mlayer/permute.hpp>
#include <mlayer/pooling.hpp>
#include <mlayer/power.hpp>
#include <mlayer/relu.hpp>
#include <mlayer/reshape.hpp>
#include <mlayer/rmsnorm.hpp>
#include <mlayer/scale.hpp>
#include <mlayer/shufflechannel.hpp>
#include <mlayer/sigmoid.hpp>
#include <mlayer/slice.hpp>
#include <mlayer/softmax.hpp>
#include <mlayer/split.hpp>
#include <mlayer/swish.hpp>
#include <mlayer/tanh.hpp>
#include <mlayer/threshold.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/quantize.hpp>

namespace mnet {
namespace {

using MatT = mnet::Mat;
using OptionT = mnet::Option;
using ParamDictT = mformat::ParamDict;
using ModelBinT = mformat::ModelBin;

using ConvolutionT = mlayer::Convolution<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, false, mnet::activation_ss,
                                         mnet::quantize_int8, mnet::copy_make_border>;
using PoolingT = mlayer::Pooling<Layer, MatT, OptionT, ParamDictT, false, mnet::copy_make_border>;
using InnerProductT = mlayer::InnerProduct<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, false,
                                           mnet::activation_ss, mnet::quantize_int8>;
using ConvDepthWiseT =
    mlayer::ConvolutionDepthWise<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, false, mnet::activation_ss,
                                 mnet::quantize_int8, mnet::copy_make_border, mnet::flatten>;

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
    add("Convolution", &make_layer<ConvolutionT>);
    add("Pooling", &make_layer<PoolingT>);
    add("InnerProduct", &make_layer<InnerProductT>);
    add("Concat", &make_layer<mlayer::Concat<Layer, MatT, OptionT, ParamDictT>>);
    add("Split", &make_layer<mlayer::Split<Layer, MatT, OptionT>>);
    add("ConvolutionDepthWise", &make_layer<ConvDepthWiseT>);
    add("BatchNorm", &make_layer<mlayer::BatchNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("ShuffleChannel", &make_layer<mlayer::ShuffleChannel<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Slice", &make_layer<mlayer::Slice<Layer, MatT, OptionT, ParamDictT, false>>);
    add("RMSNorm", &make_layer<mlayer::RMSNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("BinaryOp", &make_layer<mlayer::BinaryOp<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Permute", &make_layer<mlayer::Permute<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Reshape", &make_layer<mlayer::Reshape<Layer, MatT, OptionT, ParamDictT>>);
    add("MatMul", &make_layer<mlayer::MatMul<Layer, MatT, OptionT, ParamDictT>>);
    add("Interp", &make_layer<mlayer::Interp<Layer, MatT, OptionT, ParamDictT>>);
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
