// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <mnet/layer_registry.hpp>

#include <cstdint>

#include <string>
#include <vector>

#include <mformat/model_bin.hpp>
#include <mlayer/absval.hpp>
#include <mlayer/argmax.hpp>
#include <mlayer/batchnorm.hpp>
#include <mlayer/bias.hpp>
#include <mlayer/binaryop.hpp>
#include <mlayer/bnll.hpp>
#include <mlayer/cast.hpp>
#include <mlayer/celu.hpp>
#include <mlayer/clip.hpp>
#include <mlayer/concat.hpp>
#include <mlayer/convolution.hpp>
#include <mlayer/convolution1d.hpp>
#include <mlayer/convolution3d.hpp>
#include <mlayer/convolutiondepthwise.hpp>
#include <mlayer/convolutiondepthwise1d.hpp>
#include <mlayer/convolutiondepthwise3d.hpp>
#include <mlayer/copyto.hpp>
#include <mlayer/crop.hpp>
#include <mlayer/cumulativesum.hpp>
#include <mlayer/deconvolution.hpp>
#include <mlayer/deconvolution1d.hpp>
#include <mlayer/deconvolution3d.hpp>
#include <mlayer/deepcopy.hpp>
#include <mlayer/deformableconv2d.hpp>
#include <mlayer/detectionoutput.hpp>
#include <mlayer/diag.hpp>
#include <mlayer/dropout.hpp>
#include <mlayer/einsum.hpp>
#include <mlayer/eltwise.hpp>
#include <mlayer/elu.hpp>
#include <mlayer/embed.hpp>
#include <mlayer/erf.hpp>
#include <mlayer/exp.hpp>
#include <mlayer/expanddims.hpp>
#include <mlayer/flatten.hpp>
#include <mlayer/flip.hpp>
#include <mlayer/gateddeltarule.hpp>
#include <mlayer/gelu.hpp>
#include <mlayer/gemm.hpp>
#include <mlayer/glu.hpp>
#include <mlayer/gridsample.hpp>
#include <mlayer/groupnorm.hpp>
#include <mlayer/gru.hpp>
#include <mlayer/hardsigmoid.hpp>
#include <mlayer/hardswish.hpp>
#include <mlayer/innerproduct.hpp>
#include <mlayer/instancenorm.hpp>
#include <mlayer/interp.hpp>
#include <mlayer/layernorm.hpp>
#include <mlayer/log.hpp>
#include <mlayer/lrn.hpp>
#include <mlayer/lstm.hpp>
#include <mlayer/matmul.hpp>
#include <mlayer/memorydata.hpp>
#include <mlayer/mish.hpp>
#include <mlayer/multiheadattention.hpp>
#include <mlayer/mvn.hpp>
#include <mlayer/noop.hpp>
#include <mlayer/normalize.hpp>
#include <mlayer/packing.hpp>
#include <mlayer/padding.hpp>
#include <mlayer/permute.hpp>
#include <mlayer/pixelshuffle.hpp>
#include <mlayer/pooling.hpp>
#include <mlayer/pooling1d.hpp>
#include <mlayer/pooling3d.hpp>
#include <mlayer/power.hpp>
#include <mlayer/prelu.hpp>
#include <mlayer/priorbox.hpp>
#include <mlayer/proposal.hpp>
#include <mlayer/psroipooling.hpp>
#include <mlayer/quantize_layer.hpp>
#include <mlayer/reduction.hpp>
#include <mlayer/relposbias.hpp>
#include <mlayer/relu.hpp>
#include <mlayer/reorg.hpp>
#include <mlayer/reshape.hpp>
#include <mlayer/rmsnorm.hpp>
#include <mlayer/rnn.hpp>
#include <mlayer/roialign.hpp>
#include <mlayer/roipooling.hpp>
#include <mlayer/rotaryembed.hpp>
#include <mlayer/scale.hpp>
#include <mlayer/sdpa.hpp>
#include <mlayer/selu.hpp>
#include <mlayer/shortconv.hpp>
#include <mlayer/situ.hpp>
#include <mlayer/stft.hpp>
#include <mlayer/shrink.hpp>
#include <mlayer/shufflechannel.hpp>
#include <mlayer/sigmoid.hpp>
#include <mlayer/slice.hpp>
#include <mlayer/softmax.hpp>
#include <mlayer/softplus.hpp>
#include <mlayer/spectrogram.hpp>
#include <mlayer/split.hpp>
#include <mlayer/spp.hpp>
#include <mlayer/squeeze.hpp>
#include <mlayer/statisticspooling.hpp>
#include <mlayer/swish.hpp>
#include <mlayer/tanh.hpp>
#include <mlayer/threshold.hpp>
#include <mlayer/tile.hpp>
#include <mlayer/unaryop.hpp>
#include <mlayer/unfold.hpp>
#include <mlayer/yolodetectionoutput.hpp>
#include <mlayer/yolov3detectionoutput.hpp>
#include <mlayer/yolov5focus.hpp>
#include <mnet/activation.hpp>
#include <mnet/border.hpp>
#include <mnet/quantize.hpp>

namespace mnet {
namespace {

using MatT = mnet::Mat;
using OptionT = mnet::Option;
using ParamDictT = mformat::ParamDict;
using ModelBinT = mformat::ModelBin;

using ConvolutionT = mlayer::Convolution<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, true, mnet::activation_ss,
                                         mnet::quantize_int8, mnet::copy_make_border>;
using PoolingT = mlayer::Pooling<Layer, MatT, OptionT, ParamDictT, false, mnet::copy_make_border>;
using InnerProductT = mlayer::InnerProduct<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, true,
                                           mnet::activation_ss, mnet::quantize_int8>;
using ConvDepthWiseT =
    mlayer::ConvolutionDepthWise<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, true, mnet::activation_ss,
                                 mnet::quantize_int8, mnet::copy_make_border, mnet::flatten>;
using DeconvolutionT = mlayer::Deconvolution<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss,
                                             mnet::copy_cut_border, mnet::flatten>;

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
    add("ConvolutionDepthWise1D",
        &make_layer<
            mlayer::ConvolutionDepthWise1D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("ConvolutionDepthWise3D",
        &make_layer<
            mlayer::ConvolutionDepthWise3D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("BatchNorm", &make_layer<mlayer::BatchNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("ShuffleChannel", &make_layer<mlayer::ShuffleChannel<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Slice", &make_layer<mlayer::Slice<Layer, MatT, OptionT, ParamDictT, false>>);
    add("RMSNorm", &make_layer<mlayer::RMSNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("AbsVal", &make_layer<mlayer::AbsVal<Layer, MatT, OptionT>>);
    add("BNLL", &make_layer<mlayer::BNLL<Layer, MatT, OptionT>>);
    add("Clip", &make_layer<mlayer::Clip<Layer, MatT, OptionT, ParamDictT>>);
    add("CELU", &make_layer<mlayer::CELU<Layer, MatT, OptionT, ParamDictT>>);
    add("Shrink", &make_layer<mlayer::Shrink<Layer, MatT, OptionT, ParamDictT>>);
    add("DeepCopy", &make_layer<mlayer::DeepCopy<Layer, MatT, OptionT>>);
    add("ArgMax", &make_layer<mlayer::ArgMax<Layer, MatT, OptionT, ParamDictT>>);
    add("GLU", &make_layer<mlayer::GLU<Layer, MatT, OptionT, ParamDictT>>);
    add("Cast", &make_layer<mlayer::Cast<Layer, MatT, OptionT, ParamDictT>>);
    add("CumulativeSum", &make_layer<mlayer::CumulativeSum<Layer, MatT, OptionT, ParamDictT>>);
    add("CopyTo", &make_layer<mlayer::CopyTo<Layer, MatT, OptionT, ParamDictT>>);
    add("Diag", &make_layer<mlayer::Diag<Layer, MatT, OptionT, ParamDictT>>);
    add("SPP", &make_layer<mlayer::SPP<Layer, MatT, OptionT, ParamDictT, false>>);
    add("StatisticsPooling", &make_layer<mlayer::StatisticsPooling<Layer, MatT, OptionT, ParamDictT>>);
    add("Packing", &make_layer<mlayer::Packing<Layer, MatT, OptionT, ParamDictT>>);
    add("Pooling1D", &make_layer<mlayer::Pooling1D<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Pooling3D", &make_layer<mlayer::Pooling3D<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Quantize", &make_layer<mlayer::Quantize<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Dequantize", &make_layer<mlayer::Dequantize<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Requantize",
        &make_layer<mlayer::Requantize<Layer, MatT, OptionT, ParamDictT, ModelBinT, mnet::activation_ss>>);
    add("Flip", &make_layer<mlayer::Flip<Layer, MatT, OptionT, ParamDictT>>);
    add("InstanceNorm", &make_layer<mlayer::InstanceNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("MVN", &make_layer<mlayer::MVN<Layer, MatT, OptionT, ParamDictT>>);
    add("Exp", &make_layer<mlayer::Exp<Layer, MatT, OptionT, ParamDictT>>);
    add("Log", &make_layer<mlayer::Log<Layer, MatT, OptionT, ParamDictT>>);
    add("LRN", &make_layer<mlayer::LRN<Layer, MatT, OptionT, ParamDictT>>);
    add("HardSigmoid", &make_layer<mlayer::HardSigmoid<Layer, MatT, OptionT, ParamDictT>>);
    add("HardSwish", &make_layer<mlayer::HardSwish<Layer, MatT, OptionT, ParamDictT>>);
    add("GroupNorm", &make_layer<mlayer::GroupNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT, false>>);
    add("LayerNorm", &make_layer<mlayer::LayerNorm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("PReLU", &make_layer<mlayer::PReLU<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Normalize", &make_layer<mlayer::Normalize<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Deconvolution", &make_layer<DeconvolutionT>);
    add("DeconvolutionDepthWise", &make_layer<DeconvolutionT>);
    add("Deconvolution1D",
        &make_layer<mlayer::Deconvolution1D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("DeconvolutionDepthWise1D",
        &make_layer<mlayer::Deconvolution1D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("Deconvolution3D",
        &make_layer<mlayer::Deconvolution3D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("DeconvolutionDepthWise3D",
        &make_layer<mlayer::Deconvolution3D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("Crop", &make_layer<mlayer::Crop<Layer, MatT, OptionT, ParamDictT, false>>);
    add("YoloDetectionOutput", &make_layer<mlayer::YoloDetectionOutput<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Yolov3DetectionOutput", &make_layer<mlayer::Yolov3DetectionOutput<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Squeeze", &make_layer<mlayer::Squeeze<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Padding", &make_layer<mlayer::Padding<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Tile", &make_layer<mlayer::Tile<Layer, MatT, OptionT, ParamDictT>>);
    add("Unfold", &make_layer<mlayer::Unfold<Layer, MatT, OptionT, ParamDictT>>);
    add("DeformableConv2D",
        &make_layer<mlayer::DeformableConv2D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss>>);
    add("Fold", &make_layer<mlayer::Fold<Layer, MatT, OptionT, ParamDictT, mnet::copy_cut_border>>);
    add("GridSample", &make_layer<mlayer::GridSample<Layer, MatT, OptionT, ParamDictT, false>>);
    add("LSTM", &make_layer<mlayer::LSTM<Layer, MatT, OptionT, ParamDictT, ModelBinT, false>>);
    add("RNN", &make_layer<mlayer::RNN<Layer, MatT, OptionT, ParamDictT, ModelBinT, false>>);
    add("Spectrogram", &make_layer<mlayer::Spectrogram<Layer, MatT, OptionT, ParamDictT, false>>);
    add("InverseSpectrogram", &make_layer<mlayer::InverseSpectrogram<Layer, MatT, OptionT, ParamDictT, false>>);
    add("GRU", &make_layer<mlayer::GRU<Layer, MatT, OptionT, ParamDictT, ModelBinT, false>>);
    add("ShortConv", &make_layer<mlayer::ShortConv<Layer, MatT, OptionT, ParamDictT>>);
    add("GatedDeltaRule", &make_layer<mlayer::GatedDeltaRule<Layer, MatT, OptionT>>);
    add("Proposal", &make_layer<mlayer::Proposal<Layer, MatT, OptionT, ParamDictT, false>>);
    add("ROIPooling", &make_layer<mlayer::ROIPooling<Layer, MatT, OptionT, ParamDictT>>);
    add("ROIAlign", &make_layer<mlayer::ROIAlign<Layer, MatT, OptionT, ParamDictT, false>>);
    add("PSROIPooling", &make_layer<mlayer::PSROIPooling<Layer, MatT, OptionT, ParamDictT, false>>);
    add("ELU", &make_layer<mlayer::ELU<Layer, MatT, OptionT, ParamDictT>>);
    add("Situ", &make_layer<mlayer::Situ<Layer, MatT, OptionT, ParamDictT>>);
    add("RelPosBias", &make_layer<mlayer::RelPosBias<Layer, MatT, OptionT, ParamDictT>>);
    add("Stft", &make_layer<mlayer::Stft<Layer, MatT, OptionT, ParamDictT>>);
    add("Erf", &make_layer<mlayer::Erf<Layer, MatT, OptionT>>);
    add("Mish", &make_layer<mlayer::Mish<Layer, MatT, OptionT>>);
    add("SELU", &make_layer<mlayer::SELU<Layer, MatT, OptionT, ParamDictT>>);
    add("Softplus", &make_layer<mlayer::Softplus<Layer, MatT, OptionT>>);
    add("PixelShuffle", &make_layer<mlayer::PixelShuffle<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Reorg", &make_layer<mlayer::Reorg<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Convolution1D",
        &make_layer<mlayer::Convolution1D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false, mnet::activation_ss,
                                          mnet::copy_make_border, mnet::flatten>>);
    add("Convolution3D", &make_layer<mlayer::Convolution3D<Layer, MatT, OptionT, ParamDictT, ModelBinT, false,
                                                           mnet::activation_ss, mnet::copy_make_border_3d>>);
    add("MultiHeadAttention",
        &make_layer<mlayer::MultiHeadAttention<Layer, MatT, OptionT, ParamDictT, ModelBinT, false>>);
    add("YoloV5Focus", &make_layer<mlayer::YoloV5Focus<Layer, MatT, OptionT>>);
    add("MemoryData", &make_layer<mlayer::MemoryData<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("PriorBox", &make_layer<mlayer::PriorBox<Layer, MatT, OptionT, ParamDictT, false>>);
    add("DetectionOutput", &make_layer<mlayer::DetectionOutput<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Eltwise", &make_layer<mlayer::Eltwise<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Noop", &make_layer<mlayer::Noop<Layer, MatT, OptionT>>);
    add("UnaryOp", &make_layer<mlayer::UnaryOp<Layer, MatT, OptionT, ParamDictT, false>>);
    add("ExpandDims", &make_layer<mlayer::ExpandDims<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Reduction", &make_layer<mlayer::Reduction<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Einsum", &make_layer<mlayer::Einsum<Layer, MatT, OptionT, ParamDictT, false>>);
    add("BinaryOp", &make_layer<mlayer::BinaryOp<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Permute", &make_layer<mlayer::Permute<Layer, MatT, OptionT, ParamDictT, false>>);
    add("Reshape", &make_layer<mlayer::Reshape<Layer, MatT, OptionT, ParamDictT>>);
    add("MatMul", &make_layer<mlayer::MatMul<Layer, MatT, OptionT, ParamDictT>>);
    add("Interp", &make_layer<mlayer::Interp<Layer, MatT, OptionT, ParamDictT>>);
    add("Gemm", &make_layer<mlayer::Gemm<Layer, MatT, OptionT, ParamDictT, ModelBinT>>);
    add("Embed", &make_layer<mlayer::Embed<Layer, MatT, OptionT, ParamDictT, ModelBinT, true>>);
    add("RotaryEmbed", &make_layer<mlayer::RotaryEmbed<Layer, MatT, OptionT, ParamDictT>>);
    add("SDPA", &make_layer<mlayer::SDPA<Layer, MatT, OptionT, ParamDictT>>);
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

Layer* create_layer_by_index(std::int32_t type_index) {
    // Complete mapping for every registered layer that has a stable ncnn
    // binary type index (ncnn layer_type_enum.h). Layers newer than the enum
    // (YoloV5Focus, ShortConv, GatedDeltaRule) are text-only for now.
    switch (type_index) {
        case 0:
            return create_layer("AbsVal");
        case 2:
            return create_layer("BatchNorm");
        case 3:
            return create_layer("Bias");
        case 4:
            return create_layer("BNLL");
        case 5:
            return create_layer("Concat");
        case 6:
            return create_layer("Convolution");
        case 7:
            return create_layer("Crop");
        case 8:
            return create_layer("Deconvolution");
        case 9:
            return create_layer("Dropout");
        case 10:
            return create_layer("Eltwise");
        case 11:
            return create_layer("ELU");
        case 12:
            return create_layer("Embed");
        case 13:
            return create_layer("Exp");
        case 14:
            return create_layer("Flatten");
        case 15:
            return create_layer("InnerProduct");
        case 16:
            return create_layer("Input");
        case 17:
            return create_layer("Log");
        case 18:
            return create_layer("LRN");
        case 19:
            return create_layer("MemoryData");
        case 21:
            return create_layer("Pooling");
        case 22:
            return create_layer("Power");
        case 23:
            return create_layer("PReLU");
        case 24:
            return create_layer("Proposal");
        case 25:
            return create_layer("Reduction");
        case 26:
            return create_layer("ReLU");
        case 27:
            return create_layer("Reshape");
        case 28:
            return create_layer("ROIPooling");
        case 29:
            return create_layer("Scale");
        case 30:
            return create_layer("Sigmoid");
        case 31:
            return create_layer("Slice");
        case 32:
            return create_layer("Softmax");
        case 33:
            return create_layer("Split");
        case 35:
            return create_layer("TanH");
        case 36:
            return create_layer("Threshold");
        case 40:
            return create_layer("BinaryOp");
        case 41:
            return create_layer("UnaryOp");
        case 42:
            return create_layer("ConvolutionDepthWise");
        case 44:
            return create_layer("Squeeze");
        case 45:
            return create_layer("ExpandDims");
        case 46:
            return create_layer("Normalize");
        case 47:
            return create_layer("Permute");
        case 48:
            return create_layer("PriorBox");
        case 49:
            return create_layer("DetectionOutput");
        case 50:
            return create_layer("Interp");
        case 51:
            return create_layer("DeconvolutionDepthWise");
        case 52:
            return create_layer("ShuffleChannel");
        case 54:
            return create_layer("Clip");
        case 55:
            return create_layer("Reorg");
        case 56:
            return create_layer("YoloDetectionOutput");
        case 59:
            return create_layer("Yolov3DetectionOutput");
        case 60:
            return create_layer("PSROIPooling");
        case 65:
            return create_layer("HardSigmoid");
        case 66:
            return create_layer("SELU");
        case 67:
            return create_layer("HardSwish");
        case 68:
            return create_layer("Noop");
        case 69:
            return create_layer("PixelShuffle");
        case 71:
            return create_layer("Mish");
        case 73:
            return create_layer("Swish");
        case 74:
            return create_layer("Gemm");
        case 75:
            return create_layer("GroupNorm");
        case 76:
            return create_layer("LayerNorm");
        case 77:
            return create_layer("Softplus");
        case 79:
            return create_layer("MultiHeadAttention");
        case 80:
            return create_layer("GELU");
        case 81:
            return create_layer("Convolution1D");
        case 84:
            return create_layer("Convolution3D");
        case 87:
            return create_layer("MatMul");
        case 92:
            return create_layer("Einsum");
        case 100:
            return create_layer("Erf");
        case 104:
            return create_layer("RMSNorm");
        case 108:
            return create_layer("SDPA");
        case 109:
            return create_layer("RotaryEmbed");
        default:
            return nullptr;
    }
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
