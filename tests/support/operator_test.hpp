// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mformat/model_bin.hpp>
#include <mformat/param_dict.hpp>
#include <mnet/core/layer_registry.hpp>

namespace mnet::test {

struct TensorShape {
    std::int32_t rank;
    std::int32_t width;
    std::int32_t height = 1;
    std::int32_t depth = 1;
    std::int32_t channels = 1;

    bool operator==(const TensorShape&) const = default;
};

inline void PrintTo(const TensorShape& shape, std::ostream* stream) {
    *stream << "rank=" << shape.rank << ", w=" << shape.width << ", h=" << shape.height << ", d=" << shape.depth
            << ", c=" << shape.channels;
}

inline constexpr std::int32_t MNET_TEST_MATRIX_RANK = 2;
inline constexpr std::int32_t MNET_TEST_VOLUME_RANK = 3;
inline constexpr std::int32_t MNET_TEST_HYPERVOLUME_RANK = 4;
inline constexpr float MNET_TEST_FLOAT_TOLERANCE = 1e-5F;

inline TensorShape shape_of(const Mat& mat) {
    return {mat.dims, mat.w, mat.h, mat.d, mat.c};
}

inline Mat tensor(TensorShape shape, const std::vector<float>& values) {
    auto mat = Mat{};
    switch (shape.rank) {
        case 1: {
            mat.create(shape.width);
            break;
        }
        case MNET_TEST_MATRIX_RANK: {
            mat.create(shape.width, shape.height);
            break;
        }
        case MNET_TEST_VOLUME_RANK: {
            mat.create(shape.width, shape.height, shape.channels);
            break;
        }
        case MNET_TEST_HYPERVOLUME_RANK: {
            mat.create(shape.width, shape.height, shape.depth, shape.channels);
            break;
        }
        default: {
            throw std::invalid_argument("Invalid tensor fixture");
        }
    }
    const auto plane = static_cast<std::size_t>(shape.width * shape.height * shape.depth);
    if (mat.empty() || values.size() != plane * static_cast<std::size_t>(shape.channels)) {
        throw std::invalid_argument("Invalid fixture size");
    }
    // Initialize padding for deterministic fixtures, but compare only logical elements.
    std::fill_n(static_cast<float*>(mat), mat.total(), 0.0F);
    for (auto q = 0; q < shape.channels; ++q) {
        auto* dst = static_cast<float*>(mat) + static_cast<std::size_t>(q) * mat.cstep;
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(q) * plane), plane, dst);
    }
    return mat;
}

inline Mat vector(const std::vector<float>& values) {
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Invalid reference fixture");
    }
    return tensor({1, static_cast<std::int32_t>(values.size())}, values);
}

inline Mat tensor(TensorShape shape, std::initializer_list<float> values) {
    return tensor(shape, std::vector<float>(values));
}

inline Mat vector(std::initializer_list<float> values) {
    return vector(std::vector<float>(values));
}

inline Mat indices(std::initializer_list<std::int32_t> values) {
    auto mat = Mat(static_cast<std::int32_t>(values.size()), sizeof(std::int32_t));
    if (mat.empty()) {
        ADD_FAILURE() << "Could not allocate index fixture";
        return {};
    }
    std::copy(values.begin(), values.end(), static_cast<std::int32_t*>(mat.data));
    return mat;
}

inline std::vector<float> values_of(const Mat& mat) {
    const auto plane = static_cast<std::size_t>(mat.w * mat.h * mat.d);
    auto values = std::vector<float>{};
    for (auto q = 0; q < mat.c; ++q) {
        const auto* src = static_cast<const float*>(mat) + static_cast<std::size_t>(q) * mat.cstep;
        values.insert(values.end(), src, src + plane);
    }
    return values;
}

inline void expect_tensor(const Mat& actual, const Mat& expected, float tolerance = MNET_TEST_FLOAT_TOLERANCE) {
    if (expected.empty()) {
        EXPECT_TRUE(actual.empty());
        return;
    }
    ASSERT_FALSE(actual.empty());
    ASSERT_EQ(shape_of(actual), shape_of(expected));
    ASSERT_EQ(actual.elempack, 1);
    ASSERT_EQ(actual.elemsize, sizeof(float));
    const auto actual_values = values_of(actual);
    const auto reference = values_of(expected);
    if (actual_values.size() != reference.size()) {
        ADD_FAILURE() << "Logical element count mismatch: " << actual_values.size() << " vs " << reference.size();
        return;
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(actual_values[i], reference[i], tolerance) << "Flat index: " << i;
    }
}

// A logical tensor in Mat's storage order: [w], [h, w], [c, h, w] or [c, d, h, w].
// Values are channel-major and dense, so alignment padding is never referenced.
inline constexpr std::size_t MNET_TEST_REFERENCE_MAX_BYTES = 64 * 1024 * 1024;

struct ReferenceTensor {
    std::vector<std::size_t> shape;
    std::vector<float> values;

    ReferenceTensor(const std::vector<std::size_t>& dimensions, const std::vector<float>& data)
            : shape(dimensions), values(data) {
        if (dimensions.empty() || dimensions.size() > 4 || data.size() != element_count(dimensions)) {
            throw std::invalid_argument("Invalid reference tensor");
        }
    }

    static ReferenceTensor full(const std::vector<std::size_t>& shape, float value) {
        return {shape, std::vector<float>(element_count(shape), value)};
    }

    std::size_t element_count() const { return element_count(shape); }

    static std::size_t element_count(const std::vector<std::size_t>& dimensions) {
        auto count = std::size_t{1};
        for (const auto dimension : dimensions) {
            if (dimension == 0 || count > MNET_TEST_REFERENCE_MAX_BYTES / sizeof(float) / dimension) {
                throw std::invalid_argument("Invalid reference tensor");
            }
            count *= dimension;
        }
        return count;
    }

    float& at(std::size_t i) { return values[i]; }

    const float& at(std::size_t i) const { return values[i]; }
};

inline Mat convert(const ReferenceTensor& tensor) {
    for (const auto dimension : tensor.shape) {
        if (dimension > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Invalid reference tensor");
        }
    }
    const auto dim = [&tensor](std::size_t axis) { return static_cast<std::int32_t>(tensor.shape[axis]); };
    const auto channels = tensor.shape.size() >= 3 ? dim(0) : 1;
    auto result = Mat{};
    switch (tensor.shape.size()) {
        case 1: {
            result.create(dim(0));
            break;
        }
        case MNET_TEST_MATRIX_RANK: {
            result.create(dim(1), dim(0));
            break;
        }
        case MNET_TEST_VOLUME_RANK: {
            result.create(dim(2), dim(1), channels);
            break;
        }
        case MNET_TEST_HYPERVOLUME_RANK: {
            result.create(dim(3), dim(2), dim(1), channels);
            break;
        }
        default: {
            throw std::invalid_argument("Invalid reference tensor");
        }
    }
    if (result.empty()) {
        throw std::invalid_argument("Invalid reference tensor");
    }
    const auto plane =
        static_cast<std::size_t>(result.w) * static_cast<std::size_t>(result.h) * static_cast<std::size_t>(result.d);
    for (std::int32_t channel = 0; channel < result.c; ++channel) {
        std::copy_n(tensor.values.data() + static_cast<std::size_t>(channel) * plane, plane,
                    static_cast<float*>(result) + static_cast<std::size_t>(channel) * result.cstep);
    }
    return result;
}

inline void expect_tensor(const Mat& actual, const ReferenceTensor& expected,
                          float tolerance = MNET_TEST_FLOAT_TOLERANCE) {
    ASSERT_FALSE(expected.shape.empty());
    if (expected.values.empty()) {
        EXPECT_TRUE(actual.empty());
        return;
    }
    ASSERT_FALSE(actual.empty());
    EXPECT_EQ(actual.elempack, 1);
    EXPECT_EQ(actual.elemsize, sizeof(float));
    const auto dim = [&expected](std::size_t axis) { return static_cast<std::int32_t>(expected.shape[axis]); };
    ASSERT_EQ(actual.dims, static_cast<std::int32_t>(expected.shape.size()));
    EXPECT_EQ(actual.w, dim(expected.shape.size() - 1));
    if (expected.shape.size() >= MNET_TEST_MATRIX_RANK) {
        EXPECT_EQ(actual.h, dim(expected.shape.size() - 2));
    }
    if (expected.shape.size() >= MNET_TEST_VOLUME_RANK) {
        EXPECT_EQ(actual.c, dim(0));
    }
    if (expected.shape.size() == MNET_TEST_HYPERVOLUME_RANK) {
        EXPECT_EQ(actual.d, dim(1));
    }
    const auto actual_values = values_of(actual);
    ASSERT_EQ(actual_values.size(), expected.values.size());
    for (std::size_t i = 0; i < expected.values.size(); ++i) {
        EXPECT_NEAR(actual_values[i], expected.values[i], tolerance) << "Flat index: " << i;
    }
}

// Bounded top-k that matches ArgMax's tie rule (higher value, then lower index).
inline ReferenceTensor topk_reference(const ReferenceTensor& input, std::int32_t top_k, bool with_values) {
    auto ranked = std::vector<std::pair<float, std::size_t>>{};
    ranked.reserve(input.values.size());
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        ranked.emplace_back(input.values[i], i);
    }
    const auto count = std::min(static_cast<std::size_t>(top_k), ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(count), ranked.end(),
                      [](const auto& left, const auto& right) {
                          return left.first > right.first || (left.first == right.first && left.second < right.second);
                      });
    auto output =
        ReferenceTensor::full({with_values ? std::size_t{2} : std::size_t{1}, static_cast<std::size_t>(top_k)}, 0.0F);
    for (std::size_t i = 0; i < count; ++i) {
        if (with_values) {
            output.at(i) = ranked[i].first;
            output.at(static_cast<std::size_t>(top_k) + i) = static_cast<float>(ranked[i].second);
        } else {
            output.at(i) = static_cast<float>(ranked[i].second);
        }
    }
    return output;
}

struct ForwardCase {
    const char* type;
    std::vector<Mat> inputs;
    std::vector<Mat> outputs;
    mformat::ParamDict params{};
    std::vector<Mat> weights{};
    float tolerance = MNET_TEST_FLOAT_TOLERANCE;
};

inline void expect_forward(const ForwardCase& test) {
    SCOPED_TRACE(test.type);
    auto layer = std::unique_ptr<Layer>(create_layer(test.type));
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->load_param(test.params), 0);
    if (!test.weights.empty()) {
        ASSERT_EQ(layer->load_model(mformat::ModelBinFromMatArray(test.weights.data())), 0);
    }
    const auto opt = Option{};
    ASSERT_EQ(layer->create_pipeline(opt), 0);
    auto outputs = std::vector<Mat>(test.outputs.size());
    const auto single_blob = layer->one_blob_only;
    if (single_blob) {
        ASSERT_EQ(test.inputs.size(), 1);
        ASSERT_EQ(outputs.size(), 1);
    }
    const auto status = single_blob ? layer->forward(test.inputs.front(), outputs.front(), opt)
                                    : layer->forward(test.inputs, outputs, opt);
    EXPECT_EQ(layer->destroy_pipeline(opt), 0);
    ASSERT_EQ(status, 0);
    ASSERT_EQ(outputs.size(), test.outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        SCOPED_TRACE(i);
        expect_tensor(outputs[i], test.outputs[i], test.tolerance);
    }
}

inline void expect_rejected(const char* type, const std::vector<Mat>& inputs, const mformat::ParamDict& params = {}) {
    SCOPED_TRACE(type);
    auto layer = std::unique_ptr<Layer>(create_layer(type));
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->load_param(params), 0);
    auto outputs = std::vector<Mat>(1);
    const auto opt = Option{};
    ASSERT_EQ(layer->create_pipeline(opt), 0);
    const auto single_blob = layer->one_blob_only;
    if (single_blob) {
        ASSERT_EQ(inputs.size(), 1);
    }
    const auto status =
        single_blob ? layer->forward(inputs.front(), outputs.front(), opt) : layer->forward(inputs, outputs, opt);
    EXPECT_EQ(layer->destroy_pipeline(opt), 0);
    EXPECT_NE(status, 0);
}

template<typename T>
inline mformat::ParamDict params(std::initializer_list<std::pair<std::int32_t, T>> entries) {
    auto dict = mformat::ParamDict{};
    for (const auto& [key, value] : entries) {
        dict.set(key, value);
    }
    return dict;
}

}  // namespace mnet::test
