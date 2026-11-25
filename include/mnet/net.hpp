// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_NET_HPP
#define MNET_NET_HPP

#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include <mnet/layer.hpp>
#include <mnet/mat.hpp>
#include <mnet/option.hpp>

namespace mnet {

struct NetPrivate;

// Standalone CPU runtime: graph loading through mgraph/mformat and layer
// execution through mlayer. It never includes or links ncnn.
class Net {
public:
    Net();
    ~Net();

    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

    std::int32_t load_param(const char* param_path);
    std::int32_t load_model(const char* bin_path);
    std::int32_t load(const char* param_path, const char* bin_path);

    [[nodiscard]] Option& opt() { return this->opt_; }
    [[nodiscard]] const Option& opt() const { return this->opt_; }

    class Extractor {
    public:
        std::int32_t input(const char* blob_name, const Mat& in);
        std::int32_t input(std::int32_t blob_index, const Mat& in);
        std::int32_t extract(const char* blob_name, Mat& out);
        std::int32_t extract(std::int32_t blob_index, Mat& out);

    private:
        friend class Net;
        explicit Extractor(const Net* net);

        const Net* net_;
        std::vector<Mat> blob_mats_;
    };

    [[nodiscard]] Extractor create_extractor() const { return Extractor(this); }

    [[nodiscard]] std::vector<std::string> input_names() const;
    [[nodiscard]] std::vector<std::string> output_names() const;

private:
    Option opt_;
    std::unique_ptr<NetPrivate> d;
};

}  // namespace mnet

#endif  // MNET_NET_HPP
