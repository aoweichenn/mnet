// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_LAYER_HPP
#define MNET_LAYER_HPP

#include <mgraph/layer.hpp>
#include <mnet/mat.hpp>
#include <mnet/option.hpp>

namespace mnet {

using Layer = mgraph::Layer<Mat, Option>;

}  // namespace mnet

#endif  // MNET_LAYER_HPP
