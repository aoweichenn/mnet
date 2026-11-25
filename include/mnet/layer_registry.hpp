// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_LAYER_REGISTRY_HPP
#define MNET_LAYER_REGISTRY_HPP

#include <cstdint>

#include <mnet/layer.hpp>

namespace mnet {

using LayerCreator = Layer* (*)();

// Create a built-in layer by param type name. Returns nullptr for unknown types.
[[nodiscard]] Layer* create_layer(const char* type);

// Create a built-in layer from the ncnn binary param type index.
[[nodiscard]] Layer* create_layer_by_index(std::int32_t type_index);

// Register or replace a type name. The creator owns the returned layer.
std::int32_t register_layer(const char* type, LayerCreator creator);

}  // namespace mnet

#endif  // MNET_LAYER_REGISTRY_HPP
