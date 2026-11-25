// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_QUANTIZE_HPP
#define MNET_QUANTIZE_HPP

#include <mnet/mat.hpp>
#include <mnet/option.hpp>

namespace mnet {

// Int8 quantization hook. The fp32 runtime never calls it; it exists so the
// mlayer kernels can be instantiated with Int8=false.
inline void quantize_int8(const Mat&, Mat&, const Mat&, const Option&) {}

}  // namespace mnet

#endif  // MNET_QUANTIZE_HPP
