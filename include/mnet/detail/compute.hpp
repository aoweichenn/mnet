// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_DETAIL_COMPUTE_HPP
#define MNET_DETAIL_COMPUTE_HPP

// Device-neutral compute entry points. Callers (Reader) describe the problem
// and never learn whether a CPU or a GPU executed it: a backend registers
// itself at start-up through set_matvec_q8_0_backend() and matvec_q8_0() picks
// it when it can handle the type, otherwise it runs the CPU loop. Adding a new
// device (Metal, CUDA, ...) only adds one registration in its own translation
// unit; no model code changes.

#include <cstdint>

namespace mnet::detail::compute {

// Executes y = scale * W x for a GGML type that uses Q8_0 activations.
// Returns false when this device cannot handle the type.
using MatvecQ8_0Fn = bool (*)(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows,
                              std::uint64_t columns, std::uint64_t row_bytes, float scale,
                              const std::int8_t* q8, const float* q8_scales, float* out);

void set_matvec_q8_0_backend(MatvecQ8_0Fn backend);

// Implemented by a device translation unit when one is compiled in, otherwise
// the portable no-op in src/compute/backend.cpp.
void register_platform_backends();

// Always computes the full matrix-vector product.
void matvec_q8_0(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                 std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales, float* out);

}  // namespace mnet::detail::compute

#endif  // MNET_DETAIL_COMPUTE_HPP
