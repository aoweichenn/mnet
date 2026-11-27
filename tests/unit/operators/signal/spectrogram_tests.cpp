// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Spectrogram, PreservesComplexPhaseAndConjugateSymmetry) {
    const auto input = vector({0, 1, 0, 0, 2, 0});
    const auto expected = tensor({3, 2, 2, 1, 4}, {1, 0, 2, 0, 0, -1, -2, 0, -1, 0, 2, 0, 0, 1, -2, 0});
    expect_forward({"Spectrogram", {input}, {expected}, params<std::int32_t>({{0, 4}, {2, 2}, {5, 0}, {8, 0}})});
}

// assertions-in-helper: expect_forward
TEST(InverseSpectrogram, ReconstructsIndependentSpectrumWithOverlapNormalization) {
    const auto spectrum = tensor({3, 2, 2, 1, 3}, {1, 0, 2, 0, 0, -1, -2, 0, -1, 0, 2, 0});
    expect_forward({"InverseSpectrogram",
                    {spectrum},
                    {vector({0, 1, 0, 0, 2, 0})},
                    params<std::int32_t>({{0, 4}, {1, 1}, {2, 2}, {5, 0}})});
}

}  // namespace
