// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <operator_test.hpp>

namespace {

using mnet::test::expect_forward;
using mnet::test::params;
using mnet::test::tensor;
using mnet::test::vector;

// assertions-in-helper: expect_forward
TEST(Deconvolution, AccumulatesOverlappingKernelContributions) {
    expect_forward({"Deconvolution",
                    {tensor({3, 2, 1, 1, 1}, {2, 3})},
                    {tensor({3, 3, 2, 1, 1}, {2, 7, 6, 6, 17, 12})},
                    params<std::int32_t>({{0, 1}, {1, 2}, {6, 4}}),
                    {vector({1, 2, 3, 4})}});
}

// assertions-in-helper: expect_forward
TEST(DeconvolutionDepthWise, KeepsGroupsIndependent) {
    expect_forward({"DeconvolutionDepthWise",
                    {tensor({3, 2, 1, 1, 2}, {2, 3, -1, 4})},
                    {tensor({3, 3, 1, 1, 2}, {2, 7, 6, 3, -13, 4})},
                    params<std::int32_t>({{0, 2}, {1, 2}, {11, 1}, {6, 4}, {7, 2}}),
                    {vector({1, 2, -3, 1})}});
}

// assertions-in-helper: expect_forward
TEST(Deconvolution1D, LeavesStrideGapsAtBiasValue) {
    expect_forward({"Deconvolution1D",
                    {tensor({2, 2, 1}, {2, 3})},
                    {tensor({2, 5, 1}, {3, 5, 1, 4, 7})},
                    params<std::int32_t>({{0, 1}, {1, 2}, {3, 3}, {5, 1}, {6, 2}}),
                    {vector({1, 2}), vector({1})}});
}

// assertions-in-helper: expect_forward
TEST(DeconvolutionDepthWise1D, KeepsGroupsIndependent) {
    expect_forward({"DeconvolutionDepthWise1D",
                    {tensor({2, 2, 2}, {2, 3, -1, 4})},
                    {tensor({2, 3, 2}, {2, 7, 6, 3, -13, 4})},
                    params<std::int32_t>({{0, 2}, {1, 2}, {6, 4}, {7, 2}}),
                    {vector({1, 2, -3, 1})}});
}

// assertions-in-helper: expect_forward
TEST(Deconvolution3D, AccumulatesOverlapAlongDepth) {
    expect_forward({"Deconvolution3D",
                    {tensor({4, 1, 1, 2, 1}, {2, 3})},
                    {tensor({4, 1, 1, 3, 1}, {2, 7, 6})},
                    params<std::int32_t>({{0, 1}, {1, 1}, {21, 2}, {6, 2}}),
                    {vector({1, 2})}});
}

// assertions-in-helper: expect_forward
TEST(DeconvolutionDepthWise3D, KeepsDepthVolumesIndependent) {
    expect_forward({"DeconvolutionDepthWise3D",
                    {tensor({4, 1, 1, 2, 2}, {2, 3, -1, 4})},
                    {tensor({4, 1, 1, 3, 2}, {2, 7, 6, 3, -13, 4})},
                    params<std::int32_t>({{0, 2}, {1, 1}, {21, 2}, {6, 4}, {7, 2}}),
                    {vector({1, 2, -3, 1})}});
}

}  // namespace
