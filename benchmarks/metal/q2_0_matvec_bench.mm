// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Standalone Metal benchmark for the Q2_0 matvec: packs a real tensor into
// 24-byte records (16 code bytes + half scale), runs a llama-style kernel
// (R rows per threadgroup, L lanes per row, simd_sum reduction) and reports
// GB/s plus the mismatch against the CPU dot. Not part of the CMake build;
// build on the Mac with:
//
//   clang++ -std=c++20 -O2 -x objective-c++ -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/q2_0_matvec_bench.mm -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o q2_0_bench
//
// Measured on the M4 Pro (Bonsai 27B Q2_0):
//   blk.0.attn_gate (6144x5120)   8.8 MB   17 GB/s
//   blk.0.ffn_gate (17408x5120)  25.1 MB   21 GB/s
//   token_embd     (248320x5120) 357 MB    99 GB/s
// i.e. the kernel reaches llama.cpp's class once the tensor is large; the
// per-layer tensors are latency-bound because each submission only has a few
// MB of work (an empty submit+wait on this GPU is 0.108 ms).

#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

static const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void q2_0_f32(device const uchar* w     [[buffer(0)]],
                     device const float* qf    [[buffer(1)]],
                     device const float* qd    [[buffer(2)]],
                     device float* out         [[buffer(3)]],
                     constant uint& blocks     [[buffer(4)]],
                     constant float& scale     [[buffer(5)]],
                     constant uint& rows       [[buffer(7)]],
                     constant uint& rows_per_group [[buffer(6)]],
                     constant uint& lanes          [[buffer(8)]],
                     uint tgid                 [[threadgroup_position_in_grid]],
                     uint tid                  [[thread_position_in_threadgroup]],
                     uint tgsize               [[threads_per_threadgroup]]) {
    const uint lane = tid % lanes;
    const uint row = tgid * rows_per_group + tid / lanes;
    if (row >= rows) {
        return;
    }
    const ulong base = (ulong)row * blocks * 24u;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += lanes) {
        const ulong lo = *reinterpret_cast<device const ulong*>(w + base + (ulong)i * 24u);
        const ulong hi = *reinterpret_cast<device const ulong*>(w + base + (ulong)i * 24u + 8u);
        const float dx = float(*reinterpret_cast<device const half*>(w + base + (ulong)i * 24u + 16u));
        for (uint k = 0; k < 2u; ++k) {
            const float d = dx * qd[2u * i + k];
            const ulong word = k == 0u ? lo : hi;
            device const float* q = qf + (ulong)(2u * i + k) * 32u;
            float s = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                const uint byte = uint((word >> (8u * j)) & 0xFFu);
                s = fma(float(int((byte >> 0) & 3u) - 1), q[j * 4u + 0u], s);
                s = fma(float(int((byte >> 2) & 3u) - 1), q[j * 4u + 1u], s);
                s = fma(float(int((byte >> 4) & 3u) - 1), q[j * 4u + 2u], s);
                s = fma(float(int((byte >> 6) & 3u) - 1), q[j * 4u + 3u], s);
            }
            acc = fma(d, s, acc);
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[row] = scale * total;
    }
}
)MSL";

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:options error:&error];
    if (library == nil) {
        std::printf("compile failed: %s\n", [[error description] UTF8String]);
        return 1;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"q2_0_f32"] error:&error];

    mnet::gguf::Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    for (const char* name : {"blk.0.attn_gate.weight", "blk.0.ffn_gate.weight", "token_embd.weight"}) {
        const auto matrix = reader.quantized(reader.tensor(name));
        const std::uint64_t blocks = matrix.columns / 64U;
        const std::uint64_t row_bytes = blocks * 18U;
        std::vector<std::uint8_t> packed(matrix.rows * blocks * 24U);
        for (std::uint64_t r = 0; r < matrix.rows; ++r) {
            for (std::uint64_t b = 0; b < blocks; ++b) {
                const std::uint8_t* src = matrix.blocks.data() + r * row_bytes + b * 18U;
                std::uint8_t* dst = packed.data() + (r * blocks + b) * 24U;
                std::memcpy(dst, src + 2U, 16U);
                dst[16] = src[0];
                dst[17] = src[1];
            }
        }
        std::vector<float> input(matrix.columns);
        std::uint32_t state = 3U;
        for (auto& value : input) {
            state = state * 1664525U + 1013904223U;
            value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
        }
        const auto q80 = mnet::gguf::make_q8_0_blocks(input.data(), matrix.columns);
        std::vector<float> qf(q80.qs.size());
        for (std::size_t i = 0; i < qf.size(); ++i) {
            qf[i] = static_cast<float>(q80.qs[i]);
        }
        std::vector<float> out(matrix.rows);
        id<MTLBuffer> wbuf = [device newBufferWithBytes:packed.data()
                                                length:packed.size()
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> qfbuf = [device newBufferWithBytes:qf.data()
                                                  length:qf.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> qdbuf = [device newBufferWithBytes:q80.d.data()
                                                  length:q80.d.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> obuf = [device newBufferWithLength:out.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
        const float scale = matrix.scale;
        const auto run = [&] {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:wbuf offset:0 atIndex:0];
            [encoder setBuffer:qfbuf offset:0 atIndex:1];
            [encoder setBuffer:qdbuf offset:0 atIndex:2];
            [encoder setBuffer:obuf offset:0 atIndex:3];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
            const std::uint32_t rows32 = static_cast<std::uint32_t>(matrix.rows);
            const std::uint32_t rows32 = static_cast<std::uint32_t>(matrix.rows);
            const std::uint32_t rows_per_group = 4U;
            const std::uint32_t lanes = 8U;
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:7];
            [encoder setBytes:&rows_per_group length:sizeof(rows_per_group) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake((matrix.rows + rows_per_group - 1U) / rows_per_group, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(lanes * rows_per_group, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
        };
        run();
        std::memcpy(out.data(), obuf.contents, out.size() * sizeof(float));
        std::uint64_t bad = 0;
        double worst = 0.0;
        for (std::uint64_t r = 0; r < matrix.rows; ++r) {
            const float cpu = matrix.scale * mnet::gguf::dot_q2_0_q8_0(matrix.blocks.data() + r * row_bytes, q80, blocks);
            if (cpu != out[r]) {
                ++bad;
                worst = std::max(worst, std::fabs(static_cast<double>(cpu) - static_cast<double>(out[r])) /
                                            std::max(1e-30, std::fabs(static_cast<double>(cpu))));
            }
        }
        const int reps = 20;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            run();
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
        const double bytes = static_cast<double>(matrix.rows) * static_cast<double>(row_bytes);
        std::printf("%-24s rows %6llu  %8.3f ms  %6.2f GB/s  mismatched %llu worst %.2e\n", name,
                    static_cast<unsigned long long>(matrix.rows), seconds * 1e3, bytes / seconds / 1e9,
                    static_cast<unsigned long long>(bad), worst);
    }
    return 0;
}
