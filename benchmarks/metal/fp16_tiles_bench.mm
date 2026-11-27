// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Would a different kernel class beat the Q2_0 GEMV for a single-token decode?
// The candidate is "dequantise to fp16 tiles and use simdgroup_matrix" (the
// prefill shape llama.cpp uses). For batch 1 that has to be a loss on paper:
// Q2_0 stores 18 bytes per 64 weights (0.281 B/weight) while fp16 stores
// 2 B/weight, i.e. seven times the traffic with no reuse to amortise it. This
// bench measures the real numbers instead of trusting that argument:
//
//   q2_0   the production shape (one warp per row, fp32 activation)
//   fp16   the same GEMV against a dequantised fp16 copy of the same tensor
//          (one warp per row, half4 loads widened to fp32)
//
// It also times the one-off dequantisation, so the batched case can be
// estimated: cost(n tokens) = dequant + n * f16_read versus n * q2_0_read.
//
// Measured on the M4 Pro, blk.0.ffn_gate 17408x5120 (one warp per row, warm):
//   q2_0   0.470 ms   53.4 GB/s of weight bytes   189.7 Gweight/s
//   fp16   0.844 ms  211.3 GB/s of weight bytes   105.7 Gweight/s
//   one-off Q2_0 -> fp16 dequantisation            2.95 ms
// So the fp16 form reads 7.1x the bytes, gets 2.2x the raw bandwidth and still
// delivers 1.8x fewer weights per second: for a single-token decode the tile
// class cannot win, and its one-off dequantisation alone costs 6 token
// equivalents. It would only pay for prefill, where one dequantisation feeds
// many tokens. The same arithmetic kills simdgroup_matrix for batch 1: a
// one-column vector wastes 7/8 of every 8x8 tile while reading the same 2 bytes
// per weight, so the kernel is not wired into the fast path.
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -x objective-c++ -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/fp16_tiles_bench.mm -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o fp16_tiles_bench

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

namespace {

const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// One warp per row, fp32 activation: the production shape.
kernel void mv_q2_0(device const uchar* w      [[buffer(0)]],
                    device const float* x      [[buffer(1)]],
                    device float* out          [[buffer(3)]],
                    constant uint& blocks      [[buffer(4)]],
                    constant float& scale      [[buffer(5)]],
                    uint tgid                  [[threadgroup_position_in_grid]],
                    uint lane                  [[thread_position_in_threadgroup]]) {
    device const uchar* row = w + (ulong)tgid * blocks * 18u;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += 32u) {
        device const uchar* block = row + (ulong)i * 18u;
        const float dx = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
        for (uint k = 0; k < 2u; ++k) {
            device const float* xy = x + (ulong)(2u * i + k) * 32u;
            device const uchar* q = block + 2u + k * 8u;
            float sum = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                const uchar byte = q[j];
                sum += float(int(byte & 3u) - 1) * xy[j * 4u + 0u];
                sum += float(int((byte >> 2u) & 3u) - 1) * xy[j * 4u + 1u];
                sum += float(int((byte >> 4u) & 3u) - 1) * xy[j * 4u + 2u];
                sum += float(int((byte >> 6u) & 3u) - 1) * xy[j * 4u + 3u];
            }
            acc = fma(dx, sum, acc);
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[tgid] = scale * total;
    }
}

// One warp per row over an fp16 copy: half4 loads, fp16 products accumulated in
// fp32 pairs. This is the cheapest form of the "fp16 tile" idea that still does
// the same math; a simdgroup_matrix version would waste 7/8 of each tile on a
// one-column vector and read the same 2 bytes per weight.
kernel void mv_f16(device const half* w           [[buffer(0)]],
                   device const float* x         [[buffer(1)]],
                   device float* out             [[buffer(3)]],
                   constant uint& blocks         [[buffer(4)]],
                   constant float& scale         [[buffer(5)]],
                   uint tgid                     [[threadgroup_position_in_grid]],
                   uint lane                     [[thread_position_in_threadgroup]]) {
    device const half* row = w + (ulong)tgid * blocks * 64u;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += 32u) {
        device const half* block = row + (ulong)i * 64u;
        for (uint k = 0; k < 2u; ++k) {
            device const float* xy = x + (ulong)(2u * i + k) * 32u;
            device const half4* packed = reinterpret_cast<device const half4*>(block + k * 32u);
            float sum = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                const half4 v = packed[j];
                sum += float(v.x) * xy[j * 4u + 0u];
                sum += float(v.y) * xy[j * 4u + 1u];
                sum += float(v.z) * xy[j * 4u + 2u];
                sum += float(v.w) * xy[j * 4u + 3u];
            }
            acc += sum;
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[tgid] = scale * total;
    }
}

// Q2_0 -> fp16, one thread per weight; the dequantisation a tile-based path
// pays once per tensor.
kernel void dequant_f16(device const uchar* w   [[buffer(0)]],
                        device half* out        [[buffer(1)]],
                        constant uint& blocks   [[buffer(2)]],
                        constant float& scale   [[buffer(3)]],
                        uint gid                [[thread_position_in_grid]]) {
    const uint block = gid / 64u;
    const uint within = gid % 64u;
    device const uchar* record = w + (ulong)block * 18u;
    const float dx = float(as_type<half>(ushort(record[0] | (ushort(record[1]) << 8))));
    const uchar byte = record[2u + within / 4u];
    const int code = int((byte >> (2u * (within % 4u))) & 3u) - 1;
    out[gid] = half(dx * scale * float(code));
}
)MSL";

}  // namespace

int main() {
    using mnet::gguf::Reader;
    Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto gate = reader.quantized(reader.tensor("blk.0.ffn_gate.weight"));
    const std::uint64_t rows = gate.rows;
    const std::uint64_t columns = gate.columns;
    const std::uint64_t blocks = columns / 64U;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                                   options:options
                                                     error:&error];
    if (library == nil) {
        std::printf("compile failed: %s\n", [[error description] UTF8String]);
        return 1;
    }
    const auto pipeline_for = [&](const char* name) {
        return [device newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:name]]
                                                     error:&error];
    };
    id<MTLComputePipelineState> mv_q2_0 = pipeline_for("mv_q2_0");
    id<MTLComputePipelineState> mv_f16 = pipeline_for("mv_f16");
    id<MTLComputePipelineState> dequant = pipeline_for("dequant_f16");

    id<MTLBuffer> q2_0 = [device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(gate.data())
                                                   length:static_cast<NSUInteger>(rows * blocks * 18U)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    id<MTLBuffer> f16 = [device newBufferWithLength:static_cast<NSUInteger>(rows * columns * sizeof(std::uint16_t))
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:static_cast<NSUInteger>(rows * sizeof(float))
                                            options:MTLResourceStorageModeShared];
    std::vector<float> activation(columns);
    std::uint32_t state = 3U;
    for (auto& value : activation) {
        state = state * 1664525U + 1013904223U;
        value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
    }
    id<MTLBuffer> x = [device newBufferWithBytes:activation.data()
                                         length:columns * sizeof(float)
                                        options:MTLResourceStorageModeShared];

    const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
    const float scale = gate.scale;
    const std::uint64_t q2_0_bytes = rows * blocks * 18U;
    const std::uint64_t f16_bytes = rows * columns * sizeof(std::uint16_t);

    // One-off dequantisation cost (the price a tile-based path pays).
    double dequant_ms = 0.0;
    {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:dequant];
        [encoder setBuffer:q2_0 offset:0 atIndex:0];
        [encoder setBuffer:f16 offset:0 atIndex:1];
        [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:2];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(rows * columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const int reps = 3;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            id<MTLCommandBuffer> again = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [again computeCommandEncoder];
            [enc setComputePipelineState:dequant];
            [enc setBuffer:q2_0 offset:0 atIndex:0];
            [enc setBuffer:f16 offset:0 atIndex:1];
            [enc setBytes:&blocks32 length:sizeof(blocks32) atIndex:2];
            [enc setBytes:&scale length:sizeof(scale) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(rows * columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [again commit];
            [again waitUntilCompleted];
        }
        dequant_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() * 1e3 / reps;
    }

    const auto measure = [&](const char* label, id<MTLComputePipelineState> pipeline, id<MTLBuffer> weights,
                             std::uint64_t bytes) {
        const auto run = [&] {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weights offset:0 atIndex:0];
            [encoder setBuffer:x offset:0 atIndex:1];
            [encoder setBuffer:out offset:0 atIndex:3];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
        };
        for (int i = 0; i < 3; ++i) {
            run();
        }
        const int reps = 20;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            run();
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
        const double weight_count = static_cast<double>(rows) * static_cast<double>(columns);
        std::printf("  %-8s %7.3f ms  %7.2f GB/s  %6.1f Gweight/s\n", label, seconds * 1e3,
                    static_cast<double>(bytes) / seconds / 1e9, weight_count / seconds / 1e9);
    };

    std::printf("blk.0.ffn_gate %llu x %llu, Q2_0 %llu bytes -> fp16 %llu bytes (%.1fx)\n",
                static_cast<unsigned long long>(rows), static_cast<unsigned long long>(columns),
                static_cast<unsigned long long>(q2_0_bytes), static_cast<unsigned long long>(f16_bytes),
                static_cast<double>(f16_bytes) / static_cast<double>(q2_0_bytes));
    measure("q2_0", mv_q2_0, q2_0, q2_0_bytes);
    measure("fp16", mv_f16, f16, f16_bytes);
    std::printf("  one-off Q2_0 -> fp16 dequantisation: %.3f ms\n", dequant_ms);
    std::printf("  per token: q2_0 reads %.1f MB, fp16 reads %.1f MB (+%.1f MB dequant on the first token)\n",
                static_cast<double>(q2_0_bytes) / 1e6, static_cast<double>(f16_bytes) / 1e6,
                static_cast<double>(q2_0_bytes) / 1e6);
    return 0;
}
