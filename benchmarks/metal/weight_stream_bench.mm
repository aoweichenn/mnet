// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Roofline for the forward: how fast can this GPU walk the Q2_0 weight stream
// with the *same* access pattern the mat-mat uses, when nothing else happens?
// The mat-mat's decode reads a 64-row tile by having two threads per row pull
// eight code bytes per k-step; this kernel does exactly that and nothing else
// (no mma, no staging), so its GB/s is the ceiling the forward's aggregate
// 7.67 GB / 463 ms = 16.6 GB/s can be compared against.
//
// Build on the Mac from the repository root:
//   clang++ -std=c++20 -O2 -x objective-c++ -isysroot \
//     /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/weight_stream_bench.mm -x none -framework Metal \
//     -framework Foundation -lobjc -o weight_stream_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

static const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Two threads per row, eight code bytes each, 64 rows per threadgroup: the
// mat-mat's weight read, with the decode reduced to a byte sum so the kernel is
// purely a reader.
kernel void weight_stream(device const uchar* w        [[buffer(0)]],
                          constant uint& row_bytes     [[buffer(1)]],
                          constant uint& blocks        [[buffer(2)]],
                          constant uint& rows          [[buffer(3)]],
                          device float* out            [[buffer(4)]],
                          uint tgid                    [[threadgroup_position_in_grid]],
                          uint tiitg                   [[thread_index_in_threadgroup]]) {
    const uint row = tgid * 64u + tiitg / 2u;
    const uint part = tiitg % 2u;
    if (row >= rows) {
        return;
    }
    device const uchar* rowp = w + (ulong)row * row_bytes;
    uint acc = 0u;
    for (uint i = 0; i < blocks; ++i) {
        device const uchar* q = rowp + (ulong)i * 18u + 2u + part * 8u;
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; ++j) {
            acc += q[j];
        }
    }
    if (tiitg == 0u) {
        out[tgid] = float(acc);
    }
}
)MSL";

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                                  options:nil
                                                    error:&error];
    if (library == nil) {
        std::fprintf(stderr, "compile failed: %s\n", [[error description] UTF8String]);
        return 1;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"weight_stream"] error:&error];
    if (pipeline == nil) {
        std::fprintf(stderr, "pipeline failed: %s\n", [[error description] UTF8String]);
        return 1;
    }

    // The model's geometry: 1440-byte rows (5120 columns of Q2_0), and enough
    // rows to walk the model's 7.67 GB.
    const std::uint32_t columns = 5120u;
    const std::uint32_t row_bytes = columns / 64u * 18u;
    const std::uint32_t blocks = columns / 64u;
    const std::uint64_t target = 7673194848ULL;
    const std::uint32_t rows = static_cast<std::uint32_t>(target / row_bytes);
    const std::uint64_t bytes = static_cast<std::uint64_t>(rows) * row_bytes;
    std::vector<std::uint8_t> host(bytes, 0x5Au);
    id<MTLBuffer> w = [device newBufferWithBytes:host.data()
                                          length:static_cast<NSUInteger>(bytes)
                                         options:MTLResourceStorageModeShared];
    std::vector<std::uint8_t>().swap(host);
    id<MTLBuffer> out = [device newBufferWithLength:(rows / 64u + 1u) * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    if (w == nil || out == nil) {
        std::fprintf(stderr, "allocation failed for %llu bytes\n", static_cast<unsigned long long>(bytes));
        return 1;
    }

    const auto encode = [&](id<MTLCommandBuffer> command) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:w offset:0 atIndex:0];
        [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:1];
        [encoder setBytes:&blocks length:sizeof(blocks) atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBuffer:out offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 63u) / 64u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    };
    // Warm up once, then three timed passes.
    {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
    }
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 3; ++pass) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 3.0;
    const double gbs = static_cast<double>(bytes) / seconds / 1e9;
    std::printf("%.3f GB walked in %.1f ms = %.1f GB/s (warm)\n", static_cast<double>(bytes) / 1e9,
                seconds * 1000.0, gbs);
    std::printf("model aggregate: 7.67 GB / 463 ms = 16.6 GB/s\n");
    return 0;
}
