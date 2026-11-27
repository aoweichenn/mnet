// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Is there a win left in amortising the Q2_0 decode over the 32 tokens of a
// prefill? The GEMV decodes 64 weights per 18-byte block for one token and
// repeats that for every token; a prefill kernel can decode the block once and
// multiply it with all 32 activations, cutting instructions per
// (weight, token) from ~5 to ~2.
//
// This bench measures exactly that, on blk.0.ffn_gate (17408 x 5120):
//   gemv   32 separate GEMVs, the current shape (one warp per row)
//   prefill one warp per row, lane = token: 64 codes decoded per block into
//           threadgroup memory, then each lane multiplies its token
// Both read the same weights and the same 32 x 5120 activation matrix, so the
// difference is the decode cost alone.
//
// Measured on the M4 Pro (blk.0.ffn_gate 17408x5120, 32 tokens):
//   gemv     6.759 ms   422.0 Gweight/s  (0.211 ms per token matvec, 119 GB/s)
//   prefill  8.847 ms   322.4 Gweight/s  (0.76x = slower)
// So amortising the decode over the tokens is *not* the win: the batched GEMV
// sequence already pipelines 32 dispatches of the same shape, while the prefill
// kernel puts 32 tokens in one warp and pays threadgroup barriers per block.
// The number worth keeping is the 119 GB/s: a run of same-sized independent
// dispatches reaches ~119 GB/s on the same tensor where the three-dispatch
// fused FFN block only gets 93.8, because the swiglu/hadamard chain between the
// up and down matvecs breaks the pipeline. That points at batching the FFN over
// the layer's tokens in one submission, not at a different kernel.
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -x objective-c++ -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/prefill_amortise_bench.mm -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o prefill_amortise_bench

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

namespace {

constexpr std::uint32_t kTokens = 32;

const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Current shape: one warp per row, one token.
kernel void mv_gemv(device const uchar* w        [[buffer(0)]],
                    device const float* x        [[buffer(1)]],
                    device float* out            [[buffer(3)]],
                    constant uint& blocks        [[buffer(4)]],
                    constant float& scale        [[buffer(5)]],
                    uint tgid                    [[threadgroup_position_in_grid]],
                    uint lane                    [[thread_position_in_threadgroup]]) {
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

// Prefill shape: one warp per row, lane = token. Each block's 64 codes are
// decoded once into threadgroup memory and used by all 32 tokens.
kernel void mv_prefill(device const uchar* w           [[buffer(0)]],
                       device const float* x           [[buffer(1)]],
                       device float* out               [[buffer(3)]],
                       constant uint& blocks           [[buffer(4)]],
                       constant float& scale           [[buffer(5)]],
                       constant uint& columns          [[buffer(6)]],
                       uint tgid                       [[threadgroup_position_in_grid]],
                       uint lane                       [[thread_position_in_threadgroup]]) {
    threadgroup float codes[64];
    device const uchar* row = w + (ulong)tgid * blocks * 18u;
    device const float* xa = x + (ulong)lane * columns;
    float acc = 0.0f;
    for (uint i = 0; i < blocks; ++i) {
        device const uchar* block = row + (ulong)i * 18u;
        const float dx = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
        // 64 codes, two per lane.
        for (uint part = 0; part < 2u; ++part) {
            const uint index = part * 32u + lane;
            const uchar byte = block[2u + index / 4u];
            codes[index] = float(int((byte >> (2u * (index % 4u))) & 3u) - 1) * dx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float* xy = xa + (ulong)i * 64u;
        float sum = 0.0f;
        for (uint j = 0; j < 64u; j += 8u) {
            sum = fma(codes[j + 0u], xy[j + 0u], sum);
            sum = fma(codes[j + 1u], xy[j + 1u], sum);
            sum = fma(codes[j + 2u], xy[j + 2u], sum);
            sum = fma(codes[j + 3u], xy[j + 3u], sum);
            sum = fma(codes[j + 4u], xy[j + 4u], sum);
            sum = fma(codes[j + 5u], xy[j + 5u], sum);
            sum = fma(codes[j + 6u], xy[j + 6u], sum);
            sum = fma(codes[j + 7u], xy[j + 7u], sum);
        }
        acc += sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[(ulong)tgid * 32u + lane] = scale * acc;
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
    id<MTLComputePipelineState> gemv = pipeline_for("mv_gemv");
    id<MTLComputePipelineState> prefill = pipeline_for("mv_prefill");

    id<MTLBuffer> weights = [device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(gate.data())
                                                      length:static_cast<NSUInteger>(rows * blocks * 18U)
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    std::vector<float> activation(kTokens * columns);
    std::uint32_t state = 5U;
    for (auto& value : activation) {
        state = state * 1664525U + 1013904223U;
        value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
    }
    id<MTLBuffer> x = [device newBufferWithBytes:activation.data()
                                          length:activation.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_gemv = [device newBufferWithLength:static_cast<NSUInteger>(rows * kTokens * sizeof(float))
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_prefill = [device newBufferWithLength:static_cast<NSUInteger>(rows * kTokens * sizeof(float))
                                                    options:MTLResourceStorageModeShared];
    const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
    const std::uint32_t columns32 = static_cast<std::uint32_t>(columns);
    const float scale = gate.scale;

    const auto run_gemv = [&] {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:gemv];
        for (std::uint32_t t = 0; t < kTokens; ++t) {
            [encoder setBuffer:weights offset:0 atIndex:0];
            [encoder setBuffer:x offset:static_cast<NSUInteger>(t * columns * sizeof(float)) atIndex:1];
            [encoder setBuffer:out_gemv offset:static_cast<NSUInteger>(t * rows * sizeof(float)) atIndex:3];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
    };
    const auto run_prefill = [&] {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:prefill];
        [encoder setBuffer:weights offset:0 atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:out_prefill offset:0 atIndex:3];
        [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
    };

    const auto measure = [&](const char* label, const std::function<void()>& run, id<MTLBuffer> out) {
        for (int i = 0; i < 2; ++i) {
            run();
        }
        const int reps = 10;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            run();
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
        const double weight_bytes = static_cast<double>(rows * blocks * 18U);
        const double pairs = static_cast<double>(rows) * static_cast<double>(columns) * kTokens;
        std::printf("  %-8s %7.3f ms for %u tokens  %7.2f GB/s  %6.1f Gweight/s\n", label, seconds * 1e3, kTokens,
                    weight_bytes / seconds / 1e9, pairs / seconds / 1e9);
        (void)out;
        return seconds;
    };

    std::printf("blk.0.ffn_gate %llu x %llu, %u tokens\n", static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(columns), kTokens);
    const double gemv_s = measure("gemv", run_gemv, out_gemv);
    const double prefill_s = measure("prefill", run_prefill, out_prefill);

    // Same tensors, so the two outputs must agree; report the worst relative gap.
    const auto* a = static_cast<const float*>(out_gemv.contents);
    const auto* b = static_cast<const float*>(out_prefill.contents);
    double worst = 0.0;
    for (std::uint64_t i = 0; i < rows * kTokens; ++i) {
        // out_gemv is token-major, out_prefill row-major.
        const std::uint64_t t = i / rows;
        const std::uint64_t r = i % rows;
        const double ref = a[i];
        const double other = b[r * kTokens + t];
        worst = std::max(worst, std::abs(ref - other) / std::max(1e-12, std::abs(ref)));
    }
    std::printf("  worst relative difference between the two shapes: %.3e\n", worst);
    std::printf("  prefill speedup: %.2fx\n", gemv_s / prefill_s);
    return 0;
}
