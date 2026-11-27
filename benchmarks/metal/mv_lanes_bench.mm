// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// How many lanes per row should the Q2_0 matvec use? The fused SSM block runs
// its three matvecs at half the fused FFN's bandwidth, and those matvecs have
// fewer rows (10240, 6144, 5120) than the FFN's (17408), so the hypothesis was
// that one 32-lane warp per row leaves too little parallelism per row to hide
// the latency of reading weights from the file mapping.
//
// Measured on the M4 Pro, warm page cache, one submission:
//   ssm shapes  32 lanes 0.659 ms 49.2 GB/s | 64 lanes 0.433 ms 74.9 GB/s
//   ffn shapes  32 lanes 0.550 ms 91.2 GB/s | 64 lanes 0.548 ms 91.5 GB/s
// (a cold first run reads 0.884 / 0.615 ms, so warm up before trusting these).
// Two warps per row are worth 52% on the SSM shapes and nothing on the FFN's,
// yet in the model the SSM block only moved 1.413 -> 1.398 s and the change
// flipped the fifth top-5 token, so it was not kept. The matvecs in that block
// are evidently not the exposed critical path once the whole layer pipeline is
// running, even though the isolated shape claims otherwise.
//
// The bench runs each real tensor through two kernels in one submission:
//   mv32 : 32 lanes per row, one warp per threadgroup (the production shape)
//   mv64 : 64 lanes per row, two warps per threadgroup, one threadgroup-level
//          reduction per row
// Weights are aliased from the GGUF mapping exactly like the backend does, so
// the numbers are comparable to the in-model block times.
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -x objective-c++ -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/mv_lanes_bench.mm -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o mv_lanes_bench

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mnet/gguf/format.hpp>
#include <mnet/gguf/kernels.hpp>
#include <mnet/gguf/reader.hpp>

namespace {

const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline void decode_block(device const uchar* block, device const float* xy, thread float& sum) {
    const uchar* q = block;
    for (uint j = 0; j < 8u; ++j) {
        const uchar byte = q[j];
        sum += float(int(byte & 3u)) * xy[j * 4u + 0u];
        sum += float(int((byte >> 2u) & 3u)) * xy[j * 4u + 1u];
        sum += float(int((byte >> 4u) & 3u)) * xy[j * 4u + 2u];
        sum += float(int((byte >> 6u) & 3u)) * xy[j * 4u + 3u];
    }
}

inline float block_scale(device const uchar* block) {
    return float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
}

// One warp per row.
kernel void mv32(device const uchar* w        [[buffer(0)]],
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
        for (uint k = 0; k < 2u; ++k) {
            const float d = block_scale(block);
            float sum = 0.0f;
            decode_block(block + 2u + k * 8u, x + (ulong)(2u * i + k) * 32u, sum);
            acc = fma(d, sum, acc);
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[tgid] = scale * total;
    }
}

// Two warps per row: each lane handles half as many blocks, and the two
// partial sums meet in threadgroup memory.
kernel void mv64(device const uchar* w        [[buffer(0)]],
                 device const float* x        [[buffer(1)]],
                 device float* out            [[buffer(3)]],
                 constant uint& blocks        [[buffer(4)]],
                 constant float& scale        [[buffer(5)]],
                 uint tgid                    [[threadgroup_position_in_grid]],
                 uint lane                    [[thread_position_in_threadgroup]]) {
    threadgroup float partial[2];
    device const uchar* row = w + (ulong)tgid * blocks * 18u;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += 64u) {
        device const uchar* block = row + (ulong)i * 18u;
        for (uint k = 0; k < 2u; ++k) {
            const float d = block_scale(block);
            float sum = 0.0f;
            decode_block(block + 2u + k * 8u, x + (ulong)(2u * i + k) * 32u, sum);
            acc = fma(d, sum, acc);
        }
    }
    const float total = simd_sum(acc);
    if ((lane & 31u) == 0u) {
        partial[lane >> 5u] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        out[tgid] = scale * (partial[0] + partial[1]);
    }
}
)MSL";

struct Job {
    const char* name;
    const mnet::gguf::Reader::Quantized* weights;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t blocks;
};

}  // namespace

int main() {
    using mnet::gguf::Reader;
    Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto qkv = reader.quantized(reader.tensor("blk.0.attn_qkv.weight"));
    const auto gate = reader.quantized(reader.tensor("blk.0.attn_gate.weight"));
    const auto ssm_out = reader.quantized(reader.tensor("blk.0.ssm_out.weight"));
    const auto ffn_gate = reader.quantized(reader.tensor("blk.0.ffn_gate.weight"));
    const auto ffn_down = reader.quantized(reader.tensor("blk.0.ffn_down.weight"));

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
    id<MTLComputePipelineState> mv32 = pipeline_for("mv32");
    id<MTLComputePipelineState> mv64 = pipeline_for("mv64");

    // Alias the whole file like the backend does.
    if (reader.mapping() == nullptr) {
        std::printf("no mapping\n");
        return 1;
    }
    id<MTLBuffer> region = [device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(reader.mapping())
                                                     length:static_cast<NSUInteger>(reader.mapping_size())
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (region == nil) {
        std::printf("cannot alias the mapping\n");
        return 1;
    }

    const auto measure = [&](const char* label, const std::vector<Job>& jobs, id<MTLComputePipelineState> pipeline,
                             NSUInteger lanes) {
        std::uint64_t total_bytes = 0;
        std::uint64_t total_outs = 0;
        for (const auto& job : jobs) {
            total_bytes += job.rows * job.blocks * 18U;
            total_outs += job.rows;
        }
        id<MTLBuffer> out = [device newBufferWithLength:static_cast<NSUInteger>(total_outs * sizeof(float))
                                                options:MTLResourceStorageModeShared];
        // Activations: the widest job decides the buffer.
        std::uint64_t widest = 0;
        for (const auto& job : jobs) {
            widest = std::max(widest, job.columns);
        }
        std::vector<float> activation(widest);
        std::uint32_t state = 9U;
        for (auto& value : activation) {
            state = state * 1664525U + 1013904223U;
            value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
        }
        // Using the raw values as the "activation" is fine for a bandwidth test.
        id<MTLBuffer> x = [device newBufferWithBytes:activation.data()
                                             length:widest * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        const auto run = [&] {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            std::uint64_t offset = 0;
            for (const auto& job : jobs) {
                const auto job_offset = static_cast<std::uint64_t>(job.weights->data() - reader.mapping());
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:region offset:static_cast<NSUInteger>(job_offset) atIndex:0];
                [encoder setBuffer:x offset:0 atIndex:1];
                [encoder setBuffer:out offset:static_cast<NSUInteger>(offset * sizeof(float)) atIndex:3];
                const std::uint32_t blocks32 = static_cast<std::uint32_t>(job.blocks);
                [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
                const float scale = job.weights->scale;
                [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(job.rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
                offset += job.rows;
            }
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
        std::printf("  %-22s %6.3f ms  %6.2f GB/s\n", label, seconds * 1e3,
                    static_cast<double>(total_bytes) / seconds / 1e9);
    };

    std::vector<Job> ssm = {{"qkv", &qkv, qkv.rows, qkv.columns, qkv.columns / 64U},
                            {"gate", &gate, gate.rows, gate.columns, gate.columns / 64U},
                            {"ssm_out", &ssm_out, ssm_out.rows, ssm_out.columns, ssm_out.columns / 64U}};
    std::vector<Job> ffn = {{"ffn_gate", &ffn_gate, ffn_gate.rows, ffn_gate.columns, ffn_gate.columns / 64U},
                            {"ffn_down", &ffn_down, ffn_down.rows, ffn_down.columns, ffn_down.columns / 64U}};
    std::printf("SSM block shapes (qkv %llu x %llu, gate %llu, ssm_out %llu x %llu)\n",
                static_cast<unsigned long long>(qkv.rows), static_cast<unsigned long long>(qkv.columns),
                static_cast<unsigned long long>(gate.rows), static_cast<unsigned long long>(ssm_out.rows),
                static_cast<unsigned long long>(ssm_out.columns));
    measure("ssm 32 lanes/row", ssm, mv32, 32);
    measure("ssm 64 lanes/row", ssm, mv64, 64);
    std::printf("FFN shapes for reference\n");
    measure("ffn 32 lanes/row", ffn, mv32, 32);
    measure("ffn 64 lanes/row", ffn, mv64, 64);
    return 0;
}
