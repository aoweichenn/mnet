// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Isolated kernel bench for the Q2_0 matvec, on the real FFN tensors of one
// layer and in the shape the model uses: gate/up/down encoded into one command
// buffer with one wait. Per-tensor timing hides the decode cost behind the
// per-submission latency, so this measures the block instead.
//
// Two kernels share the 18-byte Q2_0 record:
//   mv_int    : integer codes dotted with i8 activations (the production
//               kernel: one byte load plus a shift/mask/sub per weight)
//   mv_f32act : the same decode, but the activation is dequantised to fp32
//               once per submission, so a weight costs shifts + one fma and no
//               activation load
//
// Build on the Mac from the repository root:
//
//   clang++ -std=c++20 -O2 -x objective-c++ -I include \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/q2_0_kernel_bench.mm -x none build-metal/libmnet.a \
//     build-metal/mgraph/libmgraph.a build-metal/mformat/libmformat.a \
//     build-metal/mformat/mmat/libmmat.a -framework Metal -framework Foundation \
//     -lobjc -o q2_0_kernel_bench

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

constexpr std::uint32_t kLanes = 32;

const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Production integer kernel: 18-byte record, i8 activation.
kernel void mv_int(device const uchar* w      [[buffer(0)]],
                   device const char* q8      [[buffer(1)]],
                   device const float* qd     [[buffer(2)]],
                   device float* out          [[buffer(3)]],
                   constant uint& blocks      [[buffer(4)]],
                   constant float& scale      [[buffer(5)]],
                   constant uint& row_base    [[buffer(6)]],
                   uint tgid                  [[threadgroup_position_in_grid]],
                   uint lane                  [[thread_position_in_threadgroup]]) {
    device const uchar* row = w + (ulong)tgid * blocks * 18u;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += 32u) {
        device const uchar* block = row + (ulong)i * 18u;
        const float dx = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
        for (uint k = 0; k < 2u; ++k) {
            const float d = dx * qd[2u * i + k];
            device const char* qy = q8 + (ulong)(2u * i + k) * 32u;
            device const uchar* q = block + 2u + k * 8u;
            int sum = 0;
            for (uint j = 0; j < 8u; ++j) {
                const uchar byte = q[j];
                sum += (int((byte >> 0) & 3u) - 1) * int(qy[j * 4u + 0u]) +
                       (int((byte >> 2) & 3u) - 1) * int(qy[j * 4u + 1u]) +
                       (int((byte >> 4) & 3u) - 1) * int(qy[j * 4u + 2u]) +
                       (int((byte >> 6) & 3u) - 1) * int(qy[j * 4u + 3u]);
            }
            acc = fma(d, float(sum), acc);
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[row_base + tgid] = scale * total;
    }
}

// Dequantise the activation once per submission: x = q8 * block scale.
kernel void dequant_q8(device const char* q8    [[buffer(0)]],
                       device const float* qd   [[buffer(1)]],
                       device float* out        [[buffer(2)]],
                       uint gid                 [[thread_position_in_grid]]) {
    out[gid] = float(q8[gid]) * qd[gid / 32u];
}

// Same decode, fp32 activation: no activation load and no per-block scale mul
// in the inner loop; the weight scale is folded once per 64-value block.
kernel void mv_f32act(device const uchar* w         [[buffer(0)]],
                      device const float* x         [[buffer(1)]],
                      device float* out             [[buffer(3)]],
                      constant uint& blocks         [[buffer(4)]],
                      constant float& scale         [[buffer(5)]],
                      constant uint& row_base       [[buffer(6)]],
                      uint tgid                     [[threadgroup_position_in_grid]],
                      uint lane                     [[thread_position_in_threadgroup]]) {
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
                sum += float(int((byte >> 0) & 3u) - 1) * xy[j * 4u + 0u];
                sum += float(int((byte >> 2) & 3u) - 1) * xy[j * 4u + 1u];
                sum += float(int((byte >> 4) & 3u) - 1) * xy[j * 4u + 2u];
                sum += float(int((byte >> 6) & 3u) - 1) * xy[j * 4u + 3u];
            }
            acc = fma(dx, sum, acc);
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0u) {
        out[row_base + tgid] = scale * total;
    }
}
)MSL";

}  // namespace

int main() {
    using mnet::gguf::Reader;
    Reader reader("/Users/aoweichen/mnet_p4/models/bonsai-q2_0.gguf");
    const auto gate = reader.quantized(reader.tensor("blk.0.ffn_gate.weight"));
    const auto up = reader.quantized(reader.tensor("blk.0.ffn_up.weight"));
    const auto down = reader.quantized(reader.tensor("blk.0.ffn_down.weight"));

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
    id<MTLComputePipelineState> mv_int = pipeline_for("mv_int");
    id<MTLComputePipelineState> mv_f32 = pipeline_for("mv_f32act");
    id<MTLComputePipelineState> dequant = pipeline_for("dequant_q8");

    const std::uint64_t columns = gate.columns;
    const std::uint64_t blocks = columns / 64U;
    const std::uint64_t row_bytes = blocks * 18U;
    std::vector<float> input(columns);
    std::uint32_t state = 5U;
    for (auto& value : input) {
        state = state * 1664525U + 1013904223U;
        value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
    }
    const auto q80 = mnet::gguf::make_q8_0_blocks(input.data(), columns);
    std::vector<float> activation(columns);
    for (std::uint64_t i = 0; i < columns; ++i) {
        activation[i] = static_cast<float>(q80.qs[i]) * q80.d[i / 32U];
    }

    // One submission = the whole FFN block, exactly like the model's fused path.
    struct Job {
        const mnet::gguf::Reader::Quantized* weights;
        std::uint64_t rows;
        std::uint64_t row_base;
        std::uint64_t columns;
        std::uint64_t blocks;
    };
    std::vector<Job> jobs = {{&gate, gate.rows, 0, gate.columns, gate.columns / 64U},
                             {&up, up.rows, gate.rows, up.columns, up.columns / 64U},
                             {&down, down.rows, gate.rows + up.rows, down.columns, down.columns / 64U}};
    std::uint64_t total_rows = 0;
    std::uint64_t total_bytes = 0;
    for (const auto& job : jobs) {
        total_rows += job.rows;
        total_bytes += job.rows * job.weights->columns / 64U * 18U;
    }

    std::vector<id<MTLBuffer>> weight_buffers;
    std::vector<id<MTLBuffer>> q8_buffers;
    std::vector<id<MTLBuffer>> qd_buffers;
    std::vector<id<MTLBuffer>> act_buffers;
    std::vector<mnet::gguf::Q8_0Blocks> activations;
    std::vector<std::vector<float>> activation_f32;
    for (const auto& job : jobs) {
        const std::size_t bytes = static_cast<std::size_t>(job.rows * job.weights->columns / 64U * 18U);
        weight_buffers.push_back([device newBufferWithBytes:job.weights->blocks.data()
                                                     length:bytes
                                                    options:MTLResourceStorageModeShared]);
    }
    for (const auto& job : jobs) {
        // Each matrix gets its own activation so every dispatch is real work.
        std::vector<float> values(job.columns);
        for (auto& value : values) {
            state = state * 1664525U + 1013904223U;
            value = (static_cast<float>((state >> 8U) & 0xFFFFU) / 65536.0F - 0.5F) * 2.0F;
        }
        auto blocks_here = mnet::gguf::make_q8_0_blocks(values.data(), job.columns);
        std::vector<float> widened(job.columns);
        for (std::uint64_t i = 0; i < job.columns; ++i) {
            widened[i] = static_cast<float>(blocks_here.qs[i]) * blocks_here.d[i / 32U];
        }
        q8_buffers.push_back([device newBufferWithBytes:blocks_here.qs.data()
                                                 length:job.columns
                                                options:MTLResourceStorageModeShared]);
        qd_buffers.push_back([device newBufferWithBytes:blocks_here.d.data()
                                                 length:(job.columns / 32U) * sizeof(float)
                                                options:MTLResourceStorageModeShared]);
        act_buffers.push_back([device newBufferWithBytes:widened.data()
                                                  length:job.columns * sizeof(float)
                                                 options:MTLResourceStorageModeShared]);
        activations.push_back(std::move(blocks_here));
        activation_f32.push_back(std::move(widened));
    }
    id<MTLBuffer> out_buffer = [device newBufferWithLength:total_rows * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    const auto run = [&](id<MTLComputePipelineState> pipeline, bool float_activation) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            if (float_activation) {
                [encoder setComputePipelineState:dequant];
                [encoder setBuffer:q8_buffers[i] offset:0 atIndex:0];
                [encoder setBuffer:qd_buffers[i] offset:0 atIndex:1];
                [encoder setBuffer:act_buffers[i] offset:0 atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(jobs[i].columns, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weight_buffers[i] offset:0 atIndex:0];
            [encoder setBuffer:float_activation ? act_buffers[i] : q8_buffers[i] offset:0 atIndex:1];
            if (!float_activation) {
                [encoder setBuffer:qd_buffers[i] offset:0 atIndex:2];
            }
            [encoder setBuffer:out_buffer offset:0 atIndex:3];
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(jobs[i].blocks);
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
            const float scale = jobs[i].weights->scale;
            [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
            const std::uint32_t row_base = static_cast<std::uint32_t>(jobs[i].row_base);
            [encoder setBytes:&row_base length:sizeof(row_base) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(jobs[i].rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kLanes, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
    };

    const auto measure = [&](const char* label, id<MTLComputePipelineState> pipeline, bool float_activation) {
        for (int i = 0; i < 3; ++i) {
            run(pipeline, float_activation);
        }
        const int reps = 10;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            run(pipeline, float_activation);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
        std::printf("%-10s %6.3f ms  %6.2f GB/s  (%llu rows, %.1f MB per submission)\n", label, seconds * 1e3,
                    static_cast<double>(total_bytes) / seconds / 1e9,
                    static_cast<unsigned long long>(total_rows),
                    static_cast<double>(total_bytes) / 1e6);
    };

    // Sanity: the two kernels must agree with the CPU dot to within tolerance.
    const auto check = [&](id<MTLComputePipelineState> pipeline, bool float_activation) {
        run(pipeline, float_activation);
        const auto* output = static_cast<const float*>(out_buffer.contents);
        double worst = 0.0;
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto& job = jobs[j];
            const std::uint64_t matrix_blocks = job.columns / 64U;
            for (std::uint64_t r = 0; r < job.rows; r += 977) {
                const float cpu = job.weights->scale * mnet::gguf::dot_q2_0_q8_0(
                                                          job.weights->blocks.data() + r * matrix_blocks * 18U,
                                                          activations[j], matrix_blocks);
                const float gpu = output[job.row_base + r];
                worst = std::max(worst, std::abs(static_cast<double>(cpu) - static_cast<double>(gpu)) /
                                            std::max(1e-30, std::abs(static_cast<double>(cpu))));
            }
        }
        std::printf("  worst relative difference vs the CPU dot: %.3e\n", worst);
    };

    std::printf("Q2_0 block: gate %llu x %llu, up %llu, down %llu x %llu\n",
                static_cast<unsigned long long>(gate.rows), static_cast<unsigned long long>(gate.columns),
                static_cast<unsigned long long>(up.rows), static_cast<unsigned long long>(down.rows),
                static_cast<unsigned long long>(down.columns));
    measure("mv_int", mv_int, false);
    measure("mv_f32act", mv_f32, true);
    check(mv_int, false);
    check(mv_f32, true);
    (void)row_bytes;
    return 0;
}
