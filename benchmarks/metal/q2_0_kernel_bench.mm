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

// The activation chain between the up matvec and the down matvec, copied from
// the backend: swiglu -> sign*scale -> block FWHT -> Q8_0 requantise.
kernel void swiglu_f32(device const float* gate [[buffer(0)]],
                       device const float* up   [[buffer(1)]],
                       device float* out        [[buffer(2)]],
                       constant uint& count     [[buffer(3)]],
                       uint gid                 [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    const float x = gate[gid];
    out[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

kernel void sign_scale(device const float* in  [[buffer(0)]],
                       device const int* signs  [[buffer(1)]],
                       device float* out        [[buffer(2)]],
                       constant float& scale    [[buffer(3)]],
                       constant uint& count     [[buffer(4)]],
                       uint gid                 [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    out[gid] = (in[gid] * float(signs[gid])) * scale;
}

kernel void fwht_block(device float* data       [[buffer(0)]],
                       constant uint& block_size [[buffer(1)]],
                       uint tgid                 [[threadgroup_position_in_grid]],
                       uint lane                 [[thread_position_in_threadgroup]]) {
    threadgroup float v[1024];
    const uint base = tgid * block_size;
    v[lane] = data[base + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1u; len < block_size; len <<= 1u) {
        if ((lane & len) == 0u) {
            const float a = v[lane];
            const float b = v[lane + len];
            v[lane] = a + b;
            v[lane + len] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    data[base + lane] = v[lane];
}

kernel void q8_0_quantize(device const float* x    [[buffer(0)]],
                          device char* qs          [[buffer(1)]],
                          device float* scales     [[buffer(2)]],
                          uint gid                 [[thread_position_in_grid]]) {
    const uint base = gid * 32u;
    float amax = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        amax = fmax(amax, fabs(x[base + j]));
    }
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        qs[base + j] = char(rint(x[base + j] * id));
    }
    scales[gid] = float(half(amax / 127.0f));
}

// Widen the activation and, in the same pass, sum each 32-value block: the
// matvec then extracts the 2-bit code unsigned and subtracts the block sum
// once, which removes the per-weight bias subtraction.
kernel void widen_q8(device const char* q8      [[buffer(0)]],
                     device const float* qd     [[buffer(1)]],
                     device float* out          [[buffer(2)]],
                     device float* block_sum    [[buffer(3)]],
                     uint tgid                  [[threadgroup_position_in_grid]],
                     uint lane                  [[thread_position_in_threadgroup]]) {
    const uint index = tgid * 32u + lane;
    const float value = float(q8[index]) * qd[tgid];
    out[index] = value;
    float total = simd_sum(value);
    if (lane == 0u) {
        block_sum[tgid] = total;
    }
}

// Same as mv_f32act but with the bias folded per block: one op less per weight.
kernel void mv_f32sum(device const uchar* w         [[buffer(0)]],
                      device const float* x         [[buffer(1)]],
                      device const float* block_sum [[buffer(2)]],
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
                sum += float(int(byte & 3u)) * xy[j * 4u + 0u];
                sum += float(int((byte >> 2u) & 3u)) * xy[j * 4u + 1u];
                sum += float(int((byte >> 4u) & 3u)) * xy[j * 4u + 2u];
                sum += float(int((byte >> 6u) & 3u)) * xy[j * 4u + 3u];
            }
            acc = fma(dx, sum - block_sum[2u * i + k], acc);
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
    id<MTLComputePipelineState> swiglu = pipeline_for("swiglu_f32");
    id<MTLComputePipelineState> sign_scale = pipeline_for("sign_scale");
    id<MTLComputePipelineState> fwht = pipeline_for("fwht_block");
    id<MTLComputePipelineState> quantize = pipeline_for("q8_0_quantize");
    id<MTLComputePipelineState> widen = pipeline_for("widen_q8");
    id<MTLComputePipelineState> mv_sum = pipeline_for("mv_f32sum");

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
    std::vector<id<MTLBuffer>> sum_buffers;
    for (const auto& job : jobs) {
        sum_buffers.push_back([device newBufferWithLength:(job.columns / 32U) * sizeof(float)
                                                  options:MTLResourceStorageModeShared]);
    }
    id<MTLBuffer> out_buffer = [device newBufferWithLength:total_rows * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    // Extra buffers for the activation chain between the up and down matvecs.
    const std::uint64_t hidden = up.rows;
    id<MTLBuffer> act_f32 = [device newBufferWithLength:hidden * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> act_signs = [device newBufferWithLength:hidden * sizeof(std::int32_t)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> act_q8 = [device newBufferWithLength:hidden options:MTLResourceStorageModeShared];
    id<MTLBuffer> act_scales = [device newBufferWithLength:(hidden / 32U) * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> act_sums = [device newBufferWithLength:(hidden / 32U) * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    {
        auto* signs = static_cast<std::int32_t*>(act_signs.contents);
        for (std::uint64_t i = 0; i < hidden; ++i) {
            signs[i] = ((i * 2654435761U) >> 7U) & 1U ? 1 : -1;
        }
    }
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    const std::uint32_t block_size = 1024;
    const float hadamard_scale = 1.0F / std::sqrt(static_cast<float>(block_size));

    // phase 0: gate+up matvecs; 1: + activation chain; 2: + down matvec.
    // mode 0 = integer activation kernel, 1 = fp32 activation, 2 = fp32 with the
    // per-block bias folded out.
    const auto run = [&](int phase, int mode, const std::vector<id<MTLBuffer>>& weights) {
        const bool float_activation = mode != 0;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        const auto encode_matvec = [&](std::size_t i, id<MTLBuffer> activation_buffer, id<MTLBuffer> scale_buffer) {
            if (mode == 2) {
                [encoder setComputePipelineState:mv_sum];
            } else {
                [encoder setComputePipelineState:float_activation ? mv_f32 : mv_int];
            }
            [encoder setBuffer:weights[i] offset:0 atIndex:0];
            [encoder setBuffer:activation_buffer offset:0 atIndex:1];
            [encoder setBuffer:mode == 2 ? sum_buffers[i] : scale_buffer offset:0 atIndex:2];
            [encoder setBuffer:out_buffer offset:0 atIndex:3];
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(jobs[i].blocks);
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:4];
            const float scale = jobs[i].weights->scale;
            [encoder setBytes:&scale length:sizeof(scale) atIndex:5];
            const std::uint32_t row_base = static_cast<std::uint32_t>(jobs[i].row_base);
            [encoder setBytes:&row_base length:sizeof(row_base) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(jobs[i].rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kLanes, 1, 1)];
        };
        const auto encode_dequant = [&](std::size_t i, id<MTLBuffer> destination) {
            if (mode == 2) {
                [encoder setComputePipelineState:widen];
                [encoder setBuffer:q8_buffers[i] offset:0 atIndex:0];
                [encoder setBuffer:qd_buffers[i] offset:0 atIndex:1];
                [encoder setBuffer:destination offset:0 atIndex:2];
                [encoder setBuffer:sum_buffers[i] offset:0 atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(jobs[i].columns / 32U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                return;
            }
            [encoder setComputePipelineState:dequant];
            [encoder setBuffer:q8_buffers[i] offset:0 atIndex:0];
            [encoder setBuffer:qd_buffers[i] offset:0 atIndex:1];
            [encoder setBuffer:destination offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(jobs[i].columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        };

        for (std::size_t i = 0; i < 2; ++i) {
            if (float_activation) {
                encode_dequant(i, act_buffers[i]);
            }
            encode_matvec(i, float_activation ? act_buffers[i] : q8_buffers[i], qd_buffers[i]);
        }
        if (phase >= 1) {
            [encoder setComputePipelineState:swiglu];
            [encoder setBuffer:out_buffer offset:0 atIndex:0];
            [encoder setBuffer:out_buffer offset:static_cast<NSUInteger>(gate.rows) * sizeof(float) atIndex:1];
            [encoder setBuffer:act_f32 offset:0 atIndex:2];
            [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [encoder setComputePipelineState:sign_scale];
            [encoder setBuffer:act_f32 offset:0 atIndex:0];
            [encoder setBuffer:act_signs offset:0 atIndex:1];
            [encoder setBuffer:act_f32 offset:0 atIndex:2];
            [encoder setBytes:&hadamard_scale length:sizeof(hadamard_scale) atIndex:3];
            [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [encoder setComputePipelineState:fwht];
            [encoder setBuffer:act_f32 offset:0 atIndex:0];
            [encoder setBytes:&block_size length:sizeof(block_size) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden / block_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];

            [encoder setComputePipelineState:quantize];
            [encoder setBuffer:act_f32 offset:0 atIndex:0];
            [encoder setBuffer:act_q8 offset:0 atIndex:1];
            [encoder setBuffer:act_scales offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(hidden / 32U, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        if (phase >= 2) {
            if (mode == 2) {
                [encoder setComputePipelineState:widen];
                [encoder setBuffer:act_q8 offset:0 atIndex:0];
                [encoder setBuffer:act_scales offset:0 atIndex:1];
                [encoder setBuffer:act_f32 offset:0 atIndex:2];
                [encoder setBuffer:act_sums offset:0 atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(hidden / 32U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else if (float_activation) {
                [encoder setComputePipelineState:dequant];
                [encoder setBuffer:act_q8 offset:0 atIndex:0];
                [encoder setBuffer:act_scales offset:0 atIndex:1];
                [encoder setBuffer:act_f32 offset:0 atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            encode_matvec(2, float_activation ? act_f32 : act_q8, act_scales);
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
    };

    const auto measure = [&](const char* label, int phase, int mode) {
        for (int i = 0; i < 3; ++i) {
            run(phase, mode, weight_buffers);
        }
        const int reps = 10;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            run(phase, mode, weight_buffers);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
        std::printf("%-22s %6.3f ms  %6.2f GB/s\n", label, seconds * 1e3,
                    static_cast<double>(total_bytes) / seconds / 1e9);
        return seconds * 1e3;
    };

    const auto check = [&](int mode) {
        run(2, mode, weight_buffers);
        const auto* output = static_cast<const float*>(out_buffer.contents);
        double worst = 0.0;
        for (std::size_t j = 0; j < 2; ++j) {  // the down matvec reads the chain's output
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
        std::printf("  matvec-only stages vs the CPU dot: %.3e\n", worst);
    };

    std::printf("Q2_0 block: gate %llu x %llu, up %llu, down %llu x %llu (%.1f MB)\n",
                static_cast<unsigned long long>(gate.rows), static_cast<unsigned long long>(gate.columns),
                static_cast<unsigned long long>(up.rows), static_cast<unsigned long long>(down.rows),
                static_cast<unsigned long long>(down.columns), static_cast<double>(total_bytes) / 1e6);
    measure("mv_int   full", 2, 0);
    const double full = measure("mv_f32   full", 2, 1);
    const double no_down = measure("mv_f32   gate+up+act", 1, 1);
    const double up_only = measure("mv_f32   gate+up", 0, 1);
    std::printf("  activation chain %.3f ms, down matvec %.3f ms (of %.3f ms)\n", no_down - up_only, full - no_down, full);
    const double sum_full = measure("mv_f32sum full", 2, 2);
    std::printf("  per-block bias folded: %.3f ms vs %.3f ms (%.2fx)\n", sum_full, full, full / sum_full);

    // Sustained load: is the in-model 0.94 ms/call (versus 0.76 here) a clock
    // effect rather than a bubble? Run the block for ~2 s and compare the
    // first and the last window.
    {
        const int reps = 2600;
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            run(2, 1, weight_buffers);
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        }
        const auto average = [&](std::size_t from, std::size_t to) {
            double total = 0.0;
            for (std::size_t i = from; i < to; ++i) {
                total += samples[i];
            }
            return total / static_cast<double>(to - from);
        };
        const double first = average(20, 220);
        const double last = average(samples.size() - 200, samples.size());
        std::printf("  sustained: first 200 reps %.3f ms, last 200 reps %.3f ms (%.2fx slower, %.1f GB/s)\n",
                    first * 1e3, last * 1e3, last / first, static_cast<double>(total_bytes) / last / 1e9);
    }
    check(1);
    check(2);

    // The model streams 64 different layers' weights (6.7 GB) instead of the
    // same 75 MB; reproduce that by allocating N copies and cycling through
    // them, one submission per copy (like one layer-token).
    {
        const char* cycle_env = std::getenv("MNET_BENCH_CYCLE");
        const int copies = cycle_env != nullptr ? std::atoi(cycle_env) : 0;
        if (copies > 1) {
            std::vector<std::vector<id<MTLBuffer>>> sets;
            sets.reserve(static_cast<std::size_t>(copies));
            for (int c = 0; c < copies; ++c) {
                std::vector<id<MTLBuffer>> set;
                for (std::size_t i = 0; i < jobs.size(); ++i) {
                    const std::size_t bytes = static_cast<std::size_t>(jobs[i].rows * jobs[i].columns / 64U * 18U);
                    set.push_back([device newBufferWithBytes:weight_buffers[i].contents
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared]);
                }
                sets.push_back(std::move(set));
            }
            const double per_cycle = static_cast<double>(copies) * static_cast<double>(total_bytes);
            const auto cycle = [&] {
                for (const auto& set : sets) {
                    run(2, 1, set);
                }
            };
            for (int i = 0; i < 2; ++i) {
                cycle();
            }
            const int reps = 6;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                cycle();
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(reps);
            std::printf("%-22s %6.3f ms/layer  %6.2f GB/s  (%d layers, %.1f GB)\n", "cycle (64 layers)", seconds * 1e3,
                        per_cycle / seconds / 1e9, copies, per_cycle / 1e9);
        }
    }
    (void)row_bytes;
    return 0;
}
