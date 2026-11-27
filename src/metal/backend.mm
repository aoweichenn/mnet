// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Metal provider for the device-neutral compute layer. Only this file knows
// about Metal: it registers a Q2_0 matvec with mnet::detail::compute and is
// linked only when mnet_enable_metal is ON. The kernel reproduces the CPU
// accumulation order byte for byte (eight byte-lanes, two alternating block
// accumulators, the same hsum tree), so results match the CPU path exactly.

#include <mnet/detail/compute.hpp>

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mnet::detail::compute {
namespace {

constexpr const char* kQ2_0Source = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void q2_0_matvec(device const uchar* w        [[buffer(0)]],
                        device const char* q8       [[buffer(1)]],
                        device const float* qd      [[buffer(2)]],
                        device float* out           [[buffer(3)]],
                        constant uint& row_bytes    [[buffer(4)]],
                        constant uint& blocks       [[buffer(5)]],
                        constant float& scale       [[buffer(6)]],
                        uint gid                    [[thread_position_in_grid]]) {
    device const uchar* base = w + (ulong)gid * row_bytes;
    float acc0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < blocks; ++i) {
        device const uchar* block = base + (ulong)i * 18u;
        const float dx = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
        thread float* acc = ((i & 1u) != 0u) ? acc1 : acc0;
        for (uint k = 0; k < 2u; ++k) {
            const float d = dx * qd[2u * i + k];
            device const char* qy = q8 + (ulong)(2u * i + k) * 32u;
            device const uchar* q = block + 2u + k * 8u;
            for (uint j = 0; j < 8u; ++j) {
                const uchar byte = q[j];
                const int lane = (int((byte >> 0) & 3u) - 1) * int(qy[j * 4u + 0u]) +
                                 (int((byte >> 2) & 3u) - 1) * int(qy[j * 4u + 1u]) +
                                 (int((byte >> 4) & 3u) - 1) * int(qy[j * 4u + 2u]) +
                                 (int((byte >> 6) & 3u) - 1) * int(qy[j * 4u + 3u]);
                acc[j] = fma(d, float(lane), acc[j]);
            }
        }
    }
    for (uint j = 0; j < 8u; ++j) {
        acc0[j] += acc1[j];
    }
    const float sum = ((acc0[0] + acc0[4]) + (acc0[2] + acc0[6])) + ((acc0[1] + acc0[5]) + (acc0[3] + acc0[7]));
    out[gid] = scale * sum;
}
)MSL";

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::unordered_map<const void*, id<MTLBuffer>> weight_buffers;
    id<MTLBuffer> q8_buffer = nil;
    id<MTLBuffer> scale_buffer = nil;
    id<MTLBuffer> out_buffer = nil;
    std::size_t q8_capacity = 0;
    std::size_t scale_capacity = 0;
    std::size_t out_capacity = 0;
    bool ready = false;
};

void initialise(MetalState& state) {
    state.device = MTLCreateSystemDefaultDevice();
    if (state.device == nil) {
        return;
    }
    state.queue = [state.device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [state.device newLibraryWithSource:[NSString stringWithUTF8String:kQ2_0Source]
                                                        options:nil
                                                          error:&error];
    if (library == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
        std::fprintf(stderr, "mnet: metal compile failed: %s\n", [[error description] UTF8String]);
    }
    id<MTLFunction> function = library != nil ? [library newFunctionWithName:@"q2_0_matvec"] : nil;
    if (function != nil) {
        state.pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
    }
    state.ready = state.pipeline != nil && state.queue != nil;
}

MetalState& state() {
    static MetalState instance;
    static std::once_flag once;
    std::call_once(once, [] { initialise(instance); });
    return instance;
}

id<MTLBuffer> ensure_buffer(id<MTLDevice> device, id<MTLBuffer>& buffer, std::size_t& capacity, std::size_t bytes) {
    if (bytes <= capacity) {
        return buffer;
    }
    buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    capacity = buffer != nil ? bytes : 0;
    return buffer;
}

bool matvec_q2_0(const std::uint8_t* weights, std::uint64_t rows, std::uint64_t columns, std::uint64_t row_bytes,
                 float scale, const std::int8_t* q8, const float* q8_scales, float* out) {
    MetalState& s = state();
    if (!s.ready || rows == 0 || columns == 0 || (columns % 64U) != 0U) {
        return false;
    }
    const std::uint64_t blocks = columns / 64U;

    id<MTLBuffer>& weight_buffer = s.weight_buffers[weights];
    if (weight_buffer == nil) {
        weight_buffer = [s.device newBufferWithLength:static_cast<NSUInteger>(rows * row_bytes)
                                             options:MTLResourceStorageModeShared];
        if (weight_buffer == nil) {
            s.weight_buffers.erase(weights);
            return false;
        }
        std::memcpy(weight_buffer.contents, weights, static_cast<std::size_t>(rows * row_bytes));
    }

    id<MTLBuffer> q8_buffer = ensure_buffer(s.device, s.q8_buffer, s.q8_capacity, static_cast<std::size_t>(columns));
    id<MTLBuffer> scale_buffer =
        ensure_buffer(s.device, s.scale_buffer, s.scale_capacity, static_cast<std::size_t>(columns / 32U) * sizeof(float));
    id<MTLBuffer> out_buffer =
        ensure_buffer(s.device, s.out_buffer, s.out_capacity, static_cast<std::size_t>(rows) * sizeof(float));
    if (q8_buffer == nil || scale_buffer == nil || out_buffer == nil) {
        return false;
    }
    std::memcpy(q8_buffer.contents, q8, static_cast<std::size_t>(columns));
    std::memcpy(scale_buffer.contents, q8_scales, static_cast<std::size_t>(columns / 32U) * sizeof(float));

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:s.pipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:q8_buffer offset:0 atIndex:1];
    [encoder setBuffer:scale_buffer offset:0 atIndex:2];
    [encoder setBuffer:out_buffer offset:0 atIndex:3];
    const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(row_bytes);
    const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
    [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
    [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
    const NSUInteger group = s.pipeline.maxTotalThreadsPerThreadgroup < 128U ? s.pipeline.maxTotalThreadsPerThreadgroup : 128U;
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(out, out_buffer.contents, static_cast<std::size_t>(rows) * sizeof(float));
    return true;
}

bool matvec_q8_0_backend(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                         std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales,
                         float* out) {
    if (type != 42U) {  // kGgmlQ2_0; kept numeric so this TU needs no GGUF header
        return false;
    }
    return matvec_q2_0(data, rows, columns, row_bytes, scale, q8, q8_scales, out);
}

}  // namespace

void register_platform_backends() {
    if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
        std::fprintf(stderr, "mnet: metal backend %s\n", state().ready ? "ready" : "unavailable");
    }
    set_matvec_q8_0_backend(&matvec_q8_0_backend);
}

}  // namespace mnet::detail::compute
