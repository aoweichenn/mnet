// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Metal provider for the device-neutral compute layer. Only this file knows
// about Metal: it registers a Q2_0 matvec with mnet::detail::compute and is
// linked only when mnet_enable_metal is ON. The kernel reproduces the CPU
// accumulation order byte for byte (eight byte-lanes, two alternating block
// accumulators, the same hsum tree), so results match the CPU path exactly.

#include <mnet/detail/compute.hpp>

#include <mnet/gguf/format.hpp>

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mnet::detail::compute {
namespace {

constexpr const char* kQ2_0Source = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
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

// One threadgroup (32 lanes) per row; lanes stride over the blocks so a warp
// reads 32 consecutive 18-byte records per step, and the 32 partial sums are
// folded with simd_sum. This is llama.cpp's mul_mv shape (bandwidth-bound);
// the reduction order differs from the CPU kernel, so it is the fast path.
kernel void q2_0_matvec_fast(device const uchar* w        [[buffer(0)]],
                             device const char* q8       [[buffer(1)]],
                             device const float* qd      [[buffer(2)]],
                             device float* out           [[buffer(3)]],
                             constant uint& row_bytes    [[buffer(4)]],
                             constant uint& blocks       [[buffer(5)]],
                             constant float& scale       [[buffer(6)]],
                             uint tgid                   [[threadgroup_position_in_grid]],
                             uint lane                   [[thread_position_in_threadgroup]]) {
    device const uchar* row = w + (ulong)tgid * row_bytes;
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
        out[tgid] = scale * total;
    }
}


// SwiGLU: out = silu(gate) * up, one thread per element.
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

// out = in * sign * scale (the model's signed Hadamard pre-step).
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

// Normalised in-place Hadamard over one 256-value block, same butterfly order
// as the CPU fwht (a+b, a-b per stage), one threadgroup of 256 threads.
kernel void fwht256(device float* data          [[buffer(0)]],
                    uint tgid                   [[threadgroup_position_in_grid]],
                    uint lane                   [[thread_position_in_threadgroup]]) {
    threadgroup float v[256];
    const uint base = tgid * 256u;
    v[lane] = data[base + lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1u; len < 256u; len <<= 1u) {
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

// Q8_0 activation quantisation, one thread per 32-value block; matches
// make_q8_0_blocks: round-to-nearest-even codes and a half-rounded scale.
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
)MSL";

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> fast_pipeline = nil;
    id<MTLComputePipelineState> swiglu_pipeline = nil;
    id<MTLComputePipelineState> quantize_pipeline = nil;
    id<MTLBuffer> act_buffer = nil;
    id<MTLBuffer> act_q8_buffer = nil;
    id<MTLBuffer> act_scale_buffer = nil;
    id<MTLBuffer> gate_buffer = nil;
    id<MTLBuffer> up_buffer = nil;
    id<MTLBuffer> signs_buffer = nil;
    id<MTLComputePipelineState> sign_scale_pipeline = nil;
    id<MTLComputePipelineState> fwht_pipeline = nil;
    std::size_t signs_capacity = 0;
    std::size_t act_capacity = 0;
    std::size_t act_q8_capacity = 0;
    std::size_t act_scale_capacity = 0;
    std::size_t gate_capacity = 0;
    std::size_t up_capacity = 0;
    std::unordered_map<std::uint64_t, id<MTLBuffer>> weight_buffers;
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
    // Metal shades fast math by default; the kernel must keep IEEE rounding to
    // stay bit-identical to the CPU path, so turn it off explicitly.
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [state.device newLibraryWithSource:[NSString stringWithUTF8String:kQ2_0Source]
                                                        options:options
                                                          error:&error];
    if (library == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
        std::fprintf(stderr, "mnet: metal compile failed: %s\n", [[error description] UTF8String]);
    }
    id<MTLFunction> function = library != nil ? [library newFunctionWithName:@"q2_0_matvec"] : nil;
    if (function != nil) {
        state.pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
    }
    id<MTLFunction> sign_scale_function = library != nil ? [library newFunctionWithName:@"sign_scale"] : nil;
    if (sign_scale_function != nil) {
        state.sign_scale_pipeline = [state.device newComputePipelineStateWithFunction:sign_scale_function error:&error];
    }
    id<MTLFunction> fwht_function = library != nil ? [library newFunctionWithName:@"fwht256"] : nil;
    if (fwht_function != nil) {
        state.fwht_pipeline = [state.device newComputePipelineStateWithFunction:fwht_function error:&error];
    }
    id<MTLFunction> swiglu_function = library != nil ? [library newFunctionWithName:@"swiglu_f32"] : nil;
    if (swiglu_function != nil) {
        state.swiglu_pipeline = [state.device newComputePipelineStateWithFunction:swiglu_function error:&error];
    }
    id<MTLFunction> quantize_function = library != nil ? [library newFunctionWithName:@"q8_0_quantize"] : nil;
    if (quantize_function != nil) {
        state.quantize_pipeline = [state.device newComputePipelineStateWithFunction:quantize_function error:&error];
    }
    id<MTLFunction> fast_function = library != nil ? [library newFunctionWithName:@"q2_0_matvec_fast"] : nil;
    if (fast_function != nil) {
        state.fast_pipeline = [state.device newComputePipelineStateWithFunction:fast_function error:&error];
        if (state.fast_pipeline == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
            std::fprintf(stderr, "mnet: metal fast pipeline failed: %s\n", [[error description] UTF8String]);
        }
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

std::uint64_t cache_key(std::uint64_t identity, const std::uint8_t* weights) {
    // Loaded tensors carry a unique identity; raw-pointer callers (tests) use
    // the address with the top bit set so the two spaces cannot collide.
    return identity != 0 ? identity : (reinterpret_cast<std::uintptr_t>(weights) | (1ULL << 63U));
}

id<MTLBuffer> weight_buffer_for(MetalState& s, std::uint64_t identity, const std::uint8_t* weights, std::uint64_t rows,
                                std::uint64_t columns, std::uint64_t row_bytes) {
    id<MTLBuffer>& weight_buffer = s.weight_buffers[cache_key(identity, weights)];
    if (weight_buffer != nil) {
        if (weight_buffer.length != static_cast<NSUInteger>(rows * row_bytes)) {
            if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
                std::fprintf(stderr, "mnet: metal weight cache shape mismatch (%lu vs %llu)\n",
                             static_cast<unsigned long>(weight_buffer.length),
                             static_cast<unsigned long long>(rows * row_bytes));
            }
            return nil;
        }
        return weight_buffer;
    }
    weight_buffer = [s.device newBufferWithLength:static_cast<NSUInteger>(rows * row_bytes)
                                         options:MTLResourceStorageModeShared];
    if (weight_buffer == nil) {
        s.weight_buffers.erase(cache_key(identity, weights));
        return nil;
    }
    std::memcpy(weight_buffer.contents, weights, static_cast<std::size_t>(rows * row_bytes));
    return weight_buffer;
}

void encode_dispatch(MetalState& s, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> weight_buffer,
                     std::uint64_t rows, std::uint64_t columns, std::uint64_t row_bytes, float scale,
                     std::uint64_t out_offset_floats, id<MTLBuffer> activations, id<MTLBuffer> activation_scales,
                     id<MTLBuffer> output_buffer) {
    const std::uint64_t blocks = columns / 64U;
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:activations offset:0 atIndex:1];
    [encoder setBuffer:activation_scales offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:static_cast<NSUInteger>(out_offset_floats * sizeof(float)) atIndex:3];
    const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(row_bytes);
    const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
    [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
    [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
    static const bool use_fast = std::getenv("MNET_METAL_FAST") != nullptr && s.fast_pipeline != nil;
    if (use_fast) {
        [encoder setComputePipelineState:s.fast_pipeline];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [encoder setComputePipelineState:s.pipeline];
        const NSUInteger group =
            s.pipeline.maxTotalThreadsPerThreadgroup < 128U ? s.pipeline.maxTotalThreadsPerThreadgroup : 128U;
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    }
}

bool matvec_batch(const MatvecQ8_0Job* jobs, std::size_t count, std::uint64_t columns, const std::int8_t* q8,
                  const float* q8_scales);

bool matvec_q2_0(const std::uint8_t* weights, std::uint64_t rows, std::uint64_t columns, std::uint64_t row_bytes,
                 float scale, const std::int8_t* q8, const float* q8_scales, float* out, std::uint64_t identity) {
    MatvecQ8_0Job job{42U, weights, rows, row_bytes, scale, out, identity};
    return matvec_batch(&job, 1, columns, q8, q8_scales);
}

// Encode every job in one command buffer: one submission, one wait.
bool matvec_batch(const MatvecQ8_0Job* jobs, std::size_t count, std::uint64_t columns, const std::int8_t* q8,
                  const float* q8_scales) {
    MetalState& s = state();
    if (!s.ready || count == 0 || columns == 0 || (columns % 64U) != 0U) {
        return false;
    }
    std::uint64_t total_rows = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (jobs[i].type != 42U || jobs[i].rows == 0) {
            return false;
        }
        total_rows += jobs[i].rows;
    }

    id<MTLBuffer> q8_buffer = ensure_buffer(s.device, s.q8_buffer, s.q8_capacity, static_cast<std::size_t>(columns));
    id<MTLBuffer> scale_buffer = ensure_buffer(s.device, s.scale_buffer, s.scale_capacity,
                                               static_cast<std::size_t>(columns / 32U) * sizeof(float));
    id<MTLBuffer> out_buffer =
        ensure_buffer(s.device, s.out_buffer, s.out_capacity, static_cast<std::size_t>(total_rows) * sizeof(float));
    if (q8_buffer == nil || scale_buffer == nil || out_buffer == nil) {
        return false;
    }
    std::memcpy(q8_buffer.contents, q8, static_cast<std::size_t>(columns));
    std::memcpy(scale_buffer.contents, q8_scales, static_cast<std::size_t>(columns / 32U) * sizeof(float));

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        id<MTLBuffer> weight_buffer =
            weight_buffer_for(s, jobs[i].identity, jobs[i].data, jobs[i].rows, columns, jobs[i].row_bytes);
        if (weight_buffer == nil) {
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            return false;
        }
        encode_dispatch(s, encoder, weight_buffer, jobs[i].rows, columns, jobs[i].row_bytes, jobs[i].scale, offset,
                        s.q8_buffer, s.scale_buffer, s.out_buffer);
        offset += jobs[i].rows;
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(jobs[i].out,
                    static_cast<const float*>(out_buffer.contents) + static_cast<std::ptrdiff_t>(offset),
                    static_cast<std::size_t>(jobs[i].rows) * sizeof(float));
        offset += jobs[i].rows;
    }
    return true;
}

// gate/up matvec -> SwiGLU -> on-device Q8_0 -> down matvec, one submission.
bool ffn_swiglu(const MatvecQ8_0Job& gate, const MatvecQ8_0Job& up, const MatvecQ8_0Job& down, std::uint64_t columns,
                std::uint64_t hidden, const float* input, const std::int32_t* signs, std::uint32_t block_size) {
    MetalState& s = state();
    if (!s.ready || s.swiglu_pipeline == nil || s.quantize_pipeline == nil || s.sign_scale_pipeline == nil ||
        s.fwht_pipeline == nil) {
        return false;
    }
    if (block_size != 256U) {
        return false;
    }
    if (gate.type != 42U || up.type != 42U || down.type != 42U) {
        return false;
    }
    if (up.rows != hidden || down.rows == 0 || (columns % 64U) != 0U || (hidden % block_size) != 0U) {
        return false;
    }

    const auto q80 = mnet::gguf::make_q8_0_blocks(input, columns);
    id<MTLBuffer> q8_buffer = ensure_buffer(s.device, s.q8_buffer, s.q8_capacity, static_cast<std::size_t>(columns));
    id<MTLBuffer> scale_buffer = ensure_buffer(s.device, s.scale_buffer, s.scale_capacity,
                                               static_cast<std::size_t>(columns / 32U) * sizeof(float));
    id<MTLBuffer> gate_buffer =
        ensure_buffer(s.device, s.gate_buffer, s.gate_capacity, static_cast<std::size_t>(hidden) * sizeof(float));
    id<MTLBuffer> up_buffer =
        ensure_buffer(s.device, s.up_buffer, s.up_capacity, static_cast<std::size_t>(hidden) * sizeof(float));
    id<MTLBuffer> act_buffer =
        ensure_buffer(s.device, s.act_buffer, s.act_capacity, static_cast<std::size_t>(hidden) * sizeof(float));
    id<MTLBuffer> act_q8_buffer = ensure_buffer(s.device, s.act_q8_buffer, s.act_q8_capacity,
                                                static_cast<std::size_t>(hidden));
    id<MTLBuffer> act_scale_buffer = ensure_buffer(s.device, s.act_scale_buffer, s.act_scale_capacity,
                                                   static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    id<MTLBuffer> signs_buffer =
        ensure_buffer(s.device, s.signs_buffer, s.signs_capacity, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    id<MTLBuffer> out_buffer =
        ensure_buffer(s.device, s.out_buffer, s.out_capacity, static_cast<std::size_t>(down.rows) * sizeof(float));
    if (q8_buffer == nil || scale_buffer == nil || gate_buffer == nil || up_buffer == nil || act_buffer == nil ||
        act_q8_buffer == nil || act_scale_buffer == nil || signs_buffer == nil || out_buffer == nil) {
        return false;
    }
    std::memcpy(signs_buffer.contents, signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    std::memcpy(q8_buffer.contents, q80.qs.data(), static_cast<std::size_t>(columns));
    std::memcpy(scale_buffer.contents, q80.d.data(), static_cast<std::size_t>(columns / 32U) * sizeof(float));

    id<MTLBuffer> gate_weights = weight_buffer_for(s, gate.identity, gate.data, gate.rows, columns, gate.row_bytes);
    id<MTLBuffer> up_weights = weight_buffer_for(s, up.identity, up.data, up.rows, columns, up.row_bytes);
    id<MTLBuffer> down_weights =
        weight_buffer_for(s, down.identity, down.data, down.rows, hidden, down.row_bytes);
    if (gate_weights == nil || up_weights == nil || down_weights == nil) {
        return false;
    }

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_dispatch(s, encoder, gate_weights, gate.rows, columns, gate.row_bytes, gate.scale, 0, q8_buffer,
                    scale_buffer, gate_buffer);
    encode_dispatch(s, encoder, up_weights, up.rows, columns, up.row_bytes, up.scale, 0, q8_buffer, scale_buffer,
                    up_buffer);

    [encoder setComputePipelineState:s.swiglu_pipeline];
    [encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [encoder setBuffer:up_buffer offset:0 atIndex:1];
    [encoder setBuffer:act_buffer offset:0 atIndex:2];
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    [encoder setComputePipelineState:s.sign_scale_pipeline];
    [encoder setBuffer:act_buffer offset:0 atIndex:0];
    [encoder setBuffer:signs_buffer offset:0 atIndex:1];
    [encoder setBuffer:act_buffer offset:0 atIndex:2];
    const float hadamard_scale = 1.0F / std::sqrt(static_cast<float>(block_size));
    [encoder setBytes:&hadamard_scale length:sizeof(hadamard_scale) atIndex:3];
    [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    [encoder setComputePipelineState:s.fwht_pipeline];
    [encoder setBuffer:act_buffer offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(hidden / block_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];

    [encoder setComputePipelineState:s.quantize_pipeline];
    [encoder setBuffer:act_buffer offset:0 atIndex:0];
    [encoder setBuffer:act_q8_buffer offset:0 atIndex:1];
    [encoder setBuffer:act_scale_buffer offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(hidden / 32U, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    encode_dispatch(s, encoder, down_weights, down.rows, hidden, down.row_bytes, down.scale, 0, act_q8_buffer,
                    act_scale_buffer, out_buffer);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(down.out, out_buffer.contents, static_cast<std::size_t>(down.rows) * sizeof(float));
    return true;
}

bool matvec_q8_0_backend(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                         std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales,
                         float* out, std::uint64_t identity) {
    if (type != 42U) {  // kGgmlQ2_0; kept numeric so this TU needs no GGUF header
        return false;
    }
    return matvec_q2_0(data, rows, columns, row_bytes, scale, q8, q8_scales, out, identity);
}

}  // namespace

void register_platform_backends() {
    if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
        std::fprintf(stderr, "mnet: metal backend %s (fast %s, ffn %s)\n",
                     state().ready ? "ready" : "unavailable", state().fast_pipeline != nil ? "ready" : "unavailable",
                     (state().swiglu_pipeline != nil && state().quantize_pipeline != nil &&
                      state().sign_scale_pipeline != nil && state().fwht_pipeline != nil)
                         ? "ready"
                         : "unavailable");
    }
    set_matvec_q8_0_backend(&matvec_q8_0_backend);
    set_matvec_q8_0_batch_backend(&matvec_batch);
    set_ffn_swiglu_backend(&ffn_swiglu);
}

}  // namespace mnet::detail::compute
