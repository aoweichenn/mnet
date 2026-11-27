// Q2_0 tiled mat-mat, extracted verbatim from src/metal/backend.mm at the
// commit this bench was added in. The isolation experiment of log 122: one real
// dispatch, repeated, with nothing else in the command buffer, to separate the
// kernel's own rate from the per-layer dispatch chain.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void q2_0_mat_mat_f16(device const uchar* w        [[buffer(0)]],
                             device const float* x        [[buffer(1)]],
                             device float* out            [[buffer(3)]],
                             constant uint& row_bytes     [[buffer(4)]],
                             constant uint& columns       [[buffer(5)]],
                             constant float& scale        [[buffer(6)]],
                             constant uint& rows          [[buffer(8)]],
                             constant uint& tokens        [[buffer(9)]],
                             constant uint& x_stride      [[buffer(10)]],
                             constant uint& out_stride    [[buffer(11)]],
                             uint3 tgpig                  [[threadgroup_position_in_grid]],
                             uint tiitg                   [[thread_index_in_threadgroup]],
                             uint sgitg                   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = 64u;  // rows per threadgroup
    constexpr uint NR1 = 32u;  // tokens per threadgroup
    constexpr uint NK = 64u;   // columns per step, one Q2_0 block
    threadgroup half sa[NR0 * NK];  // [row][k]
    threadgroup half sb[NR1 * NK];  // [token][k]

    const uint row0 = tgpig.y * NR0;
    const uint tok0 = tgpig.x * NR1;
    const uint row_count = min(NR0, rows - row0);
    const uint tok_count = min(NR1, tokens - tok0);

    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < columns; k0 += NK) {
        // Weights: two threads per row, eight code bytes each (32 codes).
        {
            const uint row = tiitg / 2u;
            const uint part = tiitg % 2u;
            const float d = row < row_count
                                ? float(as_type<half>(ushort(w[(ulong)(row0 + row) * row_bytes + (k0 / 64u) * 18u] |
                                                             (ushort(w[(ulong)(row0 + row) * row_bytes + (k0 / 64u) * 18u + 1u]) << 8))))
                                : 0.0f;
            device const uchar* q = w + (ulong)(row0 + row) * row_bytes + (k0 / 64u) * 18u + 2u + part * 8u;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uchar byte = row < row_count ? q[i] : 0u;
                sa[(part * 32u + i * 4u + 0u) * NR0 + row] = half(float(int(byte & 3u) - 1) * d);
                sa[(part * 32u + i * 4u + 1u) * NR0 + row] = half(float(int((byte >> 2u) & 3u) - 1) * d);
                sa[(part * 32u + i * 4u + 2u) * NR0 + row] = half(float(int((byte >> 4u) & 3u) - 1) * d);
                sa[(part * 32u + i * 4u + 3u) * NR0 + row] = half(float(int((byte >> 6u) & 3u) - 1) * d);
            }
        }
        // Activations: one tile of the batch, transposed to [token][k].
        // Four consecutive columns per thread, loaded as one 16-byte vector and
        // stored as one half4. The scalar version (one float per thread per
        // iteration) was the kernel's whole bottleneck: with the tiles left
        // static, the same loop runs at 5.85 TFLOP/s instead of 1.76
        // (benchmarks/metal/q2_0_mat_mat_bench.mm).
        for (uint idx = tiitg; idx < NR1 * (NK / 4u); idx += 128u) {
            const uint tok = idx / (NK / 4u);
            const uint k = (idx % (NK / 4u)) * 4u;
            float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (tok < tok_count && k0 + k + 3u < columns) {
                v = *reinterpret_cast<device const float4*>(x + (ulong)(tok0 + tok) * x_stride + k0 + k);
            }
            *reinterpret_cast<threadgroup half4*>(sb + tok * NK + k) = half4(v.x, v.y, v.z, v.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* sa_q = sa + 32u * (sgitg & 1u);    // this simdgroup's 32 rows, tile is [k][row]
        threadgroup const half* sb_q = sb + 16u * (sgitg >> 1u) * NK;  // this simdgroup's 16 tokens
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < NK / 8u; ++kk) {
            simdgroup_half8x8 mb[2];
            simdgroup_half8x8 ma[4];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                // [row][k] -> [k][row], which is what A*B needs on the right.
                simdgroup_load(ma[i], sa_q + kk * 8u * NR0 + i * 8u, NR0, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(mb[i], sb_q + i * 8u * NK + kk * 8u, NK, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // mc[i] holds [token][row] for tokens 8*(i/4).., rows 8*(i%4).. of this
    // simdgroup's quadrant. Stage the tile, then write it out with `scale`.
    // The 8 KB weight tile is dead once the k loop ends, so the float staging
    // tile reuses it: 12 KB of threadgroup memory per threadgroup instead of 20,
    // which lets two of them share a core.
    threadgroup float* tile = reinterpret_cast<threadgroup float*>(sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_store(mc[i], tile + (16u * (sgitg >> 1u) + 8u * (i / 4u)) * NR0 + 32u * (sgitg & 1u) + 8u * (i % 4u),
                        NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tiitg; idx < NR0 * NR1; idx += 128u) {
        const uint tok = idx / NR0;
        const uint row = idx % NR0;
        if (tok < tok_count && row < row_count) {
            out[(ulong)(tok0 + tok) * out_stride + row0 + row] = tile[idx] * scale;
        }
    }
}
kernel void q2_0_mat_mat_f16_static(device const uchar* w        [[buffer(0)]],
                             device const float* x        [[buffer(1)]],
                             device float* out            [[buffer(3)]],
                             constant uint& row_bytes     [[buffer(4)]],
                             constant uint& columns       [[buffer(5)]],
                             constant float& scale        [[buffer(6)]],
                             constant uint& rows          [[buffer(8)]],
                             constant uint& tokens        [[buffer(9)]],
                             constant uint& x_stride      [[buffer(10)]],
                             constant uint& out_stride    [[buffer(11)]],
                             uint3 tgpig                  [[threadgroup_position_in_grid]],
                             uint tiitg                   [[thread_index_in_threadgroup]],
                             uint sgitg                   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = 64u;  // rows per threadgroup
    constexpr uint NR1 = 32u;  // tokens per threadgroup
    constexpr uint NK = 64u;   // columns per step, one Q2_0 block
    threadgroup half sa[NR0 * NK];  // [row][k]
    threadgroup half sb[NR1 * NK];  // [token][k]

    const uint row0 = tgpig.y * NR0;
    const uint tok0 = tgpig.x * NR1;
    const uint row_count = min(NR0, rows - row0);
    const uint tok_count = min(NR1, tokens - tok0);

    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < columns; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* sa_q = sa + 32u * (sgitg & 1u);    // this simdgroup's 32 rows, tile is [k][row]
        threadgroup const half* sb_q = sb + 16u * (sgitg >> 1u) * NK;  // this simdgroup's 16 tokens
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < NK / 8u; ++kk) {
            simdgroup_half8x8 mb[2];
            simdgroup_half8x8 ma[4];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                // [row][k] -> [k][row], which is what A*B needs on the right.
                simdgroup_load(ma[i], sa_q + kk * 8u * NR0 + i * 8u, NR0, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(mb[i], sb_q + i * 8u * NK + kk * 8u, NK, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // mc[i] holds [token][row] for tokens 8*(i/4).., rows 8*(i%4).. of this
    // simdgroup's quadrant. Stage the tile, then write it out with `scale`.
    // The 8 KB weight tile is dead once the k loop ends, so the float staging
    // tile reuses it: 12 KB of threadgroup memory per threadgroup instead of 20,
    // which lets two of them share a core.
    threadgroup float* tile = reinterpret_cast<threadgroup float*>(sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_store(mc[i], tile + (16u * (sgitg >> 1u) + 8u * (i / 4u)) * NR0 + 32u * (sgitg & 1u) + 8u * (i % 4u),
                        NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tiitg; idx < NR0 * NR1; idx += 128u) {
        const uint tok = idx / NR0;
        const uint row = idx % NR0;
        if (tok < tok_count && row < row_count) {
            out[(ulong)(tok0 + tok) * out_stride + row0 + row] = tile[idx] * scale;
        }
    }
}
)MSL";

namespace {

struct Shape {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t tokens;
    const char* name;
};

double elapsed_ms(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> weights, id<MTLBuffer> x, id<MTLBuffer> out,
                  const Shape& shape, std::uint32_t dispatches, std::uint32_t submissions) {
    const std::uint32_t row_bytes = shape.columns / 64u * 18u;
    const std::uint32_t x_stride = shape.columns;
    const std::uint32_t out_stride = shape.rows;
    const float scale = 0.0078125f;
    const auto encode = [&](id<MTLCommandBuffer> command) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (std::uint32_t i = 0; i < dispatches; ++i) {
            [encoder setBuffer:weights offset:0 atIndex:0];
            [encoder setBuffer:x offset:0 atIndex:1];
            [encoder setBuffer:out offset:0 atIndex:3];
            [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:4];
            [encoder setBytes:&shape.columns length:sizeof(shape.columns) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&shape.rows length:sizeof(shape.rows) atIndex:8];
            [encoder setBytes:&shape.tokens length:sizeof(shape.tokens) atIndex:9];
            [encoder setBytes:&x_stride length:sizeof(x_stride) atIndex:10];
            [encoder setBytes:&out_stride length:sizeof(out_stride) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake((shape.tokens + 31u) / 32u, (shape.rows + 63u) / 64u, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    };
    // Warm up.
    {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
    }
    const auto start = std::chrono::steady_clock::now();
    const std::uint32_t rounds = 5;
    for (std::uint32_t round = 0; round < rounds; ++round) {
        for (std::uint32_t s = 0; s < submissions; ++s) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            encode(command);
            [command commit];
            [command waitUntilCompleted];
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count() / rounds;
    const double macs = static_cast<double>(shape.rows) * shape.columns * shape.tokens * dispatches;
    return 2.0 * macs / seconds / 1e12;  // TFLOP/s
}

}  // namespace

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
    id<MTLFunction> static_function = [library newFunctionWithName:@"q2_0_mat_mat_f16_static"];
    id<MTLComputePipelineState> static_pipeline =
        [device newComputePipelineStateWithFunction:static_function error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"q2_0_mat_mat_f16"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        std::fprintf(stderr, "pipeline failed: %s\n", [[error description] UTF8String]);
        return 1;
    }

    // The FFN's three shapes, at 32 tokens and at 1 token.
    const Shape shapes[] = {
        {17408u, 5120u, 32u, "gate/up 17408x5120, 32 tokens"},
        {5120u, 17408u, 32u, "down    5120x17408, 32 tokens"},
        {17408u, 5120u, 1u, "gate/up 17408x5120, 1 token"},
    };
    std::printf("isolated q2_0_mat_mat_f16 (one dispatch repeated)\n");
    std::printf("%-34s %12s %12s %12s %12s\n", "shape", "1 submiss", "6 disp/sub", "1 disp/sub", "static tiles");
    for (const auto& shape : shapes) {
        const std::uint32_t row_bytes = shape.columns / 64u * 18u;
        const std::uint64_t weight_bytes = static_cast<std::uint64_t>(shape.rows) * row_bytes;
        std::vector<std::uint8_t> weights(weight_bytes, 0x55u);
        id<MTLBuffer> wbuf = [device newBufferWithBytes:weights.data()
                                                 length:static_cast<NSUInteger>(weight_bytes)
                                                options:MTLResourceStorageModeShared];
        std::vector<float> act(static_cast<std::size_t>(shape.tokens) * shape.columns, 0.01f);
        id<MTLBuffer> xbuf = [device newBufferWithBytes:act.data()
                                                 length:static_cast<NSUInteger>(act.size() * sizeof(float))
                                                options:MTLResourceStorageModeShared];
        std::vector<float> out(static_cast<std::size_t>(shape.tokens) * shape.rows, 0.0f);
        id<MTLBuffer> obuf = [device newBufferWithBytes:out.data()
                                                 length:static_cast<NSUInteger>(out.size() * sizeof(float))
                                                options:MTLResourceStorageModeShared];
        const double tflops_isolated = elapsed_ms(queue, pipeline, wbuf, xbuf, obuf, shape, 1u, 1u);
        const double tflops_chain = elapsed_ms(queue, pipeline, wbuf, xbuf, obuf, shape, 6u, 1u);
        const double tflops_per_dispatch = elapsed_ms(queue, pipeline, wbuf, xbuf, obuf, shape, 1u, 6u);
        const double tflops_static = elapsed_ms(queue, static_pipeline, wbuf, xbuf, obuf, shape, 1u, 1u);
        std::printf("%-34s %12.2f %12.2f %12.2f %12.2f\n", shape.name, tflops_isolated, tflops_chain,
                    tflops_per_dispatch, tflops_static);
    }
    return 0;
}
