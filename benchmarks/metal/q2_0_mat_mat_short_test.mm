// Isolated check for the 32-row mat-mat tile: both kernels on the same weights
// and activations, compared element by element. Written after two failed
// attempts to land the short-matrix tile in the model (log 147/148), where the
// only signal was a NaN tail in the final logits.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstring>

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
kernel void q2_0_mat_mat_f16_short(device const uchar* w        [[buffer(0)]],
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
    constexpr uint NR0 = 32u;
    constexpr uint NR1 = 32u;
    constexpr uint NK = 64u;
    threadgroup half sa[NR0 * NK];  // [k][row], stride NR0
    threadgroup half sb[NR1 * NK];  // [token][k], stride NK

    const uint row0 = tgpig.y * NR0;
    const uint tok0 = tgpig.x * NR1;
    const uint row_count = min(NR0, rows - row0);
    const uint tok_count = min(NR1, tokens - tok0);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < columns; k0 += NK) {
        {
            const uint row = tiitg % NR0;
            const uint part = tiitg / NR0;
            const bool live = row < row_count;
            const device uchar* head = w + (ulong)(row0 + row) * row_bytes + (k0 / 64u) * 18u;
            const float d = live ? float(as_type<half>(ushort(head[0] | (ushort(head[1]) << 8)))) : 0.0f;
            device const uchar* q = head + 2u + part * 4u;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                const uchar byte = live ? q[i] : 0u;
                const uint kk = part * 16u + i * 4u;
                sa[(kk + 0u) * NR0 + row] = half(float(int(byte & 3u) - 1) * d);
                sa[(kk + 1u) * NR0 + row] = half(float(int((byte >> 2u) & 3u) - 1) * d);
                sa[(kk + 2u) * NR0 + row] = half(float(int((byte >> 4u) & 3u) - 1) * d);
                sa[(kk + 3u) * NR0 + row] = half(float(int((byte >> 6u) & 3u) - 1) * d);
            }
        }
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

        const uint row_block0 = 16u * (sgitg & 1u);
        const uint tok_block0 = 16u * (sgitg >> 1u);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < NK / 8u; ++kk) {
            simdgroup_half8x8 mb[2];
            simdgroup_half8x8 ma[2];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(ma[i], sa + (kk * 8u) * NR0 + row_block0 + i * 8u, NR0, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(mb[i], sb + (tok_block0 + i * 8u) * NK + kk * 8u, NK, 0, false);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 2u], ma[i % 2u], mc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* tile = reinterpret_cast<threadgroup float*>(sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) {
        simdgroup_store(mc[i], tile + (16u * (sgitg >> 1u) + 8u * (i / 2u)) * NR0 + 16u * (sgitg & 1u) + 8u * (i % 2u),
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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::uint32_t kRows = 5120;
constexpr std::uint32_t kColumns = 6144;
constexpr std::uint32_t kTokens = 32;
constexpr std::uint32_t kRowBytes = kColumns / 64u * 18u;

struct Result {
    std::vector<float> out;
    double ms = 0.0;
};

std::uint32_t rng_state = 12345u;
std::uint32_t next_random() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8u;
}

Result run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLBuffer> weights,
           id<MTLBuffer> x, std::uint32_t row_tile) {
    const std::uint32_t rows32 = kRows;
    const std::uint32_t tokens32 = kTokens;
    const std::uint32_t columns32 = kColumns;
    const std::uint32_t row_bytes32 = kRowBytes;
    const std::uint32_t x_stride32 = kColumns;
    const std::uint32_t out_stride32 = kRows;
    const float scale = 1.0f;
    Result result;
    result.out.assign(static_cast<std::size_t>(kRows) * kTokens, 0.0f);
    id<MTLBuffer> out = [queue.device newBufferWithLength:kRows * kTokens * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    const auto encode = [&](id<MTLCommandBuffer> command) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weights offset:0 atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
        [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake(kTokens / 32u, (kRows + row_tile - 1u) / row_tile, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    };
    {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
    }
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 5; ++pass) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        encode(command);
        [command commit];
        [command waitUntilCompleted];
    }
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5.0;
    std::memcpy(result.out.data(), out.contents, result.out.size() * sizeof(float));
    return result;
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
    id<MTLComputePipelineState> ref_pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"q2_0_mat_mat_f16"] error:&error];
    id<MTLComputePipelineState> short_pipeline = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"q2_0_mat_mat_f16_short"]
                                      error:&error];
    if (ref_pipeline == nil || short_pipeline == nil) {
        std::fprintf(stderr, "pipeline failed: %s\n", [[error description] UTF8String]);
        return 1;
    }

    // Q2_0 weights: a half scale plus 16 code bytes per 64 weights.
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(kRows) * kRowBytes);
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t b = 0; b < kColumns / 64u; ++b) {
            std::uint8_t* block = weights.data() + static_cast<std::size_t>(row) * kRowBytes + b * 18u;
            const std::uint16_t d = static_cast<std::uint16_t>(0x2000u + (next_random() % 0x0800u));  // ~0.03-0.06
            block[0] = static_cast<std::uint8_t>(d & 0xFFu);
            block[1] = static_cast<std::uint8_t>(d >> 8u);
            for (std::uint32_t i = 2; i < 18u; ++i) {
                block[i] = static_cast<std::uint8_t>(next_random() & 0xFFu);
            }
        }
    }
    std::vector<float> acts(static_cast<std::size_t>(kTokens) * kColumns);
    for (auto& value : acts) {
        value = (static_cast<float>(next_random() % 2000u) / 1000.0f) - 1.0f;
    }
    id<MTLBuffer> wbuf = [device newBufferWithBytes:weights.data()
                                             length:weights.size()
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> xbuf = [device newBufferWithBytes:acts.data()
                                             length:acts.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];

    const Result ref = run(queue, ref_pipeline, wbuf, xbuf, 64u);
    const Result got = run(queue, short_pipeline, wbuf, xbuf, 32u);

    double worst = 0.0;
    std::size_t first_bad = SIZE_MAX;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < ref.out.size(); ++i) {
        const double diff = std::abs(static_cast<double>(ref.out[i]) - static_cast<double>(got.out[i]));
        if (diff > 1e-3) {
            ++bad;
            if (first_bad == SIZE_MAX) {
                first_bad = i;
            }
            worst = std::max(worst, diff);
        }
    }
    std::printf("64-row %.3f ms, 32-row %.3f ms, differing %zu of %zu, worst %.4f\n", ref.ms, got.ms, bad,
                ref.out.size(), worst);
    if (first_bad != SIZE_MAX) {
        const std::size_t token = first_bad / kRows;
        const std::size_t row = first_bad % kRows;
        std::printf("first mismatch at token %zu row %zu: ref %.4f got %.4f\n", token, row, ref.out[first_bad],
                    got.out[first_bad]);
        // Which quadrant? (row % 32, token % 32) tells the tile-local position.
        std::printf("tile-local row %zu token %zu\n", row % 32u, token % 32u);
        for (std::uint32_t r = 0; r < 32u; ++r) {
            std::size_t row_bad = 0;
            for (std::uint32_t t = 0; t < kTokens; ++t) {
                if (std::abs(static_cast<double>(ref.out[static_cast<std::size_t>(t) * kRows + r]) -
                             static_cast<double>(got.out[static_cast<std::size_t>(t) * kRows + r])) > 1e-3) {
                    ++row_bad;
                }
            }
            if (row_bad > 0) {
                std::printf("  tile row %2u: %zu of %u tokens wrong\n", r, row_bad, kTokens);
            }
        }
    }
    return bad == 0 ? 0 : 3;
}
