// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// What is this GPU's actual simdgroup_matrix issue rate, and how many warps does
// it take to reach it? The tiled mat-mat (q2_0_mat_mat_f16) runs at 1.7-2.9
// TFLOP/s while llama.cpp's mul_mm reaches 5.8 on the same model, and five
// kernel-shape experiments (log 120, 121) failed to move it: decode, bank
// conflicts, double buffering and tile size were each ruled out by measurement.
// The remaining difference is per-warp concurrency, so this bench measures the
// pure mma loop - same 8 accumulators, same 6 threadgroup loads per 8 mma as the
// real kernel - while sweeping threadgroup size and grid size.
//
// Measured on the M4 Pro (TFLOP/s, 2048 iterations per threadgroup):
//
//   groups  32thr  64thr  128thr  256thr        with one barrier per k-step
//   20       0.29   0.71    1.35    2.20            1.03   1.63   3.23   5.17
//   40       0.71   1.42    2.72    3.09            2.12   3.43   5.76   8.14
//   80       1.71   2.94    7.00    6.03            4.18   6.80  11.00  13.72
//   160      3.27   7.64    6.82    9.68            8.62  11.27  15.27  16.93
//   320      5.64   8.82   11.67   14.36           11.46  14.31  17.82  19.24
//   640      8.38  15.45   18.28   22.89           12.49  16.90  20.47  21.04
//
// So the same 8-accumulator, 6-loads-per-8-mma loop the real kernel runs
// reaches 15-20 TFLOP/s at the grid sizes it actually uses (272 threadgroups of
// 128 threads for a FFN gate matvec), while q2_0_mat_mat_f16 measures ~2. That
// makes the shortfall an implementation problem, not a hardware or tile-shape
// one - and it rules out the hypotheses already tested (decode, bank conflicts,
// double buffering, tile size: log 120-121).
//
// Build on the Mac from the repository root:
//   clang++ -std=c++20 -O2 -x objective-c++ -isysroot \
//     /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/mma_rate_bench.mm -x none -framework Metal \
//     -framework Foundation -lobjc -o mma_rate_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static const char* kSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// One threadgroup: every simdgroup runs the same inner loop as
// q2_0_mat_mat_f16's k loop (4 weight-tile loads, 2 activation-tile loads,
// 8 multiply-accumulates), reading from threadgroup memory that is never
// rewritten, so the measurement is the matrix-unit rate and nothing else.
kernel void mma_loop(device float* out            [[buffer(0)]],
                     constant uint& iterations    [[buffer(1)]],
                     threadgroup half* sa         [[threadgroup(0)]],
                     threadgroup half* sb         [[threadgroup(1)]],
                     uint tiitg                   [[thread_index_in_threadgroup]],
                     uint sgitg                   [[simdgroup_index_in_threadgroup]],
                     uint sgsize                  [[simdgroups_per_threadgroup]]) {
    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    threadgroup half* wa = sa + (sgitg % 4u) * 512u;
    threadgroup half* wb = sb + (sgitg % 4u) * 512u;
    for (uint it = 0; it < iterations; ++it) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_load(ma[i], wa + ((it + i) & 7u) * 64u, 8, 0, false);
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < 2u; ++i) {
            simdgroup_load(mb[i], wb + ((it + i) & 7u) * 64u, 8, 0, false);
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
        }
    }
    // Keep the loop alive without a device store per simdgroup.
    threadgroup float* sink = reinterpret_cast<threadgroup float*>(sa);
    if (tiitg == 0u) {
        sink[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(mc[0], sink + sgitg * 64u, 8, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg < sgsize) {
        out[tiitg] = sink[tiitg * 64u];
    }
}

// Same loop with a threadgroup barrier per iteration: the real kernel needs one
// between staging a k-step and consuming it, and this isolates what that costs.
kernel void mma_loop_barrier(device float* out        [[buffer(0)]],
                             constant uint& iterations [[buffer(1)]],
                             threadgroup half* sa      [[threadgroup(0)]],
                             threadgroup half* sb      [[threadgroup(1)]],
                             uint tiitg                [[thread_index_in_threadgroup]],
                             uint sgitg                [[simdgroup_index_in_threadgroup]],
                             uint sgsize               [[simdgroups_per_threadgroup]]) {
    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    threadgroup half* wa = sa + (sgitg % 4u) * 512u;
    threadgroup half* wb = sb + (sgitg % 4u) * 512u;
    for (uint it = 0; it < iterations; ++it) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            simdgroup_load(ma[i], wa + ((it + i) & 7u) * 64u, 8, 0, false);
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < 2u; ++i) {
            simdgroup_load(mb[i], wb + ((it + i) & 7u) * 64u, 8, 0, false);
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* sink = reinterpret_cast<threadgroup float*>(sa);
    if (tiitg == 0u) {
        sink[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(mc[0], sink + sgitg * 64u, 8, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg < sgsize) {
        out[tiitg] = sink[tiitg * 64u];
    }
}
)MSL";

namespace {

double run(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
           std::uint32_t threads, std::uint32_t groups, std::uint32_t iterations) {
    id<MTLBuffer> out = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:out offset:0 atIndex:0];
    [encoder setBytes:&iterations length:sizeof(iterations) atIndex:1];
    [encoder setThreadgroupMemoryLength:4096 atIndex:0];
    [encoder setThreadgroupMemoryLength:4096 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return -1.0;
    }
    // Two passes timed together to cover the launch latency.
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 3; ++pass) {
        id<MTLCommandBuffer> timed = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [timed computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBytes:&iterations length:sizeof(iterations) atIndex:1];
        [enc setThreadgroupMemoryLength:4096 atIndex:0];
        [enc setThreadgroupMemoryLength:4096 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [timed commit];
        [timed waitUntilCompleted];
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count() / 3.0;
    const double simdgroups = static_cast<double>(groups) * (threads / 32.0);
    const double macs = simdgroups * static_cast<double>(iterations) * 8.0 * 512.0;
    return 2.0 * macs / seconds / 1e12;
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
    id<MTLFunction> function = [library newFunctionWithName:@"mma_loop"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLFunction> barred_function = [library newFunctionWithName:@"mma_loop_barrier"];
    id<MTLComputePipelineState> barred = [device newComputePipelineStateWithFunction:barred_function error:&error];
    if (pipeline == nil || barred == nil) {
        std::fprintf(stderr, "pipeline failed: %s\n", [[error description] UTF8String]);
        return 1;
    }
    std::printf("M4 Pro mma rate (TFLOP/s), 8 accumulators, 6 tile loads per 8 mma\n");
    std::printf("%8s", "groups");
    for (const std::uint32_t threads : {32u, 64u, 128u, 256u}) {
        std::printf("%10u thr", threads);
    }
    std::printf("\n");
    for (const std::uint32_t groups : {20u, 40u, 80u, 160u, 320u, 640u}) {
        std::printf("%8u", groups);
        for (const std::uint32_t threads : {32u, 64u, 128u, 256u}) {
            const double tflops = run(device, queue, pipeline, threads, groups, 2048u);
            std::printf("%13.2f", tflops);
        }
        std::printf("\n");
    }
    std::printf("\nwith one threadgroup barrier per k-step (the real kernel's pattern)\n");
    std::printf("%8s", "groups");
    for (const std::uint32_t threads : {32u, 64u, 128u, 256u}) {
        std::printf("%10u thr", threads);
    }
    std::printf("\n");
    for (const std::uint32_t groups : {20u, 40u, 80u, 160u, 320u, 640u}) {
        std::printf("%8u", groups);
        for (const std::uint32_t threads : {32u, 64u, 128u, 256u}) {
            std::printf("%13.2f", run(device, queue, barred, threads, groups, 2048u));
        }
        std::printf("\n");
    }
    return 0;
}
