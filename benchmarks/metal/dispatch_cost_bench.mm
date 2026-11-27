// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Calibration for the "what does a dispatch cost" question: the same number of
// no-op dispatches encoded as one submission per ten, versus one per submission.
//
// Measured on the M4 Pro:
//   grid     1: 10x10 in 10 submits   1.534 ms   0.015 ms/dispatch
//   grid     1: 100x1 in 100 submits 12.329 ms   0.123 ms/dispatch
//   grid 10240: 10x10 in 10 submits   7.208 ms   0.072 ms/dispatch
//   grid 10240: 100x1 in 100 submits 17.827 ms   0.178 ms/dispatch
//
// so a submission costs ~0.12 ms and a dispatch inside one submission
// 0.015-0.072 ms depending on the grid. That means the gated-delta block's
// 11.3 ms of GPU wait for a single token cannot be submission or dispatch
// overhead: something in the real block costs an order of magnitude more, and
// the standing suspect is the per-dispatch binding of the 6.7 GB aliased weight
// buffer (this bench binds 16 bytes).
//
// Build on the Mac:
//   clang++ -std=c++20 -O2 -x objective-c++ -isysroot \
//     /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     benchmarks/metal/dispatch_cost_bench.mm -x none -framework Metal \
//     -framework Foundation -lobjc -o dispatch_cost_bench

#include <cstdio>

static const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void noop(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0u) { out[0] = 1.0f; }
}
)MSL";

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:nil error:&error];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"noop"] error:&error];
    id<MTLBuffer> buffer = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];

    const auto run = [&](int submissions, int dispatches_per_submit, NSUInteger grid, NSUInteger threads) {
        const auto begin = std::chrono::steady_clock::now();
        for (int s = 0; s < submissions; ++s) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            for (int d = 0; d < dispatches_per_submit; ++d) {
                [encoder dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            }
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    };

    for (int i = 0; i < 5; ++i) { run(1, 10, 1, 32); }
    const auto report = [&](const char* label, double seconds, int dispatches) {
        std::printf("%-34s %8.3f ms total  %7.3f ms/dispatch\n", label, seconds * 1e3,
                    seconds * 1e3 / static_cast<double>(dispatches));
    };
    for (const auto grid : {1U, 10240U}) {
        const int submissions = 10;
        const int per = 10;
        const double one = run(submissions, per, grid, 32);          // 10 submits x 10 dispatches
        const double many = run(submissions * per, 1, grid, 32);     // 100 submits x 1 dispatch
        char label[96];
        std::snprintf(label, sizeof(label), "grid %5u: 10x10 in 10 submits", grid);
        report(label, one, submissions * per);
        std::snprintf(label, sizeof(label), "grid %5u: 100x1 in 100 submits", grid);
        report(label, many, submissions * per);
    }
    return 0;
}
