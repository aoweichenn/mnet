"""Assemble llama.cpp's mul_mm Q2_0 kernel into a standalone Metal bench.

Assembles llama.cpp's Q2_0 mat-mat (common.h, the Q2_0 dequantiser and the plain
kernel_mul_mm) into one self-contained .mm so it can be measured on this
repository's shapes and compared with q2_0_mat_mat_f16.  Everything the model
never takes is dropped (f16/f32 weights, bf16, the expert-mapping variants) and
the FC_mul_mm_* function constants become literals.

Usage: `python3 benchmarks/metal/port_llama_mul_mm.py [llama.cpp root]` writes
`$HOME/mnet_p4/mm_port.mm`; build and run it with

    clang++ -std=c++20 -O2 -x objective-c++ \
        -isysroot "$(xcrun --show-sdk-path)" mm_port.mm -x none \
        -framework Metal -framework Foundation -lobjc -o mm_port && ./mm_port

The generated file is not committed: it is llama.cpp's source (MIT) assembled for
one measurement, not part of this library.
"""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "llama.cpp"
SRC = ROOT / "ggml/src/ggml-metal"
CONSTANTS = [
    ("FC_mul_mm_bc_inp", "false"),
    ("FC_mul_mm_bc_out", "false"),
    ("FC_mul_mm_ne12", "1"),
    ("FC_mul_mm_ne13", "1"),
    ("FC_mul_mm_r2", "1"),
    ("FC_mul_mm_r3", "1"),
    ("FC_mul_mm_id_map0_ne02", "0"),
]


def as_literal(line):
    for name, value in CONSTANTS:
        if re.match(r"constant\s+\w+\s+" + name + r"\s+\[\[function_constant", line):
            return "constant auto " + name + " = " + value + ";"
    return None


common = "\n".join(
    line
    for line in (SRC / "kernels/common.h").read_text().splitlines()
    if not line.strip().startswith("#include")
)

# Only dequantize_q2_0: the rest of dequantize.h names block types that would
# drag in all of ggml-common.h.
deq_lines = (SRC / "kernels/dequantize.h").read_text().splitlines()
start = next(i for i, l in enumerate(deq_lines) if l.startswith("void dequantize_q2_0("))
while deq_lines[start - 1].strip().startswith("template"):
    start -= 1
end = next(i for i in range(start, len(deq_lines)) if deq_lines[i].rstrip() == "}")
dequantize = "\n".join(deq_lines[start:end + 1])

blocks = """
#define QK2_0 64
typedef half ggml_half_x;
typedef struct {
    ggml_half_x d;
    uint8_t qs[QK2_0 / 4];
} block_q2_0;
"""

kargs = """
typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm;
"""

mm_lines = (SRC / "kernels/mul_mm.metal").read_text().splitlines()
decls = []
body = []
for line in mm_lines:
    literal = as_literal(line)
    if literal is not None:
        decls.append(literal)
        continue
    if line.strip().startswith("#include") or "host_name(" in line:
        continue
    body.append(line)

kernel_start = next(i for i, l in enumerate(body) if l.startswith("kernel void kernel_mul_mm("))
while body[kernel_start - 1].strip().startswith("template") or body[kernel_start - 1].strip().startswith("typename"):
    kernel_start -= 1
kernel_end = next(i for i, l in enumerate(body) if l.startswith("kernel void kernel_mul_mm_id_map0("))
slice_lines = body[kernel_start:kernel_end]
# The next function's template header sits just above its signature; drop the
# trailing blanks and that dangling `template<...>` line.
while slice_lines and (slice_lines[-1].strip() == "" or slice_lines[-1].lstrip().startswith("template")):
    slice_lines.pop()
kernel = "\n".join(slice_lines)
if len(decls) != len(CONSTANTS):
    raise SystemExit("expected %d function constants, found %d" % (len(CONSTANTS), len(decls)))

instantiation = """
typedef decltype(kernel_mul_mm<half, half, block_q2_0, 4, dequantize_q2_0, float, float>) mul_mm_t;

// llama.cpp's default (half accumulation) and the f32-precision variant.
template [[host_name("kernel_mul_mm_q2_0_f32")]]      kernel mul_mm_t kernel_mul_mm<half,  half,  block_q2_0, 4, dequantize_q2_0, float, float>;
template [[host_name("kernel_mul_mm_q2_0_f32_f32p")]] kernel mul_mm_t kernel_mul_mm<float, float, block_q2_0, 4, dequantize_q2_0, float, float>;
"""

msl = (
    "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
    "using namespace metal;\n"
    + common
    + "\n"
    + blocks
    + "\n#define QK_NL 16\n"
    + dequantize
    + "\n"
    + kargs
    + "\n"
    + "\n".join(decls)
    + "\n"
    + kernel
    + "\n"
    + instantiation
)

host_header = r"""
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} args_mul_mm;
"""

host_main = r"""
static const char * kSource = R"MSL(
__MSL__
)MSL";

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError * error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                                  options:nil
                                                    error:&error];
    if (library == nil) {
        std::fprintf(stderr, "compile failed: %s\n", [[error description] UTF8String]);
        return 1;
    }

    // The three shapes the forward actually runs, so the comparison against the
    // shipped kernel is per shape rather than only for the largest one.
    const int32_t shapes[3][2] = {{17408, 5120}, {5120, 17408}, {5120, 6144}};
    const int32_t tokens = 32;

    for (int shape = 0; shape < 3; ++shape) {
    const int32_t rows = shapes[shape][0];
    const int32_t columns = shapes[shape][1];
    const uint64_t row_bytes = static_cast<uint64_t>(columns / 64) * 18;

    std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * row_bytes);
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<std::uint8_t>(0x5A + (i & 7));
    }
    for (std::size_t i = 0; i < w.size(); i += 18) {
        w[i] = 0x00;
        w[i + 1] = 0x3C;  // fp16 1.0
    }
    std::vector<float> x(static_cast<std::size_t>(tokens) * columns, 0.01f);
    std::vector<float> out(static_cast<std::size_t>(tokens) * rows, 0.0f);

    id<MTLBuffer> wb = [device newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> xb = [device newBufferWithBytes:x.data()
                                          length:x.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [device newBufferWithBytes:out.data()
                                          length:out.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];

    const char * names[2] = {"kernel_mul_mm_q2_0_f32", "kernel_mul_mm_q2_0_f32_f32p"};
    for (int variant = 0; variant < 2; ++variant) {
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:names[variant]]];
        if (fn == nil) {
            std::fprintf(stderr, "%s: function missing\n", names[variant]);
            return 1;
        }
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];
        if (pipe == nil) {
            std::fprintf(stderr, "%s: pipeline failed: %s\n", names[variant], [[error description] UTF8String]);
            return 1;
        }

        args_mul_mm args{};
        args.ne00 = columns;
        args.ne02 = 1;
        args.nb01 = row_bytes;
        args.nb02 = row_bytes * static_cast<uint64_t>(rows);
        args.nb03 = 0;
        args.ne12 = 1;
        args.nb10 = 4;
        args.nb11 = 4ull * static_cast<uint64_t>(columns);
        args.nb12 = 0;
        args.nb13 = 0;
        args.ne0 = rows;
        args.ne1 = tokens;
        args.r2 = 1;
        args.r3 = 1;

        const std::uint32_t nr0 = 64;
        const std::uint32_t nr1 = 32;
        const std::uint32_t nsg = 4;
        const std::size_t smem = 12288;

        const auto encode = [&](id<MTLCommandBuffer> command) {
            id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wb offset:0 atIndex:1];
            [enc setBuffer:xb offset:0 atIndex:2];
            [enc setBuffer:ob offset:0 atIndex:3];
            [enc setThreadgroupMemoryLength:smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((tokens + nr1 - 1) / nr1, (rows + nr0 - 1) / nr0, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
            [enc endEncoding];
        };

        {
            id<MTLCommandBuffer> c = [queue commandBuffer];
            encode(c);
            [c commit];
            [c waitUntilCompleted];
            if (c.status == MTLCommandBufferStatusError) {
                std::fprintf(stderr, "%s: GPU error %s\n", names[variant], [[c.error description] UTF8String]);
                return 1;
            }
        }

        // Row 0, token 0 against a scalar reference over the first weight row:
        // confirms the port computes this model's Q2_0 layout, not just that it runs.
        double reference = 0.0;
        for (int32_t b = 0; b < columns / 64; ++b) {
            const std::uint8_t * blk = w.data() + static_cast<std::size_t>(b) * 18;
            for (int g = 0; g < 2; ++g) {
                for (int j = 0; j < 8; ++j) {
                    const std::uint8_t byte = blk[2 + g * 8 + j];
                    for (int k = 0; k < 4; ++k) {
                        const int code = (byte >> (2 * k)) & 3;
                        const std::size_t idx = static_cast<std::size_t>(b) * 64 + g * 32 + j * 4 + k;
                        reference += static_cast<double>(code - 1) * x[idx];
                    }
                }
            }
        }
        // The MTL buffer copied the host vector, so read the result back from
        // the buffer itself.
        const float * got = static_cast<const float *>(ob.contents);
        std::printf("%-32s out[0] = %.6f  ref = %.6f  diff = %.2e  out[last] = %.4f\n", names[variant],
                    static_cast<double>(got[0]), reference,
                    static_cast<double>(got[0]) - reference,
                    static_cast<double>(got[out.size() - 1]));

        const int reps = 10;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            id<MTLCommandBuffer> c = [queue commandBuffer];
            encode(c);
            [c commit];
            [c waitUntilCompleted];
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        const double macs = static_cast<double>(rows) * columns * tokens;
        std::printf("%-28s %5dx%-5d %-24s %.3f ms -> %.2f TFLOP/s\n", "llama mul_mm", rows, columns,
                    names[variant] + 20, ms, 2.0 * macs / (ms / 1000.0) / 1e12);
    }
    }

    std::printf("shipped q2_0_mat_mat_f16 (same harness, 32 tokens):\n"
                "  17408x5120 4.87   5120x17408 4.24   5120x6144 ~4.0 TFLOP/s\n");
    return 0;
}
"""

host = host_header + host_main.replace("__MSL__", msl)
out = Path.home() / "mnet_p4/mm_port.mm"  # scratch, never inside the tree
out.write_text(host)
print("wrote", out, len(host.splitlines()), "lines")
