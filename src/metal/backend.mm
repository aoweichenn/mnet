// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Metal provider for the device-neutral compute layer. Only this file knows
// about Metal: it registers a Q2_0 matvec with mnet::detail::compute and is
// linked only when mnet_enable_metal is ON. The kernel reproduces the CPU
// accumulation order byte for byte (eight byte-lanes, two alternating block
// accumulators, the same hsum tree), so results match the CPU path exactly.

#include <mnet/detail/compute.hpp>

#include <mnet/detail/txn_log.hpp>
#include <mnet/gguf/format.hpp>

#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mnet::detail::compute {
namespace {

// llama.cpp's mul_mm Q2_0 kernel. MIT licence, Copyright (c) 2023-2024 The ggml
// authors (https://github.com/ggml-org/llama.cpp); assembled verbatim for this
// repository by benchmarks/metal/port_llama_mul_mm.py, see docs/mat-mat-kernel.md.
// It compiles into a separate library because it brings its own common.h helpers
// and macros, which would collide with the kernels above.
//
// Used for all three FFN projections by default: it measures 4.99 TFLOP/s on
// gate/up and 4.84 on down against the fp16-tile kernel's 4.87 and 4.24, worth
// 8 ms of the forward.
constexpr const char* kLlamaMatMatSource = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;



#ifdef GGML_METAL_HAS_TENSOR

#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// E2M1 values, shared by MXFP4 and NVFP4
constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

// UE4M3: unsigned, 4 exponent bits with bias 7, 3 mantissa bits - the NVFP4 sub-block scale
static inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0x7F) {
        return 0.0f;
    }

    const int e = (x >> 3) & 0xF;
    const int m = x & 0x7;

    return e == 0 ? m*(1.0f/512.0f) : as_type<float>((uint32_t) (((e + 120) << 23) | (m << 20)));
}

static inline float dot(float x, float y) {
    return x*y;
}

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
inline T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}

template <typename T> struct value_type;
template <> struct value_type<float4x4> { typedef float type; };
template <> struct value_type<half4x4>  { typedef half  type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type<bfloat4x4> { typedef bfloat type; };
#endif

template <typename T> struct value_type_4x4;
template <> struct value_type_4x4<float>  { typedef float4x4 type; };
template <> struct value_type_4x4<half>   { typedef half4x4  type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type_4x4<bfloat> { typedef bfloat4x4 type; };
#endif

template <typename T> struct value_type_2x4;
template <> struct value_type_2x4<float>  { typedef float2x4 type; };
template <> struct value_type_2x4<half>   { typedef half2x4  type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type_2x4<bfloat> { typedef bfloat2x4 type; };
#endif

template <typename T> struct value_type_8x8;
template <> struct value_type_8x8<float>  { typedef simdgroup_float8x8 type; };
template <> struct value_type_8x8<half>   { typedef simdgroup_half8x8  type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type_8x8<bfloat> { typedef simdgroup_bfloat8x8 type; };
#endif

template <typename T> struct value_type_4;
template <> struct value_type_4<float>  { typedef float4 type; };
template <> struct value_type_4<half>   { typedef half4  type; };
template <> struct value_type_4<int32_t> { typedef int4 type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type_4<bfloat> { typedef bfloat4 type; };
#endif

template <typename T> struct value_type_2;
template <> struct value_type_2<float>  { typedef float2 type; };
template <> struct value_type_2<half>   { typedef half2  type; };
#if defined(GGML_METAL_HAS_BF16)
template <> struct value_type_2<bfloat> { typedef bfloat2 type; };
#endif

#define QK2_0 64
typedef half ggml_half_x;
typedef struct {
    ggml_half_x d;
    uint8_t qs[QK2_0 / 4];
} block_q2_0;

#define QK_NL 16
template <typename type4x4>
void dequantize_q2_0(device const block_q2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    const int byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const uint8_t b = qs[byte_offset + i];
        reg_f[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

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

constant auto FC_mul_mm_bc_inp = false;
constant auto FC_mul_mm_bc_out = false;
constant auto FC_mul_mm_ne12 = 1;
constant auto FC_mul_mm_ne13 = 1;
constant auto FC_mul_mm_r2 = 1;
constant auto FC_mul_mm_r3 = 1;
constant auto FC_mul_mm_id_map0_ne02 = 0;
template<
    typename S0, typename S1,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &),
    typename T0, typename T1>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T0_4x4 = typename value_type_4x4<T0>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (uint64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (uint64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (uint64_t) (r1 + j)*args.ne0 + (uint64_t) im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half, block_q2_0, 4, dequantize_q2_0, float, float>) mul_mm_t;

// llama.cpp's default (half accumulation) and the f32-precision variant.
template [[host_name("kernel_mul_mm_q2_0_f32")]]      kernel mul_mm_t kernel_mul_mm<half,  half,  block_q2_0, 4, dequantize_q2_0, float, float>;
template [[host_name("kernel_mul_mm_q2_0_f32_f32p")]] kernel mul_mm_t kernel_mul_mm<float, float, block_q2_0, 4, dequantize_q2_0, float, float>;
)MSL";

constexpr const char* kQ2_0Source = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tiled mat-mat: one threadgroup owns a 64-row x 32-token output tile. The
// Q2_0 weight block for a row is dequantised once per k-step into threadgroup
// memory and then multiplied against all 32 tokens with the matrix unit, so a
// weight byte serves the whole batch instead of one token. That is the
// difference of llama.cpp's mul_mm: our per-(token,row) matvec re-reads and
// re-decodes every weight for every token (log 115 measured the FFN moving
// 153.6 GB per 32-token forward against 4.8 GB here).
//
// 128 threads = 4 simdgroups. Each simdgroup accumulates 32 rows x 16 tokens as
// eight 8x8 blocks (mc[8]), exactly like llama.cpp's mul_mm quadrant split.
// Arithmetic stays in fp32 (no fp16 tiles) so the requantisation downstream
// sees the same magnitudes as the scalar path.
kernel void q2_0_mat_mat_f32(device const uchar* w        [[buffer(0)]],
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
    threadgroup float sa[NR0 * NK];  // [row][k]
    threadgroup float sb[NR1 * NK];  // [token][k]

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
                sa[row * NK + part * 32u + i * 4u + 0u] = float(int(byte & 3u) - 1) * d;
                sa[row * NK + part * 32u + i * 4u + 1u] = float(int((byte >> 2u) & 3u) - 1) * d;
                sa[row * NK + part * 32u + i * 4u + 2u] = float(int((byte >> 4u) & 3u) - 1) * d;
                sa[row * NK + part * 32u + i * 4u + 3u] = float(int((byte >> 6u) & 3u) - 1) * d;
            }
        }
        // Activations: one tile of the batch, transposed to [token][k].
        for (uint idx = tiitg; idx < NR1 * NK; idx += 128u) {
            const uint tok = idx / NK;
            const uint k = idx % NK;
            sb[idx] = (tok < tok_count && k0 + k < columns) ? x[(ulong)(tok0 + tok) * x_stride + k0 + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float* sa_q = sa + 32u * (sgitg & 1u) * NK;    // this simdgroup's 32 rows
        threadgroup const float* sb_q = sb + 16u * (sgitg >> 1u) * NK;  // this simdgroup's 16 tokens
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < NK / 8u; ++kk) {
            simdgroup_float8x8 mb[2];
            simdgroup_float8x8 ma[4];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                // [row][k] -> [k][row], which is what A*B needs on the right.
                simdgroup_load(ma[i], sa_q + i * 8u * NK + kk * 8u, NK, 0, true);
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
    threadgroup float* tile = sb;  // 8 KB, the k loop is done
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

// The same matvec for several tokens at once: `rows` rows per token, the token's
// activation at a fixed stride. One threadgroup per (token, row), so the
// accumulation order per row is unchanged and results stay bit-identical.
kernel void q2_0_matvec_f32_multi(device const uchar* w        [[buffer(0)]],
                                  device const float* x        [[buffer(1)]],
                                  device float* out            [[buffer(3)]],
                                  constant uint& row_bytes     [[buffer(4)]],
                                  constant uint& blocks        [[buffer(5)]],
                                  constant float& scale        [[buffer(6)]],
                                  constant uint& lanes         [[buffer(7)]],
                                  constant uint& rows          [[buffer(8)]],
                                  constant uint& x_stride      [[buffer(9)]],
                                  uint tgid                    [[threadgroup_position_in_grid]],
                                  uint lane                    [[thread_position_in_threadgroup]]) {
    const uint token = tgid / rows;
    const uint row_index = tgid % rows;
    device const uchar* row = w + (ulong)row_index * row_bytes;
    device const float* xt = x + (ulong)token * x_stride;
    float acc = 0.0f;
    for (uint i = lane; i < blocks; i += lanes) {
        device const uchar* block = row + (ulong)i * 18u;
        const float dx = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
        for (uint k = 0; k < 2u; ++k) {
            device const float* xy = xt + (ulong)(2u * i + k) * 32u;
            device const uchar* q = block + 2u + k * 8u;
            float sum = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                const uchar byte = q[j];
                sum += float(int(byte & 3u) - 1) * xy[j * 4u + 0u];
                sum += float(int((byte >> 2u) & 3u) - 1) * xy[j * 4u + 1u];
                sum += float(int((byte >> 4u) & 3u) - 1) * xy[j * 4u + 2u];
                sum += float(int((byte >> 6u) & 3u) - 1) * xy[j * 4u + 3u];
            }
            acc = fma(dx, sum, acc);
        }
    }
    const float total = simd_sum(acc);
    if (lanes <= 32u) {
        if (lane == 0u) {
            out[tgid] = scale * total;
        }
        return;
    }
    threadgroup float partial[8];
    if ((lane & 31u) == 0u) {
        partial[lane >> 5u] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float sum = partial[0];
        for (uint warp = 1u; warp < (lanes / 32u); ++warp) {
            sum += partial[warp];
        }
        out[tgid] = scale * sum;
    }
}

// Q8_0 quantisation of the batch's activations on the device: the same
// nearest-int trick and the same half-rounded block scale as the host's
// make_q8_0_blocks, so the codes are identical.
inline int q8_nearest_int(float value) {
    const float val = value + 12582912.0f;
    return int(as_type<uint>(val) & 0x007FFFFFu) - 0x00400000;
}

kernel void q8_0_quantize_tokens(device const float* x       [[buffer(0)]],
                                 device char* qs             [[buffer(1)]],
                                 device float* d             [[buffer(2)]],
                                 constant uint& blocks       [[buffer(3)]],
                                 constant uint& tokens       [[buffer(4)]],
                                 uint gid                    [[thread_position_in_grid]]) {
    const uint token = gid / blocks;
    const uint block = gid % blocks;
    const ulong slot = (ulong)token * blocks + block;
    device const float* xb = x + slot * 32u;
    float amax = 0.0f;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 32u; ++j) {
        amax = max(amax, fabs(xb[j]));
    }
    const float scale = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    d[slot] = float(half(scale));
    device char* out = qs + slot * 32u;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 32u; ++j) {
        out[j] = char(clamp(q8_nearest_int(xb[j] * id), -127, 127));
    }
}

// Dequantise the Q8_0 activation to fp32 once per dispatch: q8_0_dequantize
// runs before every fast matvec and turns the per-weight activation byte load
// plus per-block scale multiply into a plain fma.
kernel void q8_0_dequantize(device const char* q8    [[buffer(0)]],
                            device const float* qd   [[buffer(1)]],
                            device float* out        [[buffer(2)]],
                            uint gid                 [[thread_position_in_grid]]) {
    out[gid] = float(q8[gid]) * qd[gid / 32u];
}

// Fast matvec against an fp32 activation. Same 18-byte record and the same
// integer decode of the 2-bit codes, but no activation load per weight and the
// per-block scale is folded once: measured 92 GB/s versus 55 GB/s for the
// integer form on the fused-FFN shape (benchmarks/metal/q2_0_kernel_bench.mm).
kernel void q2_0_matvec_f32(device const uchar* w        [[buffer(0)]],
                            device const float* x        [[buffer(1)]],
                            device float* out            [[buffer(3)]],
                            constant uint& row_bytes     [[buffer(4)]],
                            constant uint& blocks        [[buffer(5)]],
                            constant float& scale        [[buffer(6)]],
                            uint tgid                    [[threadgroup_position_in_grid]],
                            uint lane                    [[thread_position_in_threadgroup]]) {
    device const uchar* row = w + (ulong)tgid * row_bytes;
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

// residual += delta, element-wise. The device-resident residual stream needs
// this in place of the model's host `for (i) x[i] += delta[i]` loops.
kernel void residual_add(device float* x            [[buffer(0)]],
                         device const float* delta  [[buffer(1)]],
                         constant uint& count       [[buffer(2)]],
                         uint gid                   [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    x[gid] += delta[gid];
}

// RMSNorm on the device: `size` values, one threadgroup, strided loads and a
// threadgroup-wide reduction. The host reference sums the squares in double
// (Apple GPUs have no fp64), so this is the one place where the fast path
// deliberately differs from the host in the last bits.
kernel void rms_norm_f32(device const float* x        [[buffer(0)]],
                         device const float* weight   [[buffer(1)]],
                         device float* out            [[buffer(2)]],
                         constant uint& size          [[buffer(3)]],
                         constant float& eps          [[buffer(4)]],
                         threadgroup float* partial   [[threadgroup(0)]],
                         uint tid                     [[thread_position_in_threadgroup]],
                         uint group_size              [[threads_per_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < size; i += group_size) {
        sum = fma(x[i], x[i], sum);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(size);
    const float scale = 1.0f / sqrt(mean + eps);
    for (uint i = tid; i < size; i += group_size) {
        out[i] = (x[i] * scale) * weight[i];
    }
}

// Signed, normalised block Hadamard followed by Q8_0 requantisation, with the
// block kept in threadgroup memory: one threadgroup per block instead of three
// separate dispatches (sign*scale, FWHT, quantise).
kernel void hadamard_quantize(device const float* in     [[buffer(0)]],
                              device const int* signs     [[buffer(1)]],
                              device char* qs             [[buffer(2)]],
                              device float* scales        [[buffer(3)]],
                              constant uint& block_size   [[buffer(4)]],
                              uint tgid                   [[threadgroup_position_in_grid]],
                              uint lane                   [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    const uint base = tgid * block_size;
    const float scale = 1.0f / sqrt(float(block_size));
    values[lane] = (in[base + lane] * float(signs[base + lane])) * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1u; len < block_size; len <<= 1u) {
        if ((lane & len) == 0u) {
            const float a = values[lane];
            const float b = values[lane + len];
            values[lane] = a + b;
            values[lane + len] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The first block_size / 32 threads quantise one 32-value block each.
    if (lane < block_size / 32u) {
        const uint offset = lane * 32u;
        float amax = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            amax = fmax(amax, fabs(values[offset + j]));
        }
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            qs[base + offset + j] = char(rint(values[offset + j] * id));
        }
        scales[(base + offset) / 32u] = float(half(amax / 127.0f));
    }
}

// ---- attention block -----------------------------------------------------

constant uint kAttentionMaxHeadDim = 256u;
constant uint kAttentionSimdGroups = 8u;  // kAttentionMaxHeadDim / 32

// Per-head RMSNorm + RoPE, one head per threadgroup of head_dim threads.
// `src` rows are src_stride apart: for q that is 2 * head_dim and the second
// half is the gate half the caller reads separately. With half_store set the
// result is rounded to f16 before the store, which is what the KV cache holds.
kernel void attn_head_prep(device const float* src      [[buffer(0)]],
                           device const float* norm     [[buffer(1)]],
                           device const float* cos_t    [[buffer(2)]],
                           device const float* sin_t    [[buffer(3)]],
                           device float* dst            [[buffer(4)]],
                           constant uint& head_dim      [[buffer(5)]],
                           constant uint& rope_dims     [[buffer(6)]],
                           constant float& eps          [[buffer(7)]],
                           constant uint& src_stride    [[buffer(8)]],
                           constant uint& half_store    [[buffer(9)]],
                           uint tgid                    [[threadgroup_position_in_grid]],
                           uint tid                     [[thread_position_in_threadgroup]]) {
    threadgroup float head[kAttentionMaxHeadDim];
    threadgroup float squares[kAttentionSimdGroups];
    const float x = src[tgid * src_stride + tid];
    const float partial = simd_sum(x * x);
    if ((tid & 31u) == 0u) {
        squares[tid >> 5u] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < kAttentionSimdGroups; ++i) {
            total += squares[i];
        }
        squares[0] = total / float(head_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(squares[0] + eps);
    head[tid] = (x * scale) * norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rope_dims / 2u) {
        const float c = cos_t[tid];
        const float s = sin_t[tid];
        const float x0 = head[tid];
        const float x1 = head[tid + rope_dims / 2u];
        head[tid] = x0 * c - x1 * s;
        head[tid + rope_dims / 2u] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst[tgid * head_dim + tid] = half_store != 0u ? float(half(head[tid])) : head[tid];
}
kernel void attn_head_prep_batch(device const float* src      [[buffer(0)]],
                           device const float* norm     [[buffer(1)]],
                           device const float* cos_t    [[buffer(2)]],
                           device const float* sin_t    [[buffer(3)]],
                           device float* dst            [[buffer(4)]],
                           constant uint& head_dim      [[buffer(5)]],
                           constant uint& rope_dims     [[buffer(6)]],
                           constant float& eps          [[buffer(7)]],
                           constant uint& src_stride    [[buffer(8)]],
                           constant uint& half_store    [[buffer(9)]],
                           constant uint& heads         [[buffer(10)]],
                           constant uint& cos_stride    [[buffer(11)]],
                           uint tgid                    [[threadgroup_position_in_grid]],
                           uint tid                     [[thread_position_in_threadgroup]]) {
    threadgroup float head[kAttentionMaxHeadDim];
    threadgroup float squares[kAttentionSimdGroups];
    // One dispatch for every token: tgid covers (token, head).
    const uint token = tgid / heads;
    const uint head_index = tgid % heads;
    const uint slot = token * heads + head_index;
    const float x = src[(ulong)slot * src_stride + tid];
    const float partial = simd_sum(x * x);
    if ((tid & 31u) == 0u) {
        squares[tid >> 5u] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < kAttentionSimdGroups; ++i) {
            total += squares[i];
        }
        squares[0] = total / float(head_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(squares[0] + eps);
    head[tid] = (x * scale) * norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rope_dims / 2u) {
        const float c = cos_t[(ulong)token * cos_stride + tid];
        const float s = sin_t[(ulong)token * cos_stride + tid];
        const float x0 = head[tid];
        const float x1 = head[tid + rope_dims / 2u];
        head[tid] = x0 * c - x1 * s;
        head[tid + rope_dims / 2u] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst[(ulong)slot * head_dim + tid] = half_store != 0u ? float(half(head[tid])) : head[tid];
}
kernel void attn_gate_sigmoid_batch(device const float* src   [[buffer(0)]],
                                    device float* dst         [[buffer(1)]],
                                    constant uint& head_dim   [[buffer(2)]],
                                    constant uint& per_token  [[buffer(3)]],
                                    constant uint& tokens     [[buffer(4)]],
                                    uint gid                  [[thread_position_in_grid]]) {
    if (gid >= per_token * tokens) {
        return;
    }
    const uint token = gid / per_token;
    const uint local = gid % per_token;
    const uint head = local / head_dim;
    const float x = src[(ulong)token * 2u * per_token + head * 2u * head_dim + head_dim + (local % head_dim)];
    dst[(ulong)token * per_token + local] = 1.0f / (1.0f + exp(-x));
}

kernel void attn_store_half_batch(device const float* src  [[buffer(0)]],
                                  device float* dst        [[buffer(1)]],
                                  constant uint& per_token [[buffer(2)]],
                                  constant uint& tokens    [[buffer(3)]],
                                  uint gid                 [[thread_position_in_grid]]) {
    if (gid >= per_token * tokens) {
        return;
    }
    dst[gid] = float(half(src[gid]));
}


kernel void attn_gate_sigmoid(device const float* src   [[buffer(0)]],
                              device float* dst         [[buffer(1)]],
                              constant uint& head_dim   [[buffer(2)]],
                              constant uint& count      [[buffer(3)]],
                              uint gid                  [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    const uint head = gid / head_dim;
    const float x = src[head * 2u * head_dim + head_dim + (gid % head_dim)];
    dst[gid] = 1.0f / (1.0f + exp(-x));
}

// f16 round on store, one thread per element (the V side of the KV cache).
kernel void attn_store_half(device const float* src  [[buffer(0)]],
                            device float* dst        [[buffer(1)]],
                            constant uint& count     [[buffer(2)]],
                            uint gid                 [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    dst[gid] = float(half(src[gid]));
}

// Online-softmax attention over the whole cache, one simdgroup per head.
// Lane l owns the elements [l * elems, (l + 1) * elems) of the head, so the
// value accumulator stays in registers and the score is one simd_sum.
kernel void attn_flash(device const float* q         [[buffer(0)]],
                       device const float* k_cache   [[buffer(1)]],
                       device const float* v_cache   [[buffer(2)]],
                       device const float* gate      [[buffer(3)]],
                       device float* out             [[buffer(4)]],
                       constant uint& head_dim       [[buffer(5)]],
                       constant uint& tokens         [[buffer(6)]],
                       constant uint& heads_kv       [[buffer(7)]],
                       constant uint& group          [[buffer(8)]],
                       constant float& attn_scale    [[buffer(9)]],
                       uint tgid                     [[threadgroup_position_in_grid]],
                       uint lane                     [[thread_position_in_threadgroup]]) {
    const uint elems = head_dim / 32u;
    const uint kv_head = tgid / group;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint first = lane * elems;
    float max_score = -INFINITY;
    float sum = 0.0f;
    for (uint p = 0; p < tokens; ++p) {
        const uint base = (p * heads_kv + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint j = 0; j < elems; ++j) {
            dot = fma(k_cache[base + first + j], q[tgid * head_dim + first + j], dot);
        }
        const float score = simd_sum(dot) * attn_scale;
        float ms = 1.0f;
        float vs = 1.0f;
        if (score > max_score) {
            ms = exp(max_score - score);
            max_score = score;
        } else {
            vs = exp(score - max_score);
        }
        for (uint j = 0; j < elems; ++j) {
            acc[j] = fma(v_cache[base + first + j], vs, acc[j] * ms);
        }
        sum = fma(sum, ms, vs);
    }
    const float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (uint j = 0; j < elems; ++j) {
        out[tgid * head_dim + first + j] = (acc[j] * inv) * gate[tgid * head_dim + first + j];
    }
}
kernel void attn_flash_batch(device const float* q         [[buffer(0)]],
                       device const float* k_cache   [[buffer(1)]],
                       device const float* v_cache   [[buffer(2)]],
                       device const float* gate      [[buffer(3)]],
                       device float* out             [[buffer(4)]],
                       constant uint& head_dim       [[buffer(5)]],
                       constant uint& heads          [[buffer(6)]],
                       constant uint& heads_kv       [[buffer(7)]],
                       constant uint& group          [[buffer(8)]],
                       constant float& attn_scale    [[buffer(9)]],
                       constant uint& cache_tokens   [[buffer(10)]],
                       constant uint& hidden         [[buffer(11)]],
                       uint tgid                     [[threadgroup_position_in_grid]],
                       uint lane                     [[thread_position_in_threadgroup]]) {
    const uint elems = head_dim / 32u;
    // One dispatch for every token: tgid covers (token, head).
    const uint token = tgid / heads;
    const uint head = tgid % heads;
    const uint kv_head = head / group;
    const uint visible = cache_tokens + token + 1u;
    device const float* qh = q + (ulong)token * hidden + head * head_dim;
    device float* outh = out + (ulong)token * hidden + head * head_dim;
    device const float* gh = gate + (ulong)token * hidden + head * head_dim;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint first = lane * elems;
    float max_score = -INFINITY;
    float sum = 0.0f;
    for (uint p = 0; p < visible; ++p) {
        const uint base = (p * heads_kv + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint j = 0; j < elems; ++j) {
            dot = fma(k_cache[base + first + j], qh[first + j], dot);
        }
        const float score = simd_sum(dot) * attn_scale;
        float ms = 1.0f;
        float vs = 1.0f;
        if (score > max_score) {
            ms = exp(max_score - score);
            max_score = score;
        } else {
            vs = exp(score - max_score);
        }
        for (uint j = 0; j < elems; ++j) {
            acc[j] = fma(v_cache[base + first + j], vs, acc[j] * ms);
        }
        sum = fma(sum, ms, vs);
    }
    const float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (uint j = 0; j < elems; ++j) {
        outh[first + j] = (acc[j] * inv) * gh[first + j];
    }
}
// Same as hadamard_quantize, but `sign_blocks` tells it how many Hadamard
// blocks one token has, so a dispatch can cover several tokens whose slabs are
// contiguous.
kernel void hadamard_quantize_multi(device const float* in     [[buffer(0)]],
                                    device const int* signs     [[buffer(1)]],
                                    device char* qs             [[buffer(2)]],
                                    device float* scales        [[buffer(3)]],
                                    constant uint& block_size   [[buffer(4)]],
                                    constant uint& sign_blocks  [[buffer(5)]],
                                    uint tgid                   [[threadgroup_position_in_grid]],
                                    uint lane                   [[thread_position_in_threadgroup]]) {
    threadgroup float values[1024];
    const uint base = tgid * block_size;
    const uint sign_base = (tgid % sign_blocks) * block_size;
    const float scale = 1.0f / sqrt(float(block_size));
    values[lane] = (in[base + lane] * float(signs[sign_base + lane])) * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1u; len < block_size; len <<= 1u) {
        if ((lane & len) == 0u) {
            const float a = values[lane];
            const float b = values[lane + len];
            values[lane] = a + b;
            values[lane + len] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < block_size / 32u) {
        const uint offset = lane * 32u;
        float amax = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            amax = fmax(amax, fabs(values[offset + j]));
        }
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            qs[base + offset + j] = char(rint(values[offset + j] * id));
        }
        scales[(base + offset) / 32u] = float(half(amax / 127.0f));
    }
}

// The beta/alpha head projections and the sigmoid/softplus gate in one
// dispatch: one threadgroup per (token, head), 32 lanes. Both dots keep the
// row-strided simd_sum accumulation of ssm_rows_dot, so results match.
kernel void ssm_head_gates(device const float* x        [[buffer(0)]],
                           device const float* weights  [[buffer(1)]],
                           device const float* dt       [[buffer(2)]],
                           device const float* ssm_a    [[buffer(3)]],
                           device float* beta           [[buffer(4)]],
                           device float* gate           [[buffer(5)]],
                           constant uint& columns       [[buffer(6)]],
                           constant uint& heads         [[buffer(7)]],
                           constant uint& stride        [[buffer(8)]],
                           uint tgid                    [[threadgroup_position_in_grid]],
                           uint lane                    [[thread_position_in_threadgroup]]) {
    const uint head = tgid % heads;
    const uint token = tgid / heads;
    device const float* xt = x + (ulong)token * columns;
    device const float* wb = weights + (ulong)head * columns;
    device const float* wa = weights + (ulong)(heads + head) * columns;
    float sum_beta = 0.0f;
    float sum_alpha = 0.0f;
    for (uint j = lane; j < columns; j += 32u) {
        sum_beta = fma(wb[j], xt[j], sum_beta);
        sum_alpha = fma(wa[j], xt[j], sum_alpha);
    }
    sum_beta = simd_sum(sum_beta);
    sum_alpha = simd_sum(sum_alpha);
    if (lane == 0u) {
        const uint index = token * stride + head;
        beta[index] = 1.0f / (1.0f + exp(-sum_beta));
        const float z = sum_alpha + dt[head];
        const float sp = z > 20.0f ? z : log(1.0f + exp(z));
        gate[index] = sp * ssm_a[head];
    }
}

// Same as ssm_gate_norm, with `heads` (value heads per token) telling it how to
// split the threadgroup index into (token, head).
kernel void ssm_gate_norm_multi(device const float* delta    [[buffer(0)]],
                                device const float* z        [[buffer(1)]],
                                device const float* norm     [[buffer(2)]],
                                device float* out            [[buffer(3)]],
                                constant uint& head_dim      [[buffer(4)]],
                                constant uint& key_heads     [[buffer(5)]],
                                constant uint& heads_per_key [[buffer(6)]],
                                constant float& eps          [[buffer(7)]],
                                constant uint& heads         [[buffer(8)]],
                                uint tgid                    [[threadgroup_position_in_grid]],
                                uint tid                     [[thread_position_in_threadgroup]]) {
    threadgroup float squares[kAttentionSimdGroups];
    const float x = delta[tgid * head_dim + tid];
    const float partial = simd_sum(x * x);
    if ((tid & 31u) == 0u) {
        squares[tid >> 5u] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint groups = (head_dim + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) {
            total += squares[i];
        }
        squares[0] = total / float(head_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(squares[0] + eps);
    const float zh = z[tgid * head_dim + tid];
    const float activated = zh / (1.0f + exp(-zh));
    const uint head = tgid % heads;
    const uint token = tgid / heads;
    const uint key_head = head % key_heads;
    const uint rep = head / key_heads;
    const uint dst_head = rep + heads_per_key * key_head;
    out[(token * heads + dst_head) * head_dim + tid] = ((x * scale) * norm[tid]) * activated;
}

// ---- gated delta-rule (SSM) block ---------------------------------------

// beta/alpha projection rows: one simdgroup per row, simd_sum reduction.
kernel void ssm_rows_dot(device const float* w      [[buffer(0)]],
                         device const float* x      [[buffer(1)]],
                         device float* raw          [[buffer(2)]],
                         constant uint& columns     [[buffer(3)]],
                         uint tgid                  [[threadgroup_position_in_grid]],
                         uint lane                  [[thread_position_in_threadgroup]]) {
    float acc = 0.0f;
    for (uint j = lane; j < columns; j += 32u) {
        acc = fma(w[tgid * columns + j], x[j], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        raw[tgid] = acc;
    }
}

// beta = sigmoid(beta_raw), gate = softplus(alpha_raw + dt) * ssm_a, one thread
// per value head.
kernel void ssm_activation(device const float* raw   [[buffer(0)]],
                           device const float* dt    [[buffer(1)]],
                           device const float* ssm_a [[buffer(2)]],
                           device float* beta        [[buffer(3)]],
                           device float* gate        [[buffer(4)]],
                           constant uint& heads      [[buffer(5)]],
                           uint gid                  [[thread_position_in_grid]]) {
    if (gid >= heads) {
        return;
    }
    beta[gid] = 1.0f / (1.0f + exp(-raw[gid]));
    const float z = raw[heads + gid] + dt[gid];
    // MSL has no log1p; log(1 + exp(z)) is within an ulp over the useful range.
    const float sp = z > 20.0f ? z : log(1.0f + exp(z));
    gate[gid] = sp * ssm_a[gid];
}

// Causal depthwise convolution + SiLU, and the per-head L2 norm of the q and k
// blocks, in one kernel: one threadgroup per head-sized block of the
// convolution output, so no second dispatch is needed for the norm. Every
// product is materialised (volatile) so the result matches the CPU path, which
// rounds each product before adding it.
kernel void ssm_conv_norm(device const float* window     [[buffer(0)]],
                          device const float* conv_w     [[buffer(1)]],
                          device float* out              [[buffer(2)]],
                          constant uint& conv_dim        [[buffer(3)]],
                          constant uint& kernel_size     [[buffer(4)]],
                          constant uint& ring            [[buffer(5)]],
                          constant uint& norm_blocks     [[buffer(6)]],
                          constant float& eps            [[buffer(7)]],
                          constant uint& head_dim        [[buffer(8)]],
                          constant uint& wrap            [[buffer(9)]],
                          uint tgid                      [[threadgroup_position_in_grid]],
                          uint tid                       [[thread_position_in_threadgroup]]) {
    threadgroup float squares[8];
    const uint index = tgid * head_dim + tid;
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        // `wrap` is the ring size; zero means the window is a straight run of
        // rows with `ring` the oldest one (the batched path keeps the whole
        // layer's sequence).
        uint row = ring;
        if (wrap != 0u) {
            row = (k + 1u < kernel_size) ? ((ring + 1u + k) % wrap) : ring;
        } else {
            row = ring + k;
        }
        const volatile float product = window[row * conv_dim + index] * conv_w[index * kernel_size + k];
        sum += product;
    }
    float value = sum / (1.0f + exp(-sum));
    if (tgid < norm_blocks) {
        // llama's build_gdn_l2_norm: scale(x, 1/sqrt(mean + eps/n)) then
        // scale(., 1/sqrt(n)), as two separate multiplies.
        const float partial = simd_sum(value * value);
        if ((tid & 31u) == 0u) {
            squares[tid >> 5u] = partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            const uint groups = (head_dim + 31u) / 32u;
            for (uint i = 0; i < groups; ++i) {
                total += squares[i];
            }
            squares[0] = total / float(head_dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float n = float(head_dim);
        const float rms = 1.0f / sqrt(squares[0] + eps / n);
        const float extra = 1.0f / sqrt(n);
        value = (value * rms) * extra;
    }
    out[index] = value;
}
kernel void ssm_conv_norm_batch(device const float* window     [[buffer(0)]],
                          device const float* conv_w     [[buffer(1)]],
                          device float* out              [[buffer(2)]],
                          constant uint& conv_dim        [[buffer(3)]],
                          constant uint& kernel_size     [[buffer(4)]],
                          constant uint& ring            [[buffer(5)]],
                          constant uint& norm_blocks     [[buffer(6)]],
                          constant float& eps            [[buffer(7)]],
                          constant uint& head_dim        [[buffer(8)]],
                          constant uint& wrap            [[buffer(9)]],
                          constant uint& blocks_per_token [[buffer(10)]],
                          uint tgid                      [[threadgroup_position_in_grid]],
                          uint tid                       [[thread_position_in_threadgroup]]) {
    threadgroup float squares[8];
    const uint token = tgid / blocks_per_token;
    const uint block = tgid % blocks_per_token;
    const uint index = block * head_dim + tid;
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        // `wrap` is the ring size; zero means the window is a straight run of
        // rows with `ring` the oldest one (the batched path keeps the whole
        // layer's sequence).
        // Linear window: token t's taps are rows t .. t+kernel-1.
        const uint row = token + k;
        const volatile float product = window[row * conv_dim + index] * conv_w[index * kernel_size + k];
        sum += product;
    }
    float value = sum / (1.0f + exp(-sum));
    if (block < norm_blocks) {
        // llama's build_gdn_l2_norm: scale(x, 1/sqrt(mean + eps/n)) then
        // scale(., 1/sqrt(n)), as two separate multiplies.
        const float partial = simd_sum(value * value);
        if ((tid & 31u) == 0u) {
            squares[tid >> 5u] = partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float total = 0.0f;
            const uint groups = (head_dim + 31u) / 32u;
            for (uint i = 0; i < groups; ++i) {
                total += squares[i];
            }
            squares[0] = total / float(head_dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float n = float(head_dim);
        const float rms = 1.0f / sqrt(squares[0] + eps / n);
        const float extra = 1.0f / sqrt(n);
        value = (value * rms) * extra;
    }
    out[token * conv_dim + index] = value;
}

// One delta-rule step per value head: decay the state, compute the delta from
// the key, rank-1 update, then read the query out. One threadgroup per head,
// thread `j` owning output component j. The state is held transposed
// (state[i * head_dim + j]) so every thread reads and writes consecutive
// addresses. There is no cross-thread dependency, so no barrier is needed.
//
// A column-tiled variant that loads each 32-column state tile into threadgroup
// memory once (decay, key dot, update and query dot on the resident tile, so
// the 64 KB per-head state is read and written once instead of four times and
// twice) was measured: ssm 1.420-1.422 s and GPU wait 0.871-0.873 ms per call,
// identical to this version. The block is not state-traffic bound, so the
// simpler single-pass form stays.
kernel void ssm_scan(device const float* q            [[buffer(0)]],
                     device const float* k            [[buffer(1)]],
                     device const float* v            [[buffer(2)]],
                     device const float* beta         [[buffer(3)]],
                     device const float* gate         [[buffer(4)]],
                     device float* state              [[buffer(5)]],
                     device float* delta              [[buffer(6)]],
                     constant uint& head_dim          [[buffer(7)]],
                     constant uint& key_heads         [[buffer(8)]],
                     constant float& inv_dim          [[buffer(9)]],
                     uint tgid                        [[threadgroup_position_in_grid]],
                     uint tid                         [[thread_position_in_threadgroup]]) {
    const uint key_head = tgid % key_heads;
    device const float* qh = q + key_head * head_dim;
    device const float* kh = k + key_head * head_dim;
    device const float* vh = v + tgid * head_dim;
    device float* sh = state + tgid * head_dim * head_dim;
    const float decay = exp(gate[tgid]);
    const float b = beta[tgid];
    for (uint i = 0; i < head_dim; ++i) {
        sh[i * head_dim + tid] *= decay;
    }
    float d = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
        d = fma(kh[i], sh[i * head_dim + tid], d);
    }
    const float delta_coefficient = (vh[tid] - d) * b;
    for (uint i = 0; i < head_dim; ++i) {
        sh[i * head_dim + tid] = fma(kh[i], delta_coefficient, sh[i * head_dim + tid]);
    }
    float acc = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
        acc = fma(qh[i], sh[i * head_dim + tid], acc);
    }
    delta[tgid * head_dim + tid] = acc * inv_dim;
}
kernel void ssm_scan_batch(device const float* q            [[buffer(0)]],
                     device const float* k            [[buffer(1)]],
                     device const float* v            [[buffer(2)]],
                     device const float* beta         [[buffer(3)]],
                     device const float* gate         [[buffer(4)]],
                     device float* state              [[buffer(5)]],
                     device float* delta              [[buffer(6)]],
                     constant uint& head_dim          [[buffer(7)]],
                     constant uint& key_heads         [[buffer(8)]],
                     constant float& inv_dim          [[buffer(9)]],
                     constant uint& tokens            [[buffer(10)]],
                     constant uint& qkv_stride        [[buffer(11)]],
                     constant uint& gate_stride       [[buffer(12)]],
                     constant uint& delta_stride      [[buffer(13)]],
                     uint tgid                        [[threadgroup_position_in_grid]],
                     uint tid                         [[thread_position_in_threadgroup]]) {
    // One threadgroup per value head, all tokens in order: the recurrence
    // carries the state, so the loop stays inside the threadgroup and the state
    // stops being re-read once per token dispatch.
    const uint key_head = tgid % key_heads;
    device float* sh = state + tgid * head_dim * head_dim;
    for (uint t = 0; t < tokens; ++t) {
    device const float* qh = q + (ulong)t * qkv_stride + key_head * head_dim;
    device const float* kh = k + (ulong)t * qkv_stride + key_head * head_dim;
    device const float* vh = v + (ulong)t * qkv_stride + tgid * head_dim;
    const float decay = exp(gate[(ulong)t * gate_stride + tgid]);
    const float b = beta[(ulong)t * gate_stride + tgid];
    for (uint i = 0; i < head_dim; ++i) {
        sh[i * head_dim + tid] *= decay;
    }
    float d = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
        d = fma(kh[i], sh[i * head_dim + tid], d);
    }
    const float delta_coefficient = (vh[tid] - d) * b;
    for (uint i = 0; i < head_dim; ++i) {
        sh[i * head_dim + tid] = fma(kh[i], delta_coefficient, sh[i * head_dim + tid]);
    }
    float acc = 0.0f;
    for (uint i = 0; i < head_dim; ++i) {
        acc = fma(qh[i], sh[i * head_dim + tid], acc);
    }
    delta[(ulong)t * delta_stride + tgid * head_dim + tid] = acc * inv_dim;
    }
}

// Gated RMS norm per value head and the tiled -> grouped head permute the
// folded ssm_out weight was built with, in one pass.
// Same recurrence as ssm_scan_batch, but the state block for a 32-column slice
// lives in threadgroup memory for the whole token loop: d, the rank-1 update and
// the output are all column-independent, so a 128x32 slice is self-contained.
// The state traffic drops from a full 3.1 MB read+write per token to one pass
// per layer (log 137).
kernel void ssm_scan_tiled(device const float* q            [[buffer(0)]],
                           device const float* k            [[buffer(1)]],
                           device const float* v            [[buffer(2)]],
                           device const float* beta         [[buffer(3)]],
                           device const float* gate         [[buffer(4)]],
                           device float* state              [[buffer(5)]],
                           device float* delta              [[buffer(6)]],
                           constant uint& head_dim          [[buffer(7)]],
                           constant uint& key_heads         [[buffer(8)]],
                           constant float& inv_dim          [[buffer(9)]],
                           constant uint& tokens            [[buffer(10)]],
                           constant uint& qkv_stride        [[buffer(11)]],
                           constant uint& gate_stride       [[buffer(12)]],
                           constant uint& delta_stride      [[buffer(13)]],
                           constant uint& blocks            [[buffer(14)]],
                           uint tgid                        [[threadgroup_position_in_grid]],
                           uint tid                         [[thread_position_in_threadgroup]]) {
    constexpr uint kCols = 32u;   // columns per threadgroup
    constexpr uint kRows = 128u;  // head_dim
    threadgroup float st[kRows * kCols];
    threadgroup float red[kRows];

    const uint head = tgid / blocks;
    const uint col0 = (tgid % blocks) * kCols;
    const uint col = tid & 31u;
    const uint grp = tid >> 5u;
    device float* sh = state + head * head_dim * head_dim;

#pragma clang loop unroll(full)
    for (uint idx = tid; idx < kRows * kCols; idx += 128u) {
        st[idx] = sh[(idx >> 5u) * head_dim + col0 + (idx & 31u)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < tokens; ++t) {
        device const float* qh = q + (ulong)t * qkv_stride + (head % key_heads) * head_dim;
        device const float* kh = k + (ulong)t * qkv_stride + (head % key_heads) * head_dim;
        device const float* vh = v + (ulong)t * qkv_stride + head * head_dim;
        const float decay = exp(gate[(ulong)t * gate_stride + head]);
        const float b = beta[(ulong)t * gate_stride + head];

        for (uint idx = tid; idx < kRows * kCols; idx += 128u) {
            st[idx] *= decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint r = grp * 32u; r < grp * 32u + 32u; ++r) {
            acc = fma(kh[r], st[r * kCols + col], acc);
        }
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float d = ((red[col] + red[col + 32u]) + red[col + 64u]) + red[col + 96u];

        const float coeff = (vh[col0 + col] - d) * b;
        for (uint r = grp * 32u; r < grp * 32u + 32u; ++r) {
            st[r * kCols + col] = fma(kh[r], coeff, st[r * kCols + col]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float out_acc = 0.0f;
        for (uint r = grp * 32u; r < grp * 32u + 32u; ++r) {
            out_acc = fma(qh[r], st[r * kCols + col], out_acc);
        }
        red[tid] = out_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < kCols) {
            delta[(ulong)t * delta_stride + head * head_dim + col0 + tid] =
                (((red[tid] + red[tid + 32u]) + red[tid + 64u]) + red[tid + 96u]) * inv_dim;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint idx = tid; idx < kRows * kCols; idx += 128u) {
        sh[(idx >> 5u) * head_dim + col0 + (idx & 31u)] = st[idx];
    }
}

kernel void ssm_gate_norm(device const float* delta    [[buffer(0)]],
                          device const float* z        [[buffer(1)]],
                          device const float* norm     [[buffer(2)]],
                          device float* out            [[buffer(3)]],
                          constant uint& head_dim      [[buffer(4)]],
                          constant uint& key_heads     [[buffer(5)]],
                          constant uint& heads_per_key [[buffer(6)]],
                          constant float& eps          [[buffer(7)]],
                          uint tgid                    [[threadgroup_position_in_grid]],
                          uint tid                     [[thread_position_in_threadgroup]]) {
    threadgroup float squares[kAttentionSimdGroups];
    const float x = delta[tgid * head_dim + tid];
    const float partial = simd_sum(x * x);
    if ((tid & 31u) == 0u) {
        squares[tid >> 5u] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint groups = (head_dim + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) {
            total += squares[i];
        }
        squares[0] = total / float(head_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(squares[0] + eps);
    const float zh = z[tgid * head_dim + tid];
    const float activated = zh / (1.0f + exp(-zh));
    const uint key_head = tgid % key_heads;
    const uint rep = tgid / key_heads;
    const uint dst_head = rep + heads_per_key * key_head;
    out[dst_head * head_dim + tid] = ((x * scale) * norm[tid]) * activated;
}
)MSL";

// Host-side buffers for the fused attention block; nothing here survives a
// call except the weight cache and the allocated buffers.
struct AttentionBuffers {
    id<MTLBuffer> qfull = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> k_new = nil;
    id<MTLBuffer> v_new = nil;
    id<MTLBuffer> gates = nil;
    id<MTLBuffer> attn = nil;
    id<MTLBuffer> q8 = nil;
    id<MTLBuffer> scales = nil;
    id<MTLBuffer> kv = nil;
    id<MTLBuffer> tables = nil;
    id<MTLBuffer> signs = nil;
    std::size_t qfull_capacity = 0;
    std::size_t q_capacity = 0;
    std::size_t k_new_capacity = 0;
    std::size_t v_new_capacity = 0;
    std::size_t gates_capacity = 0;
    std::size_t attn_capacity = 0;
    std::size_t q8_capacity = 0;
    std::size_t scales_capacity = 0;
    std::size_t kv_capacity = 0;
    std::size_t tables_capacity = 0;
    std::size_t signs_capacity = 0;
};

// Per-token scratch for a batched FFN: one slab of each kind, indexed by token.
struct FfnBatchBuffers {
    id<MTLBuffer> q8 = nil;
    id<MTLBuffer> scales = nil;
    id<MTLBuffer> wide = nil;
    id<MTLBuffer> gate = nil;
    id<MTLBuffer> up = nil;
    id<MTLBuffer> act = nil;
    id<MTLBuffer> act_q8 = nil;
    id<MTLBuffer> act_scales = nil;
    id<MTLBuffer> out = nil;
    std::size_t q8_capacity = 0;
    std::size_t scales_capacity = 0;
    std::size_t wide_capacity = 0;
    std::size_t gate_capacity = 0;
    std::size_t up_capacity = 0;
    std::size_t act_capacity = 0;
    std::size_t act_q8_capacity = 0;
    std::size_t act_scales_capacity = 0;
    std::size_t out_capacity = 0;
};

// True GPU time per fused block, straight from the command buffer's own
// timestamps. The wall-clock counters below charge a block for waiting on the
// block before it, which is why their totals exceed the forward; these do not.
struct GpuBlockProfile {
    std::uint64_t ns[3] = {0, 0, 0};
    std::uint64_t calls[3] = {0, 0, 0};
    ~GpuBlockProfile() {
        if (std::getenv("MNET_PROFILE") == nullptr || (this->calls[0] + this->calls[1] + this->calls[2]) == 0U) {
            return;
        }
        const char* names[3] = {"ffn", "ssm", "attention"};
        for (int i = 0; i < 3; ++i) {
            if (this->calls[i] == 0U) {
                continue;
            }
            std::fprintf(stderr, "mnet: gpu %-9s %8.2f ms total  %6.3f ms per call  %llu calls\n", names[i],
                         static_cast<double>(this->ns[i]) * 1.0e-6,
                         static_cast<double>(this->ns[i]) * 1.0e-6 / static_cast<double>(this->calls[i]),
                         static_cast<unsigned long long>(this->calls[i]));
        }
    }
};

GpuBlockProfile& gpu_block_profile() {
    static GpuBlockProfile profile;
    return profile;
}

void record_gpu_block(int block, id<MTLCommandBuffer> command) {
    if (std::getenv("MNET_PROFILE") == nullptr || command.GPUStartTime <= 0.0) {
        return;
    }
    GpuBlockProfile& profile = gpu_block_profile();
    profile.ns[block] += static_cast<std::uint64_t>((command.GPUEndTime - command.GPUStartTime) * 1.0e9);
    profile.calls[block] += 1U;
}

// Split of the batched gated-delta call: host preparation (q8 conversion of the
// layer's activations), weight binding, encoding, the GPU wait and the readback.
struct SsmBatchProfile {
    std::uint64_t prep_ns = 0;
    std::uint64_t bind_ns = 0;
    std::uint64_t encode_ns = 0;
    std::uint64_t wait_ns = 0;
    std::uint64_t tail_ns = 0;
    std::uint64_t calls = 0;
    std::uint64_t tokens = 0;
    ~SsmBatchProfile() {
        if (std::getenv("MNET_PROFILE") == nullptr || this->calls == 0U) {
            return;
        }
        const auto ms = [&](std::uint64_t ns) {
            return static_cast<double>(ns) * 1.0e-6 / static_cast<double>(this->calls);
        };
        char line[256];
        std::snprintf(line, sizeof(line),
                      "mnet: ssm batch per layer  prep %.3f  bind %.3f  encode %.3f  wait %.3f  tail %.3f ms  "
                      "(%.1f tokens/layer)\n",
                      ms(this->prep_ns), ms(this->bind_ns), ms(this->encode_ns), ms(this->wait_ns),
                      ms(this->tail_ns),
                      static_cast<double>(this->tokens) / static_cast<double>(this->calls));
        mnet::detail::write_diagnostic(std::string(line));
    }
};

SsmBatchProfile& ssm_batch_profile() {
    static SsmBatchProfile profile;
    return profile;
}

// Per-token scratch for a batched attention block.
struct AttentionBatchBuffers {
    id<MTLBuffer> q8 = nil;
    id<MTLBuffer> scales = nil;
    id<MTLBuffer> wide = nil;
    id<MTLBuffer> qfull = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> k_new = nil;
    id<MTLBuffer> v_new = nil;
    id<MTLBuffer> gates = nil;
    id<MTLBuffer> attn = nil;
    id<MTLBuffer> act_q8 = nil;
    id<MTLBuffer> act_scales = nil;
    id<MTLBuffer> out = nil;
    id<MTLBuffer> tables = nil;
    std::size_t q8_capacity = 0;
    std::size_t scales_capacity = 0;
    std::size_t wide_capacity = 0;
    std::size_t qfull_capacity = 0;
    std::size_t q_capacity = 0;
    std::size_t k_new_capacity = 0;
    std::size_t v_new_capacity = 0;
    std::size_t gates_capacity = 0;
    std::size_t attn_capacity = 0;
    std::size_t act_q8_capacity = 0;
    std::size_t act_scales_capacity = 0;
    std::size_t out_capacity = 0;
    std::size_t tables_capacity = 0;
};

// Per-token scratch for a batched gated-delta block.
struct SsmBatchBuffers {
    id<MTLBuffer> q8 = nil;
    id<MTLBuffer> scales = nil;
    id<MTLBuffer> wide = nil;
    id<MTLBuffer> activation = nil;
    id<MTLBuffer> raw = nil;
    id<MTLBuffer> gates = nil;
    id<MTLBuffer> conv = nil;
    id<MTLBuffer> z = nil;
    id<MTLBuffer> delta = nil;
    id<MTLBuffer> grouped = nil;
    id<MTLBuffer> act_q8 = nil;
    id<MTLBuffer> act_scales = nil;
    id<MTLBuffer> out = nil;
    std::size_t q8_capacity = 0;
    std::size_t scales_capacity = 0;
    std::size_t wide_capacity = 0;
    std::size_t activation_capacity = 0;
    std::size_t raw_capacity = 0;
    std::size_t gates_capacity = 0;
    std::size_t conv_capacity = 0;
    std::size_t z_capacity = 0;
    std::size_t delta_capacity = 0;
    std::size_t grouped_capacity = 0;
    std::size_t act_q8_capacity = 0;
    std::size_t act_scales_capacity = 0;
    std::size_t out_capacity = 0;
};

// Per-call scratch for the gated-delta block (kept across layers).
struct SsmBuffers {
    id<MTLBuffer> activation = nil;  // raw_input, fp32 (the beta/alpha projection input)
    id<MTLBuffer> raw = nil;
    id<MTLBuffer> gates = nil;  // beta | gate
    id<MTLBuffer> conv = nil;
    id<MTLBuffer> z = nil;
    id<MTLBuffer> delta = nil;
    id<MTLBuffer> grouped = nil;
    id<MTLBuffer> q8 = nil;
    id<MTLBuffer> scales = nil;
    id<MTLBuffer> signs = nil;
    std::size_t activation_capacity = 0;
    std::size_t raw_capacity = 0;
    std::size_t gates_capacity = 0;
    std::size_t conv_capacity = 0;
    std::size_t z_capacity = 0;
    std::size_t delta_capacity = 0;
    std::size_t grouped_capacity = 0;
    std::size_t q8_capacity = 0;
    std::size_t scales_capacity = 0;
    std::size_t signs_capacity = 0;
};

// Device-resident state of one recurrent layer. The delta-rule state is far
// too large to shuttle per token, so it lives here and is keyed by the tensor
// identity; the caller's vectors only seed it (first use) and are not updated.
struct SsmLayer {
    id<MTLBuffer> window = nil;  // conv_kernel rows of conv_dim (ring buffer)
    id<MTLBuffer> state = nil;   // value_heads * head_dim^2, transposed within a head
    id<MTLBuffer> side = nil;    // beta_w | alpha_w | dt | ssm_a | conv_w | ssm_norm
    std::uint32_t ring = 0;
};

// MNET_MAT_MAT_LLAMA: 1 both FFN projections, 2 the down projection only,
// 3 gate/up only. Declared here because the library is compiled during setup.
int llama_mat_mat_mask();

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> fast_pipeline = nil;
    id<MTLComputePipelineState> fast_f32_pipeline = nil;
    id<MTLComputePipelineState> fast_f32_multi_pipeline = nil;
    id<MTLComputePipelineState> mat_mat_pipeline = nil;
    id<MTLComputePipelineState> mat_mat_f16_pipeline = nil;
    id<MTLComputePipelineState> llama_mat_mat_pipeline = nil;
    id<MTLComputePipelineState> dequant_pipeline = nil;
    id<MTLComputePipelineState> q8_quantize_pipeline = nil;
    id<MTLComputePipelineState> swiglu_pipeline = nil;
    id<MTLBuffer> act_buffer = nil;
    id<MTLBuffer> act_q8_buffer = nil;
    id<MTLBuffer> act_scale_buffer = nil;
    id<MTLBuffer> gate_buffer = nil;
    id<MTLBuffer> up_buffer = nil;
    id<MTLBuffer> signs_buffer = nil;
    id<MTLComputePipelineState> hadamard_quantize_pipeline = nil;
    id<MTLComputePipelineState> hadamard_multi_pipeline = nil;
    id<MTLComputePipelineState> gate_norm_multi_pipeline = nil;
    id<MTLComputePipelineState> head_gates_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> residual_add_pipeline = nil;
    id<MTLComputePipelineState> attn_prep_pipeline = nil;
    id<MTLComputePipelineState> attn_prep_batch_pipeline = nil;
    id<MTLComputePipelineState> attn_gate_batch_pipeline = nil;
    id<MTLComputePipelineState> attn_store_batch_pipeline = nil;
    id<MTLComputePipelineState> attn_gate_pipeline = nil;
    id<MTLComputePipelineState> attn_flash_pipeline = nil;
    id<MTLComputePipelineState> attn_flash_batch_pipeline = nil;
    id<MTLComputePipelineState> attn_store_pipeline = nil;
    id<MTLComputePipelineState> ssm_rows_pipeline = nil;
    id<MTLComputePipelineState> ssm_act_pipeline = nil;
    id<MTLComputePipelineState> ssm_conv_norm_pipeline = nil;
    id<MTLComputePipelineState> ssm_conv_norm_batch_pipeline = nil;
    id<MTLComputePipelineState> ssm_scan_pipeline = nil;
    id<MTLComputePipelineState> ssm_scan_batch_pipeline = nil;
    id<MTLComputePipelineState> ssm_scan_tiled_pipeline = nil;
    id<MTLComputePipelineState> ssm_gate_norm_pipeline = nil;
    FfnBatchBuffers ffn_batch;
    SsmBatchBuffers ssm_batch;
    AttentionBatchBuffers attn_batch;
    AttentionBuffers attn;
    SsmBuffers ssm;
    std::unordered_map<std::uint64_t, SsmLayer> ssm_layers;
    std::size_t signs_capacity = 0;
    std::size_t act_capacity = 0;
    std::size_t act_q8_capacity = 0;
    std::size_t act_scale_capacity = 0;
    std::size_t gate_capacity = 0;
    std::size_t up_capacity = 0;
    std::unordered_map<std::uint64_t, id<MTLBuffer>> weight_buffers;
    std::unordered_map<const void*, id<MTLBuffer>> region_buffers;
    id<MTLBuffer> q8_buffer = nil;
    id<MTLBuffer> scale_buffer = nil;
    id<MTLBuffer> act_f32_buffer = nil;
    std::size_t act_f32_capacity = 0;
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
    id<MTLFunction> hadamard_function = library != nil ? [library newFunctionWithName:@"hadamard_quantize"] : nil;
    if (hadamard_function != nil) {
        state.hadamard_quantize_pipeline =
            [state.device newComputePipelineStateWithFunction:hadamard_function error:&error];
    }
    id<MTLFunction> swiglu_function = library != nil ? [library newFunctionWithName:@"swiglu_f32"] : nil;
    if (swiglu_function != nil) {
        state.swiglu_pipeline = [state.device newComputePipelineStateWithFunction:swiglu_function error:&error];
    }
    id<MTLFunction> fast_function = library != nil ? [library newFunctionWithName:@"q2_0_matvec_fast"] : nil;
    if (fast_function != nil) {
        state.fast_pipeline = [state.device newComputePipelineStateWithFunction:fast_function error:&error];
        if (state.fast_pipeline == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
            std::fprintf(stderr, "mnet: metal fast pipeline failed: %s\n", [[error description] UTF8String]);
        }
    }
    id<MTLFunction> f32_function = library != nil ? [library newFunctionWithName:@"q2_0_matvec_f32"] : nil;
    if (f32_function != nil) {
        state.fast_f32_pipeline = [state.device newComputePipelineStateWithFunction:f32_function error:&error];
        if (state.fast_f32_pipeline == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
            std::fprintf(stderr, "mnet: metal f32 matvec pipeline failed: %s\n", [[error description] UTF8String]);
        }
    }
    id<MTLFunction> multi_function = library != nil ? [library newFunctionWithName:@"q2_0_matvec_f32_multi"] : nil;
    if (multi_function != nil) {
        state.fast_f32_multi_pipeline = [state.device newComputePipelineStateWithFunction:multi_function error:&error];
    }
    if (library != nil) {
        const auto pipeline_for = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            return function != nil ? [state.device newComputePipelineStateWithFunction:function error:&error] : nil;
        };
        state.attn_prep_pipeline = pipeline_for("attn_head_prep");
        state.attn_prep_batch_pipeline = pipeline_for("attn_head_prep_batch");
        state.attn_gate_batch_pipeline = pipeline_for("attn_gate_sigmoid_batch");
        state.attn_store_batch_pipeline = pipeline_for("attn_store_half_batch");
        state.attn_gate_pipeline = pipeline_for("attn_gate_sigmoid");
        state.attn_flash_pipeline = pipeline_for("attn_flash");
        state.attn_flash_batch_pipeline = pipeline_for("attn_flash_batch");
        state.attn_store_pipeline = pipeline_for("attn_store_half");
        state.ssm_rows_pipeline = pipeline_for("ssm_rows_dot");
        state.ssm_act_pipeline = pipeline_for("ssm_activation");
        state.ssm_conv_norm_pipeline = pipeline_for("ssm_conv_norm");
        state.ssm_conv_norm_batch_pipeline = pipeline_for("ssm_conv_norm_batch");
        state.ssm_scan_pipeline = pipeline_for("ssm_scan");
        state.ssm_scan_batch_pipeline = pipeline_for("ssm_scan_batch");
        state.ssm_scan_tiled_pipeline = pipeline_for("ssm_scan_tiled");
        state.ssm_gate_norm_pipeline = pipeline_for("ssm_gate_norm");
        state.dequant_pipeline = pipeline_for("q8_0_dequantize");
        state.q8_quantize_pipeline = pipeline_for("q8_0_quantize_tokens");
        state.hadamard_multi_pipeline = pipeline_for("hadamard_quantize_multi");
        state.gate_norm_multi_pipeline = pipeline_for("ssm_gate_norm_multi");
        state.head_gates_pipeline = pipeline_for("ssm_head_gates");
        state.rms_norm_pipeline = pipeline_for("rms_norm_f32");
        state.mat_mat_pipeline = pipeline_for("q2_0_mat_mat_f32");
        state.mat_mat_f16_pipeline = pipeline_for("q2_0_mat_mat_f16");
        state.residual_add_pipeline = pipeline_for("residual_add");
        if (state.attn_flash_pipeline == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
            std::fprintf(stderr, "mnet: metal attention pipeline failed: %s\n", [[error description] UTF8String]);
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

// Lanes per row for the fast matvec. The batch path is bandwidth bound and
// prefers one warp per row; a single-token grid is 32x smaller, so the override
// exists to test whether more lanes per row helps there.
// The tiled mat-mat pipeline to use: half tiles by default (matrix unit at
// twice the fp32 rate, exact for the weight tile), MNET_MAT_MAT_F16=0 for fp32.
id<MTLComputePipelineState> tiled_mat_mat_pipeline(MetalState& s) {
    static const bool f16 = [] {
        const char* value = std::getenv("MNET_MAT_MAT_F16");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return f16 && s.mat_mat_f16_pipeline != nil ? s.mat_mat_f16_pipeline : s.mat_mat_pipeline;
}

// The argument block and launch geometry of llama.cpp's mul_mm. Kept out of the
// shared path: its summation order is not the CPU's, so it is only usable where
// the fast path already is, and it needs 64 rows, 32 tokens and 64 columns.
struct LlamaMatMatArgs {
    std::int32_t ne00;
    std::int32_t ne02;
    std::uint64_t nb01;
    std::uint64_t nb02;
    std::uint64_t nb03;
    std::int32_t ne12;
    std::uint64_t nb10;
    std::uint64_t nb11;
    std::uint64_t nb12;
    std::uint64_t nb13;
    std::int32_t ne0;
    std::int32_t ne1;
    std::int16_t r2;
    std::int16_t r3;
};

// MNET_MAT_MAT_LLAMA: 1 both FFN projections (the default), 2 the down shape
// only, 3 gate/up only, 0 our tiled kernel everywhere. Compiled on first use so
// the exact path never pays for it.
//
// Both projections leave the logits at 4.128e-3 normalised against the CPU,
// inside the fast path's 4.754e-3 bound; with the scalar attention projection
// this mask measured 5.301e-3, and tiling that projection (ba628a0) is what
// brought it back inside.
int llama_mat_mat_mask() {
    static const int mask = [] {
        const char* value = std::getenv("MNET_MAT_MAT_LLAMA");
        return value != nullptr ? std::atoi(value) : 1;
    }();
    return mask;
}

void ensure_llama_mat_mat(MetalState& s) {
    static std::once_flag once;
    std::call_once(once, [&s] {
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        id<MTLLibrary> library =
            [s.device newLibraryWithSource:[NSString stringWithUTF8String:kLlamaMatMatSource]
                                    options:options
                                      error:&error];
        id<MTLFunction> function =
            library != nil ? [library newFunctionWithName:@"kernel_mul_mm_q2_0_f32_f32p"] : nil;
        if (function != nil) {
            s.llama_mat_mat_pipeline = [s.device newComputePipelineStateWithFunction:function error:&error];
        }
        if (s.llama_mat_mat_pipeline == nil && std::getenv("MNET_METAL_DEBUG") != nullptr) {
            std::fprintf(stderr, "mnet: llama mat-mat unavailable: %s\n", [[error description] UTF8String]);
        }
    });
}

bool llama_mat_mat_ok(MetalState& s, std::size_t rows, std::size_t columns, std::size_t count, float scale,
                      bool down) {
    const int mask = llama_mat_mat_mask();
    const bool wanted = down ? (mask == 1 || mask == 2) : (mask == 1 || mask == 3);
    if (!wanted || rows % 64U != 0U || columns % 64U != 0U || count % 32U != 0U || scale != 1.0F) {
        return false;
    }
    ensure_llama_mat_mat(s);
    return s.llama_mat_mat_pipeline != nil;
}

void encode_llama_mat_mat(id<MTLComputeCommandEncoder> encoder, MetalState& s, id<MTLBuffer> weights,
                          NSUInteger weight_offset, id<MTLBuffer> x, id<MTLBuffer> out, std::size_t rows,
                          std::size_t columns, std::size_t row_bytes, std::size_t out_stride, std::size_t count) {
    LlamaMatMatArgs args{};
    args.ne00 = static_cast<std::int32_t>(columns);
    args.ne02 = 1;
    args.nb01 = row_bytes;
    args.nb02 = row_bytes * rows;
    args.nb10 = sizeof(float);
    args.nb11 = sizeof(float) * columns;
    args.ne0 = static_cast<std::int32_t>(out_stride);
    args.ne1 = static_cast<std::int32_t>(count);
    args.r2 = 1;
    args.r3 = 1;
    [encoder setComputePipelineState:s.llama_mat_mat_pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:weights offset:weight_offset atIndex:1];
    [encoder setBuffer:x offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:12U * 1024U atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (rows + 63U) / 64U, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

// Whether the caller wants the tiled mat-mat at all (MNET_MAT_MAT=0 disables).
bool tiled_mat_mat_enabled(MetalState& s, std::size_t count) {
    const char* value = std::getenv("MNET_MAT_MAT");
    const int mode = value != nullptr ? std::atoi(value) : 1;
    return mode > 0 && s.mat_mat_pipeline != nil && count >= 8U;
}

std::uint32_t matvec_lanes() {
    static const std::uint32_t lanes = [] {
        const char* value = std::getenv("MNET_MATVEC_LANES");
        const auto parsed = value != nullptr ? static_cast<std::uint32_t>(std::atoi(value)) : 32U;
        if (parsed != 32U && parsed != 64U) {
            return 32U;
        }
        return parsed;
    }();
    return lanes;
}

std::uint64_t cache_key(std::uint64_t identity, const std::uint8_t* weights) {
    // Loaded tensors carry a unique identity; raw-pointer callers (tests) use
    // the address with the top bit set so the two spaces cannot collide.
    return identity != 0 ? identity : (reinterpret_cast<std::uintptr_t>(weights) | (1ULL << 63U));
}

// A weight tensor on the device: a whole buffer plus the byte offset inside it.
struct WeightBinding {
    id<MTLBuffer> buffer = nil;
    std::uint64_t offset = 0;
};

// Weights borrowed from the reader's read-only file mapping are aliased with a
// single no-copy buffer over the whole mapping, so the model is never copied
// into device memory; the per-tensor offset is applied when the kernel is
// encoded. Tensors that arrive as owned vectors (synthetic callers) keep the
// copy path.
WeightBinding owned_weight_buffer(MetalState& s, const MatvecQ8_0Job& job, std::uint64_t rows,
                                  std::uint64_t row_bytes) {
    const std::uint64_t bytes = rows * row_bytes;
    const std::uint64_t key = cache_key(job.identity, job.data);
    id<MTLBuffer>& weight_buffer = s.weight_buffers[key];
    if (weight_buffer != nil) {
        if (weight_buffer.length != static_cast<NSUInteger>(bytes)) {
            if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
                std::fprintf(stderr, "mnet: metal weight cache shape mismatch (%lu vs %llu)\n",
                             static_cast<unsigned long>(weight_buffer.length),
                             static_cast<unsigned long long>(bytes));
            }
            return {};
        }
        return {weight_buffer, 0U};
    }
    weight_buffer = [s.device newBufferWithLength:static_cast<NSUInteger>(bytes)
                                         options:MTLResourceStorageModeShared];
    if (weight_buffer == nil) {
        s.weight_buffers.erase(key);
        return {};
    }
    std::memcpy(weight_buffer.contents, job.data, static_cast<std::size_t>(bytes));
    return {weight_buffer, 0U};
}

WeightBinding weight_buffer_for(MetalState& s, const MatvecQ8_0Job& job, std::uint64_t rows,
                                std::uint64_t row_bytes, bool allow_alias) {
    const std::uint64_t bytes = rows * row_bytes;
    static const bool aliasing = std::getenv("MNET_NO_ALIAS") == nullptr;
    if (!allow_alias || !aliasing || job.region == nullptr || job.data == nullptr || job.region_bytes == 0U) {
        return owned_weight_buffer(s, job, rows, row_bytes);
    }
    // One page-aligned no-copy buffer per tensor. Measured against a single
    // buffer over the whole 6.7 GB mapping: gated-delta 1.106 -> 0.722 s
    // (per-layer GPU wait 22.07 -> 14.12 ms), forward 2580 -> 2522 ms, at the
    // cost of the FFN going 1.269 -> 1.577 s; net faster, and the mapping
    // buffer is 6.7 GB of address space the driver had to rebind per dispatch.
    const std::uint64_t key = cache_key(job.identity, job.data) ^ 0x9E3779B97F4A7C15ULL;
    id<MTLBuffer>& weight_buffer = s.weight_buffers[key];
    const std::uint64_t page = 16U * 1024U;
    const auto address = reinterpret_cast<std::uintptr_t>(job.data);
    const auto region_address = reinterpret_cast<std::uintptr_t>(job.region);
    const std::uint64_t offset = address - region_address;
    if (offset + bytes > job.region_bytes) {
        return {};
    }
    const std::uint64_t aligned = offset / page * page;
    const std::uint64_t head = offset - aligned;
    const std::uint64_t length = (head + bytes + page - 1U) / page * page;
    if (weight_buffer != nil) {
        if (weight_buffer.length != static_cast<NSUInteger>(length)) {
            if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
                std::fprintf(stderr, "mnet: metal alias cache shape mismatch (%lu vs %llu)\n",
                             static_cast<unsigned long>(weight_buffer.length),
                             static_cast<unsigned long long>(length));
            }
            return {};
        }
        return {weight_buffer, head};
    }
    weight_buffer = [s.device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(job.data - head)
                                                length:static_cast<NSUInteger>(length)
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (weight_buffer == nil) {
        s.weight_buffers.erase(key);
        return {};
    }
    return {weight_buffer, head};
}

void encode_dispatch(MetalState& s, id<MTLComputeCommandEncoder> encoder, const WeightBinding& weights,
                     std::uint64_t rows, std::uint64_t columns, std::uint64_t row_bytes, float scale,
                     std::uint64_t out_offset_floats, id<MTLBuffer> activations, id<MTLBuffer> activation_scales,
                     id<MTLBuffer> output_buffer) {
    const std::uint64_t blocks = columns / 64U;
    [encoder setBuffer:weights.buffer offset:static_cast<NSUInteger>(weights.offset) atIndex:0];
    [encoder setBuffer:activations offset:0 atIndex:1];
    [encoder setBuffer:activation_scales offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:static_cast<NSUInteger>(out_offset_floats * sizeof(float)) atIndex:3];
    const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(row_bytes);
    const std::uint32_t blocks32 = static_cast<std::uint32_t>(blocks);
    [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
    [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
    static const bool use_fast = std::getenv("MNET_METAL_FAST") != nullptr && s.fast_pipeline != nil;
    static const bool use_f32 = use_fast && s.fast_f32_pipeline != nil && s.dequant_pipeline != nil;
    if (use_f32) {
        // Widen the activation once, then read fp32 in the inner loop.
        id<MTLBuffer> wide =
            ensure_buffer(s.device, s.act_f32_buffer, s.act_f32_capacity, static_cast<std::size_t>(columns) * sizeof(float));
        if (wide != nil) {
            [encoder setComputePipelineState:s.dequant_pipeline];
            [encoder setBuffer:activations offset:0 atIndex:0];
            [encoder setBuffer:activation_scales offset:0 atIndex:1];
            [encoder setBuffer:wide offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder setComputePipelineState:s.fast_f32_pipeline];
            [encoder setBuffer:weights.buffer offset:static_cast<NSUInteger>(weights.offset) atIndex:0];
            [encoder setBuffer:wide offset:0 atIndex:1];
            [encoder setBuffer:output_buffer offset:static_cast<NSUInteger>(out_offset_floats * sizeof(float)) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            return;
        }
    }
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
                 float scale, const std::int8_t* q8, const float* q8_scales, float* out, std::uint64_t identity,
                 const std::uint8_t* region, std::uint64_t region_bytes) {
    MatvecQ8_0Job job{42U, weights, rows, row_bytes, scale, out, identity, region, region_bytes};
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
        const WeightBinding weight_buffer = weight_buffer_for(s, jobs[i], jobs[i].rows, jobs[i].row_bytes, true);
        if (weight_buffer.buffer == nil) {
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

// MNET_PROFILE splits the fused FFN call into host prep, submission and
// readback so the per-call overhead can be compared with the GPU work.
struct FfnProfile {
    std::uint64_t q8_ns = 0;
    std::uint64_t copy_ns = 0;
    std::uint64_t prep_ns = 0;
    std::uint64_t submit_ns = 0;
    std::uint64_t wait_ns = 0;
    std::uint64_t tail_ns = 0;
    std::uint64_t calls = 0;
    std::uint64_t misses = 0;
    ~FfnProfile() {
        if (std::getenv("MNET_PROFILE") == nullptr || this->calls == 0U) {
            return;
        }
        const auto per_call = [&](std::uint64_t ns) {
            return static_cast<double>(ns) * 1.0e-6 / static_cast<double>(this->calls);
        };
        char line[256];
        std::snprintf(line, sizeof(line),
                      "mnet: ffn per call  q8 %.3f  buffers %.3f  weights %.3f  encode %.3f  wait %.3f  tail %.3f ms\n",
                      per_call(this->q8_ns), per_call(this->copy_ns), per_call(this->prep_ns),
                      per_call(this->submit_ns), per_call(this->wait_ns), per_call(this->tail_ns));
        mnet::detail::write_diagnostic(std::string(line));
        char second[192];
        const auto& map = state().weight_buffers;
        std::snprintf(second, sizeof(second), "mnet: weight cache %zu entries, %zu buckets, %zu ffn misses\n",
                      map.size(), map.bucket_count(), static_cast<std::size_t>(this->misses));
        mnet::detail::write_diagnostic(std::string(second));
    }
};

FfnProfile& ffn_profile() {
    static FfnProfile profile;
    return profile;
}

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Same split for the gated-delta block, so the alias and state costs can be
// attributed: host prep, weight binding, encoding, then the GPU wait.
struct SsmProfile {
    std::uint64_t prep_ns = 0;
    std::uint64_t weights_ns = 0;
    std::uint64_t encode_ns = 0;
    std::uint64_t wait_ns = 0;
    std::uint64_t calls = 0;
    ~SsmProfile() {
        if (std::getenv("MNET_PROFILE") == nullptr || this->calls == 0U) {
            return;
        }
        const auto per_call = [&](std::uint64_t ns) {
            return static_cast<double>(ns) * 1.0e-6 / static_cast<double>(this->calls);
        };
        char line[256];
        std::snprintf(line, sizeof(line), "mnet: ssm per call  prep %.3f  weights %.3f  encode %.3f  wait %.3f ms\n",
                      per_call(this->prep_ns), per_call(this->weights_ns), per_call(this->encode_ns),
                      per_call(this->wait_ns));
        mnet::detail::write_diagnostic(std::string(line));
    }
};

SsmProfile& ssm_profile() {
    static SsmProfile profile;
    return profile;
}

// gate/up matvec -> SwiGLU -> on-device Q8_0 -> down matvec, one submission.
bool ffn_swiglu(const MatvecQ8_0Job& gate, const MatvecQ8_0Job& up, const MatvecQ8_0Job& down, std::uint64_t columns,
                std::uint64_t hidden, const float* input, const std::int32_t* signs, std::uint32_t block_size) {
    MetalState& s = state();
    if (!s.ready || s.swiglu_pipeline == nil || s.hadamard_quantize_pipeline == nil) {
        return false;
    }
    if (block_size < 64U || block_size > 1024U || (block_size & (block_size - 1U)) != 0U) {
        return false;
    }
    if (gate.type != 42U || up.type != 42U || down.type != 42U) {
        return false;
    }
    if (up.rows != hidden || down.rows == 0 || (columns % 64U) != 0U || (hidden % block_size) != 0U) {
        return false;
    }

    const std::uint64_t profile_start = now_ns();
    const auto q80 = mnet::gguf::make_q8_0_blocks(input, columns);
    const std::uint64_t profile_q8 = now_ns();
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

    const std::uint64_t profile_copy = now_ns();
    const std::size_t cache_before = s.weight_buffers.size();
    const WeightBinding gate_weights = weight_buffer_for(s, gate, gate.rows, gate.row_bytes, true);
    const WeightBinding up_weights = weight_buffer_for(s, up, up.rows, up.row_bytes, true);
    const WeightBinding down_weights = weight_buffer_for(s, down, down.rows, down.row_bytes, true);
    if (gate_weights.buffer == nil || up_weights.buffer == nil || down_weights.buffer == nil) {
        return false;
    }

    const std::uint64_t profile_prep = now_ns();
    {
        FfnProfile& profile = ffn_profile();
        profile.misses += s.weight_buffers.size() - cache_before;
        profile.q8_ns += profile_q8 - profile_start;
        profile.copy_ns += profile_copy - profile_q8;
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

    // Signed Hadamard and requantisation in one dispatch (was three).
    [encoder setComputePipelineState:s.hadamard_quantize_pipeline];
    [encoder setBuffer:act_buffer offset:0 atIndex:0];
    [encoder setBuffer:signs_buffer offset:0 atIndex:1];
    [encoder setBuffer:act_q8_buffer offset:0 atIndex:2];
    [encoder setBuffer:act_scale_buffer offset:0 atIndex:3];
    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(hidden / block_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];

    encode_dispatch(s, encoder, down_weights, down.rows, hidden, down.row_bytes, down.scale, 0, act_q8_buffer,
                    act_scale_buffer, out_buffer);
    [encoder endEncoding];
    [command commit];
    const std::uint64_t profile_submit = now_ns();
    [command waitUntilCompleted];
    const std::uint64_t profile_wait = now_ns();
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(down.out, out_buffer.contents, static_cast<std::size_t>(down.rows) * sizeof(float));
    {
        FfnProfile& profile = ffn_profile();
        profile.prep_ns += profile_prep - profile_q8;
        profile.submit_ns += profile_submit - profile_prep;
        profile.wait_ns += profile_wait - profile_submit;
        profile.tail_ns += now_ns() - profile_wait;
        ++profile.calls;
    }
    if (const char* dir = std::getenv("MNET_FFN_DEBUG_DIR"); dir != nullptr) {
        const std::string prefix(dir);
        const auto write = [&](const char* name, const void* data, std::size_t bytes) {
            if (std::FILE* file = std::fopen((prefix + "/" + name).c_str(), "wb")) {
                std::fwrite(data, 1, bytes, file);
                std::fclose(file);
            }
        };
        write("gpu-gate.f32", gate_buffer.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-up.f32", up_buffer.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-act.f32", act_buffer.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-actq8.bin", act_q8_buffer.contents, static_cast<std::size_t>(hidden));
        write("gpu-actscale.f32", act_scale_buffer.contents, static_cast<std::size_t>(hidden / 32U) * sizeof(float));
        write("gpu-out.f32", out_buffer.contents, static_cast<std::size_t>(down.rows) * sizeof(float));
    }
    return true;
}

// q/k/v matvec -> split + head RMSNorm + RoPE + f16 KV append -> attention ->
// gate -> signed Hadamard -> on-device Q8_0 -> out matvec, one submission.
bool attention(const AttentionQ2_0Job& job) {
    MetalState& s = state();
    if (!s.ready || s.attn_prep_pipeline == nil || s.attn_flash_pipeline == nil || s.attn_gate_pipeline == nil ||
        s.attn_store_pipeline == nil || s.hadamard_quantize_pipeline == nil) {
        return false;
    }
    if (job.q.type != 42U || job.k.type != 42U || job.v.type != 42U || job.out.type != 42U) {
        return false;
    }
    const std::uint64_t head_dim = job.head_dim;
    const std::uint64_t heads = job.n_head;
    const std::uint64_t heads_kv = job.n_head_kv;
    if (head_dim == 0U || head_dim > 256U || (head_dim % 32U) != 0U) {
        return false;
    }
    if (heads == 0U || heads_kv == 0U || (heads % heads_kv) != 0U) {
        return false;
    }
    if (job.rope_dims == 0U || (job.rope_dims % 2U) != 0U || job.rope_dims > head_dim) {
        return false;
    }
    if (job.block_size < 64U || job.block_size > 1024U || (job.block_size & (job.block_size - 1U)) != 0U) {
        return false;
    }
    if (job.columns == 0U || (job.columns % 64U) != 0U || (job.input == nullptr)) {
        return false;
    }
    const std::uint64_t hidden = heads * head_dim;
    if ((hidden % job.block_size) != 0U || job.out.rows != job.columns) {
        return false;
    }
    const std::uint64_t kv_head_elems = heads_kv * head_dim;
    const std::uint64_t tokens = static_cast<std::uint64_t>(job.cache_tokens) + 1U;
    const std::uint64_t kv_plane_bytes = tokens * kv_head_elems * sizeof(float);

    AttentionBuffers& b = s.attn;
    b.qfull = ensure_buffer(s.device, b.qfull, b.qfull_capacity, static_cast<std::size_t>(2U * hidden * sizeof(float)));
    b.q = ensure_buffer(s.device, b.q, b.q_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.k_new = ensure_buffer(s.device, b.k_new, b.k_new_capacity, static_cast<std::size_t>(kv_head_elems * sizeof(float)));
    b.v_new = ensure_buffer(s.device, b.v_new, b.v_new_capacity, static_cast<std::size_t>(kv_head_elems * sizeof(float)));
    b.gates = ensure_buffer(s.device, b.gates, b.gates_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.attn = ensure_buffer(s.device, b.attn, b.attn_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.q8 = ensure_buffer(s.device, b.q8, b.q8_capacity, static_cast<std::size_t>(job.columns));
    b.scales = ensure_buffer(s.device, b.scales, b.scales_capacity,
                             static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
    b.kv = ensure_buffer(s.device, b.kv, b.kv_capacity, static_cast<std::size_t>(2U * kv_plane_bytes));
    b.tables = ensure_buffer(s.device, b.tables, b.tables_capacity,
                             static_cast<std::size_t>((2U * head_dim + job.rope_dims) * sizeof(float)));
    b.signs = ensure_buffer(s.device, b.signs, b.signs_capacity,
                            static_cast<std::size_t>(hidden * sizeof(std::int32_t)));
    id<MTLBuffer> act_q8 = ensure_buffer(s.device, s.act_q8_buffer, s.act_q8_capacity,
                                         static_cast<std::size_t>(hidden));
    id<MTLBuffer> act_scales = ensure_buffer(s.device, s.act_scale_buffer, s.act_scale_capacity,
                                             static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    id<MTLBuffer> out_buffer = ensure_buffer(s.device, s.out_buffer, s.out_capacity,
                                             static_cast<std::size_t>(job.out.rows) * sizeof(float));
    if (b.qfull == nil || b.q == nil || b.k_new == nil || b.v_new == nil || b.gates == nil || b.attn == nil ||
        b.q8 == nil || b.scales == nil || b.kv == nil || b.tables == nil || b.signs == nil || act_q8 == nil ||
        act_scales == nil || out_buffer == nil) {
        return false;
    }
    std::memcpy(b.signs.contents, job.signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));

    const WeightBinding q_weights = weight_buffer_for(s, job.q, job.q.rows, job.q.row_bytes, true);
    const WeightBinding k_weights = weight_buffer_for(s, job.k, job.k.rows, job.k.row_bytes, true);
    const WeightBinding v_weights = weight_buffer_for(s, job.v, job.v.rows, job.v.row_bytes, true);
    const WeightBinding out_weights = weight_buffer_for(s, job.out, job.out.rows, job.out.row_bytes, true);
    if (q_weights.buffer == nil || k_weights.buffer == nil || v_weights.buffer == nil ||
        out_weights.buffer == nil) {
        return false;
    }

    const auto q80 = mnet::gguf::make_q8_0_blocks(job.input, job.columns);
    std::memcpy(b.q8.contents, q80.qs.data(), static_cast<std::size_t>(job.columns));
    std::memcpy(b.scales.contents, q80.d.data(), static_cast<std::size_t>(job.columns / 32U) * sizeof(float));

    const std::uint64_t half_rope = job.rope_dims / 2U;
    auto* tables = static_cast<float*>(b.tables.contents);
    const std::size_t q_norm_offset = 0U;
    const std::size_t k_norm_offset = static_cast<std::size_t>(head_dim);
    const std::size_t cos_offset = static_cast<std::size_t>(2U * head_dim);
    const std::size_t sin_offset = cos_offset + static_cast<std::size_t>(half_rope);
    std::memcpy(tables + q_norm_offset, job.q_norm, static_cast<std::size_t>(head_dim) * sizeof(float));
    std::memcpy(tables + k_norm_offset, job.k_norm, static_cast<std::size_t>(head_dim) * sizeof(float));
    std::memcpy(tables + cos_offset, job.rope_cos, static_cast<std::size_t>(half_rope) * sizeof(float));
    std::memcpy(tables + sin_offset, job.rope_sin, static_cast<std::size_t>(half_rope) * sizeof(float));

    // The host cache stays authoritative: seed the device planes with it and
    // copy the freshly appended token back once the submission is done.
    const std::uint64_t cached_bytes = static_cast<std::uint64_t>(job.cache_tokens) * kv_head_elems * sizeof(float);
    auto* kv = static_cast<std::uint8_t*>(b.kv.contents);
    if (cached_bytes != 0U) {
        std::memcpy(kv, job.k_cache, static_cast<std::size_t>(cached_bytes));
        std::memcpy(kv + kv_plane_bytes, job.v_cache, static_cast<std::size_t>(cached_bytes));
    }

    const std::uint32_t head_dim32 = static_cast<std::uint32_t>(head_dim);
    const std::uint32_t rope32 = job.rope_dims;
    const std::uint32_t tokens32 = static_cast<std::uint32_t>(tokens);
    const std::uint32_t heads_kv32 = static_cast<std::uint32_t>(heads_kv);
    const std::uint32_t group32 = static_cast<std::uint32_t>(heads / heads_kv);
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    const std::uint32_t stride_q = static_cast<std::uint32_t>(2U * head_dim);
    const std::uint32_t half_store_off = 0U;
    const std::uint32_t half_store_on = 1U;
    const float eps = job.eps;
    const float attn_scale = job.attention_scale;

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_dispatch(s, encoder, q_weights, job.q.rows, job.columns, job.q.row_bytes, job.q.scale, 0, b.q8, b.scales,
                    b.qfull);
    encode_dispatch(s, encoder, k_weights, job.k.rows, job.columns, job.k.row_bytes, job.k.scale, 0, b.q8, b.scales,
                    b.k_new);
    encode_dispatch(s, encoder, v_weights, job.v.rows, job.columns, job.v.row_bytes, job.v.scale, 0, b.q8, b.scales,
                    b.v_new);

    // Query half: RMSNorm + RoPE into b.q; gate half: sigmoid into b.gates.
    [encoder setComputePipelineState:s.attn_prep_pipeline];
    [encoder setBuffer:b.qfull offset:0 atIndex:0];
    [encoder setBuffer:b.tables offset:q_norm_offset * sizeof(float) atIndex:1];
    [encoder setBuffer:b.tables offset:cos_offset * sizeof(float) atIndex:2];
    [encoder setBuffer:b.tables offset:sin_offset * sizeof(float) atIndex:3];
    [encoder setBuffer:b.q offset:0 atIndex:4];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
    [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
    [encoder setBytes:&stride_q length:sizeof(stride_q) atIndex:8];
    [encoder setBytes:&half_store_off length:sizeof(half_store_off) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

    [encoder setComputePipelineState:s.attn_gate_pipeline];
    [encoder setBuffer:b.qfull offset:0 atIndex:0];
    [encoder setBuffer:b.gates offset:0 atIndex:1];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:2];
    [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    // Key half: RMSNorm + RoPE, f16 rounded straight into the device plane.
    [encoder setComputePipelineState:s.attn_prep_pipeline];
    [encoder setBuffer:b.k_new offset:0 atIndex:0];
    [encoder setBuffer:b.tables offset:k_norm_offset * sizeof(float) atIndex:1];
    [encoder setBuffer:b.tables offset:cos_offset * sizeof(float) atIndex:2];
    [encoder setBuffer:b.tables offset:sin_offset * sizeof(float) atIndex:3];
    [encoder setBuffer:b.kv offset:static_cast<NSUInteger>(cached_bytes) atIndex:4];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
    [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
    [encoder setBytes:&half_store_on length:sizeof(half_store_on) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(heads_kv, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

    // Value half: f16 round into the second plane.
    const std::uint32_t kv_elems32 = static_cast<std::uint32_t>(kv_head_elems);
    [encoder setComputePipelineState:s.attn_store_pipeline];
    [encoder setBuffer:b.v_new offset:0 atIndex:0];
    [encoder setBuffer:b.kv offset:static_cast<NSUInteger>(kv_plane_bytes + cached_bytes) atIndex:1];
    [encoder setBytes:&kv_elems32 length:sizeof(kv_elems32) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(kv_head_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    [encoder setComputePipelineState:s.attn_flash_pipeline];
    [encoder setBuffer:b.q offset:0 atIndex:0];
    [encoder setBuffer:b.kv offset:0 atIndex:1];
    [encoder setBuffer:b.kv offset:static_cast<NSUInteger>(kv_plane_bytes) atIndex:2];
    [encoder setBuffer:b.gates offset:0 atIndex:3];
    [encoder setBuffer:b.attn offset:0 atIndex:4];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
    [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:6];
    [encoder setBytes:&heads_kv32 length:sizeof(heads_kv32) atIndex:7];
    [encoder setBytes:&group32 length:sizeof(group32) atIndex:8];
    [encoder setBytes:&attn_scale length:sizeof(attn_scale) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [encoder setComputePipelineState:s.hadamard_quantize_pipeline];
    [encoder setBuffer:b.attn offset:0 atIndex:0];
    [encoder setBuffer:b.signs offset:0 atIndex:1];
    [encoder setBuffer:act_q8 offset:0 atIndex:2];
    [encoder setBuffer:act_scales offset:0 atIndex:3];
    [encoder setBytes:&job.block_size length:sizeof(job.block_size) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(hidden / job.block_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(job.block_size, 1, 1)];

    encode_dispatch(s, encoder, out_weights, job.out.rows, hidden, job.out.row_bytes, job.out.scale, 0, act_q8,
                    act_scales, out_buffer);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(job.result, out_buffer.contents, static_cast<std::size_t>(job.out.rows) * sizeof(float));
    const std::uint64_t new_token_bytes = kv_head_elems * sizeof(float);
    const auto new_token_offset = static_cast<std::ptrdiff_t>(job.cache_tokens * kv_head_elems);
    std::memcpy(job.k_cache + new_token_offset, kv + cached_bytes, static_cast<std::size_t>(new_token_bytes));
    std::memcpy(job.v_cache + new_token_offset, kv + kv_plane_bytes + cached_bytes,
                static_cast<std::size_t>(new_token_bytes));

    if (const char* dir = std::getenv("MNET_ATTN_DEBUG_DIR"); dir != nullptr) {
        const std::string prefix(dir);
        const auto write = [&](const char* name, const void* data, std::size_t bytes) {
            if (std::FILE* file = std::fopen((prefix + "/" + name).c_str(), "wb")) {
                std::fwrite(data, 1, bytes, file);
                std::fclose(file);
            }
        };
        write("gpu-qfull.f32", b.qfull.contents, static_cast<std::size_t>(2U * hidden) * sizeof(float));
        write("gpu-qprep.f32", b.q.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-gatesig.f32", b.gates.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-kcache.f32", kv, static_cast<std::size_t>(tokens * kv_head_elems) * sizeof(float));
        write("gpu-vcache.f32", kv + kv_plane_bytes, static_cast<std::size_t>(tokens * kv_head_elems) * sizeof(float));
        write("gpu-attn.f32", b.attn.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-out.f32", out_buffer.contents, static_cast<std::size_t>(job.out.rows) * sizeof(float));
    }
    return true;
}

// qkv/gate matvec -> head gates -> causal depthwise conv -> delta-rule scan ->
// gated RMS norm with the grouped permute -> signed Hadamard -> on-device
// Q8_0 -> ssm_out matvec, one submission per layer-token.
bool ssm_block(const SsmQ2_0Job& job) {
    MetalState& s = state();
    if (!s.ready || s.ssm_rows_pipeline == nil || s.ssm_act_pipeline == nil || s.ssm_conv_norm_pipeline == nil ||
        s.ssm_scan_pipeline == nil || s.ssm_gate_norm_pipeline == nil || s.hadamard_quantize_pipeline == nil) {
        return false;
    }
    if (job.qkv.type != 42U || job.gate.type != 42U || job.out.type != 42U) {
        return false;
    }
    const std::uint64_t ssm_prep_start = now_ns();
    const std::uint64_t head_dim = job.head_dim;
    const std::uint64_t key_heads = job.key_heads;
    const std::uint64_t value_heads = job.value_heads;
    const std::uint64_t conv_dim = job.conv_dim;
    if (head_dim == 0U || head_dim > 256U || (head_dim % 32U) != 0U || key_heads == 0U || value_heads == 0U ||
        (value_heads % key_heads) != 0U) {
        return false;
    }
    if (job.conv_kernel < 2U || job.conv_kernel > 8U || (head_dim % 32U) != 0U) {
        return false;
    }
    const std::uint64_t hidden = value_heads * head_dim;
    if (conv_dim != 2U * key_heads * head_dim + hidden) {
        return false;
    }
    if (job.block_size < 64U || job.block_size > 1024U || (job.block_size & (job.block_size - 1U)) != 0U ||
        (hidden % job.block_size) != 0U) {
        return false;
    }
    if (job.columns == 0U || (job.columns % 64U) != 0U || job.qkv.rows != conv_dim || job.gate.rows != hidden ||
        job.out.rows != job.columns) {
        return false;
    }

    SsmBuffers& b = s.ssm;
    b.activation =
        ensure_buffer(s.device, b.activation, b.activation_capacity, static_cast<std::size_t>(job.columns) * sizeof(float));
    b.raw = ensure_buffer(s.device, b.raw, b.raw_capacity, static_cast<std::size_t>(2U * value_heads * sizeof(float)));
    b.gates = ensure_buffer(s.device, b.gates, b.gates_capacity,
                            static_cast<std::size_t>(2U * value_heads * sizeof(float)));
    b.conv = ensure_buffer(s.device, b.conv, b.conv_capacity, static_cast<std::size_t>(conv_dim * sizeof(float)));
    b.z = ensure_buffer(s.device, b.z, b.z_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.delta = ensure_buffer(s.device, b.delta, b.delta_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.grouped = ensure_buffer(s.device, b.grouped, b.grouped_capacity, static_cast<std::size_t>(hidden * sizeof(float)));
    b.q8 = ensure_buffer(s.device, b.q8, b.q8_capacity, static_cast<std::size_t>(job.columns));
    b.scales = ensure_buffer(s.device, b.scales, b.scales_capacity,
                             static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
    b.signs = ensure_buffer(s.device, b.signs, b.signs_capacity,
                            static_cast<std::size_t>(hidden * sizeof(std::int32_t)));
    id<MTLBuffer> act_q8 = ensure_buffer(s.device, s.act_q8_buffer, s.act_q8_capacity, static_cast<std::size_t>(hidden));
    id<MTLBuffer> act_scales = ensure_buffer(s.device, s.act_scale_buffer, s.act_scale_capacity,
                                             static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    id<MTLBuffer> out_buffer = ensure_buffer(s.device, s.out_buffer, s.out_capacity,
                                             static_cast<std::size_t>(job.out.rows) * sizeof(float));
    if (b.activation == nil || b.raw == nil || b.gates == nil || b.conv == nil || b.z == nil || b.delta == nil ||
        b.grouped == nil || b.q8 == nil || b.scales == nil || b.signs == nil || act_q8 == nil || act_scales == nil ||
        out_buffer == nil) {
        return false;
    }
    std::memcpy(b.signs.contents, job.signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));

    const std::uint64_t ssm_bind = now_ns();
    // Aliasing hurts this block specifically (its scan is latency bound, so
    // weight reads through the mapping land on the critical path: 0.871 vs
    // 0.561 ms of GPU wait per call). Copying only this block back was measured
    // worse overall (3.60 s vs 3.41 s) because 1.9 GB of device copies push the
    // FFN's file pages out of the page cache, so everything aliases.
    const std::uint64_t profile_bind = now_ns();
    const WeightBinding qkv_weights = weight_buffer_for(s, job.qkv, job.qkv.rows, job.qkv.row_bytes, true);
    const WeightBinding gate_weights = weight_buffer_for(s, job.gate, job.gate.rows, job.gate.row_bytes, true);
    const WeightBinding out_weights = weight_buffer_for(s, job.out, job.out.rows, job.out.row_bytes, true);
    if (qkv_weights.buffer == nil || gate_weights.buffer == nil || out_weights.buffer == nil) {
        return false;
    }
    const std::uint64_t ssm_weights = now_ns();

    // Resident state, seeded from the caller's vectors on first use.
    const std::uint64_t window_floats = static_cast<std::uint64_t>(job.conv_kernel) * conv_dim;
    const std::uint64_t side_floats = 2U * value_heads * job.columns + 2U * value_heads +
                                      conv_dim * job.conv_kernel + head_dim;
    const std::uint64_t state_floats = value_heads * head_dim * head_dim;
    SsmLayer& layer = s.ssm_layers[cache_key(job.identity, job.qkv.data)];
    if (layer.window == nil) {
        layer.window = [s.device newBufferWithLength:static_cast<NSUInteger>(window_floats * sizeof(float))
                                             options:MTLResourceStorageModeShared];
        layer.state = [s.device newBufferWithLength:static_cast<NSUInteger>(state_floats * sizeof(float))
                                            options:MTLResourceStorageModeShared];
        layer.side = [s.device newBufferWithLength:static_cast<NSUInteger>(side_floats * sizeof(float))
                                           options:MTLResourceStorageModeShared];
        if (layer.window == nil || layer.state == nil || layer.side == nil) {
            layer = SsmLayer{};
            return false;
        }
        auto* side = static_cast<float*>(layer.side.contents);
        const std::size_t beta_offset = 0U;
        const std::size_t alpha_offset = static_cast<std::size_t>(value_heads * job.columns);
        const std::size_t dt_offset = 2U * alpha_offset;
        const std::size_t a_offset = dt_offset + static_cast<std::size_t>(value_heads);
        const std::size_t conv_w_offset = a_offset + static_cast<std::size_t>(value_heads);
        const std::size_t norm_offset = conv_w_offset + static_cast<std::size_t>(conv_dim * job.conv_kernel);
        std::memcpy(side + beta_offset, job.beta_w, static_cast<std::size_t>(value_heads * job.columns) * sizeof(float));
        std::memcpy(side + alpha_offset, job.alpha_w,
                    static_cast<std::size_t>(value_heads * job.columns) * sizeof(float));
        std::memcpy(side + dt_offset, job.dt, static_cast<std::size_t>(value_heads) * sizeof(float));
        std::memcpy(side + a_offset, job.ssm_a, static_cast<std::size_t>(value_heads) * sizeof(float));
        std::memcpy(side + conv_w_offset, job.conv_w,
                    static_cast<std::size_t>(conv_dim * job.conv_kernel) * sizeof(float));
        std::memcpy(side + norm_offset, job.ssm_norm, static_cast<std::size_t>(head_dim) * sizeof(float));

        auto* window = static_cast<float*>(layer.window.contents);
        std::memcpy(window, job.conv_state,
                    static_cast<std::size_t>((job.conv_kernel - 1U) * conv_dim) * sizeof(float));
        layer.ring = job.conv_kernel - 1U;
        auto* resident = static_cast<float*>(layer.state.contents);
        for (std::uint64_t h = 0; h < value_heads; ++h) {
            const float* source = job.ssm_state + h * head_dim * head_dim;
            float* target = resident + h * head_dim * head_dim;
            for (std::uint64_t j = 0; j < head_dim; ++j) {
                for (std::uint64_t i = 0; i < head_dim; ++i) {
                    target[i * head_dim + j] = source[j * head_dim + i];
                }
            }
        }
    }
    auto* side = static_cast<const float*>(layer.side.contents);
    const std::size_t beta_offset = 0U;
    const std::size_t alpha_offset = static_cast<std::size_t>(value_heads * job.columns);
    const std::size_t dt_offset = 2U * alpha_offset;
    const std::size_t a_offset = dt_offset + static_cast<std::size_t>(value_heads);
    const std::size_t conv_w_offset = a_offset + static_cast<std::size_t>(value_heads);
    const std::size_t norm_offset = conv_w_offset + static_cast<std::size_t>(conv_dim * job.conv_kernel);

    const auto q80 = mnet::gguf::make_q8_0_blocks(job.input, job.columns);
    std::memcpy(b.q8.contents, q80.qs.data(), static_cast<std::size_t>(job.columns));
    std::memcpy(b.scales.contents, q80.d.data(), static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
    std::memcpy(b.activation.contents, job.raw_input, static_cast<std::size_t>(job.columns) * sizeof(float));

    const std::uint32_t head_dim32 = static_cast<std::uint32_t>(head_dim);
    const std::uint32_t key_heads32 = static_cast<std::uint32_t>(key_heads);
    const std::uint32_t value_heads32 = static_cast<std::uint32_t>(value_heads);
    const std::uint32_t conv_dim32 = static_cast<std::uint32_t>(conv_dim);
    const std::uint32_t kernel32 = job.conv_kernel;
    const std::uint32_t ring32 = layer.ring;
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    const std::uint32_t columns32 = static_cast<std::uint32_t>(job.columns);
    const std::uint32_t heads_per_key32 = static_cast<std::uint32_t>(value_heads / key_heads);
    const float eps = job.eps;
    const float inv_dim = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const std::uint32_t q_offset = 0U;
    const std::uint32_t k_offset = static_cast<std::uint32_t>(key_heads * head_dim);
    const std::uint32_t v_offset = static_cast<std::uint32_t>(2U * key_heads * head_dim);

    // MNET_SSM_SKIP=<phase> drops one phase for timing; results are garbage.
    static const std::string skip =
        std::getenv("MNET_SSM_SKIP") != nullptr ? std::string(std::getenv("MNET_SSM_SKIP")) : std::string();
    const bool skip_all = skip == "all";
    const bool only_matvec = skip == "onlymatvec";
    const bool skip_matvec = skip_all || skip == "matvec";
    const bool skip_gate = skip_all || only_matvec || skip == "gate";
    const bool skip_conv = skip_all || only_matvec || skip == "conv";
    const bool skip_scan = skip_all || only_matvec || skip == "scan";
    const bool skip_norm = skip_all || only_matvec || skip == "norm";

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!skip_matvec) {
    encode_dispatch(s, encoder, qkv_weights, job.qkv.rows, job.columns, job.qkv.row_bytes, job.qkv.scale,
                    static_cast<std::uint64_t>(ring32) * conv_dim, b.q8, b.scales, layer.window);
    encode_dispatch(s, encoder, gate_weights, job.gate.rows, job.columns, job.gate.row_bytes, job.gate.scale, 0, b.q8,
                    b.scales, b.z);
    }

    if (!skip_gate) {
    [encoder setComputePipelineState:s.ssm_rows_pipeline];
    [encoder setBuffer:layer.side offset:beta_offset * sizeof(float) atIndex:0];
    [encoder setBuffer:b.activation offset:0 atIndex:1];
    [encoder setBuffer:b.raw offset:0 atIndex:2];
    [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(2U * value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [encoder setComputePipelineState:s.ssm_act_pipeline];
    [encoder setBuffer:b.raw offset:0 atIndex:0];
    [encoder setBuffer:layer.side offset:dt_offset * sizeof(float) atIndex:1];
    [encoder setBuffer:layer.side offset:a_offset * sizeof(float) atIndex:2];
    [encoder setBuffer:b.gates offset:0 atIndex:3];
    [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(value_heads) * sizeof(float) atIndex:4];
    [encoder setBytes:&value_heads32 length:sizeof(value_heads32) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    if (!skip_conv) {
    // Convolution and the q/k L2 norms in one dispatch.
    const std::uint32_t norm_blocks32 = static_cast<std::uint32_t>(2U * key_heads);
    [encoder setComputePipelineState:s.ssm_conv_norm_pipeline];
    [encoder setBuffer:layer.window offset:0 atIndex:0];
    [encoder setBuffer:layer.side offset:conv_w_offset * sizeof(float) atIndex:1];
    [encoder setBuffer:b.conv offset:0 atIndex:2];
    [encoder setBytes:&conv_dim32 length:sizeof(conv_dim32) atIndex:3];
    [encoder setBytes:&kernel32 length:sizeof(kernel32) atIndex:4];
    [encoder setBytes:&ring32 length:sizeof(ring32) atIndex:5];
    [encoder setBytes:&norm_blocks32 length:sizeof(norm_blocks32) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
    [encoder setBytes:&kernel32 length:sizeof(kernel32) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(conv_dim / head_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    }

    if (!skip_scan) {
    [encoder setComputePipelineState:s.ssm_scan_pipeline];
    [encoder setBuffer:b.conv offset:q_offset * sizeof(float) atIndex:0];
    [encoder setBuffer:b.conv offset:k_offset * sizeof(float) atIndex:1];
    [encoder setBuffer:b.conv offset:v_offset * sizeof(float) atIndex:2];
    [encoder setBuffer:b.gates offset:0 atIndex:3];
    [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(value_heads) * sizeof(float) atIndex:4];
    [encoder setBuffer:layer.state offset:0 atIndex:5];
    [encoder setBuffer:b.delta offset:0 atIndex:6];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:7];
    [encoder setBytes:&key_heads32 length:sizeof(key_heads32) atIndex:8];
    [encoder setBytes:&inv_dim length:sizeof(inv_dim) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    }

    if (!skip_norm) {
    [encoder setComputePipelineState:s.ssm_gate_norm_pipeline];
    [encoder setBuffer:b.delta offset:0 atIndex:0];
    [encoder setBuffer:b.z offset:0 atIndex:1];
    [encoder setBuffer:layer.side offset:norm_offset * sizeof(float) atIndex:2];
    [encoder setBuffer:b.grouped offset:0 atIndex:3];
    [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:4];
    [encoder setBytes:&key_heads32 length:sizeof(key_heads32) atIndex:5];
    [encoder setBytes:&heads_per_key32 length:sizeof(heads_per_key32) atIndex:6];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

    [encoder setComputePipelineState:s.hadamard_quantize_pipeline];
    [encoder setBuffer:b.grouped offset:0 atIndex:0];
    [encoder setBuffer:b.signs offset:0 atIndex:1];
    [encoder setBuffer:act_q8 offset:0 atIndex:2];
    [encoder setBuffer:act_scales offset:0 atIndex:3];
    [encoder setBytes:&job.block_size length:sizeof(job.block_size) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(hidden / job.block_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(job.block_size, 1, 1)];

    encode_dispatch(s, encoder, out_weights, job.out.rows, hidden, job.out.row_bytes, job.out.scale, 0, act_q8,
                    act_scales, out_buffer);
    }
    [encoder endEncoding];
    [command commit];
    const std::uint64_t ssm_encode = now_ns();
    [command waitUntilCompleted];
    const std::uint64_t ssm_wait = now_ns();
    {
        SsmProfile& profile = ssm_profile();
        profile.prep_ns += ssm_bind - ssm_prep_start;
        profile.weights_ns += ssm_weights - ssm_bind;
        profile.encode_ns += ssm_encode - ssm_weights;
        profile.wait_ns += ssm_wait - ssm_encode;
        ++profile.calls;
    }
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(job.result, out_buffer.contents, static_cast<std::size_t>(job.out.rows) * sizeof(float));
    layer.ring = (layer.ring + 1U) % job.conv_kernel;

    if (const char* dir = std::getenv("MNET_SSM_DEBUG_DIR"); dir != nullptr) {
        const std::string prefix(dir);
        const auto write = [&](const char* name, const void* data, std::size_t bytes) {
            if (std::FILE* file = std::fopen((prefix + "/" + name).c_str(), "wb")) {
                std::fwrite(data, 1, bytes, file);
                std::fclose(file);
            }
        };
        write("gpu-qkv.f32", static_cast<const float*>(layer.window.contents) + ring32 * conv_dim,
              static_cast<std::size_t>(conv_dim) * sizeof(float));
        write("gpu-z.f32", b.z.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-raw.f32", b.raw.contents, static_cast<std::size_t>(2U * value_heads) * sizeof(float));
        write("gpu-beta.f32", b.gates.contents, static_cast<std::size_t>(value_heads) * sizeof(float));
        write("gpu-gate.f32", static_cast<const float*>(b.gates.contents) + value_heads,
              static_cast<std::size_t>(value_heads) * sizeof(float));
        write("gpu-conv.f32", b.conv.contents, static_cast<std::size_t>(conv_dim) * sizeof(float));
        write("gpu-delta.f32", b.delta.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-grouped.f32", b.grouped.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-out.f32", out_buffer.contents, static_cast<std::size_t>(job.out.rows) * sizeof(float));
        write("gpu-state.f32", layer.state.contents, static_cast<std::size_t>(state_floats) * sizeof(float));
        // Logical order (oldest first) so it lines up with the step-by-step path.
        {
            std::vector<float> ordered(static_cast<std::size_t>(window_floats));
            const auto* window = static_cast<const float*>(layer.window.contents);
            for (std::uint32_t k = 0; k + 1U < kernel32; ++k) {
                const std::uint32_t row = (ring32 + 1U + k) % kernel32;
                std::memcpy(ordered.data() + k * conv_dim, window + static_cast<std::size_t>(row) * conv_dim,
                            static_cast<std::size_t>(conv_dim) * sizeof(float));
            }
            std::memcpy(ordered.data() + (kernel32 - 1U) * conv_dim,
                        window + static_cast<std::size_t>(ring32) * conv_dim,
                        static_cast<std::size_t>(conv_dim) * sizeof(float));
            write("gpu-window.f32", ordered.data(), ordered.size() * sizeof(float));
        }
    }
    return true;
}

// Several tokens' FFN in one submission: every gate matvec, then every up
// matvec, then each token's swiglu and Hadamard/requantisation, then every down
// matvec. The per-token arithmetic is identical to ffn_swiglu; running the
// tokens together lets the independent matvecs of different tokens overlap,
// which the per-token block cannot because its chain is serial.
bool ffn_batch_block(const FfnSwigluEntry* entries, std::size_t count, std::uint64_t columns, std::uint64_t hidden,
                      const std::int32_t* signs, std::uint32_t block_size) {
    MetalState& s = state();
    if (!s.ready || s.swiglu_pipeline == nil || s.hadamard_quantize_pipeline == nil || s.fast_f32_pipeline == nil ||
        s.dequant_pipeline == nil || count == 0U) {
        return false;
    }
    if (block_size < 64U || block_size > 1024U || (block_size & (block_size - 1U)) != 0U ||
        (columns % 64U) != 0U || (hidden % block_size) != 0U) {
        return false;
    }
    const auto& first = entries[0];
    if (first.up.rows != hidden || first.gate.rows != hidden || first.down.rows != columns ||
        first.gate.type != 42U || first.up.type != 42U || first.down.type != 42U) {
        return false;
    }
    const WeightBinding gate_weights = weight_buffer_for(s, first.gate, first.gate.rows, first.gate.row_bytes, true);
    const WeightBinding up_weights = weight_buffer_for(s, first.up, first.up.rows, first.up.row_bytes, true);
    const WeightBinding down_weights = weight_buffer_for(s, first.down, first.down.rows, first.down.row_bytes, true);
    if (gate_weights.buffer == nil || up_weights.buffer == nil || down_weights.buffer == nil) {
        return false;
    }
    id<MTLBuffer> signs_buffer =
        ensure_buffer(s.device, s.signs_buffer, s.signs_capacity, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    if (signs_buffer == nil) {
        return false;
    }
    std::memcpy(signs_buffer.contents, signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));

    FfnBatchBuffers& b = s.ffn_batch;
    const std::size_t n = count;
    b.q8 = ensure_buffer(s.device, b.q8, b.q8_capacity, n * static_cast<std::size_t>(columns));
    b.scales = ensure_buffer(s.device, b.scales, b.scales_capacity,
                             n * static_cast<std::size_t>(columns / 32U) * sizeof(float));
    // The widened activation is reused for the down matvec, whose input is
    // hidden wide, so the slab has to cover the larger of the two.
    const std::uint64_t widest = columns > hidden ? columns : hidden;
    b.wide = ensure_buffer(s.device, b.wide, b.wide_capacity, n * static_cast<std::size_t>(widest) * sizeof(float));
    b.gate = ensure_buffer(s.device, b.gate, b.gate_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.up = ensure_buffer(s.device, b.up, b.up_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.act = ensure_buffer(s.device, b.act, b.act_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.act_q8 = ensure_buffer(s.device, b.act_q8, b.act_q8_capacity, n * static_cast<std::size_t>(hidden));
    b.act_scales = ensure_buffer(s.device, b.act_scales, b.act_scales_capacity,
                                 n * static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    b.out = ensure_buffer(s.device, b.out, b.out_capacity, n * static_cast<std::size_t>(columns) * sizeof(float));
    if (b.q8 == nil || b.scales == nil || b.wide == nil || b.gate == nil || b.up == nil || b.act == nil ||
        b.act_q8 == nil || b.act_scales == nil || b.out == nil) {
        return false;
    }
    // Quantise the batch's activations on the device: upload the fp32 inputs and
    // let one dispatch produce the codes, instead of 0.144 ms of host work per
    // layer (log 152/166).
    const bool device_q8 = s.q8_quantize_pipeline != nil;
    if (device_q8) {
        auto* staging = static_cast<float*>(b.wide.contents);
        for (std::size_t t = 0; t < count; ++t) {
            std::memcpy(staging + t * columns, entries[t].input, static_cast<std::size_t>(columns) * sizeof(float));
        }
    } else {
        auto* q8_destination = static_cast<std::uint8_t*>(b.q8.contents);
        auto* scale_destination = static_cast<float*>(b.scales.contents);
        for (std::size_t t = 0; t < count; ++t) {
            const auto q80 = mnet::gguf::make_q8_0_blocks(entries[t].input, columns);
            std::memcpy(q8_destination + t * columns, q80.qs.data(), static_cast<std::size_t>(columns));
            std::memcpy(scale_destination + t * (columns / 32U), q80.d.data(),
                        static_cast<std::size_t>(columns / 32U) * sizeof(float));
        }
    }

    const std::uint32_t block32 = block_size;
    // MNET_FFN_SKIP=<phase> drops one phase for timing; results are garbage.
    static const std::string ffn_skip =
        std::getenv("MNET_FFN_SKIP") != nullptr ? std::string(std::getenv("MNET_FFN_SKIP")) : std::string();
    const bool skip_all = ffn_skip == "all";
    const bool skip_matvec = skip_all || ffn_skip == "matvec";
    const bool skip_swiglu = skip_all || ffn_skip == "swiglu";
    const bool skip_hadamard = skip_all || ffn_skip == "hadamard";
    const bool skip_dequant = skip_all || ffn_skip == "dequant";

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (device_q8) {
        const std::uint32_t q8_blocks = static_cast<std::uint32_t>(columns / 32U);
        const std::uint32_t q8_tokens = static_cast<std::uint32_t>(count);
        [encoder setComputePipelineState:s.q8_quantize_pipeline];
        [encoder setBuffer:b.wide offset:0 atIndex:0];
        [encoder setBuffer:b.q8 offset:0 atIndex:1];
        [encoder setBuffer:b.scales offset:0 atIndex:2];
        [encoder setBytes:&q8_blocks length:sizeof(q8_blocks) atIndex:3];
        [encoder setBytes:&q8_tokens length:sizeof(q8_tokens) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake((columns / 32U) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    // Widen every token's activation once: the slabs are contiguous, so this is
    // a single dispatch over all tokens.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // From eight tokens on, one threadgroup computes a 64-row x 32-token tile
    // with the matrix unit, so each weight is decoded once for the batch. Below
    // that the per-token matvec has the better occupancy.
    // The tiled path changes the fp32 summation order: ~1e-6 per projection,
    // ~4e-3 in the final logits after 64 layers. It is the default because that
    // is still closer to the CPU reference than the scalar fast path
    // (3.944e-3 vs 4.754e-3), but it can reorder a near-tie: the reference's
    // 5th and 6th logits are 5.6e-4 apart (normalised) and this path swaps
    // them. MNET_MAT_MAT=0 restores the bit-identical scalar path, =2 tiles
    // gate/up only.
    const bool tiled = tiled_mat_mat_enabled(s, count);
    const char* mat_mat = std::getenv("MNET_MAT_MAT");
    const bool tiled_down = tiled && (mat_mat == nullptr || std::atoi(mat_mat) != 2);
    id<MTLComputePipelineState> tiled_pipeline = tiled_mat_mat_pipeline(s);
    if (tiled && !skip_matvec) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(first.gate.row_bytes);
        const std::uint32_t columns32 = static_cast<std::uint32_t>(columns);
        const std::uint32_t rows32 = static_cast<std::uint32_t>(hidden);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(columns);
        const std::uint32_t out_stride32 = static_cast<std::uint32_t>(hidden);
        const bool llama_ok = llama_mat_mat_ok(s, hidden, columns, count, 1.0F, false);
        for (std::size_t pass = 0; pass < 2U; ++pass) {
            const float scale = pass == 0U ? first.gate.scale : first.up.scale;
            if (llama_ok) {
                encode_llama_mat_mat(encoder, s, (pass == 0U ? gate_weights.buffer : up_weights.buffer),
                                     static_cast<NSUInteger>(pass == 0U ? gate_weights.offset : up_weights.offset),
                                     b.wide, (pass == 0U ? b.gate : b.up), hidden, columns,
                                     first.gate.row_bytes, hidden, count);
                continue;
            }
            [encoder setComputePipelineState:tiled_pipeline];
            [encoder setBuffer:(pass == 0U ? gate_weights.buffer : up_weights.buffer)
                         offset:static_cast<NSUInteger>(pass == 0U ? gate_weights.offset : up_weights.offset)
                        atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:(pass == 0U ? b.gate : b.up) offset:0 atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
            [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (hidden + 63U) / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
    // Every gate, then every up matvec.
    for (std::size_t pass = 0; !skip_matvec && tiled == false && pass < 2U; ++pass) {
        for (std::size_t t = 0; t < count; ++t) {
            [encoder setComputePipelineState:s.fast_f32_pipeline];
            // Aliased weights live at an offset inside the mapping buffer.
            [encoder setBuffer:(pass == 0U ? gate_weights.buffer : up_weights.buffer)
                         offset:static_cast<NSUInteger>(pass == 0U ? gate_weights.offset : up_weights.offset)
                        atIndex:0];
            [encoder setBuffer:b.wide offset:static_cast<NSUInteger>(t * columns * sizeof(float)) atIndex:1];
            [encoder setBuffer:(pass == 0U ? b.gate : b.up)
                         offset:static_cast<NSUInteger>(t * hidden * sizeof(float))
                        atIndex:3];
            const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(first.gate.row_bytes);
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(columns / 64U);
            const float scale = pass == 0U ? first.gate.scale : first.up.scale;
            const std::uint32_t lanes = matvec_lanes();
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
        }
    }
    // SwiGLU and the Hadamard/requantisation are element-wise over the whole
    // per-token slab, so one dispatch each covers every token.
    {
        const std::uint32_t slab = static_cast<std::uint32_t>(hidden * count);
        if (!skip_swiglu) {
        [encoder setComputePipelineState:s.swiglu_pipeline];
        [encoder setBuffer:b.gate offset:0 atIndex:0];
        [encoder setBuffer:b.up offset:0 atIndex:1];
        [encoder setBuffer:b.act offset:0 atIndex:2];
        [encoder setBytes:&slab length:sizeof(slab) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        }
        if (!skip_hadamard) {
        [encoder setComputePipelineState:s.hadamard_multi_pipeline];
        [encoder setBuffer:b.act offset:0 atIndex:0];
        [encoder setBuffer:signs_buffer offset:0 atIndex:1];
        [encoder setBuffer:b.act_q8 offset:0 atIndex:2];
        [encoder setBuffer:b.act_scales offset:0 atIndex:3];
        [encoder setBytes:&block_size length:sizeof(block_size) atIndex:4];
        const std::uint32_t sign_blocks = static_cast<std::uint32_t>(hidden / block_size);
        [encoder setBytes:&sign_blocks length:sizeof(sign_blocks) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((hidden / block_size) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
        }
    }
    // The down matvec's activation is widened for every token in one dispatch,
    // then projected in one dispatch.
    if (!skip_dequant) {
        [encoder setComputePipelineState:s.dequant_pipeline];
        [encoder setBuffer:b.act_q8 offset:0 atIndex:0];
        [encoder setBuffer:b.act_scales offset:0 atIndex:1];
        [encoder setBuffer:b.wide offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (tiled_down && llama_mat_mat_ok(s, columns, hidden, count, first.down.scale, true)) {
        encode_llama_mat_mat(encoder, s, down_weights.buffer, static_cast<NSUInteger>(down_weights.offset), b.wide,
                             b.out, columns, hidden, first.down.row_bytes, columns, count);
    } else if (tiled_down) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(first.down.row_bytes);
        const std::uint32_t columns32 = static_cast<std::uint32_t>(hidden);
        const float scale = first.down.scale;
        const std::uint32_t rows32 = static_cast<std::uint32_t>(columns);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        const std::uint32_t out_stride32 = static_cast<std::uint32_t>(columns);
        [encoder setComputePipelineState:tiled_pipeline];
        [encoder setBuffer:down_weights.buffer offset:static_cast<NSUInteger>(down_weights.offset) atIndex:0];
        [encoder setBuffer:b.wide offset:0 atIndex:1];
        [encoder setBuffer:b.out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
        [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (columns + 63U) / 64U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (!skip_matvec) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(first.down.row_bytes);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
        const float scale = first.down.scale;
        const std::uint32_t lanes = matvec_lanes();
        const std::uint32_t rows32 = static_cast<std::uint32_t>(columns);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        [encoder setComputePipelineState:s.fast_f32_multi_pipeline];
        [encoder setBuffer:down_weights.buffer offset:static_cast<NSUInteger>(down_weights.offset) atIndex:0];
        [encoder setBuffer:b.wide offset:0 atIndex:1];
        [encoder setBuffer:b.out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(columns * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    record_gpu_block(0, command);
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    const auto* results = static_cast<const float*>(b.out.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(entries[t].output, results + t * columns, static_cast<std::size_t>(columns) * sizeof(float));
    }
    if (const char* dir = std::getenv("MNET_BATCH_DUMP"); dir != nullptr) {
        static int calls = 0;
        if (calls++ == 0) {
            const auto write = [&](const char* name, const void* data, std::size_t bytes) {
                if (std::FILE* file = std::fopen((std::string(dir) + "/" + name).c_str(), "wb")) {
                    std::fwrite(data, 1, bytes, file);
                    std::fclose(file);
                }
            };
            write("batch-gate.f32", b.gate.contents, static_cast<std::size_t>(hidden) * count * sizeof(float));
            write("batch-up.f32", b.up.contents, static_cast<std::size_t>(hidden) * count * sizeof(float));
            write("batch-out.f32", b.out.contents, static_cast<std::size_t>(columns) * count * sizeof(float));
        }
    }
    if (const char* dir = std::getenv("MNET_FFN_DEBUG_DIR"); dir != nullptr) {
        const std::string prefix(dir);
        const auto write = [&](const char* name, const void* data, std::size_t bytes) {
            if (std::FILE* file = std::fopen((prefix + "/" + name).c_str(), "wb")) {
                std::fwrite(data, 1, bytes, file);
                std::fclose(file);
            }
        };
        write("gpu-gate.f32", b.gate.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-up.f32", b.up.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-act.f32", b.act.contents, static_cast<std::size_t>(hidden) * sizeof(float));
        write("gpu-actq8.bin", b.act_q8.contents, static_cast<std::size_t>(hidden));
        write("gpu-actscale.f32", b.act_scales.contents, static_cast<std::size_t>(hidden / 32U) * sizeof(float));
        write("gpu-out.f32", b.out.contents, static_cast<std::size_t>(columns) * sizeof(float));
        write("gpu-wide.f32", b.wide.contents, static_cast<std::size_t>(columns) * sizeof(float));
    }
    return true;
}

// Gated-delta block for several tokens of one layer. The token-independent
// part (qkv/gate matvec, head gates, convolution) is encoded for all tokens
// first so those dispatches overlap; the scan and what reads it stay in token
// order because the recurrent state carries across tokens. The convolution
// window is kept as a straight run of (kernel-1 + count) rows instead of a
// ring, so each token's four-tap window is just its own row and the three
// before it.
bool ssm_batch_block(const SsmQ2_0Job& job, const SsmBatchEntry* entries, std::size_t count) {
    MetalState& s = state();
    if (!s.ready || s.ssm_rows_pipeline == nil || s.ssm_act_pipeline == nil || s.ssm_conv_norm_pipeline == nil ||
        s.ssm_scan_pipeline == nil || s.ssm_gate_norm_pipeline == nil || s.hadamard_quantize_pipeline == nil ||
        s.fast_f32_pipeline == nil || s.fast_f32_multi_pipeline == nil || s.dequant_pipeline == nil ||
        entries == nullptr || count == 0U) {
        return false;
    }
    if (job.qkv.type != 42U || job.gate.type != 42U || job.out.type != 42U) {
        return false;
    }
    const std::uint64_t profile_start = now_ns();
    const std::uint64_t head_dim = job.head_dim;
    const std::uint64_t key_heads = job.key_heads;
    const std::uint64_t value_heads = job.value_heads;
    const std::uint64_t conv_dim = job.conv_dim;
    const std::uint64_t hidden = value_heads * head_dim;
    if (head_dim == 0U || head_dim > 256U || (head_dim % 32U) != 0U || key_heads == 0U || value_heads == 0U ||
        (value_heads % key_heads) != 0U || job.conv_kernel < 2U || job.conv_kernel > 8U ||
        conv_dim != 2U * key_heads * head_dim + hidden || (job.columns % 64U) != 0U ||
        job.block_size < 64U || job.block_size > 1024U || (job.block_size & (job.block_size - 1U)) != 0U ||
        (hidden % job.block_size) != 0U || job.qkv.rows != conv_dim || job.gate.rows != hidden ||
        job.out.rows != job.columns) {
        return false;
    }

    const WeightBinding qkv_weights = weight_buffer_for(s, job.qkv, job.qkv.rows, job.qkv.row_bytes, true);
    const WeightBinding gate_weights = weight_buffer_for(s, job.gate, job.gate.rows, job.gate.row_bytes, true);
    const WeightBinding out_weights = weight_buffer_for(s, job.out, job.out.rows, job.out.row_bytes, true);
    if (qkv_weights.buffer == nil || gate_weights.buffer == nil || out_weights.buffer == nil) {
        return false;
    }

    // Resident per-layer state, seeded from the caller's vectors on first use.
    const std::uint64_t window_rows = job.conv_kernel - 1U + count;
    const std::uint64_t window_floats = window_rows * conv_dim;
    const std::uint64_t side_floats = 2U * value_heads * job.columns + 2U * value_heads +
                                      conv_dim * job.conv_kernel + head_dim;
    const std::uint64_t state_floats = value_heads * head_dim * head_dim;
    SsmLayer& layer = s.ssm_layers[cache_key(job.identity, job.qkv.data)];
    if (layer.window == nil) {
        layer.window = [s.device newBufferWithLength:static_cast<NSUInteger>(window_floats * sizeof(float))
                                             options:MTLResourceStorageModeShared];
        layer.state = [s.device newBufferWithLength:static_cast<NSUInteger>(state_floats * sizeof(float))
                                            options:MTLResourceStorageModeShared];
        layer.side = [s.device newBufferWithLength:static_cast<NSUInteger>(side_floats * sizeof(float))
                                           options:MTLResourceStorageModeShared];
        if (layer.window == nil || layer.state == nil || layer.side == nil) {
            layer = SsmLayer{};
            return false;
        }
        auto* side = static_cast<float*>(layer.side.contents);
        const std::size_t alpha_offset = static_cast<std::size_t>(value_heads * job.columns);
        const std::size_t dt_offset = 2U * alpha_offset;
        const std::size_t a_offset = dt_offset + static_cast<std::size_t>(value_heads);
        const std::size_t conv_w_offset = a_offset + static_cast<std::size_t>(value_heads);
        const std::size_t norm_offset = conv_w_offset + static_cast<std::size_t>(conv_dim * job.conv_kernel);
        std::memcpy(side, job.beta_w, static_cast<std::size_t>(value_heads * job.columns) * sizeof(float));
        std::memcpy(side + alpha_offset, job.alpha_w,
                    static_cast<std::size_t>(value_heads * job.columns) * sizeof(float));
        std::memcpy(side + dt_offset, job.dt, static_cast<std::size_t>(value_heads) * sizeof(float));
        std::memcpy(side + a_offset, job.ssm_a, static_cast<std::size_t>(value_heads) * sizeof(float));
        std::memcpy(side + conv_w_offset, job.conv_w,
                    static_cast<std::size_t>(conv_dim * job.conv_kernel) * sizeof(float));
        std::memcpy(side + norm_offset, job.ssm_norm, static_cast<std::size_t>(head_dim) * sizeof(float));
        auto* window = static_cast<float*>(layer.window.contents);
        std::memcpy(window, job.conv_state,
                    static_cast<std::size_t>((job.conv_kernel - 1U) * conv_dim) * sizeof(float));
        auto* resident = static_cast<float*>(layer.state.contents);
        for (std::uint64_t h = 0; h < value_heads; ++h) {
            const float* source = job.ssm_state + h * head_dim * head_dim;
            float* target = resident + h * head_dim * head_dim;
            for (std::uint64_t j = 0; j < head_dim; ++j) {
                for (std::uint64_t i = 0; i < head_dim; ++i) {
                    target[i * head_dim + j] = source[j * head_dim + i];
                }
            }
        }
    } else if (layer.window.length < static_cast<NSUInteger>(window_floats * sizeof(float))) {
        // A later batch with more tokens needs a longer window.
        id<MTLBuffer> grown = [s.device newBufferWithLength:static_cast<NSUInteger>(window_floats * sizeof(float))
                                                    options:MTLResourceStorageModeShared];
        if (grown == nil) {
            return false;
        }
        std::memcpy(grown.contents, layer.window.contents,
                    static_cast<std::size_t>(job.conv_kernel - 1U) * conv_dim * sizeof(float));
        layer.window = grown;
    }
    auto* side = static_cast<const float*>(layer.side.contents);
    const std::size_t alpha_offset = static_cast<std::size_t>(value_heads * job.columns);
    const std::size_t dt_offset = 2U * alpha_offset;
    const std::size_t a_offset = dt_offset + static_cast<std::size_t>(value_heads);
    const std::size_t conv_w_offset = a_offset + static_cast<std::size_t>(value_heads);
    const std::size_t norm_offset = conv_w_offset + static_cast<std::size_t>(conv_dim * job.conv_kernel);
    id<MTLBuffer> signs_buffer =
        ensure_buffer(s.device, s.signs_buffer, s.signs_capacity, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    if (signs_buffer == nil) {
        return false;
    }
    std::memcpy(signs_buffer.contents, job.signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));

    SsmBatchBuffers& b = s.ssm_batch;
    const std::size_t n = count;
    const std::uint64_t widest = job.columns > hidden ? job.columns : hidden;
    b.q8 = ensure_buffer(s.device, b.q8, b.q8_capacity, n * static_cast<std::size_t>(job.columns));
    b.scales = ensure_buffer(s.device, b.scales, b.scales_capacity,
                             n * static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
    b.wide = ensure_buffer(s.device, b.wide, b.wide_capacity, n * static_cast<std::size_t>(widest) * sizeof(float));
    b.activation = ensure_buffer(s.device, b.activation, b.activation_capacity,
                                 n * static_cast<std::size_t>(job.columns) * sizeof(float));
    b.raw = ensure_buffer(s.device, b.raw, b.raw_capacity, n * static_cast<std::size_t>(2U * value_heads) * sizeof(float));
    b.gates =
        ensure_buffer(s.device, b.gates, b.gates_capacity, n * static_cast<std::size_t>(2U * value_heads) * sizeof(float));
    b.conv = ensure_buffer(s.device, b.conv, b.conv_capacity, n * static_cast<std::size_t>(conv_dim) * sizeof(float));
    b.z = ensure_buffer(s.device, b.z, b.z_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.delta = ensure_buffer(s.device, b.delta, b.delta_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.grouped = ensure_buffer(s.device, b.grouped, b.grouped_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.act_q8 = ensure_buffer(s.device, b.act_q8, b.act_q8_capacity, n * static_cast<std::size_t>(hidden));
    b.act_scales = ensure_buffer(s.device, b.act_scales, b.act_scales_capacity,
                                 n * static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    b.out = ensure_buffer(s.device, b.out, b.out_capacity, n * static_cast<std::size_t>(job.columns) * sizeof(float));
    if (b.q8 == nil || b.scales == nil || b.wide == nil || b.activation == nil || b.raw == nil || b.gates == nil ||
        b.conv == nil || b.z == nil || b.delta == nil || b.grouped == nil || b.act_q8 == nil ||
        b.act_scales == nil || b.out == nil) {
        return false;
    }
    auto* activation_destination = static_cast<float*>(b.activation.contents);
    // The prepped inputs go up as fp32 and are quantised on the device, the same
    // way the FFN block does it (log 167): b.grouped is free until the norm
    // writes it, and count * hidden floats covers count * columns.
    auto* staging = static_cast<float*>(b.grouped.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(staging + t * job.columns, entries[t].input,
                    static_cast<std::size_t>(job.columns) * sizeof(float));
        std::memcpy(activation_destination + t * job.columns, entries[t].raw_input,
                    static_cast<std::size_t>(job.columns) * sizeof(float));
    }

    const std::uint32_t head_dim32 = static_cast<std::uint32_t>(head_dim);
    const std::uint32_t key_heads32 = static_cast<std::uint32_t>(key_heads);
    const std::uint32_t value_heads32 = static_cast<std::uint32_t>(value_heads);
    const std::uint32_t conv_dim32 = static_cast<std::uint32_t>(conv_dim);
    const std::uint32_t kernel32 = job.conv_kernel;
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    const std::uint32_t columns32 = static_cast<std::uint32_t>(job.columns);
    const std::uint32_t heads_per_key32 = static_cast<std::uint32_t>(value_heads / key_heads);
    const std::uint32_t norm_blocks32 = static_cast<std::uint32_t>(2U * key_heads);
    const std::uint32_t block32 = job.block_size;
    const std::uint32_t wrap_linear = 0U;
    const float eps = job.eps;
    const float inv_dim = 1.0F / std::sqrt(static_cast<float>(head_dim));

    // MNET_SSM_SKIP=<phase> drops one phase for timing; results are garbage.
    static const std::string skip =
        std::getenv("MNET_SSM_SKIP") != nullptr ? std::string(std::getenv("MNET_SSM_SKIP")) : std::string();
    const bool skip_all = skip == "all";
    const bool skip_matvec = skip_all || skip == "matvec";
    const bool skip_gate = skip_all || skip == "gate";
    const bool skip_conv = skip_all || skip == "conv";
    const bool skip_scan = skip_all || skip == "scan";
    const bool skip_norm = skip_all || skip == "norm";
    // The norm phase is two dispatches with very different costs; these split it.
    const bool skip_gate_norm = skip_norm || skip == "gatenorm";
    const bool skip_hadamard = skip_norm || skip == "hadamard";

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (s.q8_quantize_pipeline != nil) {
        const std::uint32_t q8_blocks = static_cast<std::uint32_t>(job.columns / 32U);
        const std::uint32_t q8_tokens = static_cast<std::uint32_t>(count);
        [encoder setComputePipelineState:s.q8_quantize_pipeline];
        [encoder setBuffer:b.grouped offset:0 atIndex:0];
        [encoder setBuffer:b.q8 offset:0 atIndex:1];
        [encoder setBuffer:b.scales offset:0 atIndex:2];
        [encoder setBytes:&q8_blocks length:sizeof(q8_blocks) atIndex:3];
        [encoder setBytes:&q8_tokens length:sizeof(q8_tokens) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake((job.columns / 32U) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    // Widen every token's activation once, then run all qkv and all gate
    // matvecs.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(job.columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // One dispatch per matvec for all tokens: the per-token activation slab is
    // contiguous, so every threadgroup only needs its (token, row) pair.
    if (!skip_matvec && tiled_mat_mat_enabled(s, count)) {
        const std::uint32_t columns32 = static_cast<std::uint32_t>(job.columns);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(job.columns);
        const id<MTLComputePipelineState> pipeline = tiled_mat_mat_pipeline(s);
        for (std::size_t pass = 0; pass < 2U; ++pass) {
            const std::uint64_t rows = pass == 0U ? conv_dim : hidden;
            const std::uint32_t row_bytes32 =
                static_cast<std::uint32_t>(pass == 0U ? job.qkv.row_bytes : job.gate.row_bytes);
            const float scale = pass == 0U ? job.qkv.scale : job.gate.scale;
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            const std::uint32_t out_stride32 = static_cast<std::uint32_t>(rows);
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:(pass == 0U ? qkv_weights.buffer : gate_weights.buffer)
                         offset:static_cast<NSUInteger>(pass == 0U ? qkv_weights.offset : gate_weights.offset)
                        atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:(pass == 0U ? layer.window : b.z)
                         offset:static_cast<NSUInteger>(pass == 0U ? (job.conv_kernel - 1U) * conv_dim * sizeof(float)
                                                                   : 0U)
                        atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
            [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (rows + 63U) / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
    if (!skip_matvec && tiled_mat_mat_enabled(s, count) == false) {
        for (std::size_t pass = 0; pass < 2U; ++pass) {
            const std::uint64_t rows = pass == 0U ? conv_dim : hidden;
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(job.columns / 64U);
            const std::uint32_t row_bytes32 =
                static_cast<std::uint32_t>(pass == 0U ? job.qkv.row_bytes : job.gate.row_bytes);
            const float scale = pass == 0U ? job.qkv.scale : job.gate.scale;
            const std::uint32_t lanes = matvec_lanes();
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            const std::uint32_t x_stride32 = static_cast<std::uint32_t>(job.columns);
            [encoder setComputePipelineState:s.fast_f32_multi_pipeline];
            [encoder setBuffer:(pass == 0U ? qkv_weights.buffer : gate_weights.buffer)
                         offset:static_cast<NSUInteger>(pass == 0U ? qkv_weights.offset : gate_weights.offset)
                        atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:(pass == 0U ? layer.window : b.z)
                         offset:static_cast<NSUInteger>(pass == 0U ? (job.conv_kernel - 1U) * conv_dim * sizeof(float)
                                                                   : 0U)
                        atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(rows * count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
        }
    }
    // Head gates for every token in one dispatch: the kernel already splits
    // tgid into (token, head), so the per-token loop only ever kept 48
    // threadgroups busy. The gate/alpha slab is interleaved per token, hence the
    // stride parameter.
    if (!skip_gate) {
        const std::uint32_t stride32 = static_cast<std::uint32_t>(2U * value_heads);
        [encoder setComputePipelineState:s.head_gates_pipeline];
        [encoder setBuffer:b.activation offset:0 atIndex:0];
        [encoder setBuffer:layer.side offset:0 atIndex:1];
        [encoder setBuffer:layer.side offset:dt_offset * sizeof(float) atIndex:2];
        [encoder setBuffer:layer.side offset:a_offset * sizeof(float) atIndex:3];
        [encoder setBuffer:b.gates offset:0 atIndex:4];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(value_heads * sizeof(float)) atIndex:5];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:6];
        [encoder setBytes:&value_heads32 length:sizeof(value_heads32) atIndex:7];
        [encoder setBytes:&stride32 length:sizeof(stride32) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(value_heads * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    // Head gates and the convolution for every token.
    if (!skip_conv && s.ssm_conv_norm_batch_pipeline != nil) {
        const std::uint32_t ring32 = 0U;
        const std::uint32_t blocks_per_token = static_cast<std::uint32_t>(conv_dim / head_dim);
        [encoder setComputePipelineState:s.ssm_conv_norm_batch_pipeline];
        [encoder setBuffer:layer.window offset:0 atIndex:0];
        [encoder setBuffer:layer.side offset:conv_w_offset * sizeof(float) atIndex:1];
        [encoder setBuffer:b.conv offset:0 atIndex:2];
        [encoder setBytes:&conv_dim32 length:sizeof(conv_dim32) atIndex:3];
        [encoder setBytes:&kernel32 length:sizeof(kernel32) atIndex:4];
        [encoder setBytes:&ring32 length:sizeof(ring32) atIndex:5];
        [encoder setBytes:&norm_blocks32 length:sizeof(norm_blocks32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
        [encoder setBytes:&wrap_linear length:sizeof(wrap_linear) atIndex:9];
        [encoder setBytes:&blocks_per_token length:sizeof(blocks_per_token) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks_per_token * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    } else if (!skip_conv) {
        for (std::size_t t = 0; t < count; ++t) {
            const std::uint32_t ring32 = static_cast<std::uint32_t>(t);
            [encoder setComputePipelineState:s.ssm_conv_norm_pipeline];
            [encoder setBuffer:layer.window offset:0 atIndex:0];
            [encoder setBuffer:layer.side offset:conv_w_offset * sizeof(float) atIndex:1];
            [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(t * conv_dim * sizeof(float)) atIndex:2];
            [encoder setBytes:&conv_dim32 length:sizeof(conv_dim32) atIndex:3];
            [encoder setBytes:&kernel32 length:sizeof(kernel32) atIndex:4];
            [encoder setBytes:&ring32 length:sizeof(ring32) atIndex:5];
            [encoder setBytes:&norm_blocks32 length:sizeof(norm_blocks32) atIndex:6];
            [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
            [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
            [encoder setBytes:&wrap_linear length:sizeof(wrap_linear) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(conv_dim / head_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        }
    }
    // Token order from here: the scan carries state and the rest reads it.
    const std::uint32_t q_offset = 0U;
    const std::uint32_t k_offset = static_cast<std::uint32_t>(key_heads * head_dim);
    const std::uint32_t v_offset = static_cast<std::uint32_t>(2U * key_heads * head_dim);
    const bool tiled_scan = head_dim == 128U && s.ssm_scan_tiled_pipeline != nil;
    if (!skip_scan) {
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t qkv_stride32 = static_cast<std::uint32_t>(conv_dim);
        const std::uint32_t gate_stride32 = static_cast<std::uint32_t>(2U * value_heads);
        const std::uint32_t delta_stride32 = static_cast<std::uint32_t>(hidden);
        [encoder setComputePipelineState:tiled_scan ? s.ssm_scan_tiled_pipeline : s.ssm_scan_batch_pipeline];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(q_offset * sizeof(float)) atIndex:0];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(k_offset * sizeof(float)) atIndex:1];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(v_offset * sizeof(float)) atIndex:2];
        [encoder setBuffer:b.gates offset:0 atIndex:3];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(value_heads * sizeof(float)) atIndex:4];
        [encoder setBuffer:layer.state offset:0 atIndex:5];
        [encoder setBuffer:b.delta offset:0 atIndex:6];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:7];
        [encoder setBytes:&key_heads32 length:sizeof(key_heads32) atIndex:8];
        [encoder setBytes:&inv_dim length:sizeof(inv_dim) atIndex:9];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:10];
        [encoder setBytes:&qkv_stride32 length:sizeof(qkv_stride32) atIndex:11];
        [encoder setBytes:&gate_stride32 length:sizeof(gate_stride32) atIndex:12];
        [encoder setBytes:&delta_stride32 length:sizeof(delta_stride32) atIndex:13];
        if (tiled_scan) {
            const std::uint32_t blocks32 = 4U;
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:14];
            [encoder dispatchThreadgroups:MTLSizeMake(value_heads * blocks32, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        }
    }
    if (!skip_gate_norm) {
        [encoder setComputePipelineState:s.gate_norm_multi_pipeline];
        [encoder setBuffer:b.delta offset:0 atIndex:0];
        [encoder setBuffer:b.z offset:0 atIndex:1];
        [encoder setBuffer:layer.side offset:norm_offset * sizeof(float) atIndex:2];
        [encoder setBuffer:b.grouped offset:0 atIndex:3];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:4];
        [encoder setBytes:&key_heads32 length:sizeof(key_heads32) atIndex:5];
        [encoder setBytes:&heads_per_key32 length:sizeof(heads_per_key32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&value_heads32 length:sizeof(value_heads32) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(value_heads * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    }
    if (!skip_hadamard) {
        [encoder setComputePipelineState:s.hadamard_multi_pipeline];
        [encoder setBuffer:b.grouped offset:0 atIndex:0];
        [encoder setBuffer:signs_buffer offset:0 atIndex:1];
        [encoder setBuffer:b.act_q8 offset:0 atIndex:2];
        [encoder setBuffer:b.act_scales offset:0 atIndex:3];
        [encoder setBytes:&block32 length:sizeof(block32) atIndex:4];
        const std::uint32_t sign_blocks = static_cast<std::uint32_t>(hidden / job.block_size);
        [encoder setBytes:&sign_blocks length:sizeof(sign_blocks) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((hidden / job.block_size) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(job.block_size, 1, 1)];
    }
    // The output projections are independent across tokens; running them after
    // the whole norm/requantisation chain lets them overlap instead of each one
    // waiting behind the next token's scan.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.act_q8 offset:0 atIndex:0];
    [encoder setBuffer:b.act_scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!skip_norm && tiled_mat_mat_enabled(s, count)) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t columns32 = static_cast<std::uint32_t>(hidden);
        const float scale = job.out.scale;
        const std::uint32_t rows32 = static_cast<std::uint32_t>(job.out.rows);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        const std::uint32_t out_stride32 = static_cast<std::uint32_t>(job.out.rows);
        [encoder setComputePipelineState:tiled_mat_mat_pipeline(s)];
        [encoder setBuffer:out_weights.buffer offset:static_cast<NSUInteger>(out_weights.offset) atIndex:0];
        [encoder setBuffer:b.wide offset:0 atIndex:1];
        [encoder setBuffer:b.out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
        [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (job.out.rows + 63U) / 64U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (!skip_norm && tiled_mat_mat_enabled(s, count) == false) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t columns32 = static_cast<std::uint32_t>(hidden);
        const float scale = job.out.scale;
        const std::uint32_t rows32 = static_cast<std::uint32_t>(job.out.rows);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        const std::uint32_t out_stride32 = static_cast<std::uint32_t>(job.out.rows);
        if (tiled_mat_mat_enabled(s, count)) {
            [encoder setComputePipelineState:tiled_mat_mat_pipeline(s)];
            [encoder setBuffer:out_weights.buffer offset:static_cast<NSUInteger>(out_weights.offset) atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:b.out offset:0 atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
            [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (job.out.rows + 63U) / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
            const std::uint32_t lanes = matvec_lanes();
            [encoder setComputePipelineState:s.fast_f32_multi_pipeline];
            [encoder setBuffer:out_weights.buffer offset:static_cast<NSUInteger>(out_weights.offset) atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:b.out offset:0 atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(job.out.rows * count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
        }
    }
    [encoder endEncoding];
    const std::uint64_t profile_commit = now_ns();
    // Carry the last tokens' rows to the front for the next batch.
    {
        const std::uint64_t tail_rows = job.conv_kernel - 1U;
        const std::uint64_t source_offset = static_cast<std::uint64_t>(count) * conv_dim * sizeof(float);
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit copyFromBuffer:layer.window
                sourceOffset:static_cast<NSUInteger>(source_offset)
                    toBuffer:layer.window
           destinationOffset:0
                        size:static_cast<NSUInteger>(tail_rows * conv_dim * sizeof(float))];
        [blit endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    record_gpu_block(1, command);
    const std::uint64_t profile_wait = now_ns();
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    const auto* results = static_cast<const float*>(b.out.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(entries[t].output, results + t * job.columns, static_cast<std::size_t>(job.columns) * sizeof(float));
    }
    {
        SsmBatchProfile& profile = ssm_batch_profile();
        // Host preparation and encoding together, then the GPU wait.
        profile.encode_ns += profile_commit - profile_start;
        profile.wait_ns += profile_wait - profile_commit;
        profile.tail_ns += now_ns() - profile_wait;
        ++profile.calls;
        profile.tokens += count;
    }
    return true;
}

// Attention block for several tokens of one layer: every q/k/v matvec, every
// head norm and RoPE, every f16 KV append and every output projection are
// encoded for all tokens; each token's attention then runs over the cache up to
// its own position, so the token order is preserved where it matters.
bool attention_batch_block(const AttentionQ2_0Job& job, const AttentionBatchEntry* entries, std::size_t count) {
    MetalState& s = state();
    if (!s.ready || s.attn_prep_pipeline == nil || s.attn_flash_pipeline == nil || s.attn_gate_pipeline == nil ||
        s.attn_store_pipeline == nil || s.hadamard_quantize_pipeline == nil || s.fast_f32_pipeline == nil ||
        s.dequant_pipeline == nil || entries == nullptr || count == 0U) {
        return false;
    }
    if (job.q.type != 42U || job.k.type != 42U || job.v.type != 42U || job.out.type != 42U) {
        return false;
    }
    const std::uint64_t head_dim = job.head_dim;
    const std::uint64_t heads = job.n_head;
    const std::uint64_t heads_kv = job.n_head_kv;
    const std::uint64_t hidden = heads * head_dim;
    if (head_dim == 0U || head_dim > 256U || (head_dim % 32U) != 0U || heads == 0U || heads_kv == 0U ||
        (heads % heads_kv) != 0U || job.columns == 0U || (job.columns % 64U) != 0U ||
        job.rope_dims == 0U || (job.rope_dims % 2U) != 0U || job.rope_dims > head_dim || job.block_size < 64U ||
        job.block_size > 1024U || (job.block_size & (job.block_size - 1U)) != 0U || (hidden % job.block_size) != 0U ||
        job.out.rows != job.columns) {
        return false;
    }
    const std::uint64_t kv_head_elems = heads_kv * head_dim;
    const std::uint64_t tokens_total = static_cast<std::uint64_t>(job.cache_tokens) + count;
    const std::uint64_t kv_plane_bytes = tokens_total * kv_head_elems * sizeof(float);

    const WeightBinding q_weights = weight_buffer_for(s, job.q, job.q.rows, job.q.row_bytes, true);
    const WeightBinding k_weights = weight_buffer_for(s, job.k, job.k.rows, job.k.row_bytes, true);
    const WeightBinding v_weights = weight_buffer_for(s, job.v, job.v.rows, job.v.row_bytes, true);
    const WeightBinding out_weights = weight_buffer_for(s, job.out, job.out.rows, job.out.row_bytes, true);
    if (q_weights.buffer == nil || k_weights.buffer == nil || v_weights.buffer == nil || out_weights.buffer == nil) {
        return false;
    }
    AttentionBuffers& shared = s.attn;
    id<MTLBuffer> kv = ensure_buffer(s.device, shared.kv, shared.kv_capacity, static_cast<std::size_t>(2U * kv_plane_bytes));
    // Head norm weights live next to each other in the shared table buffer.
    id<MTLBuffer> norm_tables = ensure_buffer(s.device, shared.tables, shared.tables_capacity,
                                              static_cast<std::size_t>(2U * head_dim) * sizeof(float));
    if (norm_tables == nil) {
        return false;
    }
    {
        auto* table = static_cast<float*>(norm_tables.contents);
        std::memcpy(table, job.q_norm, static_cast<std::size_t>(head_dim) * sizeof(float));
        std::memcpy(table + head_dim, job.k_norm, static_cast<std::size_t>(head_dim) * sizeof(float));
    }
    const std::size_t q_norm_offset = 0U;
    const std::size_t k_norm_offset = static_cast<std::size_t>(head_dim);
    id<MTLBuffer> signs_buffer =
        ensure_buffer(s.device, shared.signs, shared.signs_capacity, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    if (kv == nil || signs_buffer == nil) {
        return false;
    }
    std::memcpy(signs_buffer.contents, job.signs, static_cast<std::size_t>(hidden) * sizeof(std::int32_t));

    AttentionBatchBuffers& b = s.attn_batch;
    const std::size_t n = count;
    const std::size_t half_rope = job.rope_dims / 2U;
    b.q8 = ensure_buffer(s.device, b.q8, b.q8_capacity, n * static_cast<std::size_t>(job.columns));
    b.scales = ensure_buffer(s.device, b.scales, b.scales_capacity,
                             n * static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
    // The widened activation is reused for the output projection, whose input is
    // hidden wide, so the slab covers the larger of the two.
    const std::uint64_t attn_widest = job.columns > hidden ? job.columns : hidden;
    b.wide = ensure_buffer(s.device, b.wide, b.wide_capacity,
                           n * static_cast<std::size_t>(attn_widest) * sizeof(float));
    b.qfull = ensure_buffer(s.device, b.qfull, b.qfull_capacity, n * static_cast<std::size_t>(2U * hidden) * sizeof(float));
    b.q = ensure_buffer(s.device, b.q, b.q_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.k_new = ensure_buffer(s.device, b.k_new, b.k_new_capacity, n * static_cast<std::size_t>(kv_head_elems) * sizeof(float));
    b.v_new = ensure_buffer(s.device, b.v_new, b.v_new_capacity, n * static_cast<std::size_t>(kv_head_elems) * sizeof(float));
    b.gates = ensure_buffer(s.device, b.gates, b.gates_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.attn = ensure_buffer(s.device, b.attn, b.attn_capacity, n * static_cast<std::size_t>(hidden) * sizeof(float));
    b.act_q8 = ensure_buffer(s.device, b.act_q8, b.act_q8_capacity, n * static_cast<std::size_t>(hidden));
    b.act_scales = ensure_buffer(s.device, b.act_scales, b.act_scales_capacity,
                                 n * static_cast<std::size_t>(hidden / 32U) * sizeof(float));
    b.out = ensure_buffer(s.device, b.out, b.out_capacity, n * static_cast<std::size_t>(job.columns) * sizeof(float));
    b.tables = ensure_buffer(s.device, b.tables, b.tables_capacity,
                             n * static_cast<std::size_t>(2U * half_rope) * sizeof(float));
    if (b.q8 == nil || b.scales == nil || b.wide == nil || b.qfull == nil || b.q == nil || b.k_new == nil ||
        b.v_new == nil || b.gates == nil || b.attn == nil || b.act_q8 == nil || b.act_scales == nil || b.out == nil ||
        b.tables == nil) {
        return false;
    }
    auto* tables = static_cast<float*>(b.tables.contents);
    // Same device quantisation as the FFN and SSM blocks; b.wide is free until
    // the dequantise step writes it.
    auto* staging = static_cast<float*>(b.wide.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(staging + t * job.columns, entries[t].input,
                    static_cast<std::size_t>(job.columns) * sizeof(float));
        std::memcpy(tables + t * 2U * half_rope, entries[t].rope_cos, half_rope * sizeof(float));
        std::memcpy(tables + t * 2U * half_rope + half_rope, entries[t].rope_sin, half_rope * sizeof(float));
    }
    auto* kv_bytes = static_cast<std::uint8_t*>(kv.contents);
    const std::uint64_t cached_bytes = static_cast<std::uint64_t>(job.cache_tokens) * kv_head_elems * sizeof(float);
    if (cached_bytes != 0U) {
        std::memcpy(kv_bytes, job.k_cache, static_cast<std::size_t>(cached_bytes));
        std::memcpy(kv_bytes + kv_plane_bytes, job.v_cache, static_cast<std::size_t>(cached_bytes));
    }

    const std::uint32_t head_dim32 = static_cast<std::uint32_t>(head_dim);
    const std::uint32_t heads_kv32 = static_cast<std::uint32_t>(heads_kv);
    const std::uint32_t group32 = static_cast<std::uint32_t>(heads / heads_kv);
    const std::uint32_t hidden32 = static_cast<std::uint32_t>(hidden);
    const std::uint32_t stride_q = static_cast<std::uint32_t>(2U * head_dim);
    const std::uint32_t rope32 = job.rope_dims;
    const std::uint32_t half_store_off = 0U;
    const std::uint32_t half_store_on = 1U;
    const std::uint32_t block32 = job.block_size;
    const std::uint32_t kv_elems32 = static_cast<std::uint32_t>(kv_head_elems);
    const float eps = job.eps;
    const float attn_scale = job.attention_scale;

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (s.q8_quantize_pipeline != nil) {
        const std::uint32_t q8_blocks = static_cast<std::uint32_t>(job.columns / 32U);
        const std::uint32_t q8_tokens = static_cast<std::uint32_t>(count);
        [encoder setComputePipelineState:s.q8_quantize_pipeline];
        [encoder setBuffer:b.wide offset:0 atIndex:0];
        [encoder setBuffer:b.q8 offset:0 atIndex:1];
        [encoder setBuffer:b.scales offset:0 atIndex:2];
        [encoder setBytes:&q8_blocks length:sizeof(q8_blocks) atIndex:3];
        [encoder setBytes:&q8_tokens length:sizeof(q8_tokens) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake((job.columns / 32U) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    // MNET_ATTN_SKIP=<phase> drops one phase for timing; results are garbage.
    static const std::string attn_skip =
        std::getenv("MNET_ATTN_SKIP") != nullptr ? std::string(std::getenv("MNET_ATTN_SKIP")) : std::string();
    const bool as_all = attn_skip == "all";
    const bool as_matvec = as_all || attn_skip == "matvec";
    const bool as_prep = as_all || attn_skip == "prep";
    const bool as_flash = as_all || attn_skip == "flash";
    const bool as_out = as_all || attn_skip == "out";

    // Widen every token's activation once, then all q, all k and all v matvecs.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(job.columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!as_matvec && tiled_mat_mat_enabled(s, count)) {
        const std::uint32_t columns32 = static_cast<std::uint32_t>(job.columns);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(job.columns);
        const id<MTLComputePipelineState> pipeline = tiled_mat_mat_pipeline(s);
        for (std::size_t pass = 0; pass < 3U; ++pass) {
            const WeightBinding& weights = pass == 0U ? q_weights : (pass == 1U ? k_weights : v_weights);
            id<MTLBuffer> target = pass == 0U ? b.qfull : (pass == 1U ? b.k_new : b.v_new);
            const std::uint64_t rows = pass == 0U ? 2U * hidden : kv_head_elems;
            const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(pass == 0U ? job.q.row_bytes
                                                                                    : (pass == 1U ? job.k.row_bytes
                                                                                                  : job.v.row_bytes));
            const float scale = pass == 0U ? job.q.scale : (pass == 1U ? job.k.scale : job.v.scale);
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            const std::uint32_t out_stride32 = static_cast<std::uint32_t>(rows);
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weights.buffer offset:static_cast<NSUInteger>(weights.offset) atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:target offset:0 atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
            [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (rows + 63U) / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
    for (std::size_t pass = 0; !as_matvec && tiled_mat_mat_enabled(s, count) == false && pass < 3U; ++pass) {
        const WeightBinding& weights = pass == 0U ? q_weights : (pass == 1U ? k_weights : v_weights);
        id<MTLBuffer> target = pass == 0U ? b.qfull : (pass == 1U ? b.k_new : b.v_new);
        const std::uint64_t rows = pass == 0U ? 2U * hidden : kv_head_elems;
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(pass == 0U ? job.q.row_bytes
                                                                                 : (pass == 1U ? job.k.row_bytes
                                                                                               : job.v.row_bytes));
        const float scale = pass == 0U ? job.q.scale : (pass == 1U ? job.k.scale : job.v.scale);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(job.columns / 64U);
        const std::uint32_t lanes = matvec_lanes();
        {
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            const std::uint32_t x_stride32 = static_cast<std::uint32_t>(job.columns);
            [encoder setComputePipelineState:s.fast_f32_multi_pipeline];
            [encoder setBuffer:weights.buffer offset:static_cast<NSUInteger>(weights.offset) atIndex:0];
            [encoder setBuffer:b.wide offset:0 atIndex:1];
            [encoder setBuffer:target offset:0 atIndex:3];
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
            [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
            [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(rows * count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
        }
    }
    // Head norms, RoPE, gate sigmoid and the f16 KV append for every token.
    const float hadamard_scale = 1.0F / std::sqrt(static_cast<float>(job.block_size));
    (void)hadamard_scale;
    if (!as_prep && s.attn_prep_batch_pipeline != nil && s.attn_gate_batch_pipeline != nil &&
        s.attn_store_batch_pipeline != nil) {
        // Every token's prep, gate, K append and V store, one dispatch each.
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t stride_q2 = static_cast<std::uint32_t>(2U * head_dim);
        const std::uint32_t cos_stride32 = static_cast<std::uint32_t>(2U * half_rope);
        const std::uint32_t heads32 = static_cast<std::uint32_t>(heads);

        [encoder setComputePipelineState:s.attn_prep_batch_pipeline];
        [encoder setBuffer:b.qfull offset:0 atIndex:0];
        [encoder setBuffer:norm_tables offset:q_norm_offset * sizeof(float) atIndex:1];
        [encoder setBuffer:b.tables offset:0 atIndex:2];
        [encoder setBuffer:b.tables offset:static_cast<NSUInteger>(half_rope * sizeof(float)) atIndex:3];
        [encoder setBuffer:b.q offset:0 atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&stride_q2 length:sizeof(stride_q2) atIndex:8];
        [encoder setBytes:&half_store_off length:sizeof(half_store_off) atIndex:9];
        [encoder setBytes:&heads32 length:sizeof(heads32) atIndex:10];
        [encoder setBytes:&cos_stride32 length:sizeof(cos_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake(heads * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

        [encoder setComputePipelineState:s.attn_gate_batch_pipeline];
        [encoder setBuffer:b.qfull offset:0 atIndex:0];
        [encoder setBuffer:b.gates offset:0 atIndex:1];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:2];
        [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        const std::uint32_t heads_kv32b = static_cast<std::uint32_t>(heads_kv);
        [encoder setComputePipelineState:s.attn_prep_batch_pipeline];
        [encoder setBuffer:b.k_new offset:0 atIndex:0];
        [encoder setBuffer:norm_tables offset:k_norm_offset * sizeof(float) atIndex:1];
        [encoder setBuffer:b.tables offset:0 atIndex:2];
        [encoder setBuffer:b.tables offset:static_cast<NSUInteger>(half_rope * sizeof(float)) atIndex:3];
        [encoder setBuffer:kv offset:static_cast<NSUInteger>(cached_bytes) atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
        [encoder setBytes:&half_store_on length:sizeof(half_store_on) atIndex:9];
        [encoder setBytes:&heads_kv32b length:sizeof(heads_kv32b) atIndex:10];
        [encoder setBytes:&cos_stride32 length:sizeof(cos_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake(heads_kv * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

        [encoder setComputePipelineState:s.attn_store_batch_pipeline];
        [encoder setBuffer:b.v_new offset:0 atIndex:0];
        [encoder setBuffer:kv offset:static_cast<NSUInteger>(kv_plane_bytes + cached_bytes) atIndex:1];
        [encoder setBytes:&kv_elems32 length:sizeof(kv_elems32) atIndex:2];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(kv_head_elems * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }


    // Each token's attention over the cache up to its own position.
    const bool flash_batched = !as_flash && s.attn_flash_batch_pipeline != nil;
    if (flash_batched) {
        const std::uint32_t heads32b = static_cast<std::uint32_t>(heads);
        const std::uint32_t cache32 = static_cast<std::uint32_t>(job.cache_tokens);
        const std::uint32_t hidden32b = static_cast<std::uint32_t>(hidden);
        [encoder setComputePipelineState:s.attn_flash_batch_pipeline];
        [encoder setBuffer:b.q offset:0 atIndex:0];
        [encoder setBuffer:kv offset:0 atIndex:1];
        [encoder setBuffer:kv offset:static_cast<NSUInteger>(kv_plane_bytes) atIndex:2];
        [encoder setBuffer:b.gates offset:0 atIndex:3];
        [encoder setBuffer:b.attn offset:0 atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&heads32b length:sizeof(heads32b) atIndex:6];
        [encoder setBytes:&heads_kv32 length:sizeof(heads_kv32) atIndex:7];
        [encoder setBytes:&group32 length:sizeof(group32) atIndex:8];
        [encoder setBytes:&attn_scale length:sizeof(attn_scale) atIndex:9];
        [encoder setBytes:&cache32 length:sizeof(cache32) atIndex:10];
        [encoder setBytes:&hidden32b length:sizeof(hidden32b) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake(heads * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    for (std::size_t t = 0; !flash_batched && t < count; ++t) {
        const std::uint32_t visible =
            static_cast<std::uint32_t>(job.cache_tokens) + static_cast<std::uint32_t>(t) + 1U;
        [encoder setComputePipelineState:s.attn_flash_pipeline];
        [encoder setBuffer:b.q offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:0];
        [encoder setBuffer:kv offset:0 atIndex:1];
        [encoder setBuffer:kv offset:static_cast<NSUInteger>(kv_plane_bytes) atIndex:2];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:3];
        [encoder setBuffer:b.attn offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&visible length:sizeof(visible) atIndex:6];
        [encoder setBytes:&heads_kv32 length:sizeof(heads_kv32) atIndex:7];
        [encoder setBytes:&group32 length:sizeof(group32) atIndex:8];
        [encoder setBytes:&attn_scale length:sizeof(attn_scale) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    // Hadamard and requantisation per token, then the widened activations for
    // every token in one dispatch, then the projections.
    if (!as_out) {
        [encoder setComputePipelineState:s.hadamard_multi_pipeline];
        [encoder setBuffer:b.attn offset:0 atIndex:0];
        [encoder setBuffer:signs_buffer offset:0 atIndex:1];
        [encoder setBuffer:b.act_q8 offset:0 atIndex:2];
        [encoder setBuffer:b.act_scales offset:0 atIndex:3];
        [encoder setBytes:&block32 length:sizeof(block32) atIndex:4];
        const std::uint32_t sign_blocks = static_cast<std::uint32_t>(hidden / job.block_size);
        [encoder setBytes:&sign_blocks length:sizeof(sign_blocks) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((hidden / job.block_size) * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(job.block_size, 1, 1)];
        [encoder setComputePipelineState:s.dequant_pipeline];
        [encoder setBuffer:b.act_q8 offset:0 atIndex:0];
        [encoder setBuffer:b.act_scales offset:0 atIndex:1];
        [encoder setBuffer:b.wide offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // The tiled mat-mat reuses one weight tile across all tokens; the scalar
    // matvec beside it re-reads the whole row for every token, which at 32 tokens
    // is 32x the weight traffic (37 ms of the forward, log 176).
    if (tiled_mat_mat_enabled(s, count)) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t columns32 = static_cast<std::uint32_t>(hidden);
        const float scale = job.out.scale;
        const std::uint32_t rows32 = static_cast<std::uint32_t>(job.out.rows);
        const std::uint32_t tokens32 = static_cast<std::uint32_t>(count);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        const std::uint32_t out_stride32 = static_cast<std::uint32_t>(job.out.rows);
        [encoder setComputePipelineState:tiled_mat_mat_pipeline(s)];
        [encoder setBuffer:out_weights.buffer offset:static_cast<NSUInteger>(out_weights.offset) atIndex:0];
        [encoder setBuffer:b.wide offset:0 atIndex:1];
        [encoder setBuffer:b.out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&tokens32 length:sizeof(tokens32) atIndex:9];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:10];
        [encoder setBytes:&out_stride32 length:sizeof(out_stride32) atIndex:11];
        [encoder dispatchThreadgroups:MTLSizeMake((count + 31U) / 32U, (job.out.rows + 63U) / 64U, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
        const float scale = job.out.scale;
        const std::uint32_t lanes = matvec_lanes();
        const std::uint32_t rows32 = static_cast<std::uint32_t>(job.out.rows);
        const std::uint32_t x_stride32 = static_cast<std::uint32_t>(hidden);
        [encoder setComputePipelineState:s.fast_f32_multi_pipeline];
        [encoder setBuffer:out_weights.buffer offset:static_cast<NSUInteger>(out_weights.offset) atIndex:0];
        [encoder setBuffer:b.wide offset:0 atIndex:1];
        [encoder setBuffer:b.out offset:0 atIndex:3];
        [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
        [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
        [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
        [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:8];
        [encoder setBytes:&x_stride32 length:sizeof(x_stride32) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(job.out.rows * count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    }
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    record_gpu_block(2, command);
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    const auto* results = static_cast<const float*>(b.out.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(entries[t].output, results + t * job.columns,
                    static_cast<std::size_t>(job.columns) * sizeof(float));
    }
    const std::uint64_t new_bytes = static_cast<std::uint64_t>(count) * kv_head_elems * sizeof(float);
    std::memcpy(job.k_cache + static_cast<std::ptrdiff_t>(job.cache_tokens) * kv_head_elems, kv_bytes + cached_bytes,
                static_cast<std::size_t>(new_bytes));
    std::memcpy(job.v_cache + static_cast<std::ptrdiff_t>(job.cache_tokens) * kv_head_elems,
                kv_bytes + kv_plane_bytes + cached_bytes, static_cast<std::size_t>(new_bytes));
    return true;
}

// RMSNorm with the input and output in host memory: enough to exercise the
// kernel and to measure how far an fp32 reduction drifts from the host's fp64
// reference. The device-resident residual stream will bind these buffers
// directly and skip both copies.
bool rms_norm_host(const float* input, const float* weight, std::uint64_t size, float eps, float* output) {
    MetalState& s = state();
    if (!s.ready || s.rms_norm_pipeline == nil || size == 0U || size > 65536U) {
        return false;
    }
    id<MTLBuffer> x = [s.device newBufferWithBytes:input
                                            length:static_cast<NSUInteger>(size * sizeof(float))
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> w = [s.device newBufferWithBytes:weight
                                            length:static_cast<NSUInteger>(size * sizeof(float))
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [s.device newBufferWithLength:static_cast<NSUInteger>(size * sizeof(float))
                                              options:MTLResourceStorageModeShared];
    if (x == nil || w == nil || out == nil) {
        return false;
    }
    const std::uint32_t size32 = static_cast<std::uint32_t>(size);
    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:s.rms_norm_pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:w offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&size32 length:sizeof(size32) atIndex:3];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
    // A threadgroup buffer argument carries no size, it has to be declared.
    [encoder setThreadgroupMemoryLength:256U * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(output, out.contents, static_cast<std::size_t>(size) * sizeof(float));
    return true;
}

// Layer preparation on the device: RMSNorm, the signed Hadamard and the Q8_0
// requantisation, in two dispatches and one submission. Host buffers for now;
// the device-resident residual stream will bind them directly.
bool layer_prep_host(const float* input, const float* weight, std::uint64_t size, float eps,
                     const std::int32_t* signs, std::uint32_t block_size, std::int8_t* qs, float* scales) {
    MetalState& s = state();
    if (!s.ready || s.rms_norm_pipeline == nil || s.hadamard_quantize_pipeline == nil || size == 0U ||
        size > 65536U || block_size < 64U || block_size > 1024U || (block_size & (block_size - 1U)) != 0U ||
        (size % block_size) != 0U) {
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);
    id<MTLBuffer> x = [s.device newBufferWithBytes:input length:static_cast<NSUInteger>(bytes)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> w = [s.device newBufferWithBytes:weight length:static_cast<NSUInteger>(bytes)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> normed = [s.device newBufferWithLength:static_cast<NSUInteger>(bytes)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sign_buffer = [s.device newBufferWithBytes:signs
                                                      length:static_cast<NSUInteger>(size * sizeof(std::int32_t))
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> qs_buffer = [s.device newBufferWithLength:static_cast<NSUInteger>(size)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale_buffer = [s.device newBufferWithLength:static_cast<NSUInteger>(size / 32U * sizeof(float))
                                                       options:MTLResourceStorageModeShared];
    if (x == nil || w == nil || normed == nil || sign_buffer == nil || qs_buffer == nil || scale_buffer == nil) {
        return false;
    }
    const std::uint32_t size32 = static_cast<std::uint32_t>(size);
    const std::uint32_t blocks32 = static_cast<std::uint32_t>(size / block_size);
    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:s.rms_norm_pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:w offset:0 atIndex:1];
    [encoder setBuffer:normed offset:0 atIndex:2];
    [encoder setBytes:&size32 length:sizeof(size32) atIndex:3];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:4];
    [encoder setThreadgroupMemoryLength:256U * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:s.hadamard_quantize_pipeline];
    [encoder setBuffer:normed offset:0 atIndex:0];
    [encoder setBuffer:sign_buffer offset:0 atIndex:1];
    [encoder setBuffer:qs_buffer offset:0 atIndex:2];
    [encoder setBuffer:scale_buffer offset:0 atIndex:3];
    [encoder setBytes:&block_size length:sizeof(block_size) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks32, 1, 1) threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(qs, qs_buffer.contents, static_cast<std::size_t>(size));
    std::memcpy(scales, scale_buffer.contents, static_cast<std::size_t>(size / 32U) * sizeof(float));
    return true;
}

// x += delta with both sides in host memory, one submission. Primitive of the
// device-resident residual stream (the layer job will bind the buffers instead
// of copying them), and exercised by benchmarks/metal/layer_prep_test.cpp.
bool residual_add_host(float* x, const float* delta, std::uint64_t count) {
    MetalState& s = state();
    if (!s.ready || s.residual_add_pipeline == nil || count == 0U || count > 65536U) {
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    id<MTLBuffer> xb = [s.device newBufferWithBytes:x length:static_cast<NSUInteger>(bytes)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> db = [s.device newBufferWithBytes:delta length:static_cast<NSUInteger>(bytes)
                                            options:MTLResourceStorageModeShared];
    if (xb == nil || db == nil) {
        return false;
    }
    const std::uint32_t count32 = static_cast<std::uint32_t>(count);
    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:s.residual_add_pipeline];
    [encoder setBuffer:xb offset:0 atIndex:0];
    [encoder setBuffer:db offset:0 atIndex:1];
    [encoder setBytes:&count32 length:sizeof(count32) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    std::memcpy(x, xb.contents, bytes);
    return true;
}

bool matvec_q8_0_backend(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                         std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales,
                         float* out, std::uint64_t identity, const std::uint8_t* region,
                         std::uint64_t region_bytes) {
    if (type != 42U) {  // kGgmlQ2_0; kept numeric so this TU needs no GGUF header
        return false;
    }
    return matvec_q2_0(data, rows, columns, row_bytes, scale, q8, q8_scales, out, identity, region, region_bytes);
}

}  // namespace

void register_platform_backends() {
    if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
        std::fprintf(stderr, "mnet: metal backend %s (fast %s, ffn %s, attn %s, ssm %s)\n",
                     state().ready ? "ready" : "unavailable", state().fast_pipeline != nil ? "ready" : "unavailable",
                     (state().swiglu_pipeline != nil && state().hadamard_quantize_pipeline != nil)
                         ? "ready"
                         : "unavailable",
                     (state().attn_prep_pipeline != nil && state().attn_flash_pipeline != nil &&
                      state().attn_gate_pipeline != nil && state().attn_store_pipeline != nil)
                         ? "ready"
                         : "unavailable",
                     (state().ssm_rows_pipeline != nil && state().ssm_act_pipeline != nil &&
                      state().ssm_conv_norm_pipeline != nil && state().ssm_scan_pipeline != nil &&
                      state().ssm_gate_norm_pipeline != nil)
                         ? "ready"
                         : "unavailable");
    }
    set_matvec_q8_0_backend(&matvec_q8_0_backend);
    set_matvec_q8_0_batch_backend(&matvec_batch);
    set_ffn_swiglu_backend(&ffn_swiglu);
    set_ffn_swiglu_batch_backend(&ffn_batch_block);
    set_attention_backend(&attention);
    set_attention_batch_backend(&attention_batch_block);
    set_rms_norm_backend(&rms_norm_host);
    set_layer_prep_backend(&layer_prep_host);
    set_residual_add_backend(&residual_add_host);
    set_layer_prep_backend(&layer_prep_host);
    set_ssm_backend(&ssm_block);
    set_ssm_batch_backend(&ssm_batch_block);
}

}  // namespace mnet::detail::compute
