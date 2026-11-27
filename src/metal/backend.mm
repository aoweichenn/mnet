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
        const uint index = token * heads + head;
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

// Gated RMS norm per value head and the tiled -> grouped head permute the
// folded ssm_out weight was built with, in one pass.
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

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> fast_pipeline = nil;
    id<MTLComputePipelineState> fast_f32_pipeline = nil;
    id<MTLComputePipelineState> fast_f32_multi_pipeline = nil;
    id<MTLComputePipelineState> dequant_pipeline = nil;
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
    id<MTLComputePipelineState> attn_prep_pipeline = nil;
    id<MTLComputePipelineState> attn_gate_pipeline = nil;
    id<MTLComputePipelineState> attn_flash_pipeline = nil;
    id<MTLComputePipelineState> attn_store_pipeline = nil;
    id<MTLComputePipelineState> ssm_rows_pipeline = nil;
    id<MTLComputePipelineState> ssm_act_pipeline = nil;
    id<MTLComputePipelineState> ssm_conv_norm_pipeline = nil;
    id<MTLComputePipelineState> ssm_scan_pipeline = nil;
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
        state.attn_gate_pipeline = pipeline_for("attn_gate_sigmoid");
        state.attn_flash_pipeline = pipeline_for("attn_flash");
        state.attn_store_pipeline = pipeline_for("attn_store_half");
        state.ssm_rows_pipeline = pipeline_for("ssm_rows_dot");
        state.ssm_act_pipeline = pipeline_for("ssm_activation");
        state.ssm_conv_norm_pipeline = pipeline_for("ssm_conv_norm");
        state.ssm_scan_pipeline = pipeline_for("ssm_scan");
        state.ssm_gate_norm_pipeline = pipeline_for("ssm_gate_norm");
        state.dequant_pipeline = pipeline_for("q8_0_dequantize");
        state.hadamard_multi_pipeline = pipeline_for("hadamard_quantize_multi");
        state.gate_norm_multi_pipeline = pipeline_for("ssm_gate_norm_multi");
        state.head_gates_pipeline = pipeline_for("ssm_head_gates");
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
    id<MTLBuffer>& region = s.region_buffers[job.region];
    if (region == nil) {
        region = [s.device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(job.region)
                                            length:static_cast<NSUInteger>(job.region_bytes)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        if (region == nil) {
            if (std::getenv("MNET_METAL_DEBUG") != nullptr) {
                std::fprintf(stderr, "mnet: metal cannot alias the %llu byte mapping\n",
                             static_cast<unsigned long long>(job.region_bytes));
            }
            s.region_buffers.erase(job.region);
            return {};
        }
    }
    const auto offset = static_cast<std::uint64_t>(job.data - job.region);
    if (offset + bytes > job.region_bytes) {
        return {};
    }
    return {region, offset};
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
    auto* q8_destination = static_cast<std::uint8_t*>(b.q8.contents);
    auto* scale_destination = static_cast<float*>(b.scales.contents);
    for (std::size_t t = 0; t < count; ++t) {
        const auto q80 = mnet::gguf::make_q8_0_blocks(entries[t].input, columns);
        std::memcpy(q8_destination + t * columns, q80.qs.data(), static_cast<std::size_t>(columns));
        std::memcpy(scale_destination + t * (columns / 32U), q80.d.data(),
                    static_cast<std::size_t>(columns / 32U) * sizeof(float));
    }

    const std::uint32_t block32 = block_size;
    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    // Widen every token's activation once: the slabs are contiguous, so this is
    // a single dispatch over all tokens.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // Every gate, then every up matvec.
    for (std::size_t pass = 0; pass < 2U; ++pass) {
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
            const std::uint32_t lanes = 32U;
            [encoder setBytes:&row_bytes32 length:sizeof(row_bytes32) atIndex:4];
            [encoder setBytes:&blocks32 length:sizeof(blocks32) atIndex:5];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:6];
            [encoder setBytes:&lanes length:sizeof(lanes) atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    }
    // SwiGLU and the Hadamard/requantisation are element-wise over the whole
    // per-token slab, so one dispatch each covers every token.
    {
        const std::uint32_t slab = static_cast<std::uint32_t>(hidden * count);
        [encoder setComputePipelineState:s.swiglu_pipeline];
        [encoder setBuffer:b.gate offset:0 atIndex:0];
        [encoder setBuffer:b.up offset:0 atIndex:1];
        [encoder setBuffer:b.act offset:0 atIndex:2];
        [encoder setBytes:&slab length:sizeof(slab) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

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
    // The down matvec's activation is widened for every token in one dispatch,
    // then projected in one dispatch.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.act_q8 offset:0 atIndex:0];
    [encoder setBuffer:b.act_scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(first.down.row_bytes);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
        const float scale = first.down.scale;
        const std::uint32_t lanes = 32U;
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
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        return false;
    }
    const auto* results = static_cast<const float*>(b.out.contents);
    for (std::size_t t = 0; t < count; ++t) {
        std::memcpy(entries[t].output, results + t * columns, static_cast<std::size_t>(columns) * sizeof(float));
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
    auto* q8_destination = static_cast<std::uint8_t*>(b.q8.contents);
    auto* scale_destination = static_cast<float*>(b.scales.contents);
    auto* activation_destination = static_cast<float*>(b.activation.contents);
    for (std::size_t t = 0; t < count; ++t) {
        const auto q80 = mnet::gguf::make_q8_0_blocks(entries[t].input, job.columns);
        std::memcpy(q8_destination + t * job.columns, q80.qs.data(), static_cast<std::size_t>(job.columns));
        std::memcpy(scale_destination + t * (job.columns / 32U), q80.d.data(),
                    static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
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

    id<MTLCommandBuffer> command = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    // Widen every token's activation once, then run all qkv and all gate
    // matvecs.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(job.columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // One dispatch per matvec for all tokens: the per-token activation slab is
    // contiguous, so every threadgroup only needs its (token, row) pair.
    if (!skip_matvec) {
        for (std::size_t pass = 0; pass < 2U; ++pass) {
            const std::uint64_t rows = pass == 0U ? conv_dim : hidden;
            const std::uint32_t blocks32 = static_cast<std::uint32_t>(job.columns / 64U);
            const std::uint32_t row_bytes32 =
                static_cast<std::uint32_t>(pass == 0U ? job.qkv.row_bytes : job.gate.row_bytes);
            const float scale = pass == 0U ? job.qkv.scale : job.gate.scale;
            const std::uint32_t lanes = 32U;
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
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    }
    // Head gates and the convolution for every token.
    for (std::size_t t = 0; t < count; ++t) {
        if (skip_gate && skip_conv) {
            continue;
        }
        if (!skip_gate) {
        [encoder setComputePipelineState:s.head_gates_pipeline];
        [encoder setBuffer:b.activation offset:static_cast<NSUInteger>(t * job.columns * sizeof(float)) atIndex:0];
        [encoder setBuffer:layer.side offset:0 atIndex:1];
        [encoder setBuffer:layer.side offset:dt_offset * sizeof(float) atIndex:2];
        [encoder setBuffer:layer.side offset:a_offset * sizeof(float) atIndex:3];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(t * 2U * value_heads * sizeof(float)) atIndex:4];
        [encoder setBuffer:b.gates
                     offset:static_cast<NSUInteger>(t * 2U * value_heads * sizeof(float) +
                                                    value_heads * sizeof(float))
                    atIndex:5];
        [encoder setBytes:&columns32 length:sizeof(columns32) atIndex:6];
        [encoder setBytes:&value_heads32 length:sizeof(value_heads32) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        if (skip_conv) {
            continue;
        }
        // Linear window: token t's four taps are rows t .. t + kernel - 1.
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
    // Token order from here: the scan carries state and the rest reads it.
    const std::uint32_t q_offset = 0U;
    const std::uint32_t k_offset = static_cast<std::uint32_t>(key_heads * head_dim);
    const std::uint32_t v_offset = static_cast<std::uint32_t>(2U * key_heads * head_dim);
    for (std::size_t t = 0; t < count; ++t) {
        if (skip_scan) {
            continue;
        }
        [encoder setComputePipelineState:s.ssm_scan_pipeline];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(t * conv_dim * sizeof(float)) + q_offset * sizeof(float)
                    atIndex:0];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(t * conv_dim * sizeof(float)) + k_offset * sizeof(float)
                    atIndex:1];
        [encoder setBuffer:b.conv offset:static_cast<NSUInteger>(t * conv_dim * sizeof(float)) + v_offset * sizeof(float)
                    atIndex:2];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(t * 2U * value_heads * sizeof(float)) atIndex:3];
        [encoder setBuffer:b.gates
                     offset:static_cast<NSUInteger>(t * 2U * value_heads * sizeof(float) +
                                                    value_heads * sizeof(float))
                    atIndex:4];
        [encoder setBuffer:layer.state offset:0 atIndex:5];
        [encoder setBuffer:b.delta offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:6];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:7];
        [encoder setBytes:&key_heads32 length:sizeof(key_heads32) atIndex:8];
        [encoder setBytes:&inv_dim length:sizeof(inv_dim) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

    }
    if (!skip_norm) {
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
    if (!skip_norm) {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
        const float scale = job.out.scale;
        const std::uint32_t lanes = 32U;
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
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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
    auto* q8_destination = static_cast<std::uint8_t*>(b.q8.contents);
    auto* scale_destination = static_cast<float*>(b.scales.contents);
    for (std::size_t t = 0; t < count; ++t) {
        const auto q80 = mnet::gguf::make_q8_0_blocks(entries[t].input, job.columns);
        std::memcpy(q8_destination + t * job.columns, q80.qs.data(), static_cast<std::size_t>(job.columns));
        std::memcpy(scale_destination + t * (job.columns / 32U), q80.d.data(),
                    static_cast<std::size_t>(job.columns / 32U) * sizeof(float));
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
    // Widen every token's activation once, then all q, all k and all v matvecs.
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.q8 offset:0 atIndex:0];
    [encoder setBuffer:b.scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(job.columns * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    for (std::size_t pass = 0; pass < 3U; ++pass) {
        const WeightBinding& weights = pass == 0U ? q_weights : (pass == 1U ? k_weights : v_weights);
        id<MTLBuffer> target = pass == 0U ? b.qfull : (pass == 1U ? b.k_new : b.v_new);
        const std::uint64_t rows = pass == 0U ? 2U * hidden : kv_head_elems;
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(pass == 0U ? job.q.row_bytes
                                                                                 : (pass == 1U ? job.k.row_bytes
                                                                                               : job.v.row_bytes));
        const float scale = pass == 0U ? job.q.scale : (pass == 1U ? job.k.scale : job.v.scale);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(job.columns / 64U);
        const std::uint32_t lanes = 32U;
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
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    }
    // Head norms, RoPE, gate sigmoid and the f16 KV append for every token.
    const float hadamard_scale = 1.0F / std::sqrt(static_cast<float>(job.block_size));
    (void)hadamard_scale;
    for (std::size_t t = 0; t < count; ++t) {
        [encoder setComputePipelineState:s.attn_prep_pipeline];
        [encoder setBuffer:b.qfull offset:static_cast<NSUInteger>(t * 2U * hidden * sizeof(float)) atIndex:0];
        [encoder setBuffer:norm_tables offset:q_norm_offset * sizeof(float) atIndex:1];
        [encoder setBuffer:b.tables offset:static_cast<NSUInteger>(t * 2U * half_rope * sizeof(float)) atIndex:2];
        [encoder setBuffer:b.tables
                     offset:static_cast<NSUInteger>((t * 2U * half_rope + half_rope) * sizeof(float))
                    atIndex:3];
        [encoder setBuffer:b.q offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&stride_q length:sizeof(stride_q) atIndex:8];
        [encoder setBytes:&half_store_off length:sizeof(half_store_off) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

        [encoder setComputePipelineState:s.attn_gate_pipeline];
        [encoder setBuffer:b.qfull offset:static_cast<NSUInteger>(t * 2U * hidden * sizeof(float)) atIndex:0];
        [encoder setBuffer:b.gates offset:static_cast<NSUInteger>(t * hidden * sizeof(float)) atIndex:1];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:2];
        [encoder setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [encoder setComputePipelineState:s.attn_prep_pipeline];
        [encoder setBuffer:b.k_new offset:static_cast<NSUInteger>(t * kv_head_elems * sizeof(float)) atIndex:0];
        [encoder setBuffer:norm_tables offset:k_norm_offset * sizeof(float) atIndex:1];
        [encoder setBuffer:b.tables offset:static_cast<NSUInteger>(t * 2U * half_rope * sizeof(float)) atIndex:2];
        [encoder setBuffer:b.tables
                     offset:static_cast<NSUInteger>((t * 2U * half_rope + half_rope) * sizeof(float))
                    atIndex:3];
        [encoder setBuffer:kv
                     offset:static_cast<NSUInteger>(cached_bytes + t * kv_head_elems * sizeof(float))
                    atIndex:4];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:5];
        [encoder setBytes:&rope32 length:sizeof(rope32) atIndex:6];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:7];
        [encoder setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:8];
        [encoder setBytes:&half_store_on length:sizeof(half_store_on) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(heads_kv, 1, 1) threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];

        [encoder setComputePipelineState:s.attn_store_pipeline];
        [encoder setBuffer:b.v_new offset:static_cast<NSUInteger>(t * kv_head_elems * sizeof(float)) atIndex:0];
        [encoder setBuffer:kv
                     offset:static_cast<NSUInteger>(kv_plane_bytes + cached_bytes + t * kv_head_elems * sizeof(float))
                    atIndex:1];
        [encoder setBytes:&kv_elems32 length:sizeof(kv_elems32) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(kv_head_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    // Each token's attention over the cache up to its own position.
    for (std::size_t t = 0; t < count; ++t) {
        const std::uint32_t visible = static_cast<std::uint32_t>(job.cache_tokens) + static_cast<std::uint32_t>(t) + 1U;
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
    {
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
    }
    [encoder setComputePipelineState:s.dequant_pipeline];
    [encoder setBuffer:b.act_q8 offset:0 atIndex:0];
    [encoder setBuffer:b.act_scales offset:0 atIndex:1];
    [encoder setBuffer:b.wide offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(hidden * count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    {
        const std::uint32_t row_bytes32 = static_cast<std::uint32_t>(job.out.row_bytes);
        const std::uint32_t blocks32 = static_cast<std::uint32_t>(hidden / 64U);
        const float scale = job.out.scale;
        const std::uint32_t lanes = 32U;
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
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
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
    set_ssm_backend(&ssm_block);
    set_ssm_batch_backend(&ssm_batch_block);
}

}  // namespace mnet::detail::compute
