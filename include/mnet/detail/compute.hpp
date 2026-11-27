// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MNET_DETAIL_COMPUTE_HPP
#define MNET_DETAIL_COMPUTE_HPP

// Device-neutral compute entry points. Callers (Reader) describe the problem
// and never learn whether a CPU or a GPU executed it: a backend registers
// itself at start-up through set_matvec_q8_0_backend() and matvec_q8_0() picks
// it when it can handle the type, otherwise it runs the CPU loop. Adding a new
// device (Metal, CUDA, ...) only adds one registration in its own translation
// unit; no model code changes.

#include <cstddef>
#include <cstdint>

namespace mnet::detail::compute {

// Executes y = scale * W x for a GGML type that uses Q8_0 activations.
// Returns false when this device cannot handle the type.
using MatvecQ8_0Fn = bool (*)(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                              std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales,
                              float* out, std::uint64_t identity, const std::uint8_t* region,
                              std::uint64_t region_bytes);

// One matrix in a batched call; all jobs share the activation (q8/q8_scales).
struct MatvecQ8_0Job {
    std::uint32_t type = 0;
    const std::uint8_t* data = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t row_bytes = 0;
    float scale = 1.0F;
    float* out = nullptr;
    std::uint64_t identity = 0;  // stable tensor id; devices key their caches on it
    // When the blocks are borrowed from a read-only file mapping, `region` is
    // that mapping's base and the device can alias it instead of copying.
    // `region_bytes` is the mapping length rounded up to a page.
    const std::uint8_t* region = nullptr;
    std::uint64_t region_bytes = 0;
};

// A device can fuse several independent matvecs that share one activation into
// a single submission (one wait instead of N).
using MatvecQ8_0BatchFn = bool (*)(const MatvecQ8_0Job* jobs, std::size_t count, std::uint64_t columns,
                                   const std::int8_t* q8, const float* q8_scales);

void set_matvec_q8_0_backend(MatvecQ8_0Fn backend);
void set_matvec_q8_0_batch_backend(MatvecQ8_0BatchFn backend);

using FfnSwigluFn = bool (*)(const MatvecQ8_0Job& gate, const MatvecQ8_0Job& up, const MatvecQ8_0Job& down,
                             std::uint64_t columns, std::uint64_t hidden, const float* input, const std::int32_t* signs,
                             std::uint32_t block_size);
void set_ffn_swiglu_backend(FfnSwigluFn backend);

// One token of a batched FFN: the same three matrices, its own input and
// output. Encoding several of these into one submission lets the independent
// matvecs of different tokens pipeline, which the per-token block cannot do
// because swiglu -> Hadamard -> requantisation -> down is serial.
struct FfnSwigluEntry {
    MatvecQ8_0Job gate;
    MatvecQ8_0Job up;
    MatvecQ8_0Job down;
    const float* input = nullptr;  // n_embd, already Hadamard'ed
    float* output = nullptr;       // n_embd
    // Optional device preparation of `input`: when all three are set the block
    // reads `residual` (the raw layer output) and runs the RMSNorm with
    // post_norm, the signed Hadamard with input_signs and the Q8_0
    // requantisation itself, instead of waiting for the host to do the first two
    // (log 178). Same values go in every entry of a batch.
    const float* residual = nullptr;            // n_embd, un-normalised
    const float* post_norm = nullptr;           // n_embd
    const std::int32_t* input_signs = nullptr;  // n_embd, block_size blocks
    float eps = 0.0F;
};

using FfnSwigluBatchFn = bool (*)(const FfnSwigluEntry* entries, std::size_t count, std::uint64_t columns,
                                  std::uint64_t hidden, const std::int32_t* signs, std::uint32_t block_size);
void set_ffn_swiglu_batch_backend(FfnSwigluBatchFn backend);

[[nodiscard]] bool ffn_swiglu_batch(const FfnSwigluEntry* entries, std::size_t count, std::uint64_t columns,
                                    std::uint64_t hidden, const std::int32_t* signs, std::uint32_t block_size);

// Implemented by a device translation unit when one is compiled in, otherwise
// the portable no-op in src/compute/backend.cpp.
void register_platform_backends();

// Always computes the full matrix-vector product.
void matvec_q8_0(std::uint32_t type, const std::uint8_t* data, std::uint64_t rows, std::uint64_t columns,
                 std::uint64_t row_bytes, float scale, const std::int8_t* q8, const float* q8_scales, float* out,
                 std::uint64_t identity, const std::uint8_t* region = nullptr, std::uint64_t region_bytes = 0);

// Same, for several matrices sharing one activation. The device may execute
// the whole batch before returning; callers only see the outputs.
void matvec_q8_0_batch(const MatvecQ8_0Job* jobs, std::size_t count, std::uint64_t columns, const std::int8_t* q8,
                       const float* q8_scales);

// Fused feed-forward block: gate/up matvec, SwiGLU, the model's normalised
// Hadamard over the hidden vector (signs are +-1, block_size a power of two),
// on-device Q8_0 requantisation and the down matvec, all in one submission.
// Returns false when the device cannot take the block; the caller then keeps
// the step-by-step path.
[[nodiscard]] bool ffn_swiglu_q2_0(const MatvecQ8_0Job& gate, const MatvecQ8_0Job& up, const MatvecQ8_0Job& down,
                                   std::uint64_t columns, std::uint64_t hidden, const float* input,
                                   const std::int32_t* signs, std::uint32_t block_size);

// Fused attention block. One submission covers the q/k/v matvecs, the split of
// each query row into query and gate halves, per-head RMSNorm, RoPE, the
// f16-rounded KV append, the online-softmax attention over the whole cache, the
// sigmoid gate multiply, the model's signed Hadamard over the head vector, the
// on-device Q8_0 requantisation and the output projection.
//
// Layouts (all row-major):
//   q.rows   = 2 * n_head * head_dim   (head h: query at 2*h*head_dim, gate next)
//   k.rows   = v.rows = n_head_kv * head_dim
//   out.rows = n_embd, out columns = n_head * head_dim
//   k_cache/v_cache: [cache_tokens][n_head_kv][head_dim], already f16-rounded
//   rope_cos/rope_sin: rope_dims / 2 entries for the current position
struct AttentionQ2_0Job {
    MatvecQ8_0Job q;
    MatvecQ8_0Job k;
    MatvecQ8_0Job v;
    MatvecQ8_0Job out;
    const float* input = nullptr;         // n_embd activation, already Hadamard'ed
    const float* q_norm = nullptr;        // head_dim
    const float* k_norm = nullptr;        // head_dim
    const float* rope_cos = nullptr;      // rope_dims / 2
    const float* rope_sin = nullptr;      // rope_dims / 2
    const std::int32_t* signs = nullptr;  // n_head * head_dim
    float* k_cache = nullptr;
    float* v_cache = nullptr;
    float* result = nullptr;    // n_embd
    std::uint64_t columns = 0;  // n_embd
    std::uint32_t n_head = 0;
    std::uint32_t n_head_kv = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rope_dims = 0;
    std::uint32_t block_size = 0;
    std::uint32_t cache_tokens = 0;  // tokens already stored, before this one
    float eps = 0.0F;
    float attention_scale = 0.0F;
};

using AttentionQ2_0Fn = bool (*)(const AttentionQ2_0Job& job);
void set_attention_backend(AttentionQ2_0Fn backend);

// Returns false when the device cannot take the block; the caller then keeps
// the step-by-step path (which is also the bit-exact reference).
[[nodiscard]] bool attention_q2_0(const AttentionQ2_0Job& job);

// One token of a batched attention block. The q/k/v matvecs, the head norms and
// RoPE, the f16 KV append and the output projection are token independent; the
// attention itself only needs the cache to hold the tokens up to its own
// position.
struct AttentionBatchEntry {
    const float* input = nullptr;  // n_embd, already Hadamard'ed
    float* output = nullptr;       // n_embd
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
};

using AttentionQ2_0BatchFn = bool (*)(const AttentionQ2_0Job& job, const AttentionBatchEntry* entries,
                                      std::size_t count);
void set_attention_batch_backend(AttentionQ2_0BatchFn backend);
[[nodiscard]] bool attention_q2_0_batch(const AttentionQ2_0Job& job, const AttentionBatchEntry* entries,
                                        std::size_t count);

// Fused gated-delta (SSM) block: qkv and gate matvecs, the beta/alpha head
// projections, the causal depthwise convolution, the per-head delta rule over
// the persistent state, the gated RMS norm with the grouped-head permute, the
// signed Hadamard, on-device Q8_0 requantisation and the ssm_out projection.
//
// The state is large (value_heads * head_dim^2 floats per layer), so the device
// keeps it resident and keyed by `identity`; conv_state/ssm_state seed it on
// first use and are not written back. Callers that need the state on the host
// (state dumps, save/load) must use the step-by-step path.
struct SsmQ2_0Job {
    MatvecQ8_0Job qkv;                    // rows = conv_dim, columns = n_embd
    MatvecQ8_0Job gate;                   // rows = value_heads * head_dim, columns = n_embd
    MatvecQ8_0Job out;                    // rows = n_embd, columns = value_heads * head_dim
    const float* input = nullptr;         // n_embd, already Hadamard'ed (qkv/gate)
    const float* raw_input = nullptr;     // n_embd, before the Hadamard (beta/alpha)
    const float* beta_w = nullptr;        // value_heads * n_embd
    const float* alpha_w = nullptr;       // value_heads * n_embd
    const float* dt = nullptr;            // value_heads
    const float* ssm_a = nullptr;         // value_heads
    const float* conv_w = nullptr;        // conv_dim * conv_kernel
    const float* ssm_norm = nullptr;      // head_dim
    const std::int32_t* signs = nullptr;  // value_heads * head_dim
    const float* conv_state = nullptr;    // (conv_kernel - 1) * conv_dim, seed only
    const float* ssm_state = nullptr;     // value_heads * head_dim * head_dim, seed only
    float* result = nullptr;              // n_embd
    std::uint64_t columns = 0;            // n_embd
    std::uint32_t conv_dim = 0;
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t conv_kernel = 0;
    std::uint32_t block_size = 0;
    float eps = 0.0F;
    std::uint64_t identity = 0;  // per-layer key for the resident state
};

// One token of a batched gated-delta block. The block's qkv/gate matvecs, the
// head gates and the convolution are token independent; only the delta-rule
// scan and what reads it (gated RMS norm, Hadamard, output projection) must run
// in token order.
struct SsmBatchEntry {
    const float* input = nullptr;      // n_embd, already Hadamard'ed
    const float* raw_input = nullptr;  // n_embd, before the Hadamard
    float* output = nullptr;           // n_embd
    // Optional device preparation, as for FfnSwigluEntry: with residual set the
    // block normalises it and rotates it itself instead of reading host work.
    const float* residual = nullptr;            // n_embd
    const float* input_norm = nullptr;          // n_embd
    const std::int32_t* input_signs = nullptr;  // n_embd, block_size blocks
};

using SsmQ2_0BatchFn = bool (*)(const SsmQ2_0Job& job, const SsmBatchEntry* entries, std::size_t count);
void set_ssm_batch_backend(SsmQ2_0BatchFn backend);
[[nodiscard]] bool ssm_q2_0_batch(const SsmQ2_0Job& job, const SsmBatchEntry* entries, std::size_t count);

using SsmQ2_0Fn = bool (*)(const SsmQ2_0Job& job);
void set_ssm_backend(SsmQ2_0Fn backend);

[[nodiscard]] bool ssm_q2_0(const SsmQ2_0Job& job);

// RMSNorm executed on the device with the input and output in host memory. The
// fp32 reduction cannot match the host's fp64 accumulator, so this belongs to
// the tolerance-level fast path only; it exists so the layer pipeline can run
// the norm without returning to the CPU once the residual stream lives on the
// device.
using RmsNormFn = bool (*)(const float* input, const float* weight, std::uint64_t size, float eps, float* output);
void set_rms_norm_backend(RmsNormFn backend);
[[nodiscard]] bool rms_norm_f32(const float* input, const float* weight, std::uint64_t size, float eps, float* output);

// Layer preparation in one submission: RMSNorm, the model's signed Hadamard and
// the Q8_0 requantisation that the block matvecs consume. `size` must be a
// multiple of `block_size`.
using LayerPrepFn = bool (*)(const float* input, const float* weight, std::uint64_t size, float eps,
                             const std::int32_t* signs, std::uint32_t block_size, std::int8_t* qs, float* scales);
void set_layer_prep_backend(LayerPrepFn backend);
[[nodiscard]] bool layer_prep_f32(const float* input, const float* weight, std::uint64_t size, float eps,
                                  const std::int32_t* signs, std::uint32_t block_size, std::int8_t* qs, float* scales);

// x += delta on the device (host buffers for now; the layer job will bind them
// directly). Element-wise, so bit-identical to the host loop.
using ResidualAddFn = bool (*)(float* x, const float* delta, std::uint64_t count);
void set_residual_add_backend(ResidualAddFn backend);
[[nodiscard]] bool residual_add(float* x, const float* delta, std::uint64_t count);

// True when the caller may use the tolerance-level fast paths.
[[nodiscard]] bool fast_path_enabled();

// The fused FFN block is still being validated; it stays behind this flag so
// the fast path cannot silently regress.
[[nodiscard]] bool fused_ffn_enabled();

// Same A/B switch for the fused attention block (MNET_NO_FUSED_ATTN).
[[nodiscard]] bool fused_attention_enabled();

// Same A/B switch for the fused gated-delta block (MNET_NO_FUSED_SSM).
[[nodiscard]] bool fused_ssm_enabled();

}  // namespace mnet::detail::compute

#endif  // MNET_DETAIL_COMPUTE_HPP
