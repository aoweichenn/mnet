#pragma once

// Operator layer for the pure-C++ TTS runners: the model-independent
// VITS/StyleTTS2 building blocks (convolution, transposed convolution, linear,
// LayerNorm, AdaIN, WaveNet and ResBlock stacks). Every kernel follows ONNX
// Runtime's arithmetic bit-for-bit; the element-wise primitives it builds on
// live in <mnet/tts/tts_exact.hpp>.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mnet/tts/tts_exact.hpp>
#include <mnet/tts/tts_reference.hpp>
#include <mnet/tts/tts_tensor.hpp>

namespace mnet::tts {

constexpr float kLayerNormEps = 1.0e-5F;

// ONNX Runtime's RNN kernels call MlasGemm with prepacked weights, so SGEMM
// walks K in blocks of MLAS_SGEMM_PACKED_STRIDEK = 256 and adds the partial
// sums of the blocks (sgemm.cpp, MlasSgemmPackedOperation).
constexpr std::size_t kMlasGemmPackedStrideK = 256;

// --profile: coarse kernel accounting so performance work can be measured
// without a profiler (perf events are usually not permitted for user builds).
inline bool g_profile = false;
inline std::uint64_t g_profile_ns[8] = {0, 0, 0, 0, 0, 0, 0, 0};
inline std::uint64_t g_profile_calls[8] = {0, 0, 0, 0, 0, 0, 0, 0};
enum ProfileSlot : std::size_t {
    kProfileConv = 0,
    kProfileConvTranspose = 1,
    kProfileLinear = 2,
    kProfileLstm = 3,
    kProfileNorm = 4,
    kProfileAttention = 5,
    kProfileIstft = 6,
    kProfileHarmonic = 7,
};

class ProfileScope {
public:
    explicit ProfileScope(std::size_t slot) : slot_(slot) {
        if (g_profile) {
            this->start_ = std::chrono::steady_clock::now();
        }
    }
    ~ProfileScope() {
        if (g_profile) {
            const auto end = std::chrono::steady_clock::now();
            g_profile_ns[this->slot_] += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - this->start_).count());
            ++g_profile_calls[this->slot_];
        }
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&&) = delete;
    ProfileScope& operator=(ProfileScope&&) = delete;

private:
    std::size_t slot_;
    std::chrono::steady_clock::time_point start_{};
};

// ONNX Runtime's intra-op thread count. It decides how MLAS partitions a
// convolution's N dimension and therefore each output column's K blocking.
// The pinned Piper reference ran with the default (one thread per core); the
// pinned Kokoro reference ran with a single thread.
inline std::size_t g_ort_threads =
    std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));

// Split an independent iteration space across the machine's cores. Every
// kernel parallelised with this keeps exactly one thread per output element, so
// the per-element accumulation order (and therefore the bit-exact result) does
// not depend on the thread count. Small jobs stay serial to avoid the
// per-call thread spawn cost.
template<typename Fn>
inline void parallel_for(std::size_t count, std::size_t work_per_item, Fn&& fn) {
    constexpr std::size_t kMinWorkPerThread = std::size_t(1) << 17;
    if (count == 0) {
        return;
    }
    const std::size_t hardware =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
    const std::size_t total_work = count * std::max<std::size_t>(work_per_item, 1);
    std::size_t threads = std::min<std::size_t>(hardware, std::max<std::size_t>(1, total_work / kMinWorkPerThread));
    threads = std::min(threads, count);
    if (threads <= 1) {
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    const std::size_t per_thread = (count + threads - 1) / threads;
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (std::size_t t = 1; t < threads; ++t) {
        const std::size_t begin = t * per_thread;
        if (begin >= count) {
            break;
        }
        const std::size_t end = std::min(count, begin + per_thread);
        pool.emplace_back([&fn, begin, end] {
            for (std::size_t i = begin; i < end; ++i) {
                fn(i);
            }
        });
    }
    const std::size_t first_end = std::min(count, per_thread);
    for (std::size_t i = 0; i < first_end; ++i) {
        fn(i);
    }
    for (std::thread& thread : pool) {
        thread.join();
    }
}

// Conv1d with dilation and groups (groups == in_channels is the depthwise case
// used by DDSConv). Padding follows the reference: ((kernel-1)*dilation)/2.
[[nodiscard]] Act conv1d(const Act& input, const Weight& weight, const Weight& bias, std::size_t kernel,
                         std::size_t pad, std::size_t dilation = 1, std::size_t groups = 1, std::size_t stride = 1);

[[nodiscard]] Act layer_norm(const Act& input, const Weight& gamma, const Weight& beta, float eps = kLayerNormEps);

// LayerNorm without the affine step (AdaIN applies its own scale/shift).
[[nodiscard]] Act layer_norm_unit(const Act& input, float eps);

void apply_mask(Act& act, const std::vector<float>& mask);

void relu_in_place(Act& act);

// Piper's LayerNorm is exported as an explicit decomposition (Transpose ->
// ReduceMean over the channel axis -> Sub -> Pow(2) -> ReduceMean -> Add(eps)
// -> Sqrt -> Div -> Mul(gamma) -> Add(beta)), so it must be evaluated in that
// order rather than through the MLAS LayerNorm kernel's packed affine path.
//
// ONNX Runtime's graph optimiser rewrites the leading Transpose away: the
// ReduceMean runs with axes=[1] on the channel-major [1, C, T] tensor, so
// NoTransposeReduce1Loop walks the reduced axis with the scalar
// ReduceAggregatorMean::update() chain (one float add per channel, stride T)
// and divides once by the float count. Verified against the pinned reference:
// 2304/2304 for layer 0 norm_1 (Eigen's packet redux gives only 702/2304).
[[nodiscard]] Act piper_norm(const Act& input, const Weight& gamma, const Weight& beta, bool with_beta = true);

[[nodiscard]] Act gelu(const Act& input);

// modules.DDSConv: dilated depthwise conv -> LayerNorm -> gelu -> 1x1 conv ->
// LayerNorm -> gelu, with a residual add per layer.
[[nodiscard]] Act dds_conv(const Act& input, const Weights& weights, const std::string& prefix, const Act* cond,
                           const std::vector<float>& mask, std::size_t kernel, const Reference* ref = nullptr);

[[nodiscard]] Act conv_transpose1d(const Act& input, const Weight& weight, const Weight& bias, std::size_t stride,
                                   std::size_t kernel, std::size_t pad);

void leaky_relu(Act& act, float slope);

// modules.WN: tanh * sigmoid gated dilated convolutions with residual skips.
[[nodiscard]] Act wave_net(const Act& input, const Weights& weights, const std::string& prefix,
                           const std::vector<float>& mask, std::size_t layers, std::size_t kernel);

// modules.ResBlock2: two dilated convs with a residual add each.
[[nodiscard]] Act res_block2(const Act& input, const Weights& weights, const std::string& prefix, std::size_t kernel,
                             std::size_t d0, std::size_t d1, const Reference* ref = nullptr,
                             const std::string& label = {});

// y = x^T W + b, W is (in, out) as exported by MatMul(x, W).
[[nodiscard]] Act linear(const Act& input, const Weight& weight, const Weight* bias);

// Some exporters name a conv bias after the layer path and its weight after the
// node path; accept either spelling.
[[nodiscard]] const Weight& weight_or(const Weights& weights, const std::string& first, const std::string& second);

[[nodiscard]] Act style_act_of(const std::vector<float>& style_values);

// AdaIN1d: instance normalisation over the time axis (the F0/N blocks are
// channel-major [C][T], and the exported ReduceMean runs over axis 2 of that
// view), then the style scale/shift.
// ReduceMean's accumulation order follows the memory layout the optimizer
// leaves behind:
//  - a contiguous row uses Eigen's packet redux (Redux.h,
//    LinearVectorizedTraversal, NoUnrolling): two packet accumulators stepped
//    by 8 elements, combined, plus the trailing packet, then
//    predux = (l0+l2)+(l1+l3);
//  - the F0.0/N.0 norm1 reductions consume a fused Transpose view, so ONNX
//    Runtime walks the strided axis one element at a time.
[[nodiscard]] float eigen_redux(const float* data, std::size_t n);

[[nodiscard]] float reduce_mean_scalar(const float* data, std::size_t n);

[[nodiscard]] Act adain_norm(const Act& x, const Weight& fc_w, const Weight& fc_b, const Act& style_act,
                             bool strided_input);

[[nodiscard]] Act nearest_upsample(const Act& x, std::size_t factor);

// Depthwise ConvTranspose1d (the AdaIN blocks' "pool"): weight (C, 1, k),
// stride 2, kernel 3, padding 1, output_padding 1.
[[nodiscard]] Act depthwise_transpose(const Act& x, const Weight& weight, const Weight& bias, std::size_t stride,
                                      std::size_t pad);

}  // namespace mnet::tts
