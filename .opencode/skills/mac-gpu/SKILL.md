---
name: Mac GPU Work
description: Default host for all GPU/Metal exploration in mnet — the remote Apple Silicon Mac (M4 Pro, 20-core GPU, Metal 4). Load before any GPU planning, benchmarking or backend work.
---

# Mac GPU work

All GPU experiments for mnet run on the user's **remote Mac** (`ssh mac`). The
local Linux host is WSL2 on a Windows box with an RTX 5070 Ti; its numbers are
reference material only and must not be used as the target or as validation
for the mnet GPU backend.

## Host

- Apple M4 Pro, 20-core GPU, Metal 4, 48 GB unified memory, macOS 27.0.1.
- Unified memory means CPU and GPU share the same 48 GB; there is no separate
  VRAM budget and no host->device copy for the weights.
- Toolchain: CommandLineTools only. `cmake`/`git`/`python3` from Homebrew,
  `clang++` from `/usr/bin`. There is **no** `metal` compiler and no full
  Xcode, so build Metal kernels with the CLT clang + `-framework Metal` and
  runtime-compiled `.metal` sources, or use prebuilt binaries.
- SSH: `ssh -o BatchMode=yes mac ...`, same rules as the
  `remote-macos-release` skill (no host-key workarounds, no stored secrets).

## Workspace

- Source tree: `~/mnet_p4/mnet`; build dir `~/mnet_p4/mnet/build-p4`.
- Models: `~/mnet_p4/models` (Bonsai Q2_0, Inkling2 IQ1_S — mnet-only formats).
- GPU scratch: `~/mnet_p4/gpu` (llama.cpp prebuilt, small HF models,
  benchmark logs). Remove only files this task created.
- Build command used so far:
  `cmake -S . -B build-p4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++ -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -Dmnet_use_simple_omp=ON -Dmnet_build_tests=OFF`

## Benchmarking method

- `llama-bench` from the llama.cpp macOS arm64 release is the reference
  harness: CPU run `-ngl 0`, Metal run `-ngl 99`, same model, `-p 512 -n 128`,
  at least 3 reps. Record pp512 and tg128.
- Use the same Qwen2.5 0.5B/1.5B Q4_K_M GGUFs as the WSL reference run so the
  numbers are directly comparable.
- Always report the exact model, backend, and build string.

## Measured reference (llama.cpp b12810, Qwen2.5 Q4_K_M, macOS arm64)

llama-bench, `-p 512 -n 128`, 3-5 reps:

| model | backend | pp512 | tg128 |
|---|---|---:|---:|
| 0.5B (463 MiB) | CPU (`-ngl 0`) | 2,003 t/s | 317 t/s |
| 0.5B | Metal (`-dev MTL0`) | 7,390 t/s | 290 t/s |
| 1.5B (1.04 GiB) | CPU (`-ngl 0`) | 754 t/s | 166.8 t/s |
| 1.5B | Metal (`-dev MTL0`) | 2,264 t/s | 169.5 t/s |

Reading: Metal wins on prefill (3.0-3.7x) but decode is a tie for models this
small, because the M4 Pro CPU and GPU share the same ~273 GB/s unified memory
and Metal launch overhead eats the rest. Decode only becomes GPU-favoured when
the kernel is compute-bound (which is exactly the case for mnet's ternary
Q2_0 GEMV: 5.8 GB/s effective on the CPU today).

Quirk: this llama-bench build prints `ngl 10` in every row even with
`-ngl 0`; distinguish CPU from GPU by the timings and the `-dev` column.

## mnet Metal backend (first pilot)

- `mnet_enable_metal=OFF` by default; on Apple hosts configure with
  `-Dmnet_enable_metal=ON` and `DEVELOPER_DIR=/Library/Developer/CommandLineTools`
  (the Xcode shim otherwise trips the unaccepted-license error; CLT clang++
  works directly).
- Layering: `include/mnet/detail/compute.hpp` (contract) +
  `src/compute/backend.cpp` (CPU loop + registry) +
  `src/metal/backend.mm` (provider). Model code calls
  `compute::matvec_q8_0(...)` and never mentions Metal.
- The Q2_0 kernel mirrors the CPU accumulation order, so results are
  bit-identical; Bonsai dumps matched 2371/2371.
- `MNET_METAL_DEBUG=1` prints whether the backend initialised.
- Metal pitfalls found the hard way:
  * shaders compile with **fast-math on** by default: set
    `MTLCompileOptions.fastMathEnabled = NO` or the Q2_0 kernel loses the last
    ulps (48/2304 rows) and a 64-layer model diverges completely.
  * never key device buffers by host pointer: `Quantized` objects are loaded
    and released, addresses get reused. `Reader::Quantized::identity`
    (reader, tensor offset, expert) is the cache key.
- Two kernels: the exact one (CPU accumulation order, bit-identical) is the
  default; `MNET_METAL_FAST=1` selects the 32-lane/row + simd_sum kernel
  (llama.cpp mul_mv shape), which is ~25% faster but only tolerance-close.
- `Reader::matvec_batch` fuses independent matvecs that share an activation
  (ffn gate/up, qkv+gate, q/k/v) into one command buffer and one wait.
- `Reader::ffn_swiglu` (fused FFN) and `Reader::attention` (fused attention)
  each collapse a whole block into one submission. Both are on by default with
  `MNET_METAL_FAST=1`; `MNET_NO_FUSED_FFN=1` / `MNET_NO_FUSED_ATTN=1` select
  the step-by-step block for A/B runs, and `--dump` always uses the steps so
  the reference dump stays complete.
- Attention block kernels: `attn_head_prep` (one head per threadgroup: f32
  RMSNorm with a simdgroup sum + RoPE + optional f16 round on store),
  `attn_gate_sigmoid`, `attn_store_half` and `attn_flash` (one simdgroup per
  head, online softmax, value accumulator in registers, `simd_sum` score).
- Measured Bonsai 27B Q2_0, 32 tokens: CPU 167.7 s (5.24 s/token); Metal
  exact 8.23 s (0.26 s/token, 2371/2371 identical); Metal fast 6.47 s
  (0.20 s/token, top-5 unchanged, max rel 2e-2); fast + fused FFN + fused
  attention 5.91-5.99 s.
- Build dirs: `build-metal` (Metal ON), `build-cpu` (CPU reference) in
  `~/mnet_p4/mnet`. Dumps: `~/mnet_p4/{bonsai_cpu,bonsai_gpu}`.

## Metal without Xcode

`xcrun -f metal` is missing (CLT only). Compile MSL at runtime with
`newLibraryWithSource:` (Metal.framework), or use `metal` only if the user
installs full Xcode. No admin installs.

## Rules

- Do not install anything that needs admin rights; keep everything under the
  user's home directory.
- Never claim a GPU result from the Linux host as a macOS result.
- Clean up task-owned temporary files (downloaded archives, scratch dumps)
  when done; leave the mnet sources, models and build dir alone.
