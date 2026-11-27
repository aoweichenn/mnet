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
