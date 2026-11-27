# Device-resident layer fusion (design note)

Status: not implemented. This note exists so the work can be picked up without
re-deriving the measurements behind it.

## Why

The warm 32-token forward is 463 ms; llama.cpp reaches 297 ms on the same
machine and model. Of the 463 ms, 397 ms is device busy time (86%), measured
from `GPUEndTime - GPUStartTime` over every command buffer. The remaining 66 ms
is not GPU work, and it decomposes as:

| item | per forward |
|---|---|
| in-block CPU work (SSM 14, host Q8 9, copies 2, FFN encode 0.6) | 27 ms |
| gaps between submissions (per-layer sync + tail uploads) | ~40 ms |

Only the second row is recoverable by fusion, so the ceiling is ~423 ms
(1.42x from llama.cpp, against 1.56x today). Kernel-level changes cannot touch
either row: the mat-mat is at 4.6-4.8 TFLOP/s against a 5.9 static-tile ceiling
and ten staging variants have been measured and rejected (see the log sections
120-158 referenced below).

## What to change

One command buffer per (layer, token batch), with no upload or wait between the
blocks:

```
branch block (SSM or attention)      -> writes to a device buffer
residual add                          -> device kernel, x += branch
layer_prep_f32                        -> device: RMSNorm + Hadamard + Q8_0
FFN (gate/up -> swiglu -> hadamard -> down) -> reads the device Q8_0 codes
residual add                          -> device: x += ffn
one read-back of x at the end
```

The pieces exist and are verified:

- `compute::layer_prep_f32` (RMSNorm + signed Hadamard + Q8_0) reproduces the
  host's Q8_0 codes exactly (0 of 5120 differ), so the FFN's input codes are
  unchanged;
- `compute::residual_add` is bit-identical to the host loop;
- a fused *tail* (residual + prep + FFN in one submission) was implemented and
  produced bit-identical logits, but measured neutral on its own because the
  branch block still ended with a wait and the tail still began with an upload.

The missing piece is therefore the *device hand-off between the three blocks*.

## API sketch

The compute layer is host-pointer based today; the change is to let a block
write its result into, and read its activation from, an opaque device handle:

```cpp
struct LayerFused {
    void* residual;        // device buffer, n_embd floats, in/out
    void* branch_out;      // device buffer the branch block writes to
    // branch job (attention or ssm), then FFN parameters
};
bool layer_fused(const LayerFused& job);   // one submission, one read-back
```

Both the binding and the kernel variant must be chosen together inside the
backend (the `TiledJob`/`TiledPlan` pattern), which is what broke the default
path twice before.

## Acceptance

Only the whole-model number counts. Block-level counters have failed to predict
the forward four times (32-row tile, split-K, device tail, warp SSM norm), so:

- `--repeat 3` warm forward must be < 463 ms (target ~423 ms);
- top-5 tie rule, exact path 0 of 248320, `--dump` 2371/2371 unchanged;
- `ctest` 185/185 and the lint gates green.

## Measured context (this machine, `~/mnet_p4`)

```
mat-mat, 32 tokens, TFLOP/s:   4.81 hot   5.92 static   (llama reaches 5.8 whole)
mat-mat, 64 tokens, TFLOP/s:   5.44 hot   6.42 static
weight stream roofline:        215 GB/s    (model aggregate 16.6 GB/s)
FFN block per forward:         0.236 s     SSM 0.160     attention 0.084
generation:                    11.3 tok/s against 23.3
```
