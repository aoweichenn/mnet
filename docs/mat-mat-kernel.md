# The Q2_0 mat-mat kernel: what llama.cpp does differently

The prefill path spends about 60% of its time in `q2_0_mat_mat_f16`
(`src/metal/backend.mm`). This note records the one measurement that was still
missing: llama.cpp's own Q2_0 mat-mat, compiled from its source and run on this
model's shapes next to ours.

## Method

`benchmarks/metal/port_llama_mul_mm.py` assembles llama.cpp's mat-mat into a
self-contained Objective-C++ file: it inlines `common.h`, the Q2_0
dequantiser and the plain `kernel_mul_mm`, drops every instantiation the model
never takes (f16/f32 weights, bf16, the expert-mapping variants) and replaces
the `FC_mul_mm_*` function constants with the literals a single non-broadcast
matmul uses. The host side mirrors llama's launch exactly - 64 rows and 32
tokens per threadgroup, 4 simdgroups of 32 lanes, 12288 bytes of threadgroup
memory, grid X = token tiles and Y = row tiles.

Each variant is checked against a scalar reference over the first weight row
before it is timed, so a kernel that runs but indexes the wrong layout is not
mistaken for a fast one. (An earlier revision of the harness read the result
from the host vector instead of the `MTLBuffer`; the check caught it.)

## Result

TFLOP/s, 32 tokens, warm:

| shape | llama `_f32` (half acc.) | llama `_prec_f32` | `q2_0_mat_mat_f16` |
| --- | --- | --- | --- |
| 17408x5120 (gate/up) | 2.94 | 4.99 | 4.87 |
| 5120x17408 (down) | 4.78 | 4.84 | 4.24 |
| 5120x6144 (ssm/attn) | 3.94 | 3.96 | ~4.0 |

The port does not reach the 5.4-5.8 TFLOP/s that llama.cpp's whole-forward
number implies, so its advantage is not a decode or staging implementation
detail with a known fix. It is 2% on the gate/up shape and 12% on the down
shape - the shape where our kernel loses the most.

## What was adopted

The down projection now runs llama's kernel (separate Metal library, compiled on
first use, `MNET_MAT_MAT_LLAMA=0` to disable):

    warm 32-token forward, --repeat 3       425.5 ms -> 416.0 ms   (-2.2%)
    normalised logit deviation from CPU     4.704e-3 -> 3.959e-3

Using it for all three projections is faster again (408.8 ms) but takes the
deviation to 5.301e-3, past the fast path's bound of 4.754e-3, so that stays
opt-in (`MNET_MAT_MAT_LLAMA=1`). The exact path never reaches this code: it uses
`q2_0_matvec`, and is still 0 of 248320 elements different from the CPU.

## Why fp32 tiles are not the explanation

`MNET_MAT_MAT_F16=0` runs the *same* kernel with fp32 weight tiles instead of
fp16 and takes the forward from 425.8 ms to 910.4 ms, so the staging width is
not where llama's 12% on the down shape comes from either.

## Attribution

`kLlamaMatMatSource` contains llama.cpp code: MIT licence, Copyright (c) 2023-2024
The ggml authors (https://github.com/ggml-org/llama.cpp).
