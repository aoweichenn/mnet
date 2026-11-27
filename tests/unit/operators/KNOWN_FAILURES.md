# Operator defects

Validated on the Apple Silicon host with AppleClang 21, a Release build and
`-Dmnet_use_simple_omp=ON`:

```sh
cmake --build build --parallel 1
ctest --test-dir build --parallel 1 --output-on-failure
```

Result: 185 test entries, 100% passed, 0 build warnings. Two entries are
explicitly skipped because they need the SqueezeNet model file
(`MNET_SQUEEZENET_DIR`), and the seven end-to-end cases in `mnet_model_tests`
skip without their model assets.

## Fixed

### ArgMax counted alignment padding as input

`forward` used `Mat::total()`, which includes per-channel alignment padding, so
both `topk` validation and flattened indices ranged over storage slots. A
`topk` that only fit in padding was accepted and could return an index with no
logical element behind it.

The element count and the index base now come from `w * h * d * c`, and values
are read per channel, so padding is never visited. Ties keep the lower index.

Tests: `ArgMax.IgnoresAlignmentPaddingBetweenChannels` fills padding with a
large sentinel and requires it to be invisible; `ArgMax.MatchesOnnxTopKAcrossInputRanks`
covers ranks 1-4 and `ArgMax.KeepsTheFirstIndexForTies` pins the tie rule.

### SPP mirrored into the wrong offset

The mirror copy used `data + pyramid_num_bins`, but the second output channel
starts at `cstep`. Alignment padding sits between the two, so the mirror plane
was partly written and partly stale.

The copy now targets `data + cstep * sizeof(float)`. Tests:
`SPP.MatchesCaffeBinsOnThreeByFiveMap` and
`SPP.MatchesCaffeBinsOnFourByFourMapWithThreeLevels`. Expected values were
produced by an independent script implementing the documented Caffe/ncnn bin
geometry and `count_include_pad` average, not by this layer.

### Scalar im2col skipped full output blocks

In `convolution_im2col`, the `np == kOutputBlock` fast path contains only AVX2
and NEON code. On a scalar build the whole-block case entered that branch, ran
no arithmetic, and kept the bias initialisation. Observed on x86-64 with AVX2
disabled: max absolute error 0.77, 0.82 and 1.00 on the three 3x3 regression
cases.

The fast path is now compiled only when AVX2 or NEON exists, so scalar builds
fall through to the general loop, which handles every `np`. Verified on NEON
(the fast path still runs and all convolution tests pass); the scalar branch is
changed by inspection only, because this pass was validated on the Mac.

## Open, deliberately not changed here

### SPP is wrong for more than one input channel

`forward` allocates `top_blob` as `(pyramid_num_bins, 1, 2)` but writes
`channels * pyramid_num_bins` values per level. With `channels > 1` the writes
run past the first channel's `cstep` region into the mirror plane, so the output
is corrupted and a custom allocator can see an undersized buffer. Fixing this
changes the layer's output shape, so the suite exercises SPP only with
`channels == 1` and the gap is recorded instead of hidden.

Average pooling divides by the full kernel area rather than the number of valid
samples, so a mostly-padded bin is scaled down by that padding. This matches the
ncnn reference and the expected values encode it; it is a behaviour to know when
reading the tests, not a defect.

### Slice does not validate parameter indices

`Slice` accepts oversized or far-negative `indices`/`slices` values and produces
a zero-length or over-long slice instead of failing. The layer has no rejection
contract yet, so no negative test is asserted.

### Eltwise, Noop, Split, Concat trust their input count

`Eltwise` indexes `bottom_blobs[1]` unconditionally; a one-input graph
dereferences past the end. `Noop` and `Split` index `bottom_blobs[0]` with the
same assumption. Guards should be added with the fix; tests assert only the
supported input counts.
