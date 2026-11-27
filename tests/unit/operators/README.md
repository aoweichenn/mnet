# Operator tests

The directories mirror operator families. CTest discovers each GoogleTest case
separately. The `operators` label runs all families; `operators.<family>` selects
one family.

```sh
cmake --build build --parallel 1
ctest --test-dir build --output-on-failure -L '^operators$'
ctest --test-dir build --output-on-failure -L '^operators\.shape$'
```

When adding a registered operator, put its tests in the relevant family and
list any new source explicitly in `CMakeLists.txt`. Prefer deterministic inputs
and independently calculated expected values. Assert every output value and the
output type, and test invalid inputs where the operator defines a failure
contract.

`tests/support/operator_test.hpp` provides two fixture styles:

- `tensor({rank, width, height, depth, channels}, {...})` builds a `Mat`
  directly with the rank spelled out, which is what most expected outputs use.
- `ReferenceTensor{shape, values}` is for computed expectations. Its `shape` is
  in `Mat` storage order — `[w]`, `[h, w]`, `[c, h, w]` or `[c, d, h, w]` — with
  dense channel-major values. `convert()` and `expect_tensor()` use that same
  order, so no rank or axis inference happens anywhere.

Both compare logical elements only; per-channel alignment padding is never read
or asserted. See [operator fixes and open defects](KNOWN_FAILURES.md) before
adding a case for `SPP`, `Slice`, `Eltwise`, or another layer with a documented
gap.

Model-asset tests are optional; the small numeric fixtures require no model
downloads. Timing benchmarks are not part of CTest. Build them explicitly with
`cmake --build build --target mnet_operator_benchmarks --parallel 1`.

See [operator fixes and open defects](KNOWN_FAILURES.md) for the resolved
defects and the gaps that are documented but deliberately not changed in this
test-only change.
