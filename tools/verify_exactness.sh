#!/usr/bin/env bash
# Run mnet_tts against every pinned reference and report whether each stage and
# the decoded waveform are bit-exact. Exits non-zero on the first regression.
#
# Usage: tools/verify_exactness.sh [build-dir]
#   MODEL_ROOT   directory holding the model trees (default ~/.cache/models)
#   EXTRA_CASES  directory with the augmented references (default MODEL_ROOT/extra)
set -u

build_dir="${1:-build-avx2}"
bin="$build_dir/mnet_tts"
models="${MODEL_ROOT:-$HOME/.cache/models}"
extra="${EXTRA_CASES:-$models/extra}"
failures=0

if [[ ! -x "$bin" ]]; then
    echo "missing $bin (build it first)" >&2
    exit 2
fi

# stage_count output -> "stages=N non-exact=M" plus the waveform verdict
report() {
    local label="$1"
    shift
    local out
    out=$("$@" 2>&1)
    local stages bad waveform
    stages=$(printf '%s\n' "$out" | grep -c 'max_abs')
    bad=$(printf '%s\n' "$out" | grep 'max_abs' | grep -vcF '0.000e+00')
    waveform=$(printf '%s\n' "$out" | grep -E 'waveform( vs ref)?' | grep -F '0.000e+00' | wc -l)
    if [[ "$bad" -eq 0 && "$waveform" -ge 1 && "$stages" -gt 0 ]]; then
        printf '  PASS  %-28s stages=%-3s\n' "$label" "$stages"
    else
        printf '  FAIL  %-28s stages=%-3s non-exact=%s\n' "$label" "$stages" "$bad"
        printf '%s\n' "$out" | grep 'max_abs' | grep -vF '0.000e+00' | head -4 | sed 's/^/          /'
        failures=$((failures + 1))
    fi
}

echo "Piper (VITS)"
piper="$models/piper_lessac"
if [[ -d "$piper/ref" ]]; then
    report "pinned 12 ids" "$bin" --model "$piper" --ref "$piper/ref" --noise-scale 0 --noise-w 0
fi
for case in 13 A B; do
    ref="$extra/ref$case"
    [[ -d "$ref" ]] || continue
    ids_file="$extra/piper_ids$case.bin"
    [[ -f "$ids_file" ]] || continue
    ids=$(python3 -c "import sys,numpy as np; print(','.join(str(int(v)) for v in np.fromfile(sys.argv[1],dtype=np.float32)))" "$ids_file")
    report "extra $case ids" "$bin" --model "$piper" --ref "$ref" --ids "$ids" --noise-scale 0 --noise-w 0
done

echo "Kokoro (StyleTTS2)"
kokoro="$models/kokoro"
kokoro_ids="43,60,66,73,100,110,50,82,91,33,57,70,20"
if [[ -d "$kokoro/ref" ]]; then
    report "pinned 13 ids" "$bin" --model-type kokoro --model "$kokoro" --ref "$kokoro/ref" \
        --ids "$kokoro_ids" --style "$kokoro/style13.bin"
fi
if [[ -d "$extra/refC" ]]; then
    ids=$(grep '^ids ' "$extra/refC/manifest.txt" | cut -d' ' -f3- | tr 'x' ',')
    report "extra 40 ids" "$bin" --model-type kokoro --model "$kokoro" --ref "$extra/refC" \
        --ids "$ids" --style "$kokoro/style13.bin"
fi
if [[ -d "$extra/refD" && -f "$extra/style_row7.bin" ]]; then
    report "extra style row 7" "$bin" --model-type kokoro --model "$kokoro" --ref "$extra/refD" \
        --ids "$kokoro_ids" --style "$extra/style_row7.bin"
fi
if [[ -d "$extra/refE" ]]; then
    report "extra speed 0.85" "$bin" --model-type kokoro --model "$kokoro" --ref "$extra/refE" \
        --ids "$kokoro_ids" --style "$kokoro/style13.bin" --speed 0.85
fi

if [[ "$failures" -ne 0 ]]; then
    echo "FAILED ($failures case(s))" >&2
    exit 1
fi
echo "all exact"
