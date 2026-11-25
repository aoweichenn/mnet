#!/usr/bin/env bash
# 构建 + 零告警。
set -uo pipefail
lib="$(cd "$(dirname "$0")/.." && pwd)"
build_root="${CPPLIB_BUILD_ROOT:-/tmp/mnet_build}"
build_dir="$build_root/test"
compiler="gcc"
[ "${1:-}" = "--clang" ] && compiler="clang"

mkdir -p "$build_root"
args=(-S "$lib" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug -Dmnet_build_tests=ON)
if [ "$compiler" = "clang" ]; then
    args+=(-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang)
fi

log="$build_root/build-$compiler.log"
echo "== configure ($compiler) =="
cmake "${args[@]}" > "$log" 2>&1 || { tail -20 "$log"; exit 1; }
echo "== build =="
cmake --build "$build_dir" -j"$(nproc)" >> "$log" 2>&1 || { grep -E "error" "$log" | head -20; exit 1; }
echo "== warnings =="
python3 "$(dirname "$0")/check-warnings.py" "$log"
