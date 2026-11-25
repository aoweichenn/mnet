#!/usr/bin/env bash
# clang-format 检查/修复。
set -uo pipefail
lib="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"
files=$(find "$lib/include" "$lib/src" "$lib/tests" -name '*.hpp' -o -name '*.cpp')

if [ "$mode" = "--check" ]; then
    if ! clang-format --dry-run --Werror $files > /dev/null 2>&1; then
        echo "FAIL 格式不干净；运行 scripts/format.sh 修复" >&2
        exit 1
    fi
    echo "CHECK OK: formatting clean"
else
    clang-format -i $files
    echo "formatting applied"
fi
