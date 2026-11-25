#!/usr/bin/env bash
set -uo pipefail
lib="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
run() {
    local name="$1"; shift
    printf '\n== %s ==\n' "$name"
    "$@" || { printf '!! %s 失败\n' "$name"; fail=1; }
}

run "格式"       "$lib/scripts/format.sh" --check
run "类型"       python3 "$lib/scripts/check-types.py" --check "$lib"
run "头文件"     python3 "$lib/scripts/check-headers.py" --check "$lib"
run "无直接输出" python3 "$lib/scripts/check-no-printf.py" --check "$lib"
run "不抛异常"   python3 "$lib/scripts/check-no-throw.py" --check
run "nodiscard"  python3 "$lib/scripts/check-nodiscard.py" --check
run "用例断言"   python3 "$lib/scripts/check-test-assertions.py" --check
run "构建零告警" "$lib/scripts/build.sh"
run "安装包"     "$lib/scripts/package-test.sh"

# Optional numeric parity against an upstream ncnn build. Off by default so
# the gate stays usable on machines without ncnn.
if [ "${MNET_NCNN_PARITY:-0}" = "1" ] && [ -x "$lib/../../mcnn/tools/parity/run.sh" ]; then
    run "ncnn parity" "$lib/../../mcnn/tools/parity/run.sh"
elif [ "${MNET_NCNN_PARITY:-0}" = "1" ]; then
    printf '\n== ncnn parity ==\nSKIP: mcnn/tools/parity/run.sh not found\n'
fi

printf '\n== 测试 ==\n'
ctest --test-dir "${CPPLIB_BUILD_ROOT:-/tmp/mnet_build}/test" --output-on-failure || fail=1

printf '\n'
[ "$fail" -eq 0 ] && echo "全部通过" || echo "存在失败项"
exit "$fail"
