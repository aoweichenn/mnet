#!/usr/bin/env bash
# 安装到干净前缀，用源码树外的消费方 find_package 并编译。
set -uo pipefail
lib="$(cd "$(dirname "$0")/.." && pwd)"
build_root="${CPPLIB_BUILD_ROOT:-/tmp/mnet_build}"
build_dir="$build_root/test"
prefix="$build_root/install"
consumer_build="$build_root/consumer"

rm -rf "$prefix" "$consumer_build"
cmake --install "$build_dir" --prefix "$prefix" > /dev/null || exit 1
cmake -S "$lib/tests/package/consumer" -B "$consumer_build" -G Ninja \
      -DCMAKE_PREFIX_PATH="$prefix" > /dev/null || exit 1
cmake --build "$consumer_build" -j"$(nproc)" > /dev/null || exit 1
"$consumer_build/mnet_consumer"
echo "CHECK OK: consumer find_package + link"
