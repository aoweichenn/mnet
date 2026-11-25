#!/usr/bin/env python3
"""编译日志零告警检查。

只统计自有代码（src/zip, include/mnet, tests, benchmarks）的告警；
第三方（_deps/、zlib-ng、GTest、benchmark）不计入基线。

用法：
    python3 scripts/check-warnings.py <build.log>
    python3 scripts/check-warnings.py <build.log> --allow <allowlist.txt>
"""

import argparse
import re
import sys

# 责任范围：只有这些前缀下的告警算我们的。
OWNED_PREFIXES = (
    "/src/",
    "/include/mnet/",
    "/tests/",
    "/benchmarks/",
)

# 明确不负责的第三方来源。
EXTERNAL_MARKERS = (
    "_deps/",
    "zlib-ng",
    "/usr/include/",
    "/usr/lib64/",
    "googletest",
    "benchmark/include",
)

WARNING_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): warning: (?P<msg>.*)$")


def is_owned(path: str) -> bool:
    if any(marker in path for marker in EXTERNAL_MARKERS):
        return False
    return any(prefix in path for prefix in OWNED_PREFIXES)


def load_allowlist(path: str) -> list[str]:
    if path is None:
        return []
    entries = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--allow", default=None)
    args = parser.parse_args(argv)

    allowlist = load_allowlist(args.allow)
    owned: list[str] = []
    external = 0

    with open(args.log, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = WARNING_RE.match(raw)
            if match is None:
                continue
            path = match.group("path")
            if is_owned(path):
                owned.append(raw.rstrip())
            else:
                external += 1

    # **按内容匹配，不按数量匹配。**
    #
    # 早先的写法是 `unexpected = len(owned) - len(allowlist)`：只比条数。
    # 于是"2 条旧告警被修掉、又冒出 2 条新告警"会得到 0 → 静默通过。
    # 允许清单的意义是"这几条具体的告警我知道且接受"，必须逐条对上。
    compiled = [re.compile(entry) for entry in allowlist]
    used = [False] * len(compiled)
    unexpected: list[str] = []
    for line in owned:
        hit = False
        for index, pattern in enumerate(compiled):
            if pattern.search(line):
                used[index] = True
                hit = True
                break
        if not hit:
            unexpected.append(line)

    # 反向：清单里没有匹配任何告警的条目 = 过期条目，也要报出来。
    # 否则修掉一条旧告警之后，它的清单条目会永远留在那里，下次同名告警
    # 出现时被"合法"吞掉。
    stale = [allowlist[index] for index, was_used in enumerate(used) if not was_used]

    print(f"自有代码告警: {len(owned)}")
    print(f"第三方告警（不计入基线）: {external}")
    print(f"允许清单条目: {len(allowlist)}（命中 {sum(used)}）")

    if unexpected:
        print("\n未登记的告警：", file=sys.stderr)
        for line in unexpected[:20]:
            print(f"  {line}", file=sys.stderr)
        print(f"\n需要修掉 {len(unexpected)} 条，或写进允许清单并说明理由。", file=sys.stderr)
        return 1

    if stale:
        print("\n清单里有未命中的过期条目（对应告警已被修掉？）：", file=sys.stderr)
        for entry in stale:
            print(f"  {entry}", file=sys.stderr)
        print("\n请删掉这些条目，否则它们会掩盖将来同名的告警。", file=sys.stderr)
        return 1

    print("零告警 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
