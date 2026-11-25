#!/usr/bin/env python3
"""REQ-015：src/ 下不得直接输出。

禁止 printf / fprintf / std::cout / std::cerr / puts / fputs。
日志必须经 detail/txn_log.hpp（内部）或 mnet::init_logging（公共）。

用法: python3 scripts/check-no-printf.py --check <lib_dir>
"""

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(
    r'\b(printf|fprintf|vprintf|vfprintf|puts|fputs|putchar|fputc)\s*\(|'
    r'\bstd::(cout|cerr|clog)\b')

# 允许的例外：错误码名称表不输出；测试与工具不受此限制。
SKIP_DIRS = ("/tests/", "/benchmarks/", "/tools/")


def strip_comments(text: str) -> str:
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    problems = []

    for path in sorted(root.glob("src/**/*.cpp")) + sorted(root.glob("src/**/*.hpp")):
        rel = path.relative_to(root).as_posix()
        if any(marker.strip("/") in rel for marker in SKIP_DIRS):
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for number, line in enumerate(text.split("\n"), 1):
            for match in FORBIDDEN.finditer(line):
                problems.append(f"{rel}:{number}: 禁止直接输出 '{match.group(0)}'，改用 detail/txn_log.hpp")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n共 {len(problems)} 处", file=sys.stderr)
        return 1

    print("CHECK OK: no direct output in src/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
