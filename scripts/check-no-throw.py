#!/usr/bin/env python3
"""库代码不得用异常报告失败（REQ-012 的模板版）。

src/ 与 include/ 里 throw = 0；catch 也不允许——需要兜住 OOM 时再按
mzip 的 scripts/check-no-throw.py 加白名单与理由。
"""

import argparse
import re
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent


def strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args(argv)

    problems = []
    files = list(LIB.glob("src/**/*.cpp")) + list(LIB.glob("src/**/*.hpp")) + list(LIB.glob("include/**/*.hpp"))
    for path in sorted(files):
        raw_text = path.read_text(encoding="utf-8")
        raw_lines = raw_text.split("\n")
        body = strip_comments(raw_text)
        for number, line in enumerate(body.split("\n"), 1):
            # 唯一的白名单：内存预算的 OOM 保护，行尾写 `check-no-throw: allow <理由>`。
            if re.search(r"\bthrow\b", line) and number <= len(raw_lines):
                if "check-no-throw: allow" in raw_lines[number - 1]:
                    continue
                if number >= 2 and "check-no-throw: allow" in raw_lines[number - 2]:
                    continue
                problems.append(f"{path.relative_to(LIB)}:{number}: throw")
            if re.search(r"\bcatch\s*\(", line):
                problems.append(f"{path.relative_to(LIB)}:{number}: catch")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print("CHECK OK: no throw / catch")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
