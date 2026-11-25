#!/usr/bin/env python3
"""禁止平台相关内建类型（全局 cpp-conventions §5）。

检查 include/ src/ tests/ tools/ benchmarks/ 下的 .hpp/.cpp。
注释与字符串/字符字面量先剥离，避免 `"int"` 或 `// int` 误报。
唯一例外：语言/ABI 规定的签名 `int main(...)`、`int LLVMFuzzerTestOneInput(...)`，
以及在行尾写 `// check-types: allow <原因>` 的 ABI 强制签名。

用法: python3 scripts/check-types.py --check <lib_dir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 宽度/符号随平台变化的类型。`(?<![\w:.])` 放行 `std::int32_t` / `std::size_t`。
BANNED = re.compile(
    r"(?<![\w:.])"
    r"(?:unsigned|signed|short|long|int|size_t|ptrdiff_t|intptr_t|uintptr_t|u?int(?:8|16|32|64)_t)"
    r"\b"
)

EXEMPT_SIGNATURE = re.compile(r"\b(?:int\s+main|int\s+LLVMFuzzerTestOneInput)\s*\(")

# 确有 ABI/OS 强制签名的场合，在同一行写 `// check-types: allow <原因>`。
ALLOW_MARKER = "check-types: allow"

SCAN_DIRS = ("include", "src", "tests", "tools", "benchmarks")
SKIP_PARTS = ("third_party", "build", "_deps")


def strip_literals_and_comments(text: str) -> str:
    """把注释、字符串、字符字面量替换成空格，保留换行与列偏移。"""
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        nxt = text[index + 1] if index + 1 < size else ""
        if char == "/" and nxt == "/":
            while index < size and text[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if char == "/" and nxt == "*":
            out.append("  ")
            index += 2
            while index < size and not (text[index] == "*" and index + 1 < size and text[index + 1] == "/"):
                out.append("\n" if text[index] == "\n" else " ")
                index += 1
            if index < size:
                out.append("  ")
                index += 2
            continue
        if char in ('"', "'"):
            quote = char
            out.append(" ")
            index += 1
            while index < size:
                current = text[index]
                if current == "\\" and index + 1 < size:
                    out.append("  ")
                    index += 2
                    continue
                if current == quote:
                    out.append(" ")
                    index += 1
                    break
                out.append("\n" if current == "\n" else " ")
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def check_file(path: Path, root: Path) -> list[str]:
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    stripped = strip_literals_and_comments(raw)
    raw_lines = raw.split("\n")
    relative = path.relative_to(root).as_posix()
    for number, line in enumerate(stripped.split("\n"), 1):
        if EXEMPT_SIGNATURE.search(line):
            continue
        # 标记可以写在行尾，也可以单独写在上一行（clang-format 会折行）。
        if number <= len(raw_lines) and ALLOW_MARKER in raw_lines[number - 1]:
            continue
        if number >= 2 and ALLOW_MARKER in raw_lines[number - 2]:
            continue
        for match in BANNED.finditer(line):
            problems.append(
                f"{relative}:{number}: 禁止平台相关类型 '{match.group(0)}'"
                "（改用 std::int32_t / std::uint64_t / std::size_t / std::byte）"
            )
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    problems: list[str] = []
    for directory in SCAN_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in (".hpp", ".cpp"):
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            problems += check_file(path, root)

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n共 {len(problems)} 处", file=sys.stderr)
        return 1
    print("CHECK OK: no platform-dependent builtin types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
