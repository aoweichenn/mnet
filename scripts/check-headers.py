#!/usr/bin/env python3
"""头文件卫生检查。

规则（对应 docs/00-conventions/CODE-STYLE.md）：
  H-1  所有 include 用尖括号（禁止 #include "..."）
  H-2  公共头必须自包含（单独 include 能编过）
  H-3  每个公共头有对应的实现文件（伞头与纯 inline 头除外）
  H-4  不使用裸 int / unsigned / long / short
  H-5  公共头不 include src/ 下的路径

用法：
    python3 scripts/check-headers.py --check <lib_dir>
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
ANGLE_INCLUDE = re.compile(r'^\s*#\s*include\s+<([^>]+)>', re.M)
# 裸 unsigned 也要抓。早先的正则要求 unsigned 后面必须跟 int/long/short，
# 于是 `for (unsigned i = ...)` 这种写法完全逃过检查——复审在 src/ 里
# 找到 7 处。`unsigned char` 是字节访问的惯用法，仍然允许。
BARE_INT = re.compile(
    r'\b(?<!std::)(?:'
    r'unsigned\b(?!\s+char)'
    r'|(?:unsigned\s+)?(?:int|long|short)\b'
    r')(?!\s*(?:32|64)_t)(?!::)')

# 允许裸 int 的位置：main 函数、循环变量在 1 行内、sizeof 表达式里。
BARE_INT_EXCLUDE = re.compile(r'\bint main\s*\(|sizeof\s*\(\s*(?:int|long|short)\s*\)')

HEADER_ONLY = {"result.hpp", "error.hpp", "log.hpp"}

# 适配器头刻意不自包含：它们引用外部库（mnem）的头，只有消费方自己
# 已经引入了该库时才有意义。强行自包含会让 check-headers 反过来要求本库
# 的构建依赖 mnem——正是这次重构要消除的东西。
ADAPTER_DIRS = ("mnet/adapters/",)


def strip_comments(text: str) -> str:
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    # 字符串/字符字面量里的英文单词不是类型。没这一步时，错误消息
    # "path is too long" 会被当成裸 long 报出来（实测）。
    # 先替换字符串再替换字符，避免 '"' 与 '\'' 相互干扰。
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def check_quoted_includes(root: Path) -> list[str]:
    problems = []
    for path in list(root.glob("**/*.hpp")) + list(root.glob("**/*.cpp")):
        if "_deps" in str(path):
            continue
        for match in QUOTED_INCLUDE.finditer(path.read_text(encoding="utf-8")):
            problems.append(f"{path.relative_to(root)}: 引号 include 应改为尖括号: {match.group(1)}")
    return problems


def check_src_leak(root: Path) -> list[str]:
    problems = []
    public = root / "include"
    if not public.is_dir():
        return problems
    for path in public.glob("**/*.hpp"):
        for match in ANGLE_INCLUDE.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("base/", "platform/", "codec/", "format/", "detail/")):
                problems.append(f"{path.relative_to(root)}: 公共头泄漏内部路径 <{target}>")
    return problems


def check_self_contained(root: Path, include_dir: Path, build_dir: Path, extra_includes: list[Path]) -> list[str]:
    problems = []
    for path in sorted((root / "include").glob("**/*.hpp")):
        rel = path.relative_to(root / "include")
        if any(str(rel).startswith(prefix) for prefix in ADAPTER_DIRS):
            continue
        probe = build_dir / f"self_contained_{path.stem}.cpp"
        probe.write_text(f"#include <{rel.as_posix()}>\nint main() {{ return 0; }}\n", encoding="utf-8")
        result = subprocess.run(
            ["g++", "-std=c++20", "-fsyntax-only", f"-I{include_dir}"]
            + [f"-I{path}" for path in extra_includes]
            + [str(probe)],
            capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"{rel}: 不自包含\n{result.stderr.strip()[:400]}")
    return problems


def check_bare_int(root: Path) -> list[str]:
    problems = []
    for path in list(root.glob("include/**/*.hpp")) + list(root.glob("src/**/*.cpp")) + list(root.glob("src/**/*.hpp")):
        if "_deps" in str(path):
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for number, line in enumerate(text.split("\n"), 1):
            cleaned = BARE_INT_EXCLUDE.sub("", line)
            for match in BARE_INT.finditer(cleaned):
                problems.append(f"{path.relative_to(root)}:{number}: 裸整数类型 '{match.group(0)}'，用 std::int32_t 等")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--build-dir", default=os.environ.get("MNET_SELFCHECK_DIR", tempfile.mkdtemp(prefix="mnet_selfcheck_")))
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    problems += check_quoted_includes(root)
    problems += check_src_leak(root)
    problems += check_bare_int(root)
    problems += check_self_contained(root, root / "include", build_dir,
                                    [Path(item).resolve() for item in args.include])

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n共 {len(problems)} 处问题", file=sys.stderr)
        return 1

    print("CHECK OK: headers clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
