#!/usr/bin/env python3
"""可失败返回值必须标 [[nodiscard]]（REQ-029）。

为什么需要这条机械保证
--------------------
复审过程中反复出现的缺陷有一个共同形态：**某条路径忘了检查返回值**。
例如 `view()` 漏了尺寸校验、`Reader::u16()` 的失败被当成 0 用。

靠人眼在 review 里找这类问题不可靠。可靠的写法是让**编译器**负责：
只要每个可能失败的函数都标了 `[[nodiscard]]`，`-Wall` 里的
`-Wunused-result` 就会把"结果被丢弃"变成编译错误，再由"零告警"闸门拦住。

所以这条检查不是查代码风格，而是把"必须检查错误"从约定变成编译器强制。

检查对象
--------
`include/mnet/**/*.hpp` 与 `src/mnet/**/*.hpp` 里所有返回
`Result<...>` / `Status` / `Failure` 的函数声明，包括成员函数与自由函数。

例外（必须有理由，写在 EXEMPT 里）
--------------------------------
* 构造函数/析构函数：没有返回值。
* 运算符（`operator==` 之类）：返回值不是错误通道。
* 刻意忽略的重载：目前没有。

用法:
    python3 scripts/check-nodiscard.py --check
"""

import argparse
import re
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent

# 声明匹配。
#
# 关键细节：行首用 [ \t]* 而不是 \s*——\s 会跨行吞换行，导致匹配起点跑到
# 上一行，于是"上一行没有 [[nodiscard]]"被误判成缺失（这个坑自己踩过）。
DECL = re.compile(
    r'^[ \t]*(?P<attr>\[\[nodiscard\]\][ \t]*)?'
    r'(?:(?:static|inline|constexpr|virtual|explicit|friend)[ \t]+)*'
    r'(?P<ret>Result<[^;{}]*?>|Status|Failure)[ \t]+'
    r'(?P<name>[A-Za-z_][\w]*)[ \t]*\(',
    re.M,
)

# 名字本身不是错误通道的重载（赋值运算符等）。
EXEMPT_NAMES = {"operator=", "operator==", "operator!="}


def scan(paths: list[Path]) -> tuple[int, list[str]]:
    checked = 0
    problems: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(LIB).as_posix()
        for match in DECL.finditer(text):
            name = match.group("name")
            if name in EXEMPT_NAMES or name.startswith("operator"):
                continue
            checked += 1
            if match.group("attr") is None:
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{rel}:{line}: {match.group('ret')} {name}() 缺少 [[nodiscard]]")
    return checked, problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true", help="列出所有被检查的声明")
    args = parser.parse_args(argv)

    paths = sorted(list((LIB / "include" / "mnet").rglob("*.hpp")) + list((LIB / "src" / "mnet").rglob("*.hpp")))
    checked, problems = scan(paths)

    if args.list:
        print(f"检查了 {len(paths)} 个头文件里的 {checked} 个可失败声明")
        return 0

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n共 {len(problems)} 处缺少 [[nodiscard]]（否则丢弃返回值不会被告警）", file=sys.stderr)
        return 1

    print(f"CHECK OK: all {checked} fallible declarations are [[nodiscard]]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
