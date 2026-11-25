#!/usr/bin/env python3
"""每个用例都必须有断言（REQ-032）。

为什么需要
--------
"跑过了"和"验证过"是两件事。一个只有 `SUCCEED()` 的用例在实现完全失效时
也会通过。复审时确实发现过一个：`TxnLog.NullLoggerIsNoOp` 只写了一句
`SUCCEED() << "不得崩溃"`——如果 `log_*` 在 logger 为 nullptr 时"退回默认
logger"（那是参数语义的错误实现），该用例照样通过。

判据
----
每个 `TEST` / `TEST_F` / `TEST_P` 的函数体必须至少包含一条
`EXPECT_*` / `ASSERT_*` / `FAIL()`；只有 `SUCCEED()` 不算。

例外
----
断言写在辅助函数里的用例，在 TEST 上一行写：
    // assertions-in-helper: <辅助函数名>
这是显式声明，而不是"看起来没断言就放过"。

用法:
    python3 scripts/check-test-assertions.py --check
"""

import argparse
import re
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent

TEST_BEGIN = re.compile(r'^TEST(?:_F|_P)?\(\s*[A-Za-z0-9_]+\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\{', re.M)
REAL_ASSERT = re.compile(r'\b(?:EXPECT|ASSERT)_[A-Z_]+\s*\(|\bFAIL\s*\(')
EXEMPT_MARK = "assertions-in-helper"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    total = 0
    exempt = 0
    problems: list[str] = []

    for path in sorted(LIB.glob("tests/**/*.cpp")):
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        matches = list(TEST_BEGIN.finditer(text))
        for index, match in enumerate(matches):
            total += 1
            name = match.group(1)
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end]
            if REAL_ASSERT.search(body):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            # 紧邻上一行（允许中间有注释）里找豁免标记
            lookback = max(0, line_no - 4)
            window = "\n".join(lines[lookback:line_no])
            if EXEMPT_MARK in window:
                exempt += 1
                continue
            problems.append(f"{path.relative_to(LIB).as_posix()}:{line_no}: {name} 没有任何断言")

    if args.list:
        print(f"用例 {total} 个，其中 {exempt} 个声明断言在辅助函数里")
        return 0

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n共 {len(problems)} 个用例没有任何断言"
              f"（要么加断言，要么用 // {EXEMPT_MARK}: <函数名> 说明）", file=sys.stderr)
        return 1

    print(f"CHECK OK: all {total} test cases assert something ({exempt} declare an assertion helper)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
