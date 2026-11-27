#!/usr/bin/env python3
# Copyright 2026 mcnn authors
# SPDX-License-Identifier: BSD-3-Clause
"""Compare the stage dumps of benchmarks/metal/ssm_block_test.cpp.

Usage: compare_ssm_stages.py DIR

Each stage is compared as cpu-X.f32 against gpu-X.f32 with the worst relative
difference (absolute difference over the reference's largest magnitude).
"""

import struct
import sys
from pathlib import Path

STAGES = ["qkv", "z", "beta", "gate", "conv", "delta", "grouped", "out", "state", "window"]


def load(path):
    data = path.read_bytes()
    return struct.unpack("<%df" % (len(data) // 4), data)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    directory = Path(sys.argv[1])
    for stage in STAGES:
        cpu_path = directory / ("cpu-%s.f32" % stage)
        gpu_path = directory / ("gpu-%s.f32" % stage)
        if not cpu_path.exists() or not gpu_path.exists():
            print("%-9s missing dump (%s/%s)" % (stage, cpu_path.name, gpu_path.name))
            continue
        cpu = load(cpu_path)
        gpu = load(gpu_path)
        if len(cpu) != len(gpu):
            print("%-9s size mismatch %d vs %d" % (stage, len(cpu), len(gpu)))
            continue
        scale = max(abs(value) for value in cpu) or 1.0
        worst = 0.0
        differing = 0
        for a, b in zip(cpu, gpu):
            delta = abs(a - b)
            if delta != 0.0:
                differing += 1
            worst = max(worst, delta / scale)
        print("%-9s %6d values  differing %6d  worst rel %.3e" % (stage, len(cpu), differing, worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
