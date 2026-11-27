#!/usr/bin/env python3
# Copyright 2026 mcnn authors
# SPDX-License-Identifier: BSD-3-Clause
"""Compare the stage dumps of benchmarks/metal/attn_block_test.cpp.

Usage: compare_attn_stages.py DIR [DIR]

Each stage is compared as cpu-X.f32 against gpu-X.f32 and reported with the
worst relative difference (absolute difference over the reference's largest
magnitude, so near-zero elements do not dominate).
"""

import struct
import sys
from pathlib import Path

STAGES = ["qfull", "qprep", "gatesig", "kcache", "vcache", "attn", "out"]


def load(path):
    data = path.read_bytes()
    return struct.unpack("<%df" % (len(data) // 4), data)


def compare(directory):
    for stage in STAGES:
        cpu_path = directory / ("cpu-%s.f32" % stage)
        gpu_path = directory / ("gpu-%s.f32" % stage)
        if not cpu_path.exists() or not gpu_path.exists():
            print("%-8s missing dump (%s/%s)" % (stage, cpu_path.name, gpu_path.name))
            continue
        cpu = load(cpu_path)
        gpu = load(gpu_path)
        if len(cpu) != len(gpu):
            print("%-8s size mismatch %d vs %d" % (stage, len(cpu), len(gpu)))
            continue
        scale = max(abs(value) for value in cpu) or 1.0
        worst = 0.0
        index = -1
        differing = 0
        for i, (a, b) in enumerate(zip(cpu, gpu)):
            delta = abs(a - b)
            if delta != 0.0:
                differing += 1
            if delta / scale > worst:
                worst = delta / scale
                index = i
        print("%-8s %6d values  differing %6d  worst rel %.3e" % (stage, len(cpu), differing, worst))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    compare(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
