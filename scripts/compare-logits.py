#!/usr/bin/env python3
"""Compare two raw float32 logit dumps against the acceptance criteria.

The fast path is allowed to move logits by the same normalised distance the
existing fast path already has from the CPU dump (4.754e-3 measured on the
32-token prompt), and a rank swap between logits closer than `--tie` is
reported as a tie rather than a failure: near-ties are decided by float noise in
any path that does not reproduce the CPU's summation order exactly.

Usage:
    scripts/compare-logits.py REFERENCE.bin CANDIDATE.bin [--tie 2e-3] [--diff 4.754e-3]

Exit status is 0 when the normalised difference is within `--diff` and the top-5
is either identical or differs only by ties, 1 otherwise.
"""

from __future__ import annotations

import argparse
import struct
import sys


def load(path: str) -> list[float]:
    data = open(path, "rb").read()
    if len(data) % 4 != 0:
        raise SystemExit(f"{path}: size {len(data)} is not a multiple of 4")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def top_k(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i])[-k:][::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--tie", type=float, default=2e-3,
                        help="rank swaps inside this normalised gap count as a tie (default 2e-3)")
    parser.add_argument("--diff", type=float, default=4.754e-3,
                        help="largest allowed normalised difference (default 4.754e-3)")
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit(f"length mismatch: {len(reference)} vs {len(candidate)}")

    scale = max(abs(value) for value in reference)
    worst = 0.0
    worst_index = -1
    for index, (ref, got) in enumerate(zip(reference, candidate)):
        diff = abs(ref - got) / scale
        if diff > worst:
            worst, worst_index = diff, index

    ref_top = top_k(reference, 5)
    got_top = top_k(candidate, 5)
    if ref_top == got_top:
        verdict = "same"
    else:
        # The two lists may differ only inside the tie band around fifth place:
        # every entry that moved has to sit within `tie` of the fifth value in
        # both dumps, otherwise the ranking changed for a real reason.
        moved = set(ref_top) ^ set(got_top)
        band = args.tie * scale
        tied = all(abs(reference[entry] - reference[ref_top[-1]]) <= band and
                   abs(candidate[entry] - candidate[got_top[-1]]) <= band for entry in moved)
        verdict = "tie" if tied else "DIFFERENT"

    # The gap that decides whether fifth and sixth can swap.
    ref_six = top_k(reference, 6)
    margin = (reference[ref_six[4]] - reference[ref_six[5]]) / scale
    print(f"elements {len(reference)}  normalised max diff {worst:.3e} (at {worst_index})")
    print(f"top-5: {verdict}  (reference 5th/6th gap {margin:.3e})")
    print(f"  reference {ref_top}")
    if verdict != "same":
        print(f"  candidate {got_top}")

    ok = worst <= args.diff and verdict in ("same", "tie")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
