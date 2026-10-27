"""
Sanity check for the upper bound: every recorded play must satisfy the
shape constraints the integer program in `upper_bound.py` imposes.

    python records/bound/check_constraints.py records/bound/widening.tsv \
        records/free-order-236.txt records/above-200/*.txt
"""

import re
import sys

AREA = [10, 5, 9, 9, 8, 10, 9, 7, 8, 10]


def load(path):
    single, pair, supports = {}, {}, {}
    for line in open(path):
        p = line.split()
        if p[0] == "single":
            single[int(p[1])] = int(p[2])
        elif p[0] == "pair":
            pair[tuple(sorted(map(int, p[1:3])))] = int(p[3])
        elif p[0] in ("supports", "supports3"):
            supports[tuple(sorted(map(int, p[1:-1])))] = int(p[-1])
    return single, pair, supports


def problems(path, single, pair, supports):
    moves = re.findall(r"(\d)@(\d+):\d+:\d+:\d", open(path).read())
    levels = {}
    for v, l in moves:
        levels.setdefault(int(l), []).append(int(v))
    area = {l: sum(AREA[v] for v in vs) for l, vs in levels.items()}
    out = []
    for l in range(1, max(levels) + 1):
        below, here = tuple(sorted(levels[l - 1])), tuple(sorted(levels[l]))
        if area[l] > area[l - 1]:
            out.append(f"area grows at level {l}")
        if len(below) < 2:
            out.append(f"level {l - 1} has fewer than two tiles")
        widen = single[here[0]] if len(here) == 1 else pair[here] if len(here) == 2 else 0
        if area[l - 1] < area[l] + widen:
            out.append(f"level {l - 1} covers {area[l - 1]} < {area[l] + widen}")
        if len(below) in (2, 3) and area[l] > supports[below]:
            out.append(f"level {l - 1} {below} supports at most {supports[below]} < {area[l]}")
    return out


if __name__ == "__main__":
    tables = load(sys.argv[1])
    bad = 0
    for path in sys.argv[2:]:
        if p := problems(path, *tables):
            bad += 1
            print(path, p)
    print(f"{len(sys.argv) - 2} plays checked, {bad} violate the bound's constraints")
    sys.exit(1 if bad else 0)
