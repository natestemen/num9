"""
An upper bound on the best possible NMBR 9 score (tile order free).

Every legal play satisfies the constraints below, so the optimum of this
integer program is at least the score of any play. Shapes enter only
through `widening.tsv`, computed exhaustively by `cargo run --release --bin
widening` (the minimum extra area the level below must cover to support a
level holding one tile, or two touching tiles, and the most area two or
three connected tiles can support above them).

    pip install scipy numpy
    python records/bound/upper_bound.py records/bound/widening.tsv

Constraints, for levels 0..9 (with 20 tiles nothing can be higher: every
occupied level needs at least two tiles on each level below it):
  - two tiles of each value, each on exactly one level;
  - every cell of a level lies on a cell of the level below, so each
    level's area is at most the area of the level below;
  - a tile above the ground rests on at least two tiles, so a level below
    an occupied level holds at least two tiles;
  - if a level holds exactly one tile, or exactly two (which touch, the
    level being connected), the level below covers at least their area
    plus the computed minimum widening;
  - if a level holds exactly two or three tiles, the level above covers
    at most the computed maximum they can support (every tile above lies
    within them and rests on at least two of them); a single tile
    supports nothing.
Objective: maximise the sum of value x level.

Each level chooses a type: empty, one of the sets of one, two or three
tiles, or "four or more tiles"; that keeps the constraints linear.
"""

import itertools
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

AREA = [10, 5, 9, 9, 8, 10, 9, 7, 8, 10]
LEVELS = 10
VALUES = range(10)
PAIRS = list(itertools.combinations_with_replacement(VALUES, 2))
TRIPLES = list(itertools.combinations_with_replacement(VALUES, 3))
SMALL = [(v,) for v in VALUES] + PAIRS + TRIPLES


def load(path):
    single, pair, supports = {}, {}, {}
    for line in open(path):
        parts = line.split()
        if parts[0] == "single":
            single[int(parts[1])] = int(parts[2])
        elif parts[0] == "pair":
            a, b, w = map(int, parts[1:])
            pair[(a, b)] = pair[(b, a)] = w
        elif parts[0] == "supports":
            a, b, m = map(int, parts[1:])
            supports[(a, b)] = supports[(b, a)] = m
        elif parts[0] == "supports3":
            a, b, c, m = map(int, parts[1:])
            supports[tuple(sorted((a, b, c)))] = m
    assert len(single) == 10 and len(pair) == 100, "incomplete table"
    assert all(p in supports for p in PAIRS), "incomplete table"
    return single, pair, supports


def widening(s, single, pair):
    """Least extra area the level below must cover (0 when not computed)."""
    if len(s) == 1:
        return single[s[0]]
    return pair[s] if len(s) == 2 else 0


def bound(single, pair, supports, time_limit=3600, verbose=False):
    names = []
    index = {}

    def var(name, lo, hi):
        index[name] = len(names)
        names.append((name, lo, hi))

    for l in range(LEVELS):
        var(("empty", l), 0, 1)
        var(("big", l), 0, 1)
        for s in SMALL:
            var(("small", l, s), 0, 1)
        for v in VALUES:
            var(("y", v, l), 0, 2)  # copies of v on level l
            var(("r", v, l), 0, 2)  # of which on a "big" (4+ tiles) level
    n = len(names)
    rows, lo, hi = [], [], []

    def add(coeffs, a, b):
        r = np.zeros(n)
        for name, c in coeffs:
            r[index[name]] += c
        rows.append(r)
        lo.append(a)
        hi.append(b)

    area = lambda l, sign=1: [(("y", v, l), sign * AREA[v]) for v in VALUES]
    count = lambda l, sign=1: [(("y", v, l), sign) for v in VALUES]
    for v in VALUES:
        add([(("y", v, l), 1) for l in range(LEVELS)], 2, 2)
    for l in range(LEVELS):
        # One type per level.
        add([(("empty", l), 1), (("big", l), 1)] + [(("small", l, s), 1) for s in SMALL], 1, 1)
        # Contents: y = tiles of the chosen small set + r (only on big levels).
        for v in VALUES:
            add([(("y", v, l), 1), (("r", v, l), -1)] + [(("small", l, s), -s.count(v)) for s in SMALL], 0, 0)
            add([(("r", v, l), 1), (("big", l), -2)], -np.inf, 0)
        add([(("r", v, l), 1) for v in VALUES] + [(("big", l), -4)], 0, np.inf)
        if l > 0:
            # Nothing above an empty level.
            add([(("empty", l), 1), (("empty", l - 1), -1)], 0, np.inf)
            # Area never grows going up.
            add(area(l - 1) + area(l, -1), 0, np.inf)
            # An occupied level needs two tiles below it.
            add(count(l - 1) + [(("empty", l), 2)], 2, np.inf)
            # A level of exactly two (or three) tiles holds at most
            # `supports` cells above it; a single tile supports nothing.
            # Without a computed cap a level could hold its own area.
            def limit(s):
                if len(s) == 1:
                    return 0
                return supports.get(s, sum(AREA[v] for v in s))
            cap = [(("small", l - 1, s), -limit(s)) for s in SMALL]
            add(area(l) + cap + [(("big", l - 1), -170)], -np.inf, 0)
            # Widening under one- and two-tile levels.
            add(area(l - 1) + [(("small", l, s), -(sum(AREA[v] for v in s) + widening(s, single, pair))) for s in SMALL], 0, np.inf)
    add([(("empty", 0), 1)], 0, 0)

    obj = np.zeros(n)
    for v in VALUES:
        for l in range(LEVELS):
            obj[index[("y", v, l)]] = -v * l
    res = milp(
        obj,
        constraints=LinearConstraint(np.array(rows), lo, hi),
        integrality=np.ones(n),
        bounds=Bounds([b[1] for b in names], [b[2] for b in names]),
        options={"disp": verbose, "time_limit": time_limit},
    )
    # A proven upper bound: the optimum if solved, else the solver's bound.
    proven = -res.fun if res.status == 0 else -res.mip_dual_bound
    levels = [[v for v in VALUES for _ in range(round(res.x[index[("y", v, l)]]))] for l in range(LEVELS)]
    return proven, levels, res


if __name__ == "__main__":
    single, pair, supports = load(sys.argv[1])
    value, levels, res = bound(single, pair, supports)
    print(f"solver: {res.message}")
    print(f"upper bound on the best possible score: {value:.0f}")
    print("the relaxation's best assignment (values per level, top first):")
    for l in reversed(range(LEVELS)):
        if levels[l]:
            print(f"  level {l}: {sorted(levels[l], reverse=True)}  area {sum(AREA[v] for v in levels[l])}")
