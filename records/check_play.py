"""
Check an NMBR 9 play against the rules, independently of the engine.

    python records/check_play.py records/free-order-236.txt

The play file needs a line "score N" and the moves as
"value@level:row:col:rotation" (rotation = clockwise quarter turns of the
shapes below). Rules checked:
  - each value 0-9 is used exactly twice (20 tiles);
  - tiles never overlap a tile on the same level;
  - a tile on a level that already has tiles must share an edge with one;
  - a tile above the ground must have every cell on top of a tile on the
    level below, and must rest on at least two different tiles;
  - score = sum of value x level.
Only the standard library is needed.
"""

import re
import sys

# Tile shapes at rotation 0, as in the original num9/piece.py (1 = tile).
SHAPES = {
    0: ["111", "101", "101", "111"],
    1: ["11", "01", "01", "01"],
    2: ["011", "011", "110", "111"],
    3: ["111", "001", "011", "111"],
    4: ["011", "010", "111", "011"],
    5: ["111", "111", "001", "111"],
    6: ["110", "100", "111", "111"],
    7: ["111", "010", "110", "100"],
    8: ["011", "011", "110", "110"],
    9: ["111", "111", "110", "110"],
}


def cells(value, rotation, i, j):
    grid = [list(row) for row in SHAPES[value]]
    for _ in range(rotation):
        grid = [list(row) for row in zip(*grid[::-1])]  # clockwise
    return {(i + r, j + c) for r, row in enumerate(grid) for c, x in enumerate(row) if x == "1"}


def check(text):
    claimed = int(re.search(r"^score (\d+)", text, re.M).group(1))
    moves = re.findall(r"(\d)@(\d+):(\d+):(\d+):(\d)", text)
    levels = {}  # level -> {cell: move number}
    counts = [0] * 10
    score = 0
    for k, move in enumerate(moves):
        v, level, i, j, r = map(int, move)
        cs = cells(v, r, i, j)
        here = levels.setdefault(level, {})
        if cs & here.keys():
            raise ValueError(f"move {k}: overlaps a tile on level {level}")
        if here:
            around = {(a + da, b + db) for a, b in cs for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            if not (around - cs) & here.keys():
                raise ValueError(f"move {k}: doesn't touch a tile on level {level}")
        if level > 0:
            below = levels.get(level - 1, {})
            if not cs <= below.keys():
                raise ValueError(f"move {k}: not fully supported")
            if len({below[c] for c in cs}) < 2:
                raise ValueError(f"move {k}: rests on a single tile")
        for c in cs:
            here[c] = k
        counts[v] += 1
        score += v * level
    if len(moves) != 20 or counts != [2] * 10:
        raise ValueError(f"must use two of each value; counts {counts}")
    if score != claimed:
        raise ValueError(f"scores {score}, not the claimed {claimed}")
    return score, max(levels)


if __name__ == "__main__":
    score, top = check(open(sys.argv[1]).read())
    print(f"OK: 20 legal moves, two of each value, score {score}, top level {top}")
