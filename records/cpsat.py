"""
Search for the highest free-order NMBR 9 score with a constraint solver
(OR-Tools CP-SAT), modelling the finished stack directly.

    pip install ortools
    python records/cpsat.py [--size 12] [--levels 8] [--minutes 60]
        [--hint records/free-order-241.txt] [--at-least 242] [--out runs/cpsat]
        [--height L]

One yes/no variable for each placement of each tile (value, rotation, row,
column, level) inside a size x size grid. Constraints, all rules of a
finished stack:
  - two tiles of each value;
  - tiles on a level don't overlap;
  - every cell above the ground lies on a cell of the level below;
  - no tile rests on a single tile: a placement can't be used together with
    one on the level below that contains all its cells;
  - each level is connected: added lazily, as cuts, when a solution has a
    level in pieces (a piece and another tile both there means some tile
    must touch the piece).
Objective: the score (value x level), at least --at-least. The --hint play
is translated into the grid and given to the solver as a starting point.
With --height L, there's no score to reach: any stack with a tile on level L
will do (the grid gets levels 0 to L), so the solver finds one or shows none
fits in the grid.
Solutions are checked with check_play.py and written to --out; the solver's
bound on the best score in the grid is reported too.
"""

import argparse
import os
import re
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_play import cells, check  # noqa: E402


def shapes():
    """For each value, its distinct rotations as (rotation, cells at the origin)."""
    out = {}
    for v in range(10):
        seen = []
        for r in range(4):
            cs = frozenset(cells(v, r, 0, 0))
            if cs not in [s for _, s in seen]:
                seen.append((r, cs))
        out[v] = seen
    return out


def read_play(path):
    text = open(path).read()
    return [tuple(map(int, m)) for m in re.findall(r"(\d)@(\d+):(\d+):(\d+):(\d)", text)]


def neighbours(cs):
    return {(a + da, b + db) for a, b in cs for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))} - set(cs)


def build_order(tiles):
    """Placed tiles as a buildable play: level by level, each touching one placed."""
    out = []
    for level in sorted({t[1] for t in tiles}):
        here = [t for t in tiles if t[1] == level]
        done = [here.pop(0)]
        while here:
            near = set().union(*(neighbours(cells(v, r, i, j)) for v, _, i, j, r in done))
            k = next((k for k, (v, _, i, j, r) in enumerate(here) if cells(v, r, i, j) & near), 0)
            done.append(here.pop(k))
        out += done
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--hint", default="records/free-order-241.txt")
    ap.add_argument("--at-least", type=int, default=242)
    ap.add_argument("--out", default="runs/cpsat")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--height", type=int)
    args = ap.parse_args()
    if args.height is not None:
        args.levels = args.height + 1
    n, levels = args.size, args.levels
    os.makedirs(args.out, exist_ok=True)

    model = cp_model.CpModel()
    places = []  # (value, rotation, i, j, level, cells, var)
    for v, rots in shapes().items():
        for r, cs in rots:
            h = 1 + max(a for a, _ in cs)
            w = 1 + max(b for _, b in cs)
            for i in range(n - h + 1):
                for j in range(n - w + 1):
                    placed = frozenset((a + i, b + j) for a, b in cs)
                    for level in range(levels):
                        places.append((v, r, i, j, level, placed, model.NewBoolVar(f"t{v}r{r}i{i}j{j}l{level}")))
    print(f"{len(places)} placements on a {n}x{n} grid, {levels} levels")

    by_cell = {}
    for p in places:
        for c in p[5]:
            by_cell.setdefault((p[4], c), []).append(p)
    for v in range(10):
        model.Add(sum(p[6] for p in places if p[0] == v) == 2)
    occ = {}
    for (level, c), ps in by_cell.items():
        occ[level, c] = model.NewBoolVar("")
        model.Add(sum(p[6] for p in ps) == occ[level, c])
    for (level, c), o in occ.items():
        if level > 0:
            model.AddImplication(o, occ[level - 1, c])
    # No tile on a single tile below: not with any placement below that contains it.
    pairs = 0
    for p in places:
        if p[4] == 0:
            continue
        first = next(iter(p[5]))
        for q in by_cell.get((p[4] - 1, first), []):
            if p[5] <= q[5]:
                model.AddBoolOr([p[6].Not(), q[6].Not()])
                pairs += 1
    score = sum(p[0] * p[4] * p[6] for p in places)
    if args.height is None:
        model.Add(score >= args.at_least)
        model.Maximize(score)
    else:
        model.Add(sum(p[6] for p in places if p[4] == args.height) >= 1)
    print(f"{pairs} single-support exclusions")

    # The hint: the play moved into the middle of the grid.
    hint = read_play(args.hint)
    allc = [c for v, _, i, j, r in hint for c in cells(v, r, i, j)]
    di = (n - (max(a for a, _ in allc) - min(a for a, _ in allc) + 1)) // 2 - min(a for a, _ in allc)
    dj = (n - (max(b for _, b in allc) - min(b for _, b in allc) + 1)) // 2 - min(b for _, b in allc)
    hinted = {(l, frozenset(cells(v, r, i + di, j + dj)), v) for v, l, i, j, r in hint}
    for p in places:
        model.AddHint(p[6], (p[4], p[5], p[0]) in hinted)

    start = time.time()
    deadline = start + args.minutes * 60
    rounds = 0
    best = None
    while time.time() < deadline:
        rounds += 1
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(1.0, deadline - time.time())
        solver.parameters.num_workers = args.workers
        status = solver.Solve(model)
        bound = solver.BestObjectiveBound() if args.height is None else 0
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            goal = "is that tall" if args.height is not None else "scores that much"
            what = f"no stack in the grid {goal}" if status == cp_model.INFEASIBLE else "nothing found in time"
            print(f"round {rounds}: {what} (status {solver.StatusName(status)}, bound {bound:.0f}) after {time.time() - start:.0f}s")
            break
        chosen = [p for p in places if solver.Value(p[6])]
        got = sum(p[0] * p[4] for p in chosen)
        # Connectivity: cut every level in pieces.
        cuts = 0
        for level in range(levels):
            here = [p for p in chosen if p[4] == level]
            groups = []
            for p in here:
                touching = [g for g in groups if any(neighbours(p[5]) & q[5] for q in g)]
                merged = [p] + [q for g in touching for q in g]
                groups = [g for g in groups if g not in touching] + [merged]
            if len(groups) <= 1:
                continue
            for g in groups:
                other = next(q for h in groups if h is not g for q in h)
                gcells = set().union(*(q[5] for q in g))
                ring = neighbours(gcells)
                near = [q[6] for q in places if q[4] == level and q[5] & ring and not q[5] & gcells]
                model.Add(sum(near) >= sum(q[6] for q in g) + other[6] - len(g))
                cuts += 1
        print(f"round {rounds}: score {got} (bound {bound:.0f}), {cuts} connectivity cuts, {time.time() - start:.0f}s")
        if cuts == 0:
            tiles = [(p[0], p[4], p[2], p[3], p[1]) for p in chosen]
            moves = " ".join(f"{v}@{l}:{i}:{j}:{r}" for v, l, i, j, r in build_order(tiles))
            text = f"score {got}\nfound by: cpsat.py\n\nmoves (value@layer:row:col:rotation) {moves}\n"
            try:
                check(text)
                ok = "checked"
            except ValueError as e:
                ok = f"FAILED CHECK: {e}"
            path = os.path.join(args.out, f"{got}-cpsat.txt")
            open(path, "w").write(text)
            print(f"  a legal stack scoring {got} ({ok}), saved to {path}")
            best = got
            if args.height is not None:
                break
            if status == cp_model.OPTIMAL:
                print(f"  optimal in this grid")
                break
            model.Add(score >= got + 1)
        model.ClearHints()
        chosen_vars = {id(p[6]) for p in chosen}
        for p in places:
            model.AddHint(p[6], id(p[6]) in chosen_vars)
    print(f"done after {time.time() - start:.0f}s, {rounds} rounds; best legal stack found: {best}")


if __name__ == "__main__":
    main()
