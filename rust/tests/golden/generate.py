"""Generate the golden fixtures in this directory from the original Python engine.

The Python engine (`num9` package) was removed once the Rust port replaced
it; these fixtures preserve its behaviour so `cargo test` can keep checking
the port against it. To regenerate, check out the last commit that has the
`num9/` package (45f7d47) with `num9_core` built, and run from the repo root:

    python rust/tests/golden/generate.py

Fixture format (one record per line, `#` lines are comments). Move lists are
stored as `count:hash`, with `hash` = 64-bit FNV-1a folded over the ints
(layer, i, j, rotation of each move in order; see `fnv`).

movegen.txt: positions from driver-played games
    game <tiles...>
    pos <count:hash for find_valid_moves(v), v = 0..9>        (before each placement)
    edges <count:hash of num_edges_touching for each legal move of the drawn piece>
    place <value> <layer> <i> <j> <rotation> <score> <board hash>

strategies.txt: lockstep strategy games
    strategy <name> [<weight>=<value> ...]
    game <tiles...>
    decide <value> <moves count:hash> <tie set count:hash> <chosen layer i j rotation>
    end <score> <board hash>

The board hash folds (flat index, value) over the nonzero cells of the
(6, 160, 120) board array in C order.
"""

import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import num9.strategies as py_strategies  # noqa: E402
import num9_core  # noqa: E402
from num9 import Board, Piece  # noqa: E402

OUT = Path(__file__).resolve().parent
MASK = (1 << 64) - 1
# Strategy games patch these; keep the originals (pool workers are reused).
ORIGINAL_FIND = Board.find_valid_moves
ORIGINAL_PLACE = Board.place


def fnv(ints) -> str:
    h = 0xCBF29CE484222325
    n = 0
    for x in ints:
        h = ((h ^ int(x)) * 0x100000001B3) & MASK
        n += 1
    return f"{h:016x}"


def moves_digest(moves) -> str:
    return f"{len(moves)}:{fnv(x for m in moves for x in m)}"


def board_hash(board: Board) -> str:
    flat = board.board.ravel()
    nz = np.flatnonzero(flat)
    return fnv(x for k in nz for x in (k, flat[k]))


def tiles_for(seed: int) -> list[int]:
    tiles = list(range(10)) * 2
    random.Random(seed).shuffle(tiles)
    return tiles


def py_moves(board: Board, value: int):
    return [tuple(int(x) for x in m) for m in ORIGINAL_FIND(board, Piece(value))]


# --------------------------------------------------------------------------
# Move generation


def driver(policy: str, board: Board, value: int, moves, rng: random.Random):
    if policy == "random":
        return rng.choice(moves)
    top = max(m[0] for m in moves)
    pool = [m for m in moves if m[0] == top] if policy == "up" else moves
    if policy == "mixed" and rng.random() < 0.5:
        pool = [m for m in moves if m[0] == top]
    edges = [board.num_edges_touching(rotated(value, m[3]), m[:3]) for m in pool]
    best = max(edges)
    return rng.choice([m for m, e in zip(pool, edges) if e == best])


def rotated(value: int, rotation: int) -> Piece:
    piece = Piece(value)
    piece.rotation = rotation
    return piece


def movegen_game(args) -> list[str]:
    tiles, choose = args
    lines = ["game " + " ".join(map(str, tiles))]
    board = Board()
    for ply, value in enumerate(tiles):
        lines.append("pos " + " ".join(moves_digest(py_moves(board, v)) for v in range(10)))
        moves = py_moves(board, value)
        edges = [board.num_edges_touching(rotated(value, m[3]), m[:3]) for m in moves]
        lines.append(f"edges {len(edges)}:{fnv(edges)}")
        layer, i, j, r = choose(board, value, moves, ply)
        ORIGINAL_PLACE(board, Piece(value), (layer, i, j), rotation=r)
        lines.append(f"place {value} {layer} {i} {j} {r} {board.score()} {board_hash(board)}")
    return lines


class Driven:
    def __init__(self, policy: str, seed: int):
        self.policy, self.rng = policy, random.Random(1000 + seed)

    def __call__(self, board, value, moves, ply):
        return driver(self.policy, board, value, moves, self.rng)


class Replay:
    def __init__(self, pieces):
        self.pieces = pieces

    def __call__(self, board, value, moves, ply):
        return self.pieces[ply][1:]


def movegen_jobs():
    jobs = []
    for seed in range(8):
        for policy in ("up", "mixed", "random"):
            jobs.append((tiles_for(seed), Driven(policy, seed)))
    # Rare games that reach layer 4.
    found = 0
    for seed in range(5000):
        game = num9_core.play_game("solid_base_high_top", seed=seed)
        if max(p[1] for p in game.pieces) >= 4:
            jobs.append(([p[0] for p in game.pieces], Replay(game.pieces)))
            found += 1
            if found == 3:
                break
    return jobs


# --------------------------------------------------------------------------
# Strategies


PY_STRATEGIES = {
    "place_randomly": py_strategies.place_randomly,
    "go_up_randomly": py_strategies.go_up_randomly,
    "edges_then_up": py_strategies.choose_move_with_most_edges_touching,
    "solid_base_high_top": py_strategies.choose_solid_base_high_top,
    "maximize_lookahead": py_strategies.maximize_lookahead,
}

CUSTOM_WEIGHTS = {
    "fill_weight": 1.1,
    "hole_penalty": 0.7,
    "support_weight": 3.0,
    "edge_weight": 0.9,
    "height_weight": 11.0,
    "aspect_penalty_weight": 0.6,
    "low_height_penalty_weight": 4.0,
}


def rust_backed_find_valid_moves(board: Board, piece: Piece):
    """`find_valid_moves` answered by num9_core (proven equivalent by the
    move-generation tests); only used to make maximize_lookahead tractable."""
    cached = getattr(board, "_rs_cache", None)
    if cached is None or cached[0] != len(board.piece_sequence):
        rs = num9_core.Board()
        for entry in board.piece_sequence:
            value, layer = int(entry["name"].name), entry["layer"]
            rows, cols = np.nonzero(entry["location"][layer])
            footprint = entry["location"][layer][
                rows.min() : rows.max() + 1, cols.min() : cols.max() + 1
            ]
            rotation = next(
                r
                for r in range(4)
                if np.array_equal(footprint, np.array(num9_core.piece_array(value, r)))
            )
            rs.place(value, (layer, int(rows.min()), int(cols.min())), rotation)
        board._rs_cache = cached = (len(board.piece_sequence), rs)
    return cached[1].find_valid_moves(int(piece.name))


def strategy_game(args) -> list[str]:
    name, seed, weights = args
    rng = random.Random(2000 + seed)
    state = {}
    find = rust_backed_find_valid_moves if name == "maximize_lookahead" else ORIGINAL_FIND

    def recording_find(board, piece):
        state["moves"] = find(board, piece)
        return state["moves"]

    def recording_place(board, piece, location, rotation=None):
        state["placed"] = (*location, rotation)
        return ORIGINAL_PLACE(board, piece, location, rotation)

    def choice(seq):
        seq = list(seq)
        state["offered"] = seq
        return seq[rng.randrange(len(seq))]

    Board.find_valid_moves = recording_find
    Board.place = recording_place
    py_strategies.choice = choice

    tiles = tiles_for(seed)
    lines = ["game " + " ".join(map(str, tiles))]
    board = Board()
    for value in tiles:
        state.clear()
        kwargs = weights or {}
        if name == "maximize_lookahead":
            # find_valid_moves is also called for every lookahead position;
            # record the moves of the real position separately.
            moves = py_moves_fast(board, value)
            PY_STRATEGIES[name](board, Piece(value))
            offered = [tuple(int(x) for x in state["placed"])]
        else:
            PY_STRATEGIES[name](board, Piece(value), **kwargs)
            moves = [tuple(int(x) for x in m) for m in state["moves"]]
            offered = state["offered"]
            if not isinstance(offered[0], tuple):
                offered = [moves[k] for k in offered]
            offered = [tuple(int(x) for x in m) for m in offered]
        chosen = tuple(int(x) for x in state["placed"])
        assert chosen in offered
        lines.append(
            f"decide {value} {moves_digest(moves)} {moves_digest(offered)} "
            + " ".join(map(str, chosen))
        )
    lines.append(f"end {board.score()} {board_hash(board)}")
    return lines


def py_moves_fast(board: Board, value: int):
    return [tuple(int(x) for x in m) for m in rust_backed_find_valid_moves(board, Piece(value))]


def strategy_jobs():
    jobs = []
    for name in ("place_randomly", "go_up_randomly", "edges_then_up", "solid_base_high_top"):
        jobs += [(name, seed, None) for seed in range(12)]
    jobs += [("solid_base_high_top", seed, CUSTOM_WEIGHTS) for seed in range(3)]
    jobs += [("maximize_lookahead", seed, None) for seed in range(4)]
    return jobs


def main() -> None:
    header = [f"# generated by {Path(__file__).name} from the Python engine; see its docstring"]
    with Pool() as pool:
        games = pool.map(movegen_game, movegen_jobs())
        (OUT / "movegen.txt").write_text(
            "\n".join(header + [line for g in games for line in g]) + "\n"
        )
        print(f"movegen.txt: {len(games)} games")

        jobs = strategy_jobs()
        results = pool.map(strategy_game, jobs, chunksize=1)
    lines = list(header)
    previous = None
    for (name, _, weights), game in zip(jobs, results):
        key = (name, tuple(sorted((weights or {}).items())))
        if key != previous:
            extra = "".join(f" {k}={v!r}" for k, v in sorted((weights or {}).items()))
            lines.append(f"strategy {name}{extra}")
            previous = key
        lines += game
    (OUT / "strategies.txt").write_text("\n".join(lines) + "\n")
    print(f"strategies.txt: {len(jobs)} games")


if __name__ == "__main__":
    main()
