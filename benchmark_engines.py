"""
Compare the Python engine (`num9`) with the Rust engine (`num9_core`).

Usage:
    maturin develop --release -m rust/Cargo.toml
    python benchmark_engines.py --py-games 10 --rs-games 1000

Prints two markdown tables: move generation on identical positions, and full
games per strategy.
"""

import argparse
import os
import random
import time
from statistics import mean
from typing import Callable

import num9_core
from num9 import Board, Piece
from num9.strategies import (
    choose_move_with_most_edges_touching,
    choose_solid_base_high_top,
    go_up_randomly,
    place_randomly,
)

PY_STRATEGIES: dict[str, Callable[[Board, Piece], None]] = {
    "place_randomly": place_randomly,
    "go_up_randomly": go_up_randomly,
    "edges_then_up": choose_move_with_most_edges_touching,
    "solid_base_high_top": choose_solid_base_high_top,
}


def positions(games: int, seed: int) -> list[tuple[Board, num9_core.Board]]:
    """Every position of `games` Rust-played games, on both engines."""
    out = []
    for g in range(games):
        game = num9_core.play_game("edges_then_up", seed=seed + g)
        py, rs = Board(), num9_core.Board()
        for value, layer, i, j, r in game.pieces:
            out.append((Board(py.board.copy(), list(py.piece_sequence)), rs.copy()))
            py.place(Piece(value), (layer, i, j), rotation=r)
            rs.place(value, (layer, i, j), r)
    return out


def bench_move_generation(games: int, seed: int) -> tuple[str, float]:
    states = positions(games, seed)
    calls = len(states) * 10

    start = time.perf_counter()
    for py, _ in states:
        for value in range(10):
            py.find_valid_moves(Piece(value))
    py_us = (time.perf_counter() - start) / calls * 1e6

    reps = 20
    start = time.perf_counter()
    for _ in range(reps):
        for _, rs in states:
            for value in range(10):
                rs.find_valid_moves(value)
    rs_us = (time.perf_counter() - start) / (calls * reps) * 1e6

    lines = [
        f"`find_valid_moves` on {len(states)} positions from {games} games "
        "(all ten piece values each):",
        "",
        "| engine | µs / call | speedup |",
        "| --- | --- | --- |",
        f"| Python (`num9.Board`) | {py_us:,.0f} | 1× |",
        f"| Rust via Python (`num9_core.Board`) | {rs_us:,.1f} | {py_us / rs_us:,.0f}× |",
    ]
    return "\n".join(lines), py_us


def play_python(strategy: Callable[[Board, Piece], None], tiles: list[int]) -> int:
    board = Board()
    for value in tiles:
        strategy(board, Piece(value))
    return board.score()


def lookahead_python_estimate(py_us: float, seed: int) -> float:
    """Seconds per game the Python `maximize_lookahead` would need for its
    `find_valid_moves` calls alone (one per legal move per leftover tile)."""
    calls = 0
    for g in range(5):
        rng = random.Random(seed + g)
        tiles = list(range(10)) * 2
        rng.shuffle(tiles)
        rs = num9_core.Board()
        for k, value in enumerate(tiles):
            moves, cand = num9_core.decide("maximize_lookahead", rs, value)
            calls += len(moves) * (20 - k)
            rs.place(value, moves[cand[0]][:3], moves[cand[0]][3])
    return calls / 5 * py_us / 1e6


def rate(games_per_sec: float) -> str:
    return f"{games_per_sec:,.0f}" if games_per_sec >= 10 else f"{games_per_sec:.2f}"


def bench_games(py_games: int, rs_games: int, seed: int, py_us: float) -> str:
    threads = os.cpu_count() or 1
    rows = [
        "| strategy | Python avg | Python games/s | Rust avg | Rust games/s (1 thread) "
        f"| Rust games/s ({threads} threads) | speedup (1 thread) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in num9_core.STRATEGIES:
        rs_n = rs_games if name != "maximize_lookahead" else max(1, rs_games // 10)
        start = time.perf_counter()
        rs_scores = num9_core.play_games(name, rs_n, seed=seed, threads=1)
        rs_rate = rs_n / (time.perf_counter() - start)
        start = time.perf_counter()
        num9_core.play_games(name, rs_n, seed=seed, threads=threads)
        rs_rate_mt = rs_n / (time.perf_counter() - start)

        if name in PY_STRATEGIES:
            rng = random.Random(seed)
            py_scores = []
            start = time.perf_counter()
            for _ in range(py_games):
                tiles = list(range(10)) * 2
                rng.shuffle(tiles)
                py_scores.append(play_python(PY_STRATEGIES[name], tiles))
            py_rate = py_games / (time.perf_counter() - start)
            py_avg = f"{mean(py_scores):.1f}"
            py_rate_s = rate(py_rate)
            speedup = f"{rs_rate / py_rate:,.0f}×"
        else:
            est = lookahead_python_estimate(py_us, seed)
            py_avg = "not run"
            py_rate_s = f"≈1 per {est / 60:.0f} min (est.)"
            speedup = f"≈{rs_rate * est:,.0f}× (est.)"
        rows.append(
            f"| {name} | {py_avg} | {py_rate_s} | {mean(rs_scores):.1f} | "
            f"{rate(rs_rate)} | {rate(rs_rate_mt)} | {speedup} |"
        )
    note = (
        f"\nPython: {py_games} games per strategy. Rust: {rs_games} games per "
        f"strategy ({max(1, rs_games // 10)} for maximize_lookahead), called "
        "through the Python bindings. Averages come from different random "
        "games, so they agree only statistically. The Python maximize_lookahead "
        "figure is an estimate: its `find_valid_moves` calls times the measured "
        "Python per-call cost."
    )
    return "\n".join(rows) + "\n" + note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--py-games", type=int, default=10)
    parser.add_argument("--rs-games", type=int, default=1000)
    parser.add_argument("--position-games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table, py_us = bench_move_generation(args.position_games, args.seed)
    print("## Move generation\n")
    print(table)
    print("\n## Full games\n")
    print(bench_games(args.py_games, args.rs_games, args.seed, py_us))


if __name__ == "__main__":
    main()
