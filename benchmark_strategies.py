"""
Benchmark all strategies over many random games.

Usage:
    python benchmark_strategies.py --games 1000 --seed 1
"""

import argparse
import time
from statistics import mean, median
from typing import Callable

from num9 import Board, Piece
from tqdm import trange


def play_game(strategy: Callable[[Board, Piece], None]) -> int:
    tiles = list(range(10)) * 2
    import random

    random.shuffle(tiles)
    board = Board()
    for num in tiles:
        piece = Piece(num)
        strategy(board, piece)
    return board.score()


def run_strategy(
    name: str, strategy: Callable[[Board, Piece], None], games: int
) -> dict[str, float]:
    scores: list[int] = []
    start = time.perf_counter()
    for _ in trange(games, desc=name):
        scores.append(play_game(strategy))
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "games": games,
        "avg": mean(scores),
        "median": median(scores),
        "min": min(scores),
        "max": max(scores),
        "elapsed_s": elapsed,
        "games_per_sec": games / elapsed if elapsed > 0 else float("inf"),
    }


def render_markdown_table(results: list[dict[str, float]]) -> str:
    headers = [
        "strategy",
        "games",
        "avg",
        "median",
        "min",
        "max",
        "time (s)",
        "games/s",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for res in results:
        lines.append(
            "| "
            f"{res['name']} | "
            f"{res['games']} | "
            f"{res['avg']:.2f} | "
            f"{res['median']:.2f} | "
            f"{res['min']} | "
            f"{res['max']} | "
            f"{res['elapsed_s']:.2f} | "
            f"{res['games_per_sec']:.2f} |"
        )
    return "\n".join(lines)


def update_readme(readme_path: str, table: str) -> None:
    marker_start = "<!-- benchmarks:start -->"
    marker_end = "<!-- benchmarks:end -->"
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    block = f"{marker_start}\n{table}\n{marker_end}"

    if marker_start in content and marker_end in content:
        pre, rest = content.split(marker_start, 1)
        _, post = rest.split(marker_end, 1)
        content = pre + block + post
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + block + "\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark num9 strategies over random games."
    )
    parser.add_argument("--games", type=int, default=1000, help="Games per strategy.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="If set, write/update a benchmark table in README.md (between markers).",
    )
    parser.add_argument(
        "--readme-path",
        type=str,
        default="README.md",
        help="Path to README to update when --update-readme is set.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)

    strategies: list[tuple[str, Callable[[Board, Piece], None]]] = [
        ("place_randomly", lambda b, p: b.place_randomly(p)),
        ("go_up_randomly", lambda b, p: b.go_up_randomly(p)),
        ("edges_then_up", lambda b, p: b.choose_move_with_most_edges_touching(p)),
        ("solid_base_high_top", lambda b, p: b.choose_solid_base_high_top(p)),
        # ("maximize_lookahead", lambda b, p: b.maximize_lookahead(p)),
    ]

    results = []
    for name, strat in strategies:
        print(f"Running {name} for {args.games} games...")
        res = run_strategy(name, strat, args.games)
        results.append(res)
        print(
            f"{name}: avg={res['avg']:.2f}, median={res['median']:.2f}, "
            f"min={res['min']}, max={res['max']}, "
            f"time={res['elapsed_s']:.2f}s ({res['games_per_sec']:.2f} games/s)"
        )

    print("\nSummary:")
    for res in results:
        print(
            f"{res['name']:22s} avg={res['avg']:.2f} "
            f"median={res['median']:.2f} min={res['min']:3d} max={res['max']:3d} "
            f"time={res['elapsed_s']:.2f}s ({res['games_per_sec']:.2f} games/s)"
        )

    if args.update_readme:
        table = render_markdown_table(results)
        update_readme(args.readme_path, table)
        print(f"\nREADME updated at {args.readme_path} (benchmarks section).")


if __name__ == "__main__":
    main()
