"""
Benchmark all strategies over many random games.

Usage:
    python benchmark_strategies.py --games 1000 --seed 1
    python benchmark_strategies.py --games 1000 --threads 8 --strategies edges_then_up
"""

import argparse
import random
import time
from statistics import mean, median

import num9_core


def run_strategy(name: str, games: int, seed: int, threads: int) -> dict[str, float]:
    start = time.perf_counter()
    scores = num9_core.play_games(name, games, seed=seed, threads=threads)
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
    parser.add_argument("--games", type=int, default=100, help="Games per strategy.")
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
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Worker threads (games/s scales with this).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=num9_core.STRATEGIES,
        default=num9_core.STRATEGIES,
        help="Strategies to run (default: all).",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.getrandbits(64)

    results = []
    for name in args.strategies:
        print(f"Running {name} for {args.games} games...")
        res = run_strategy(name, args.games, seed, args.threads)
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
