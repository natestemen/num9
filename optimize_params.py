"""
Random-search tuner for choose_solid_base_high_top weights.

Usage:
    python optimize_params.py --coarse-trials 40 --refine-trials 30 --games-per-trial 5 --seed 123
"""

import argparse
import statistics
from random import Random

from num9 import Board, Piece
from num9.strategies import choose_solid_base_high_top

WEIGHT_RANGES = {
    "fill_weight": (0.2, 1.2),
    "hole_penalty": (0.2, 2.0),
    "support_weight": (2.0, 8.0),
    "edge_weight": (0.5, 4.0),
    "height_weight": (10.0, 20.0),
    "aspect_penalty_weight": (0.5, 3.0),
    "low_height_penalty_weight": (2.0, 15.0),
}


def sample_weights(rng: Random) -> dict[str, float]:
    """Sample a weight vector from coarse ranges."""
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in WEIGHT_RANGES.items()}


def sample_weights_near(
    base: dict[str, float], rng: Random, scale: float
) -> dict[str, float]:
    """Sample weights near a base point, clamped to overall ranges."""
    weights: dict[str, float] = {}
    for key, value in base.items():
        lo, hi = WEIGHT_RANGES[key]
        span = (hi - lo) * scale
        weights[key] = rng.uniform(max(lo, value - span), min(hi, value + span))
    return weights


def play_game(weights: dict[str, float], rng: Random) -> int:
    tiles = list(range(10)) * 2
    rng.shuffle(tiles)
    board = Board()
    for num in tiles:
        piece = Piece(num)
        choose_solid_base_high_top(board, piece, **weights)
    return board.score()


def evaluate_weights(weights: dict[str, float], games: int, rng: Random) -> float:
    scores = [play_game(weights, rng) for _ in range(games)]
    return statistics.mean(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune choose_solid_base_high_top weights.")
    parser.add_argument("--coarse-trials", type=int, default=30, help="Coarse random weight samples.")
    parser.add_argument("--refine-trials", type=int, default=20, help="Refinement samples around top performers.")
    parser.add_argument("--top-k", type=int, default=5, help="Top K from coarse phase to seed refinement.")
    parser.add_argument("--refine-scale", type=float, default=0.25, help="Fraction of the original ranges for refinement sampling.")
    parser.add_argument("--games-per-trial", type=int, default=5, help="Games simulated per weight set.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args()

    rng = Random(args.seed)

    best_result: dict[str, object] | None = None
    trial_records: list[dict[str, object]] = []
    # Phase 1: coarse search
    for trial in range(1, args.coarse_trials + 1):
        weights = sample_weights(rng)
        avg_score = evaluate_weights(weights, args.games_per_trial, rng)
        trial_records.append({"weights": weights, "avg_score": avg_score})

        if best_result is None or avg_score > best_result["avg_score"]:
            best_result = {"avg_score": avg_score, "weights": weights, "trial": trial}

        print(
            f"[coarse {trial}/{args.coarse_trials}] avg_score={avg_score:.2f} "
            f"weights={weights}"
        )

    # Phase 2: refine around top performers
    if trial_records and args.refine_trials > 0:
        top_k = sorted(trial_records, key=lambda t: t["avg_score"], reverse=True)[
            : max(1, min(args.top_k, len(trial_records)))
        ]
        for trial in range(1, args.refine_trials + 1):
            seed_weights = top_k[(trial - 1) % len(top_k)]["weights"]
            weights = sample_weights_near(seed_weights, rng, args.refine_scale)
            avg_score = evaluate_weights(weights, args.games_per_trial, rng)
            trial_records.append({"weights": weights, "avg_score": avg_score})

            if best_result is None or avg_score > best_result["avg_score"]:
                best_result = {
                    "avg_score": avg_score,
                    "weights": weights,
                    "trial": args.coarse_trials + trial,
                }

            print(
                f"[refine {trial}/{args.refine_trials}] avg_score={avg_score:.2f} "
                f"weights={weights}"
            )

    if trial_records:
        scores = [t["avg_score"] for t in trial_records]
        score_mean = statistics.mean(scores)
        score_std = statistics.pstdev(scores)
        print("\nParameter correlations (Pearson):")
        for key in trial_records[0]["weights"]:
            xs = [t["weights"][key] for t in trial_records]
            x_mean = statistics.mean(xs)
            x_std = statistics.pstdev(xs)
            corr = 0.0
            if x_std != 0 and score_std != 0:
                cov = sum((x - x_mean) * (s - score_mean) for x, s in zip(xs, scores)) / len(xs)
                corr = cov / (x_std * score_std)
            print(f"  {key:20s}: {corr: .3f}")

    if best_result:
        print("\nBest found:")
        print(f"  trial={best_result['trial']}")
        print(f"  avg_score={best_result['avg_score']:.2f}")
        print(f"  weights={best_result['weights']}")


if __name__ == "__main__":
    main()
