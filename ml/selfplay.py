"""
Learn to play NMBR 9 from self-play, in the style of TD-Gammon.

A small convolutional network looks at the board after a candidate move
(see `num9_core.encode_afterstates`) plus the tiles still to come, and
predicts how many more points the rest of the game will score. The player
takes the move with the best points-now + predicted-points-later, and
sometimes a random one to explore. After each game, every position it chose
is labelled with the points actually scored from there on, and the network
trains on a buffer of recent positions. Nothing but the game's score is used:
no hand-written strategy and no engine labels.

Usage (CPU is fine; needs torch and numpy, and num9_core built with maturin):
    python ml/selfplay.py --minutes 30 --out runs/first
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import num9_core as n9

PLANES, CROP = n9.PLANES, n9.CROP
GLOBALS = 11  # copies left of each value (0-2, scaled) and tiles left
TARGET_SCALE = 50.0  # future points are predicted divided by this
STANDARD_TILES = [v for v in range(10) for _ in range(2)]


class ValueNet(nn.Module):
    """Board planes + remaining tiles -> expected future points / TARGET_SCALE."""

    def __init__(self, c1: int = 16, c2: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(PLANES, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1, stride=2), nn.ReLU(),  # 16x16
            nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.ReLU(),  # 8x8
            nn.Conv2d(c2, c2, 3, padding=1, stride=2), nn.ReLU(),  # 4x4
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(c2 * 16 + GLOBALS, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(planes), globals_], dim=1)).squeeze(1)


def deal(seed: int) -> list[int]:
    tiles = list(STANDARD_TILES)
    random.Random(seed).shuffle(tiles)
    return tiles


def remaining_globals(tiles: list[int], k: int) -> np.ndarray:
    """Tiles still to come after tile `k` is placed."""
    g = np.zeros(GLOBALS, np.float32)
    for v in tiles[k + 1:]:
        g[v] += 0.5
    g[10] = (len(tiles) - k - 1) / len(tiles)
    return g


def play(model: ValueNet, tiles: list[int], eps: float, rng: random.Random, per_layer=None):
    """Play one game. Returns the final score and, for each tile but the
    last, (packed planes of the chosen afterstate, globals, score after it)."""
    board = n9.Board()
    visited = []
    for k, value in enumerate(tiles):
        moves, data = n9.encode_afterstates(board, value, per_layer)
        planes = np.frombuffer(data, np.uint8).reshape(len(moves), PLANES, CROP, CROP)
        g = remaining_globals(tiles, k)
        if len(moves) == 1:
            pick = 0
        elif rng.random() < eps:
            pick = rng.randrange(len(moves))
        else:
            with torch.inference_mode():
                future = model(
                    torch.from_numpy(planes).float(),
                    torch.from_numpy(g).expand(len(moves), GLOBALS),
                )
            now = torch.tensor([value * m[0] for m in moves], dtype=torch.float32)
            pick = int(torch.argmax(now / TARGET_SCALE + future))
        layer, i, j, rot = moves[pick]
        board.place(value, (layer, i, j), rot)
        if k + 1 < len(tiles):
            visited.append((np.packbits(planes[pick]), g, board.score()))
    return board.score(), visited


class Buffer:
    """The most recent `capacity` positions (planes bit-packed)."""

    def __init__(self, capacity: int):
        packed = PLANES * CROP * CROP // 8
        self.planes = np.zeros((capacity, packed), np.uint8)
        self.globals = np.zeros((capacity, GLOBALS), np.float32)
        self.target = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.next = capacity, 0, 0

    def add(self, packed, g, target):
        k = self.next
        self.planes[k], self.globals[k], self.target[k] = packed, g, target
        self.next = (k + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, n)
        planes = np.unpackbits(self.planes[idx], axis=1).reshape(n, PLANES, CROP, CROP)
        return (
            torch.from_numpy(planes).float(),
            torch.from_numpy(self.globals[idx]),
            torch.from_numpy(self.target[idx] / TARGET_SCALE),
        )


def evaluate(model: ValueNet, seeds: range, per_layer=None) -> float:
    """Average score playing greedily on fixed deals."""
    rng = random.Random(0)
    return float(np.mean([play(model, deal(s), 0.0, rng, per_layer)[0] for s in seeds]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--minutes", type=float, default=30, help="Wall-clock budget.")
    p.add_argument("--out", type=Path, default=Path("runs/selfplay"))
    p.add_argument("--games-per-iter", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=32, help="Gradient steps per iteration.")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=100_000, help="Positions kept for training.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=0.1, help="Chance of a random move while learning.")
    p.add_argument("--per-layer", type=int, default=None,
                   help="Only consider monte_carlo's candidate moves (this many per level).")
    p.add_argument("--eval-every", type=float, default=5, help="Minutes between evaluations.")
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    buf = Buffer(args.buffer)
    eval_seeds = range(10**9, 10**9 + args.eval_games)  # never used for training

    log = open(args.out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["minutes", "games", "positions", "loss", "train_avg_score", "eval_score"])
    start = time.time()
    next_eval = 0.0
    games, loss_avg, play_time, train_time = 0, float("nan"), 0.0, 0.0
    print(f"baselines on this kind of deal: random ~1, edges_then_up ~57, low_to_base ~63, monte_carlo ~86")
    while True:
        minutes = (time.time() - start) / 60
        if minutes >= next_eval:
            model.eval()
            t = time.time()
            score = evaluate(model, eval_seeds, args.per_layer)
            print(f"[{minutes:6.1f} min] {games:6d} games  eval avg score {score:6.2f}  "
                  f"(eval took {time.time() - t:.0f}s)", flush=True)
            writer.writerow([f"{minutes:.2f}", games, buf.size, f"{loss_avg:.4f}", "", f"{score:.2f}"])
            log.flush()
            torch.save(model.state_dict(), args.out / "model.pt")
            next_eval = minutes + args.eval_every
            if minutes >= args.minutes:
                break

        model.eval()
        t = time.time()
        scores = []
        for _ in range(args.games_per_iter):
            final, visited = play(model, deal(rng.getrandbits(62)), args.eps, rng, args.per_layer)
            scores.append(final)
            for packed, g, score_after in visited:
                buf.add(packed, g, final - score_after)
        games += args.games_per_iter
        play_time += time.time() - t

        model.train()
        t = time.time()
        losses = []
        for _ in range(args.train_steps):
            x, g, y = buf.sample(args.batch, np_rng)
            loss = F.smooth_l1_loss(model(x, g), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_time += time.time() - t
        loss_avg = float(np.mean(losses))
        minutes = (time.time() - start) / 60
        writer.writerow([f"{minutes:.2f}", games, buf.size, f"{loss_avg:.4f}", f"{np.mean(scores):.2f}", ""])
        print(f"[{minutes:6.1f} min] {games:6d} games  {games / (play_time or 1):5.1f} games/s playing  "
              f"loss {loss_avg:.4f}  self-play avg {np.mean(scores):6.2f}  "
              f"(time: {play_time / 60:.1f} min playing, {train_time / 60:.1f} min training)", flush=True)


if __name__ == "__main__":
    main()
