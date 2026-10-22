# [NMBR 9](https://boardgamegeek.com/boardgame/217449/nmbr-9)

but in rust, driven from python

The engine (board, move generation, scoring) and all strategies live in
`rust/` and are exposed to Python as `num9_core`. They started as a Python
package; the Rust port reproduces it exactly and is roughly 2,000–3,000×
faster (see [history](#history)).

## setup

```sh
pip install maturin pytest numpy tqdm
maturin develop --release -m rust/Cargo.toml   # builds and installs num9_core
```

## usage

```python
import num9_core

board = num9_core.Board()
moves = board.find_valid_moves(7)              # [(layer, i, j, rotation), ...]
board.place(7, moves[0][:3], moves[0][3])
print(board)                                   # emoji rendering of each layer
board.score()
board.undo()                                   # take back the last piece (cheaper than copy())

num9_core.STRATEGIES                           # names of the built-in strategies
num9_core.choose_move("solid_base_high_top", board, 3, seed=1)
num9_core.decide("edges_then_up", board, 3)    # (moves, indices of tied best moves)
num9_core.play_game("edges_then_up", seed=0)   # -> finished Board
num9_core.play_games("solid_base_high_top", 10_000, seed=0, threads=8)  # -> scores
num9_core.play_games("solid_base_high_top", 100, weights={"height_weight": 12.0})
```

Scripts:

- `test.py`: play and print one game
- `many.py`: play 100 games, print the good ones
- `benchmark_strategies.py`: benchmark every strategy (`--update-readme` refreshes the table below)
- `optimize_params.py`: random-search the `solid_base_high_top` weights
- `cargo run --release --manifest-path rust/Cargo.toml --bin bench`: the same benchmark without Python

## play

```sh
cargo run --release --manifest-path rust/Cargo.toml -p num9-cli
# or install it as `num9`:
cargo install --path rust/cli
num9                 # a random deal
num9 --seed 42       # the same deal every time
num9 --tiles 8,3,7,9,8,6,3,2,5,0,6,1,9,2,5,1,4,7,0,4
```

Slide the tile around with the arrow keys (or the mouse); it lands on
whatever is underneath, shows the level it would sit on and the points it
would score, and turns red with the reason when it can't go there. Space
rotates, enter places, u undoes.

Press **h** to ask the engine for the best move: it plays the rest of the
game out hundreds of times from each candidate (Monte Carlo rollouts) and
ranks them by expected final score, also rating the spot you're on. **g**
moves your tile through its top picks. At the end, the engine plays the same
deal so you can compare, and searches for the best play of that exact deal
(now that the order is known); press **b** to step through it next to your
own game. **?** shows all the keys.

To search a deal on its own: `cargo run --release --manifest-path
rust/Cargo.toml --bin solve -- --seed 42 --level 3 --per-layer 8`.

## records

The maximum possible score, with the order of the tiles chosen freely, is an
open problem ("Nmbr9 as a Constraint Programming Challenge", Lagerkvist 2020,
cites 229 as the best known, from a BoardGameGeek thread). The `tower`
command designs stacks from the top down and finds a play scoring **236**,
reaching level 6: see `records/free-order-236.txt`, and check it with
`python records/check_play.py records/free-order-236.txt` (standard library
only, independent of the engine). To step through it in the terminal game:
`cargo run --release --manifest-path rust/Cargo.toml -p num9-cli -- --replay
records/free-order-236.txt` (or `num9 --replay ...` if installed).

## testing

```sh
cargo test --release --manifest-path rust/Cargo.toml --workspace  # engine, golden and CLI tests
pytest tests                                           # Python bindings
```

`rust/tests/golden/` holds fixtures recorded from the original Python
engine: for 27 games (including rare ones reaching layer 4), the legal moves
of all ten pieces at every position, the edge counts of every legal move, and
the board after every placement; and for 55 strategy games, every decision's
legal moves, tie set and pick. `rust/tests/golden.rs` replays them, so the
engine still matches the Python behaviour it replaced (`generate.py` there
documents how they were made).

## stats

`python benchmark_strategies.py --games 1000 --seed 1 --update-readme` (one thread):

<!-- benchmarks:start -->
| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 1000 | 1.10 | 0.00 | 0 | 15 | 0.15 | 6838.79 |
| go_up_randomly | 1000 | 14.90 | 13.00 | 1 | 57 | 0.13 | 7438.11 |
| edges_then_up | 1000 | 56.78 | 55.00 | 23 | 97 | 0.16 | 6142.68 |
| solid_base_high_top | 1000 | 48.93 | 47.00 | 14 | 107 | 0.21 | 4686.81 |
| maximize_lookahead | 1000 | 60.21 | 59.00 | 30 | 113 | 64.98 | 15.39 |
<!-- benchmarks:end -->

## history

The engine was first written in Python (`num9/`, removed after the port;
see the git history). Its last benchmark, 100 games per strategy:

| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 100 | 1.05 | 0.00 | 0 | 9 | 272.12 | 0.37 |
| go_up_randomly | 100 | 15.49 | 12.00 | 2 | 44 | 174.16 | 0.57 |
| edges_then_up | 100 | 58.13 | 57.00 | 36 | 97 | 152.75 | 0.65 |
| solid_base_high_top | 100 | 50.37 | 50.00 | 26 | 81 | 210.58 | 0.47 |

Head-to-head on the same 8-core machine, before the Python engine was removed:

| | Python | Rust (via Python) | speedup |
| --- | --- | --- | --- |
| `find_valid_moves` | 103 ms / call | 44 µs / call | ~2,400× |
| edges_then_up game | 0.48 games/s | 1,167 games/s (4,192 on 8 threads) | ~2,400× |
| solid_base_high_top game | 0.40 games/s | 1,151 games/s (4,039 on 8 threads) | ~2,900× |
| maximize_lookahead game | ~47 min / game (est.) | 1.2 games/s | ~3,300× |
