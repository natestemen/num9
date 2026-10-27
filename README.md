# [NMBR 9](https://boardgamegeek.com/boardgame/217449/nmbr-9)

but in rust, driven from python

The engine (board, move generation, scoring) and all strategies live in
`rust/` and are exposed to Python as `num9_core`. They started as a Python
package; the Rust port reproduces it exactly and is roughly 2,000–3,000×
faster.

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

## website

**[natestemen.github.io/num9](https://natestemen.github.io/num9/)**: the
terminal game in the browser, running the same engine compiled to
WebAssembly (`rust/web`). Play with hints, see how the engine and its search
for the best play do on your deal, try the exactly solved small-deck
puzzles, and step through the record plays. To build it locally (needs
`rustup target add wasm32-unknown-unknown`):

```sh
site/build.sh                                  # into site/_site
python3 -m http.server -d site/_site 8000      # then open localhost:8000
```

Pushes to `main` rebuild it and publish it to the `gh-pages` branch
(`.github/workflows/website.yml`).

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

No play can score more than **354**. `records/bound/upper_bound.py` proves
it with an integer program over which tiles go on which level, using shape
facts computed exhaustively by `cargo run --release --bin widening` (how
much wider the level below one or two tiles must be, and how much area a
level of two or three tiles can hold up). The table is in
`records/bound/widening.tsv`, and `records/bound/check_constraints.py`
confirms every recorded play satisfies the constraints.

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

## strategies

Each strategy picks where the tile in hand goes, knowing which tiles are
still to come but not their order (`num9_core.STRATEGIES`; results under
[stats](#stats)).

- `place_randomly`: any legal spot, uniformly at random. The baseline for
  how much thinking helps.
- `go_up_randomly`: a random spot on the highest level the tile can reach.
- `edges_then_up`: 0s and 1s on the ground, everything else as high as it
  can go; among those spots, the one touching the most edges, which keeps
  the stack compact.
- `solid_base_high_top`: scores every spot by a weighted mix of how full and
  hole-free its level is, the support under it, edges touching, points
  scored and the stack's shape (weights tuned by `optimize_params.py`), and
  picks among the best.
- `maximize_lookahead`: looks one tile ahead, scoring each spot by its
  points plus how high the tiles still to come could then go on average
  (weighted by their values and how many of each are left).
- `low_to_base`: `edges_then_up` with a higher cut-off: tiles worth 4 or
  less go on the ground and the rest go up, except for the last four tiles,
  which all go as high as they can.
- `monte_carlo`: shortlists the snuggest few spots on each level and plays
  the rest of the game out 64 times from each, shuffling the tiles still to
  come and finishing with `low_to_base`; the best average final score wins.
  The terminal game's hints use the same method.
- `keep_support`: `low_to_base`, but a tile going up also avoids stranding
  support: cells of the level below that no tile still to come could ever
  rest on (`num9_core::support`). Spots score `edges touching - 0.15 × cells
  stranded`; on the same 10,000 deals it beats `low_to_base` by 2.41 ± 0.13
  points a game.

Three searches look for the best play outright instead, once the order is
known or when it is free to choose:

- `solve` (nested Monte Carlo search, `num9_core::solver`): finds a
  high-scoring play of a known deal by trying every candidate at each step,
  judging each with a smaller search from there, and following the best
  complete game found. It finds the "best play" shown after a game.
- `maxscore` (Nested Rollout Policy Adaptation, `num9_core::nrpa`): learns
  a weight for each move from its own best games, steering its random
  playouts towards them; for a fixed deal or with the order free.
- `tower` (`num9_core::tower`): designs stacks from the top down for the
  highest possible score: chooses what goes on top, covers it as cheaply as
  possible level by level with a beam search, then turns the design into a
  legal play. It found the 236 record.

## head to head

`cargo run --release --bin arena -- [--games N] [--baseline SPEC] SPEC...`
plays strategies on the same deals, so differences are measured on matched
games: far more precisely than comparing separate averages. A SPEC is a
strategy with optional settings, e.g. `monte_carlo:policy=keep_support`.

Does Monte Carlo get better if its simulated games are played by
`keep_support` instead of `low_to_base`? Barely: much less than spending
the same extra time on more simulations of the cheap policy (its estimates
are noisy, so more of them helps most):

<!-- arena:start -->
`arena --games 1000 --seed 1`: 1000 deals, the same for every strategy; ± is one standard error.

| strategy | avg | vs monte_carlo on the same deals | wins / ties / losses | games/s (8 threads) |
| --- | --- | --- | --- | --- |
| `monte_carlo` | 86.41 ± 0.36 |  |  | 5.4 |
| `monte_carlo:policy=keep_support` | 87.14 ± 0.35 | +0.72 ± 0.42 | 508 / 40 / 452 | 0.9 |
| `monte_carlo:rollouts=320` | 88.57 ± 0.35 | +2.15 ± 0.41 | 553 / 49 / 398 | 0.9 |
<!-- arena:end -->

## stats

`python benchmark_strategies.py --games 1000 --seed 1 --update-readme` (one thread):

<!-- benchmarks:start -->
| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 1000 | 1.10 | 0.00 | 0 | 15 | 0.09 | 10737.16 |
| go_up_randomly | 1000 | 14.90 | 13.00 | 1 | 57 | 0.10 | 10129.40 |
| edges_then_up | 1000 | 56.78 | 55.00 | 23 | 97 | 0.12 | 8046.98 |
| solid_base_high_top | 1000 | 48.93 | 47.00 | 14 | 107 | 0.16 | 6111.65 |
| maximize_lookahead | 1000 | 60.21 | 59.00 | 30 | 113 | 51.36 | 19.47 |
| low_to_base | 1000 | 62.74 | 61.00 | 33 | 115 | 0.12 | 8029.16 |
| monte_carlo | 1000 | 86.41 | 86.00 | 54 | 142 | 505.86 | 1.98 |
| keep_support | 1000 | 64.78 | 63.00 | 35 | 125 | 0.70 | 1419.30 |
<!-- benchmarks:end -->
