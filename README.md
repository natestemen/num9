# [NMBR 9](https://boardgamegeek.com/boardgame/217449/nmbr-9)

but in python


## stats

statistics here are small.
usually generated from a sequence of 100 random games.

| method     | average score | average duration (s) |
| ---------- | ------------- | -------------------- |
| random     | 1             | 2.39                 |
| up         | 14            | 1.49                 |
| edges + up | 55            | 1.17                 |

<!-- benchmarks:start -->
| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 100 | 1.05 | 0.00 | 0 | 9 | 272.12 | 0.37 |
| go_up_randomly | 100 | 15.49 | 12.00 | 2 | 44 | 174.16 | 0.57 |
| edges_then_up | 100 | 58.13 | 57.00 | 36 | 97 | 152.75 | 0.65 |
| solid_base_high_top | 100 | 50.37 | 50.00 | 26 | 81 | 210.58 | 0.47 |
<!-- benchmarks:end -->

## rust engine

`rust/` is a Rust port of the engine (`Board`, move generation, scoring) and
all strategies in `num9/strategies.py`, exposed to Python as `num9_core`.
It produces exactly the same moves, in the same order, as
`num9.Board.find_valid_moves`, and makes the same decisions as each Python
strategy. It is roughly 2,000–3,000× faster.

```sh
pip install maturin pytest pytest-xdist
maturin develop --release -m rust/Cargo.toml   # builds and installs num9_core

python -c "import num9_core; print(num9_core.play_games('edges_then_up', 1000, seed=0)[:10])"
```

```python
import num9_core

board = num9_core.Board()
moves = board.find_valid_moves(7)              # [(layer, i, j, rotation), ...]
board.place(7, moves[0][:3], moves[0][3])
num9_core.choose_move("solid_base_high_top", board, 3, seed=1)
num9_core.decide("edges_then_up", board, 3)    # (moves, indices random.choice picks from)
num9_core.play_games("solid_base_high_top", 10_000, seed=0, threads=8)
```

### testing

```sh
cargo test --release --manifest-path rust/Cargo.toml   # Rust unit tests
pytest tests -n 8                                      # Rust vs Python equivalence
NUM9_EQUIV_GAMES=50 pytest tests -n 8                  # longer sweep (~50 min)
```

`tests/test_rust_equivalence.py` replays games on both engines and checks,
at every position:

- `find_valid_moves` for all ten piece values: same moves, same order
- `num_edges_touching` for every legal move, and `is_valid_move` against
  every placement near the pieces (legal and illegal)
- identical board arrays and scores after every placement
- for each strategy, the same set of tied moves handed to `random.choice`
  (patched to make a known pick so both engines follow the same game)

It also replays rare games that reach layer 4.

### benchmarks

`python benchmark_engines.py --py-games 20 --rs-games 1000` (8-core machine):

| | Python | Rust (via Python) | speedup |
| --- | --- | --- | --- |
| `find_valid_moves` | 103 ms / call | 44 µs / call | ~2,400× |
| edges_then_up game | 0.48 games/s | 1,167 games/s (4,192 on 8 threads) | ~2,400× |
| solid_base_high_top game | 0.40 games/s | 1,151 games/s (4,039 on 8 threads) | ~2,900× |
| maximize_lookahead game | ~47 min / game (est.) | 1.2 games/s | ~3,300× |

`python benchmark_strategies.py --engine rust --games 1000 --seed 1`:

<!-- benchmarks-rust:start -->
| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 1000 | 1.10 | 0.00 | 0 | 15 | 0.90 | 1105.44 |
| go_up_randomly | 1000 | 14.90 | 13.00 | 1 | 57 | 0.85 | 1182.28 |
| edges_then_up | 1000 | 56.78 | 55.00 | 23 | 97 | 0.71 | 1409.81 |
| solid_base_high_top | 1000 | 48.93 | 47.00 | 14 | 107 | 0.84 | 1191.23 |
| maximize_lookahead | 1000 | 60.21 | 59.00 | 30 | 113 | 825.66 | 1.21 |
<!-- benchmarks-rust:end -->
