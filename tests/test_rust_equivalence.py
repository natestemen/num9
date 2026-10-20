"""Check that the Rust engine (`num9_core`) reproduces the Python one exactly.

Build the extension first: `maturin develop --release -m rust/Cargo.toml`.

The Python engine is slow, so the number of games is configurable:

    NUM9_EQUIV_GAMES=50 pytest tests/test_rust_equivalence.py -n 8

Strategy tests are cheaper and some rules (e.g. which values edges_then_up
keeps on layer 0) only show up in some games, so they default to three
times as many games (override with NUM9_EQUIV_STRATEGY_GAMES).

Python indexes the board with plain numpy slicing, so a piece within a few
cells of the edge wraps around (negative indices) or fails; the Rust engine
simply never places off the board. Twenty pieces never get near the edge in
practice, and `assert_away_from_edges` checks that every tested position
stays in the region where the two are expected to agree.
"""

import os
import random

import numpy as np
import pytest

num9_core = pytest.importorskip("num9_core")

import num9.strategies as py_strategies  # noqa: E402
from num9 import Board, Piece  # noqa: E402
from num9.board import BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH  # noqa: E402

GAMES = int(os.environ.get("NUM9_EQUIV_GAMES", "4"))
STRATEGY_GAMES = int(os.environ.get("NUM9_EQUIV_STRATEGY_GAMES", 3 * GAMES))
SHAPE = (BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH)
EDGE_MARGIN = 10

PY_STRATEGIES = {
    "place_randomly": py_strategies.place_randomly,
    "go_up_randomly": py_strategies.go_up_randomly,
    "edges_then_up": py_strategies.choose_move_with_most_edges_touching,
    "solid_base_high_top": py_strategies.choose_solid_base_high_top,
}


def tiles_for(seed: int) -> list[int]:
    tiles = list(range(10)) * 2
    random.Random(seed).shuffle(tiles)
    return tiles


def py_moves(board: Board, value: int) -> list[tuple[int, int, int, int]]:
    return [tuple(int(x) for x in m) for m in board.find_valid_moves(Piece(value))]


def assert_same_board(py: Board, rs) -> None:
    rs_values = np.array(rs.values(), dtype=np.int32).reshape(SHAPE)
    np.testing.assert_array_equal(rs_values, py.board)
    assert rs.score() == py.score()
    assert len(rs) == len(py.piece_sequence)


def assert_away_from_edges(board: Board) -> None:
    if not board.piece_sequence:
        return
    _, rows, cols = np.nonzero(board.board)
    assert rows.min() >= EDGE_MARGIN and rows.max() < BOARD_HEIGHT - EDGE_MARGIN
    assert cols.min() >= EDGE_MARGIN and cols.max() < BOARD_WIDTH - EDGE_MARGIN


def place_both(py: Board, rs, value: int, move) -> None:
    layer, i, j, r = move
    py.place(Piece(value), (layer, i, j), rotation=r)
    rs.place(value, (layer, i, j), r)
    assert_same_board(py, rs)


# --------------------------------------------------------------------------
# Pieces


@pytest.mark.parametrize("value", range(10))
def test_piece_arrays(value):
    piece = Piece(value)
    for rotation in range(4):
        piece.rotation = rotation
        assert num9_core.piece_array(value, rotation) == piece.array


def test_piece_bounding_box_is_full():
    # solid_base_high_top relies on every piece touching all four sides of
    # its array, so the piece's footprint is exactly its array's extent.
    for value in range(10):
        arr = np.array(Piece(value).array)
        assert arr[0].any() and arr[-1].any() and arr[:, 0].any() and arr[:, -1].any()


# --------------------------------------------------------------------------
# Move generation


def pick(policy: str, rs, value: int, moves, rng: random.Random):
    """Driver policies that steer games into varied (and tall) positions."""
    if policy == "random":
        return rng.choice(moves)
    top = max(m[0] for m in moves)
    pool = [m for m in moves if m[0] == top] if policy == "up" else moves
    if policy == "mixed" and rng.random() < 0.5:
        pool = [m for m in moves if m[0] == top]
    edges = [rs.num_edges_touching(value, m[:3], m[3]) for m in pool]
    best = max(edges)
    return rng.choice([m for m, e in zip(pool, edges) if e == best])


@pytest.mark.parametrize("policy", ["up", "mixed", "random"])
@pytest.mark.parametrize("seed", range(GAMES))
def test_move_generation_matches(seed, policy):
    """At every position of a game, for every piece value, both engines list
    exactly the same moves in the same order; edge counts and `is_valid_move`
    agree; boards and scores agree after every placement."""
    rng = random.Random(1000 + seed)
    py, rs = Board(), num9_core.Board()
    for value in tiles_for(seed):
        assert_away_from_edges(py)
        for other in range(10):
            assert rs.find_valid_moves(other) == py_moves(py, other), f"value {other}"

        moves = py_moves(py, value)
        legal = set(moves)
        for m in moves:
            assert rs.num_edges_touching(value, m[:3], m[3]) == py.num_edges_touching(
                _rotated(value, m[3]), m[:3]
            )

        # is_valid_move agrees with membership for every placement near the pieces.
        if py.piece_sequence:
            _, rows, cols = np.nonzero(py.board)
            top = max(m[0] for m in moves)
            for layer in range(top + 2):
                for r in range(4):
                    for i in range(rows.min() - 5, rows.max() + 2):
                        for j in range(cols.min() - 5, cols.max() + 2):
                            m = (layer, i, j, r)
                            assert rs.is_valid_move(value, m[:3], r) == (m in legal), m

        place_both(py, rs, value, pick(policy, rs, value, moves, rng))


def _rotated(value: int, rotation: int) -> Piece:
    piece = Piece(value)
    piece.rotation = rotation
    return piece


def test_tall_positions_match():
    """Find (rare) games that reach layer 4 and compare every position along them."""
    found = 0
    for seed in range(5000):
        rs_game = num9_core.play_game("solid_base_high_top", seed=seed)
        if max(p[1] for p in rs_game.pieces) < 4:
            continue
        py, rs = Board(), num9_core.Board()
        for value, layer, i, j, r in rs_game.pieces:
            for other in range(10):
                assert rs.find_valid_moves(other) == py_moves(py, other)
            place_both(py, rs, value, (layer, i, j, r))
        found += 1
        if found == 2:
            return
    pytest.fail("no game reached layer 4")


def test_first_move_and_overlap():
    py, rs = Board(), num9_core.Board()
    for value in range(10):
        assert rs.find_valid_moves(value) == py_moves(py, value) == [(0, 80, 60, 0)]
    assert rs.is_valid_move(4, (0, 80, 60), 0)
    assert not rs.is_valid_move(4, (0, 80, 61), 0)
    place_both(py, rs, 9, (0, 80, 60, 0))
    with pytest.raises(num9_core.InvalidPlacementError):
        rs.place(5, (0, 81, 60), 0)
    assert_same_board(py, rs)  # a failed placement leaves the board untouched
    with pytest.raises(num9_core.InvalidPlacementError):
        rs.place(5, (0, 158, 60), 0)
    with pytest.raises(ValueError):
        rs.place(10, (0, 20, 20), 0)


def test_copy_is_independent():
    rs = num9_core.play_game("edges_then_up", seed=5)
    clone = rs.copy()
    move = rs.find_valid_moves(3)[0]
    clone.place(3, move[:3], move[3])
    assert len(clone) == len(rs) + 1


# --------------------------------------------------------------------------
# Strategies


class Recorder:
    """Patches the Python engine so a strategy's random tie-break can be
    observed and steered: `choice` returns a deterministic pick, and the
    candidate list it was offered is recorded as moves."""

    def __init__(self, monkeypatch, rng: random.Random):
        self.rng = rng
        self.valid_moves = None
        self.offered = None
        self.placed = None
        original_find = Board.find_valid_moves
        original_place = Board.place

        def find(board, piece):
            self.valid_moves = original_find(board, piece)
            return self.valid_moves

        def place(board, piece, location, rotation=None):
            self.placed = (*location, rotation)
            return original_place(board, piece, location, rotation)

        def choice(seq):
            seq = list(seq)
            self.offered = seq
            return seq[self.rng.randrange(len(seq))]

        monkeypatch.setattr(Board, "find_valid_moves", find)
        monkeypatch.setattr(Board, "place", place)
        monkeypatch.setattr(py_strategies, "choice", choice)

    def offered_moves(self):
        if isinstance(self.offered[0], tuple):
            return [tuple(int(x) for x in m) for m in self.offered]
        return [tuple(int(x) for x in self.valid_moves[k]) for k in self.offered]


@pytest.mark.parametrize("name", list(PY_STRATEGIES))
@pytest.mark.parametrize("seed", range(STRATEGY_GAMES))
def test_strategy_decisions_match(monkeypatch, name, seed):
    """Every decision: same legal moves, same tie set handed to
    `random.choice` (same order); then both play the same pick."""
    recorder = Recorder(monkeypatch, random.Random(2000 + seed))
    strategy = PY_STRATEGIES[name]
    py, rs = Board(), num9_core.Board()
    for value in tiles_for(seed):
        assert_away_from_edges(py)
        strategy(py, Piece(value))
        moves, candidates = num9_core.decide(name, rs, value)
        assert moves == [tuple(int(x) for x in m) for m in recorder.valid_moves]
        assert [moves[k] for k in candidates] == recorder.offered_moves()

        chosen = tuple(int(x) for x in recorder.placed)
        assert chosen in [moves[k] for k in candidates]
        rs.place(value, chosen[:3], chosen[3])
        assert_same_board(py, rs)


def test_solid_base_high_top_custom_weights(monkeypatch):
    weights = {
        "fill_weight": 1.1,
        "hole_penalty": 0.7,
        "support_weight": 3.0,
        "edge_weight": 0.9,
        "height_weight": 11.0,
        "aspect_penalty_weight": 0.6,
        "low_height_penalty_weight": 4.0,
    }
    recorder = Recorder(monkeypatch, random.Random(7))
    py, rs = Board(), num9_core.Board()
    for value in tiles_for(99):
        py_strategies.choose_solid_base_high_top(py, Piece(value), **weights)
        moves, candidates = num9_core.decide("solid_base_high_top", rs, value, weights)
        assert [moves[k] for k in candidates] == recorder.offered_moves()
        chosen = tuple(int(x) for x in recorder.placed)
        rs.place(value, chosen[:3], chosen[3])
        assert_same_board(py, rs)


def _rust_backed_find_valid_moves(board: Board, piece: Piece):
    """`Board.find_valid_moves` answered by the Rust engine (shown equivalent
    by test_move_generation_matches), used to make the Python lookahead
    strategy fast enough to test."""
    cached = getattr(board, "_rs_cache", None)
    if cached is None or cached[0] != len(board.piece_sequence):
        rs = num9_core.Board()
        for entry in board.piece_sequence:
            value = int(entry["name"].name)
            layer = entry["layer"]
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


@pytest.mark.parametrize("seed", range(max(1, GAMES // 2)))
def test_maximize_lookahead_matches(monkeypatch, seed):
    monkeypatch.setattr(Board, "find_valid_moves", _rust_backed_find_valid_moves)
    placed = []
    original_place = Board.place

    def place(board, piece, location, rotation=None):
        placed.append((*location, rotation))
        return original_place(board, piece, location, rotation)

    monkeypatch.setattr(Board, "place", place)
    py, rs = Board(), num9_core.Board()
    for value in tiles_for(seed):
        py_strategies.maximize_lookahead(py, Piece(value))
        moves, candidates = num9_core.decide("maximize_lookahead", rs, value)
        assert len(candidates) == 1
        assert moves[candidates[0]] == tuple(int(x) for x in placed[-1])
        rs.place(value, placed[-1][:3], placed[-1][3])
        assert_same_board(py, rs)


# --------------------------------------------------------------------------
# Rust-side game helpers


def test_play_game_is_reproducible():
    a = num9_core.play_game("solid_base_high_top", seed=11)
    b = num9_core.play_game("solid_base_high_top", seed=11)
    assert a.pieces == b.pieces and a.score() == b.score()
    assert sorted(p[0] for p in a.pieces) == sorted(list(range(10)) * 2)


def test_play_games_threading_is_deterministic():
    one = num9_core.play_games("edges_then_up", 16, seed=3, threads=1)
    many = num9_core.play_games("edges_then_up", 16, seed=3, threads=4)
    assert one == many
    assert one[0] == num9_core.play_game("edges_then_up", seed=3).score()


def test_play_game_with_given_tiles_replays_on_python_engine():
    tiles = tiles_for(17)
    rs = num9_core.play_game("edges_then_up", seed=1, tiles=tiles)
    py = Board()
    for value, layer, i, j, r in rs.pieces:
        assert (layer, i, j, r) in py_moves(py, value)
        py.place(Piece(value), (layer, i, j), rotation=r)
    assert [p[0] for p in rs.pieces] == tiles
    assert py.score() == rs.score()


def test_bad_arguments():
    with pytest.raises(ValueError):
        num9_core.play_games("nope", 1)
    with pytest.raises(ValueError):
        num9_core.decide("edges_then_up", num9_core.Board(), 3, {"fill_weight": 1.0})
    with pytest.raises(ValueError):
        num9_core.decide("solid_base_high_top", num9_core.Board(), 3, {"typo": 1.0})
