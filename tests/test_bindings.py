"""Tests for the `num9_core` Python bindings.

Engine behaviour is checked against the original Python implementation by
the golden tests in `rust/tests/golden.rs` (`cargo test`); these tests cover
the Python-facing API. Build first: `maturin develop --release -m rust/Cargo.toml`.
"""

import random

import pytest

num9_core = pytest.importorskip("num9_core")


def test_piece_array_rotates_clockwise():
    assert num9_core.piece_array(1, 0) == [[1, 1], [0, 1], [0, 1], [0, 1]]
    assert num9_core.piece_array(1, 1) == [[0, 0, 0, 1], [1, 1, 1, 1]]
    assert num9_core.piece_array(0, 0)[1] == [10, 0, 10]  # piece 0 is stored as 10


def test_first_move_and_overlap():
    board = num9_core.Board()
    assert board.find_valid_moves(4) == [(0, 80, 60, 0)]
    assert board.is_valid_move(4, (0, 80, 60), 0)
    assert not board.is_valid_move(4, (0, 80, 61), 0)
    board.place(9, (0, 80, 60), 0)
    before = board.values()
    with pytest.raises(num9_core.InvalidPlacementError):
        board.place(5, (0, 81, 60), 0)
    assert board.values() == before  # a failed placement leaves the board untouched
    with pytest.raises(num9_core.InvalidPlacementError):
        board.place(5, (0, 158, 60), 0)
    with pytest.raises(ValueError):
        board.place(10, (0, 20, 20), 0)
    assert len(board.values()) == num9_core.DEPTH * num9_core.HEIGHT * num9_core.WIDTH


def test_copy_is_independent():
    board = num9_core.play_game("edges_then_up", seed=5)
    clone = board.copy()
    move = board.find_valid_moves(3)[0]
    clone.place(3, move[:3], move[3])
    assert len(clone) == len(board) + 1


def test_undo_restores_previous_state():
    board = num9_core.play_game("edges_then_up", seed=8)
    before = board.copy()
    move = board.find_valid_moves(6)[0]
    board.place(6, move[:3], move[3])
    assert board.undo() == (6, *move)
    assert board.values() == before.values() and board.score() == before.score()
    assert board.find_valid_moves(6) == before.find_valid_moves(6)
    assert num9_core.Board().undo() is None


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


@pytest.mark.parametrize("name", num9_core.STRATEGIES)
def test_strategies_play_legal_moves(name):
    tiles = list(range(10)) * 2
    random.Random(17).shuffle(tiles)
    game = num9_core.play_game(name, seed=1, tiles=tiles)
    assert [p[0] for p in game.pieces] == tiles
    replay = num9_core.Board()
    for value, layer, i, j, r in game.pieces:
        assert (layer, i, j, r) in replay.find_valid_moves(value)
        moves, candidates = num9_core.decide(name, replay, value)
        assert candidates and all(0 <= k < len(moves) for k in candidates)
        replay.place(value, (layer, i, j), r)
    assert replay.score() == game.score()
    assert replay.score() == sum(v * layer for v, layer, *_ in game.pieces)


def test_choose_move_picks_a_tied_move():
    board = num9_core.play_game("edges_then_up", seed=2, tiles=[5, 6, 7, 8, 9, 1])
    moves, candidates = num9_core.decide("solid_base_high_top", board, 4)
    for seed in range(5):
        move = num9_core.choose_move("solid_base_high_top", board, 4, seed=seed)
        assert move in [moves[k] for k in candidates]


def test_custom_weights_change_decisions():
    default = num9_core.play_games("solid_base_high_top", 20, seed=0)
    custom = num9_core.play_games(
        "solid_base_high_top", 20, seed=0, weights={"height_weight": 0.0}
    )
    assert default != custom


def test_str_renders_layers():
    board = num9_core.play_game("edges_then_up", seed=4)
    text = str(board)
    assert text.splitlines()[0].strip("-") == " LAYER 0 "
    assert "LAYER 1" in text
    assert str(num9_core.Board()) == ""


def test_bad_arguments():
    with pytest.raises(ValueError):
        num9_core.play_games("nope", 1)
    with pytest.raises(ValueError):
        num9_core.decide("edges_then_up", num9_core.Board(), 3, {"fill_weight": 1.0})
    with pytest.raises(ValueError):
        num9_core.decide("solid_base_high_top", num9_core.Board(), 3, {"typo": 1.0})
