from random import choice

import numpy as np

from .board import Board
from .piece import Piece


def place_randomly(board: Board, piece: Piece) -> None:
    """Find a random valid move and place the piece."""
    valid_moves = board.find_valid_moves(piece)
    layer, i, j, r = choice(valid_moves)

    board.place(piece, (layer, i, j), rotation=r)


def go_up_randomly(board: Board, piece: Piece) -> None:
    """Prioritise placing on the highest layer, choosing randomly within ties."""
    valid_moves = board.find_valid_moves(piece)
    layers = [layer for layer, *_ in valid_moves]
    max_layer = max(layers)
    option_indices = [idx for idx, layer in enumerate(layers) if layer == max_layer]
    layer, i, j, r = valid_moves[choice(option_indices)]

    board.place(piece, (layer, i, j), rotation=r)


def choose_move_with_most_edges_touching(board: Board, piece: Piece) -> None:
    """Select moves that maximise same-layer edge contacts."""
    valid_moves = board.find_valid_moves(piece)
    layers = [layer for layer, *_ in valid_moves]

    option_indices = range(len(valid_moves))
    if len(set(layers)) > 1:
        max_layer = max(layers)
        option_indices = [idx for idx, layer in enumerate(layers) if layer == max_layer]

    if piece.name in {"0", "1"}:
        option_indices = [idx for idx, layer in enumerate(layers) if layer == 0]

    edges = {}
    for idx in option_indices:
        layer, i, j, r = valid_moves[idx]
        piece.rotation = r
        edges[idx] = board.num_edges_touching(piece, (layer, i, j))

    max_edges = max(edges.values())
    moves_with_max_edges = [idx for idx, value in edges.items() if value == max_edges]
    layer, i, j, r = valid_moves[choice(moves_with_max_edges)]

    board.place(piece, (layer, i, j), rotation=r)


def choose_solid_base_high_top(
    board: Board,
    piece: Piece,
    *,
    fill_weight: float = 0.3866522473060556,
    hole_penalty: float = 0.2285685361703878,
    support_weight: float = 6.522728683569122,
    edge_weight: float = 2.209594713525341,
    height_weight: float = 13.939406411565585,
    aspect_penalty_weight: float = 2.3253084372479393,
    low_height_penalty_weight: float = 9.0,
) -> None:
    """Heuristic: keep lower layers dense with few holes; float big numbers higher."""
    valid_moves = board.find_valid_moves(piece)

    original_rotation = piece.rotation
    best_moves: list[tuple[float, int]] = []
    for idx, (layer, i, j, r) in enumerate(valid_moves):
        piece.rotation = r
        piece_height, piece_width = piece.dimensions
        piece_tiles = np.count_nonzero(piece.array)
        piece_on_board = Board.blank_board_with_piece(piece, (layer, i, j))

        layer_after = board.board[layer] + piece_on_board[layer]
        fill = int(np.count_nonzero(layer_after))

        holes_penalty = 0
        nonzero_indices = np.nonzero(layer_after)
        if nonzero_indices[0].size:
            sub_layer = layer_after[
                nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
                nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
            ]
            holes_penalty = int(np.count_nonzero(sub_layer == 0))

        edges = board.num_edges_touching(piece, (layer, i, j))

        support_ratio = 1.0
        if layer > 0 and piece_tiles:
            support_below = board.board[layer - 1][
                i : i + piece_height, j : j + piece_width
            ].copy()
            support_below[support_below > 0] = 1
            piece_mask = Board.blank_layer_with_piece(piece, (0, 0), ones=True)[
                :piece_height, :piece_width
            ]
            overlap = int(np.sum(support_below * piece_mask))
            support_ratio = overlap / piece_tiles

        height_score = int(piece.name) * layer

        footprint_indices = np.nonzero(layer_after)
        aspect_penalty = 0.0
        if footprint_indices[0].size:
            height_span = footprint_indices[0].max() - footprint_indices[0].min() + 1
            width_span = footprint_indices[1].max() - footprint_indices[1].min() + 1
            if height_span and width_span:
                aspect_ratio = max(height_span, width_span) / min(
                    height_span, width_span
                )
                aspect_penalty = aspect_ratio - 1.0

        low_height_penalty = int(piece.name) * max(0, 2 - layer)

        score = (
            fill_weight * fill
            - hole_penalty * holes_penalty
            + support_weight * support_ratio
            + edge_weight * edges
            + height_weight * height_score
            - aspect_penalty_weight * aspect_penalty
            - low_height_penalty_weight * low_height_penalty
        )
        best_moves.append((score, idx))

    piece.rotation = original_rotation
    best_score = max(best_moves, key=lambda x: x[0])[0]
    margin = abs(best_score) * 5e-2 + 1e-6
    best_indices = [idx for score, idx in best_moves if best_score - score <= margin]
    layer, i, j, r = valid_moves[choice(best_indices)]

    board.place(piece, (layer, i, j), rotation=r)


def maximize_lookahead(board: Board, piece: Piece) -> None:
    """Score moves by the number and value of future placements they enable."""
    valid_moves = board.find_valid_moves(piece)
    played = [int(p["name"].name) for p in board.piece_sequence]
    leftover = list(range(10)) * 2
    for value in played:
        leftover.remove(value)

    current_score = board.score()
    move_scores: list[float] = []
    for layer, i, j, r in valid_moves:
        piece.rotation = r
        piece_on_board = Board.blank_board_with_piece(piece, (layer, i, j))
        placement = [{"name": piece, "location": piece_on_board, "layer": layer}]
        tmp_board = Board(
            board.board + piece_on_board, piece_sequence=board.piece_sequence + placement
        )
        base_score = current_score + int(piece.name) * layer

        total_moves = 0
        weighted_layer_sum = 0
        for next_value in leftover:
            new_piece = Piece(next_value)
            next_moves = tmp_board.find_valid_moves(new_piece)
            if not next_moves:
                continue
            total_moves += len(next_moves)
            weighted_layer_sum += int(new_piece.name) * sum(
                move_layer for move_layer, *_ in next_moves
            )

        average_future_score = (
            base_score
            if total_moves == 0
            else base_score + weighted_layer_sum / total_moves
        )
        move_scores.append(average_future_score)

    best_move_index = max(enumerate(move_scores), key=lambda x: x[1])[0]
    layer, i, j, r = valid_moves[best_move_index]

    board.place(piece, (layer, i, j), rotation=r)
