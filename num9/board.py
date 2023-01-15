from itertools import product
from random import choice
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from .piece import Piece

BLANK_PIECE = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
BOARD_DEPTH = 6
BOARD_HEIGHT = 160
BOARD_WIDTH = 120


class InvalidPlacementError(Exception):
    pass


class Board:
    def __init__(
        self,
        board: npt.NDArray[np.int32] | None = None,
        piece_sequence: list[dict[str, Any]] | None = None,
    ):
        self.board = (
            np.zeros((BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)
            if board is None
            else board
        )
        self.piece_sequence = piece_sequence if piece_sequence else []

    def __str__(self):
        rows = []
        for i, layer in enumerate(self.trim_board()):
            _, width = layer.shape
            rows.append("{:-^{width}}".format(f" LAYER {i} ", width=2 * width))
            for row in layer:
                row_str = ""
                for block in row:
                    match block:
                        case 0:
                            row_str += "⬜️"
                        case 10:
                            row_str += "⬛️"
                        case 1:
                            row_str += "🟫"
                        case 2:
                            row_str += "🟧"
                        case 3:
                            row_str += "🟨"
                        case 4:
                            row_str += "🟩"
                        case 5:
                            row_str += "🔵"
                        case 6:
                            row_str += "🟦"
                        case 7:
                            row_str += "🟪"
                        case 8:
                            row_str += "💓"
                        case 9:
                            row_str += "🟥"
                        case _:
                            row_str += "🔳"
                            print(f"oh shit got strange value: {_}")
                rows.append(row_str)
        return "\n".join(rows)

    @classmethod
    def blank_board(cls):
        return np.zeros((BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)

    @classmethod
    def blank_board_with_piece(
        cls, piece: Piece, location: tuple[int, int, int], ones: bool = False
    ) -> npt.NDArray[np.int32]:
        bboard = cls.blank_board()
        layer, i, j = location
        piece_height, piece_width = piece.dimensions
        bboard[layer, i : i + piece_height, j : j + piece_width] = piece.array
        if ones:
            bboard[bboard > 1] = 1
        return bboard

    @classmethod
    def blank_layer(cls):
        return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)

    @classmethod
    def blank_layer_with_piece(
        cls, piece: Piece, location: tuple[int, int], ones: bool = False
    ) -> npt.NDArray[np.int32]:
        layer = cls.blank_layer()
        i, j = location
        piece_height, piece_width = piece.dimensions
        layer[i : i + piece_height, j : j + piece_width] = piece.array
        if ones:
            layer[layer > 1] = 1
        return layer

    @property
    def board_ones(self) -> npt.NDArray[np.int32]:
        ones = self.board.copy()
        ones[ones > 0] = 1
        return ones

    def center_coords(self) -> tuple[int, int, int]:
        center_width = BOARD_WIDTH // 2
        center_height = BOARD_HEIGHT // 2
        return 0, center_height, center_width

    def trim_board(self) -> npt.NDArray[np.int32]:
        nonzero_indices = np.nonzero(self.board)
        trimmed_board = self.board[
            nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
            nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
            nonzero_indices[2].min() : nonzero_indices[2].max() + 1,
        ]
        return np.pad(trimmed_board, ((0, 0), (1, 1), (1, 1)))

    def trim_layer(self, layer) -> npt.NDArray[np.int32]:
        nonzero_indices = np.nonzero(layer)
        trimmed_layer = layer[
            nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
            nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
        ]
        return np.pad(trimmed_layer, ((2, 2), (2, 2)))

    def layer_loop_indices(
        self, layer_index: int, piece: Piece
    ) -> tuple[int, int, int, int]:
        layer = self.board[layer_index]
        if layer.max() == 0:
            layer = self.board[layer_index - 1]

        height, width = piece.dimensions
        i_indices, j_indices = np.nonzero(layer)
        i_min, i_max = min(i_indices) - height, max(i_indices) + height
        j_min, j_max = min(j_indices) - width, max(j_indices) + width
        return i_min, i_max, j_min, j_max

    def validate_touching(self, piece: Piece, location) -> bool:
        layer_idx, i, j = location
        if self.no_pieces_on_layer(layer_idx):
            return True
        layer = self.board[layer_idx]
        surrounding_tile_indices = piece.surrounding_indices((i, j))
        touching = any(layer[idx] for idx in surrounding_tile_indices)

        indices_of_piece_on_board = set(
            (i + x, j + y) for x, y in piece.non_zero_indices()
        )
        overlapping = any(layer[idx] for idx in indices_of_piece_on_board)
        return touching and not overlapping

    def num_edges_touching(self, piece: Piece, location: tuple[int, int, int]) -> int:
        layer_idx, i, j = location
        layer = self.board[layer_idx]
        surrounding_tile_indices = piece.surrounding_indices((i, j))
        return sum(1 for idx in surrounding_tile_indices if layer[idx] > 0)

    def validate_supported(self, piece: Piece, location: tuple[int, int, int]) -> bool:
        layer_index, i, j = location
        if layer_index == 0:
            return True

        layer_below_index = layer_index - 1

        layer_ones = self.board[layer_below_index].copy()
        layer_ones[layer_ones > 1] = 1
        piece_on_layer = Board.blank_layer_with_piece(piece, (i, j), ones=True)
        maybe = layer_ones + piece_on_layer

        piece_tile_count = np.count_nonzero(piece.array)
        piece_tile_overlap_count = np.count_nonzero(maybe > 1)
        if piece_tile_count == piece_tile_overlap_count:
            # piece is supported underneath
            overlap_count = 0
            blank_board_with_piece = Board.blank_board_with_piece(
                piece, (layer_below_index, i, j), ones=True
            )
            for on_board in self.pieces_on_layer(layer_below_index, ones=True):
                maybe = on_board + blank_board_with_piece
                if maybe.max() > 1:
                    overlap_count += 1

            if overlap_count >= 2:
                if self.no_pieces_on_layer(layer_index) or self.validate_touching(
                    piece, (layer_index, i, j)
                ):
                    return True
        return False

    def no_pieces_on_layer(self, layer_index: int) -> bool:
        layer = self.board[layer_index]
        return layer.max() == 0

    def pieces_on_layer(
        self, layer_index: int, ones: bool = False
    ) -> Iterable[npt.NDArray[np.int32]]:
        for piece_data in self.piece_sequence:
            if piece_data["layer"] == layer_index:
                board = piece_data["location"].copy()
                if ones:
                    board[board > 1] = 1
                yield board

    def score(self) -> int:
        score = 0
        for p in self.piece_sequence:
            score += int(p["name"].name) * p["layer"]
        return score

    def find_valid_moves(
        self, piece: Piece
    ) -> list[tuple[int, int, int, int, npt.NDArray[np.int32]]]:
        """Finds all valid moves for given piece.

        Args:
            piece: Piece to be placed on board

        Returns: TODO: implement this:
            list of valid moves
            [
                {
                    "level": 0, 1, 2, ...
                    "location": (i, j),
                    "rotation": {0, 1, 2, 3}
                }, ...
            ]
        """
        if not self.piece_sequence:
            center = self.center_coords()
            blank_board_with_piece = Board.blank_board_with_piece(piece, center)
            final = center + (0, blank_board_with_piece)  # rotation, and board
            return [final]

        possible_moves = []
        for layer_index, layer in enumerate(self.board):
            if layer_index > 0 and self.board[layer_index - 1].max() == 0:
                break

            layer_ones = layer.copy()
            layer_ones[layer_ones > 1] = 1
            for rot in range(4):
                i_start, i_stop, j_start, j_stop = self.layer_loop_indices(
                    layer_index, piece
                )
                for i, j in product(range(i_start, i_stop), range(j_start, j_stop)):
                    if self.validate_touching(
                        piece, (layer_index, i, j)
                    ) and self.validate_supported(piece, (layer_index, i, j)):
                        blank_board_with_piece = Board.blank_board_with_piece(
                            piece, (layer_index, i, j)
                        )

                        possible_moves.append(
                            (layer_index, i, j, rot, blank_board_with_piece)
                        )

                piece.rotate_by_90()

        return possible_moves

    def place(self, piece: Piece, location: tuple[int, int, int]) -> None:
        """Places piece on board at specified location; all in place.

        Args:
            piece: piece to be placed
            location: tuple specifying corner on which to place piece
        """
        height, width = piece.dimensions
        layer, i, j = location
        self.board[layer, i : i + height, j : j + width] += piece.array
        if self.board.max() > 10:
            raise InvalidPlacementError(
                f"Placing a '{piece.name}' in that location would result in overlap"
            )

    def place_randomly(self, piece: Piece) -> None:
        """finds a random valid move, and adds the piece to the board"""
        valid_moves = self.find_valid_moves(piece)
        layer, i, j, r, piece_on_board = choice(valid_moves)
        piece.rotation = r
        self.place(piece, (layer, i, j))
        self.piece_sequence.append(
            {"name": piece, "location": piece_on_board, "layer": layer}
        )

    def go_up_randomly(self, piece: Piece) -> None:
        """picks a move at the highest level possible"""
        valid_moves = self.find_valid_moves(piece)
        layers = [layer for layer, *_ in valid_moves]
        if len(set(layers)) == 1:
            layer, i, j, r, piece_on_board = choice(valid_moves)
        else:
            max_layer = max(layers)
            option_indices = [i for i, l in enumerate(layers) if l == max_layer]
            random_index = choice(option_indices)
            layer, i, j, r, piece_on_board = valid_moves[random_index]
        piece.rotation = r
        self.place(piece, (layer, i, j))
        self.piece_sequence.append(
            {"name": piece, "location": piece_on_board, "layer": layer}
        )

    def choose_move_with_most_edges_touching(self, piece: Piece) -> None:
        valid_moves = self.find_valid_moves(piece)
        layers = [layer for layer, *_ in valid_moves]

        option_indices = range(len(valid_moves))
        if len(set(layers)) > 1:
            max_layer = max(layers)
            option_indices = [i for i, l in enumerate(layers) if l == max_layer]

        if piece.name in {"0", "1"}:
            option_indices = [i for i, layer in enumerate(layers) if layer == 0]

        edges = {}
        for idx in option_indices:
            layer, i, j, r, piece_on_board = valid_moves[idx]
            piece.rotation = r
            edges[idx] = self.num_edges_touching(piece, (layer, i, j))

        max_edges = max(edges.values())
        moves_with_max_edges = [idx for idx, v in edges.items() if v == max_edges]
        idx = choice(moves_with_max_edges)
        layer, i, j, r, piece_on_board = valid_moves[idx]

        piece.rotation = r
        self.place(piece, (layer, i, j))
        self.piece_sequence.append(
            {"name": piece, "location": piece_on_board, "layer": layer}
        )

    def maximize_lookahead(self, piece: Piece) -> None:
        valid_moves = self.find_valid_moves(piece)
        played = [int(p["name"].name) for p in self.piece_sequence]
        leftover = list(range(10)) * 2
        for p in played:
            leftover.remove(p)
        leftover = set(leftover)

        move_scores = []
        # best_scores = []
        for layer, i, j, r, piece_on_board in valid_moves:
            ps = [{"name": piece, "location": piece_on_board, "layer": layer}]
            tmp_board = Board(
                self.board + piece_on_board, piece_sequence=self.piece_sequence + ps
            )
            total_score = 0
            total_moves = 0
            # best_score = 0
            for p in leftover:
                new_piece = Piece(p)
                next_moves = tmp_board.find_valid_moves(new_piece)
                total_moves += len(next_moves)
                for layer, *_, pob in next_moves:
                    psm = [{"name": new_piece, "location": pob, "layer": layer}]
                    tmp2_board = Board(
                        tmp_board.board + pob,
                        piece_sequence=tmp_board.piece_sequence + psm,
                    )
                    total_score += tmp2_board.score()
                    # move_score = tmp2_board.score()
                    # if move_score > best_score:
                    #     best_score = move_score

            move_scores.append(total_score / total_moves)
            # best_scores.append(best_score)

        print(move_scores)
        # print(best_scores)
        best_move_index = max(enumerate(move_scores), key=lambda x: x[1])[0]
        layer, i, j, r, piece_on_board = valid_moves[best_move_index]

        piece.rotation = r
        self.place(piece, (layer, i, j))
        self.piece_sequence.append(
            {"name": piece, "location": piece_on_board, "layer": layer}
        )
