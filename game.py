from itertools import product
from random import choice, shuffle
from typing import Iterable

import numpy as np
import numpy.typing as npt


BLANK_PIECE = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
BOARD_DEPTH = 6
BOARD_HEIGHT = 160
BOARD_WIDTH = 120


class Board:
    def __init__(self, board=None):
        self.board = (
            np.zeros((BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)
            if not board
            else board
        )
        self.piece_sequence = []

    def __str__(self):
        rows = []
        small_board = self.trim_layer(self.board[0])
        for row in small_board:
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
            rows.append(row_str)
        return "\n".join(rows)

    @classmethod
    def blank_board(cls):
        return np.zeros((BOARD_DEPTH, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)

    @classmethod
    def blank_layer(cls):
        return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)

    @property
    def board_ones(self) -> npt.NDArray[np.int64]:
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
        return np.pad(trimmed_board, ((2, 2), (2, 2)), "constant")

    def trim_layer(self, layer) -> npt.NDArray[np.int32]:
        nonzero_indices = np.nonzero(layer)
        trimmed_layer = layer[
            nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
            nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
        ]
        return np.pad(trimmed_layer, ((2, 2), (2, 2)), "constant")

    def layer_loop_indices(self, layer_index: int) -> tuple[int, int, int, int]:
        # TODO: consume piece shape
        layer = self.board[layer_index]
        nonzero_indices = np.nonzero(layer)
        i_min, i_max = min(nonzero_indices[0]) - 4, max(nonzero_indices[0]) + 4
        j_min, j_max = min(nonzero_indices[1]) - 4, max(nonzero_indices[1]) + 4
        return i_min, i_max, j_min, j_max

    def find_valid_moves(self, piece: "Piece") -> list[tuple[int, int, int]]:
        """finds all valid moves for given piece

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
        if self.board[0].max() == 0:
            center = self.center_coords()
            center += (0,)  # rotation
            return [center]

        possible_moves = []
        for layer_index, layer in enumerate(self.board):
            if layer.max() == 0:  # and self.board[layer_index - 1].max() == 0:
                break

            for r in range(4):
                piece_height, piece_width = len(piece.shape), len(piece.shape[0])
                layer_ones = layer.copy()
                layer_ones[layer_ones > 1] = 1
                i_start, i_stop, j_start, j_stop = self.layer_loop_indices(layer_index)
                for i, j in product(range(i_start, i_stop), range(j_start, j_stop)):
                    blank_layer_with_piece = Board.blank_layer()
                    blank_layer_with_piece[
                        i : i + piece_height, j : j + piece_width
                    ] = piece.ones_piece()
                    maybe = layer_ones + blank_layer_with_piece
                    if maybe.max() != 1:
                        # we have overlap
                        piece_tile_count = np.count_nonzero(piece.shape)
                        overlap_section = maybe[
                            i : i + piece_height, j : j + piece_width
                        ]
                        overlap_section[overlap_section == 1] = 0
                        overlap_count = np.count_nonzero(overlap_section)
                        if piece_tile_count == overlap_count:
                            print("IT FITS ON TOP-------------")
                        # TODO: need to check if it overlaps TWO tiles now
                        continue
                    surrounding_tile_indices = surrounding_indices(maybe, (i, j), piece)
                    if any(maybe[idx] for idx in surrounding_tile_indices):
                        possible_moves.append((layer_index, i, j, r))

                piece.rotate_by_90()

        return possible_moves

    def place(self, piece: "Piece", location: tuple[int, int, int]) -> None:
        height, width = np.array(piece.shape).shape
        layer, i, j = location
        self.board[layer, i : i + height, j : j + width] += piece.shape

    def place_randomly(self, piece: "Piece") -> None:
        """finds a random valid move, and adds the piece to the board"""
        layer, i, j, r = choice(self.find_valid_moves(piece))
        for _ in range(r):
            piece.rotate_by_90()
        self.place(piece, (layer, i, j))
        self.piece_sequence.append({"name": piece})


class Piece:
    def __init__(self, piece):
        self.name = str(piece)
        self.shape: list[list[int]]
        match piece:
            case 0:
                self.shape = [
                    [10, 10, 10],
                    [10, 0, 10],
                    [10, 0, 10],
                    [10, 10, 10],
                ]
                self.color = "grey"
            case 1:
                self.shape = [
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                ]
                self.color = "brown"
            case 2:
                self.shape = [
                    [0, 2, 2],
                    [0, 2, 2],
                    [2, 2, 0],
                    [2, 2, 2],
                ]
                self.color = "orange"
            case 3:
                self.shape = [
                    [3, 3, 3],
                    [0, 0, 3],
                    [0, 3, 3],
                    [3, 3, 3],
                ]
                self.color = "yellow"
            case 4:
                self.shape = [
                    [0, 4, 4],
                    [0, 4, 0],
                    [4, 4, 4],
                    [0, 4, 4],
                ]
                self.color = "green"
            case 5:
                self.shape = [
                    [5, 5, 5],
                    [5, 5, 5],
                    [0, 0, 5],
                    [5, 5, 5],
                ]
                self.color = "light blue"
            case 6:
                self.shape = [
                    [6, 6, 0],
                    [6, 0, 0],
                    [6, 6, 6],
                    [6, 6, 6],
                ]
                self.color = "dark blue"
            case 7:
                self.shape = [
                    [7, 7, 7],
                    [0, 7, 0],
                    [7, 7, 0],
                    [7, 0, 0],
                ]
                self.color = "purple"
            case 8:
                self.shape = [
                    [0, 8, 8],
                    [0, 8, 8],
                    [8, 8, 0],
                    [8, 8, 0],
                ]
                self.color = "pink"
            case 9:
                self.shape = [
                    [9, 9, 9],
                    [9, 9, 9],
                    [9, 9, 0],
                    [9, 9, 0],
                ]
                self.color = "red"
            case _:
                raise ValueError(
                    f"piece must be an integer < 10, you passed {piece} "
                    f"of type {type(piece)}"
                )  # TODO: error is being triggered in input rather than here

    def rotate_by_90(self) -> None:
        tuples = zip(*self.shape[::-1])
        self.shape = [list(tup) for tup in tuples]

    def non_zero_indices(self) -> Iterable[tuple[int]]:
        return zip(*np.where(self.shape))

    def ones_piece(self) -> npt.NDArray[np.int32]:
        array = np.array(self.shape)
        array[array > 1] = 1
        return array


def surrounding_indices(board, index_of_piece, piece):
    i, j = index_of_piece
    indices_of_piece_on_board = set(
        map(lambda idx: (i + idx[0], j + idx[1]), piece.non_zero_indices())
    )
    possible_boundaries = set()
    for i, j in indices_of_piece_on_board:
        if i + 1 < len(board):
            possible_boundaries.add((i + 1, j))
        if i > 0:
            possible_boundaries.add((i - 1, j))
        if j + 1 < len(board[0]):
            possible_boundaries.add((i, j + 1))
        if j > 0:
            possible_boundaries.add((i, j - 1))

    return possible_boundaries - indices_of_piece_on_board
