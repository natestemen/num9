from itertools import product
from random import choice, shuffle

import numpy as np


class Board:
    def __init__(self):
        self.board = None
        self.piece_sequence = []

    def __str__(self):
        rows = []
        for row in self.board:
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

    def _trim_board(self):
        nonzero_indices = np.nonzero(self.board)
        self.board = self.board[
            nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
            nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
        ]
        self.board = np.pad(self.board, ((4, 4), (4, 4)), "constant")

    def find_valid_moves(self, piece):
        board_height, board_width = self.board.shape
        tmp_board = np.pad(self.board, ((4, 4), (4, 4)), "constant")
        tmp_board[tmp_board > 1] = 1
        new_board_height, new_board_width = tmp_board.shape

        possible = []
        for r in range(4):
            piece_one = np.array(piece.shape).copy()
            piece_one[piece_one > 1] = 1
            piece_height, piece_width = len(piece.shape), len(piece.shape[0])
            for s, t in product(
                range(new_board_height - piece_height),
                range(new_board_width - piece_width),
            ):
                blank = np.zeros((board_height + 8, board_width + 8))
                blank[s : s + piece_height, t : t + piece_width] = piece_one
                maybe = tmp_board + blank
                if maybe.max() != 1:
                    continue
                surrounding_tile_indices = surrounding_indices(maybe, (s, t), piece)
                if any(maybe[idx] for idx in surrounding_tile_indices):
                    possible.append((s, t, r))

            piece.rotate_by_90()

        return possible

    def place(self, piece, location):
        self.board = np.pad(self.board, ((4, 4), (4, 4)), "constant")
        height, width = np.array(piece.shape).shape
        i, j = location
        self.board[i : i + height, j : j + width] += piece.shape
        self._trim_board()

    def place_randomly(self, piece):
        if self.board is None:
            self.board = np.pad(piece.shape, ((4, 4), (4, 4)), "constant")
            self.piece_sequence.append({"name": piece})
        else:
            i, j, r = choice(self.find_valid_moves(piece))
            for _ in range(r):
                piece.rotate_by_90()
            self.place(piece, (i, j))
            self.piece_sequence.append({"name": piece})


class Piece:
    def __init__(self, piece):
        self.name = str(piece)
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

    def rotate_by_90(self):
        tuples = zip(*self.shape[::-1])
        self.shape = [list(tup) for tup in tuples]

    def non_zero_indices(self):
        return zip(*np.where(self.shape))


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


board = Board()

tiles = list(range(9)) * 2
shuffle(tiles)
print(tiles)
for num in tiles:
    board.place_randomly(Piece(num))

print(str(board))
