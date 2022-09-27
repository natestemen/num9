from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from random import choice

from pprint import pprint


class Board:
    def __init__(self, size=100):
        self.board = np.zeros((size, size))
        self.num_pieces = 0
        self.piece_sequence = []

    def _trim_board(self):
        nonzero_indices = np.nonzero(self.board)
        self.board = self.board[
            nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
            nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
        ]

    def surrounding_indices(self, location, piece):
        i, j = location
        indices_of_piece_on_board = set(
            map(lambda idx: (i + idx[0], j + idx[1]), piece.non_zero_indices())
        )
        possible_boundaries = set()
        for i, j in indices_of_piece_on_board:
            if i + 1 < len(self.board):
                possible_boundaries.add((i + 1, j))
            if i > 0:
                possible_boundaries.add((i - 1, j))
            if j + 1 < len(self.board[0]):
                possible_boundaries.add((i, j + 1))
            if j > 0:
                possible_boundaries.add((i, j - 1))

        return possible_boundaries - indices_of_piece_on_board

    def find_valid_moves(self, piece):
        board_height, board_width = self.board.shape
        tmp_board = np.zeros((board_height + 8, board_width + 8))
        tmp_board[4 : 4 + board_height, 4 : 4 + board_width] = self.board
        new_board_height, new_board_width = tmp_board.shape

        possible = []
        # for _ in range(4):
        piece_height, piece_width = len(piece.shape), len(piece.shape[0])
        for s, t in product(
            range(new_board_height - piece_height),
            range(new_board_width - piece_width),
        ):
            blank = np.zeros((board_height + 8, board_width + 8))
            blank[s : s + piece_height, t : t + piece_width] = piece.shape
            maybe = tmp_board + blank
            if maybe.max() != 1:
                continue
            surrounding_tile_indices = surrounding_indices(maybe, (s, t), piece)
            if any(maybe[idx] for idx in surrounding_tile_indices):
                possible.append((s, t))

            # piece.rotate_by_90()

        return possible

    def place(self, piece, location):
        board_height, board_width = self.board.shape
        tmp_board = np.zeros((board_height + 8, board_width + 8))
        tmp_board[4 : 4 + board_height, 4 : 4 + board_width] = self.board
        self.board = tmp_board
        height, width = piece.height, piece.width
        i, j = location
        self.board[i : i + height, j : j + width] = piece.shape

    def place_randomly(self, piece):
        if not self.num_pieces:
            self.board = np.array(piece.shape)
        else:
            location = choice(self.find_valid_moves(piece))
            self.place(piece, location)

        self.num_pieces += 1
        self.piece_sequence.append(piece.name)
        self._trim_board()


class Piece:
    def __init__(self, piece):
        self.name = str(piece)
        match piece:
            case 0:
                self.shape = [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
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
                    [0, 1, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
                self.color = "orange"
            case 3:
                self.shape = [
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ]
                self.color = "yellow"
            case 4:
                self.shape = [
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
                self.color = "green"
            case 5:
                self.shape = [
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                ]
                self.color = "light blue"
            case 6:
                self.shape = [
                    [1, 1, 0],
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
                self.color = "dark blue"
            case 7:
                self.shape = [
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                    [1, 0, 0],
                ]
                self.color = "purple"
            case 8:
                self.shape = [
                    [0, 1, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                ]
                self.color = "pink"
            case 9:
                self.shape = [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                ]
                self.color = "red"
            case _:
                raise ValueError(
                    f"piece must be an integer < 10, you passed {piece} "
                    f"of type {type(piece)}"
                )  # TODO: error is being triggered in input rather than here
        self.height = len(self.shape)
        self.width = len(self.shape[0])

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

board.place_randomly(Piece(5))
board.place_randomly(Piece(8))
board.place_randomly(Piece(2))

print(board.board)
