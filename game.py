import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint


class Board:
    def __init__(self, size=100):
        self.board = np.zeros((size, size))

    # def place(self, piece, index):


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

    def rotate_by_90(self):
        tuples = zip(*self.shape[::-1])
        self.shape = [list(tup) for tup in tuples]

    def non_zero_indices(self):
        return zip(*self.shape.nonzero())


def surrounding_tiles(board, index_of_piece, piece):
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


def find_valid_moves(p: Piece, q: Piece) -> list:
    """assuming p is fixed"""
    board = np.zeros((12, 9), dtype=int)
    board[4:8, 3:6] = p.shape

    possible = []
    for s in range(9):  # loop around outside of p and test placements
        for t in range(7):
            blank = np.zeros((12, 9), dtype=int)
            blank[s : s + 4, t : t + 3] = q.shape
            maybe = board + blank
            if maybe.max() != 1:
                continue
            surrounding_tile_indices = surrounding_tiles(board, (s, t), q)
            if any(board[idx] for idx in surrounding_tile_indices):
                possible.append((s, t))

    board = np.zeros((10, 11), dtype=int)
    board[3:7, 4:7] = p.shape
    q.rotate_by_90()
    for s in range(7):
        for t in range(7):
            blank = np.zeros((10, 11), dtype=int)
            blank[s : s + 3, t : t + 4] = q.shape
            maybe = board + blank
            if maybe.max() != 1:
                continue
            surrounding_tile_indices = surrounding_tiles(board, (s, t), q)
            if any(board[idx] for idx in surrounding_tile_indices):
                possible.append((s, t))

    board = np.zeros((12, 9), dtype=int)
    board[4:8, 3:6] = p.shape
    q.rotate_by_90()
    for s in range(9):
        for t in range(7):
            blank = np.zeros((12, 9), dtype=int)
            blank[s : s + 4, t : t + 3] = q.shape
            maybe = board + blank
            if maybe.max() != 1:
                continue
            surrounding_tile_indices = surrounding_tiles(board, (s, t), q)
            if any(board[idx] for idx in surrounding_tile_indices):
                possible.append((s, t))

    board = np.zeros((10, 11), dtype=int)
    board[3:7, 4:7] = p.shape
    q.rotate_by_90()
    for s in range(7):
        for t in range(7):
            blank = np.zeros((10, 11), dtype=int)
            blank[s : s + 3, t : t + 4] = q.shape
            maybe = board + blank
            if maybe.max() != 1:
                continue
            surrounding_tile_indices = surrounding_tiles(board, (s, t), q)
            if any(board[idx] for idx in surrounding_tile_indices):
                possible.append((s, t))
    return possible


first = Piece(5)
second = Piece(4)

print(len(find_valid_moves(first, second)))

# surrounding_tiles("hi", (7, 6), second)

# piece = int(input("Piece number: "))


# print(first_piece.shape)
# plt.imshow(first_piece.shape, cmap="gray")
# plt.show()
