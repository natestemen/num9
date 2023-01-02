from .piece import Piece

BOARD_HEIGHT = 160
BOARD_WIDTH = 120


def surrounding_indices(
    location: tuple[int, int], piece: Piece
) -> set[tuple[int, int]]:
    i, j = location
    indices_of_piece_on_board = set((i + x, j + y) for x, y in piece.non_zero_indices())
    possible_boundaries = set()
    for i, j in indices_of_piece_on_board:
        if i + 1 < BOARD_HEIGHT:
            possible_boundaries.add((i + 1, j))
        if i > 0:
            possible_boundaries.add((i - 1, j))
        if j + 1 < BOARD_WIDTH:
            possible_boundaries.add((i, j + 1))
        if j > 0:
            possible_boundaries.add((i, j - 1))

    return possible_boundaries - indices_of_piece_on_board
