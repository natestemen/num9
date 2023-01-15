from typing import Iterable, cast

import numpy as np

BOARD_HEIGHT = 160
BOARD_WIDTH = 120


class Piece:
    def __init__(self, piece: int):
        self.name = str(piece)
        self._array: list[list[int]]
        self.rotation = 0
        match piece:
            case 0:
                self._array = [
                    [10, 10, 10],
                    [10, 0, 10],
                    [10, 0, 10],
                    [10, 10, 10],
                ]
                self.color = "grey"
            case 1:
                self._array = [
                    [1, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ]
                self.color = "brown"
            case 2:
                self._array = [
                    [0, 2, 2],
                    [0, 2, 2],
                    [2, 2, 0],
                    [2, 2, 2],
                ]
                self.color = "orange"
            case 3:
                self._array = [
                    [3, 3, 3],
                    [0, 0, 3],
                    [0, 3, 3],
                    [3, 3, 3],
                ]
                self.color = "yellow"
            case 4:
                self._array = [
                    [0, 4, 4],
                    [0, 4, 0],
                    [4, 4, 4],
                    [0, 4, 4],
                ]
                self.color = "green"
            case 5:
                self._array = [
                    [5, 5, 5],
                    [5, 5, 5],
                    [0, 0, 5],
                    [5, 5, 5],
                ]
                self.color = "light blue"
            case 6:
                self._array = [
                    [6, 6, 0],
                    [6, 0, 0],
                    [6, 6, 6],
                    [6, 6, 6],
                ]
                self.color = "dark blue"
            case 7:
                self._array = [
                    [7, 7, 7],
                    [0, 7, 0],
                    [7, 7, 0],
                    [7, 0, 0],
                ]
                self.color = "purple"
            case 8:
                self._array = [
                    [0, 8, 8],
                    [0, 8, 8],
                    [8, 8, 0],
                    [8, 8, 0],
                ]
                self.color = "pink"
            case 9:
                self._array = [
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

    @property
    def array(self):
        array = self._array.copy()
        for _ in range(self.rotation):
            tuples = zip(*array[::-1])
            array = [list(tup) for tup in tuples]
        return array

    @property
    def dimensions(self) -> tuple[int, int]:
        return len(self.array), len(self.array[0])

    def rotate_by_90(self) -> None:
        self.rotation = (self.rotation + 1) % 4

    def non_zero_indices(self) -> Iterable[tuple[int, int]]:
        for thing in zip(*np.where(self.array)):
            yield cast(tuple[int, int], thing)

    def surrounding_indices(self, location: tuple[int, int]) -> set[tuple[int, int]]:
        i, j = location
        indices_of_piece_on_board = set(
            (i + x, j + y) for x, y in self.non_zero_indices()
        )
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
