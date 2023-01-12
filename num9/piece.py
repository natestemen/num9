from typing import Iterable, cast

import numpy as np


class Piece:
    def __init__(self, piece: int):
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

    def non_zero_indices(self) -> Iterable[tuple[int, int]]:
        for thing in zip(*np.where(self.shape)):
            yield cast(tuple[int, int], thing)
