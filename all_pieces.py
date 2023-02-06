import numpy as np
import numpy.typing as npt
from typing import Sequence
import random

PieceArray = npt.NDArray[np.int32]


def encode_random_to_array(digits: str) -> PieceArray:
    digits = "".join(d + " " for d in digits)
    arr = np.fromstring(digits, dtype=int, sep=" ").reshape((4, 3))
    return arr


def validate_size(arr: PieceArray):
    nonzero_indices = np.nonzero(arr)
    trimmed = arr[
        nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
        nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
    ]
    return trimmed.shape == (4, 3)


def surrounding_indices(location: tuple[int, int]) -> Sequence[tuple[int, int]]:
    indices = set()
    i, j = location
    for _ in range(4):
        if i < 3:
            indices.add((i + 1, j))
        if i > 0:
            indices.add((i - 1, j))
        if j < 2:
            indices.add((i, j + 1))
        if j > 0:
            indices.add((i, j - 1))
    return indices


def reachable_indices(
    arr: PieceArray, location: tuple[int, int]
) -> Sequence[tuple[int, int]]:
    arr_copy = arr.copy()
    flood_fill(arr_copy, location)
    return list(zip(*np.nonzero(arr_copy == 2)))


def flood_fill(arr: PieceArray, start: tuple[int, int]) -> PieceArray:
    i, j = start
    if i < 0 or i > 3 or j < 0 or j > 2:
        return
    if arr[start] == 0 or arr[start] == 2:
        return
    arr[start] = 2
    flood_fill(arr, (i - 1, j))
    flood_fill(arr, (i + 1, j))
    flood_fill(arr, (i, j - 1))
    flood_fill(arr, (i, j + 1))


def validate_contiguous(arr: PieceArray):
    nonzero_indices = list(zip(*np.nonzero(arr)))
    start = random.choice(nonzero_indices)
    reachable = reachable_indices(arr, start)
    return len(reachable) == len(nonzero_indices)


count = 0
pieces = []
for i in range(1, 2**12):
    bitstring = format(i, "012b")
    arr = encode_random_to_array(bitstring)
    if validate_size(arr) and validate_contiguous(arr):
        pieces.append(arr.tolist())
        count += 1

# TODO: still need to eliminate pieces that are rotations of each other
# probably want a dict {num_missing_blocks: [list of pieces]} to search
# through when looking for pieces that might be rotations to then eliminate

print(count)
import json

with open("all_pieces.json", "w") as f:
    f.write(json.dumps(pieces))
