import numpy as np
import numpy.typing as npt
from typing import Sequence
import random
import itertools

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


def rotate(arr: PieceArray):
    tuples = zip(*arr[::-1])
    array = [list(tup) for tup in tuples]
    return array


def rotational_symmetry(a1: PieceArray, a2: PieceArray) -> bool:
    return np.array_equal(a1, a2) or np.array_equal(a1, rotate(rotate(a2)))


pieces = []
num_tiles = {i: [] for i in range(6, 13)}
for i in range(1, 2**12):
    bitstring = format(i, "012b")
    arr = encode_random_to_array(bitstring)
    if validate_size(arr) and validate_contiguous(arr):
        pieces.append(arr.tolist())
        nonzero_indices = list(zip(*np.nonzero(arr)))
        num_tiles[len(nonzero_indices)].append(bitstring)

# print(num_tiles)
for num_nonzero, bitstrings in num_tiles.items():
    for b1, b2 in itertools.combinations(bitstrings, 2):
        a1, a2 = encode_random_to_array(b1), encode_random_to_array(b2)
        if rotational_symmetry(a1, a2):
            pieces.remove(a2.tolist())

print(pieces)
print(len(pieces))

import json

with open("all_pieces.json", "w") as f:
    f.write(json.dumps(pieces))
