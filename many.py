from random import shuffle

from numpy import average
from tqdm import tqdm

from num9 import Board, Piece

scores = []

tiles = list(range(10)) * 2
for _ in tqdm(range(100)):
    b = Board()
    shuffle(tiles)
    for num in tiles:
        b.choose_move_with_most_edges_touching(Piece(num))

    score = b.score()
    scores.append(b.score())
    if score > 100:
        print(str(b))


print(f"AVERAGE: {average(scores)}")
print(f"MIN/MAX: {min(scores)}/{max(scores)}")
print(scores)
