from game import Board, Piece
from random import shuffle
from tqdm import tqdm
from numpy import average


scores = []

tiles = list(range(10)) * 2
for _ in tqdm(range(100)):
    b = Board()
    shuffle(tiles)
    for num in tiles:
        b.choose_move_with_most_edges_touching(Piece(num))

    scores.append(b.score())


print(f"AVERAGE: {average(scores)}")
print(f"MIN/MAX: {min(scores)}/{max(scores)}")
print(scores)
