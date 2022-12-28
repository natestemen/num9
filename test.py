from random import shuffle
from game import Board, Piece

board = Board()

tiles = list(range(10)) * 2
# shuffle(tiles)
print(tiles)
for num in tiles:
    board.place_randomly(Piece(num))

print(str(board))
