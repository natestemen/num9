from random import shuffle
from game import Board, Piece
import numpy as np

board = Board()

tiles = list(range(10)) * 2
shuffle(tiles)
print(tiles)
for num in tiles:
    board.place_randomly(Piece(num))

print(str(board))

newboard = Board.blank_board()
for p in board.piece_sequence:
    newboard += p["location"]

print(np.allclose(newboard, board.board, atol=0.5))
