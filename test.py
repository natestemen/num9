from random import shuffle

import numpy as np

from num9 import Board, Piece

board = Board()

tiles = list(range(10)) * 2
shuffle(tiles)
print(tiles)
for num in tiles:
    # board.choose_move_with_most_edges_touching(Piece(num))
    board.maximize_lookahead(Piece(num))
    print(str(board))

print(str(board))
print(f"SCORE: {board.score()}")

# newboard = Board.blank_board()
# for p in board.piece_sequence:
#     newboard += p["location"]

# print(np.allclose(newboard, board.board))
