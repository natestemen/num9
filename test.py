from random import shuffle

from num9 import Board, Piece
from num9.strategies import (
    choose_move_with_most_edges_touching,
    choose_solid_base_high_top,
    maximize_lookahead,
    place_randomly,
)

board = Board()

tiles = list(range(10)) * 2
shuffle(tiles)
# tiles = [8, 3, 7, 9, 8, 6, 3, 2, 5, 0, 6, 1, 9, 2, 5, 1, 4, 7, 0, 4]
print(tiles)
for num in tiles:
    piece = Piece(num)
    place_randomly(board, piece)
    # choose_move_with_most_edges_touching(board, piece)
    # choose_solid_base_high_top(board, piece)
    # maximize_lookahead(board, piece)
    # print(str(board))

print(str(board))
print(f"SCORE: {board.score()}")

# newboard = Board.blank_board()
# for p in board.piece_sequence:
#     newboard += p["location"]

# print(np.allclose(newboard, board.board))
