import random

import num9_core

tiles = list(range(10)) * 2
random.shuffle(tiles)
# tiles = [8, 3, 7, 9, 8, 6, 3, 2, 5, 0, 6, 1, 9, 2, 5, 1, 4, 7, 0, 4]
print(tiles)

strategy = "place_randomly"
# strategy = "edges_then_up"
# strategy = "solid_base_high_top"
# strategy = "maximize_lookahead"
board = num9_core.play_game(strategy, seed=random.getrandbits(64), tiles=tiles)

print(str(board))
print(f"SCORE: {board.score()}")
