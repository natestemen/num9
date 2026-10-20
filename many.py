import random

from numpy import average
from tqdm import tqdm

import num9_core

scores = []

for _ in tqdm(range(100)):
    b = num9_core.play_game("edges_then_up", seed=random.getrandbits(64))

    score = b.score()
    scores.append(b.score())
    if score > 100:
        print(str(b))


print(f"AVERAGE: {average(scores)}")
print(f"MIN/MAX: {min(scores)}/{max(scores)}")
print(scores)
