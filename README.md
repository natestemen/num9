# [NMBR 9](https://boardgamegeek.com/boardgame/217449/nmbr-9)

but in python


## stats

statistics here are small.
usually generated from a sequence of 100 random games.

| method     | average score | average duration (s) |
| ---------- | ------------- | -------------------- |
| random     | 1             | 2.39                 |
| up         | 14            | 1.49                 |
| edges + up | 55            | 1.17                 |

<!-- benchmarks:start -->
| strategy | games | avg | median | min | max | time (s) | games/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| place_randomly | 100 | 1.05 | 0.00 | 0 | 9 | 272.12 | 0.37 |
| go_up_randomly | 100 | 15.49 | 12.00 | 2 | 44 | 174.16 | 0.57 |
| edges_then_up | 100 | 58.13 | 57.00 | 36 | 97 | 152.75 | 0.65 |
| solid_base_high_top | 100 | 50.37 | 50.00 | 26 | 81 | 210.58 | 0.47 |
<!-- benchmarks:end -->
