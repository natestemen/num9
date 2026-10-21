//! Full games: shuffle the twenty tiles and let a strategy place them in turn.

use crate::board::Board;
use crate::rng::Rng;
use crate::strategies::Strategy;

/// Tiles in a game: two of each value 0-9.
pub const TILES: usize = 20;

/// The twenty tiles (two of each value) in a random order.
pub fn shuffled_tiles(rng: &mut Rng) -> Vec<u8> {
    let mut tiles: Vec<u8> = (0..2).flat_map(|_| 0..10u8).collect();
    rng.shuffle(&mut tiles);
    tiles
}

/// Play `tiles` in order with `strategy`, returning the final board.
pub fn play_tiles(strategy: &Strategy, tiles: &[u8], rng: &mut Rng) -> Board {
    let mut board = Board::new();
    for &value in tiles {
        let mv = strategy.choose(&board, value, rng);
        board
            .place(value, mv)
            .expect("strategies only choose legal moves");
    }
    board
}

/// Play one game with a freshly shuffled tile order.
pub fn play_game(strategy: &Strategy, rng: &mut Rng) -> Board {
    let tiles = shuffled_tiles(rng);
    play_tiles(strategy, &tiles, rng)
}

/// Scores of `games` games, game `k` seeded with `seed + k` so results do not
/// depend on how the games are split across threads.
pub fn play_games(strategy: &Strategy, games: usize, seed: u64, threads: usize) -> Vec<i64> {
    let threads = threads.max(1).min(games.max(1));
    let run = |k: usize| play_game(strategy, &mut Rng::new(seed.wrapping_add(k as u64))).score();
    if threads == 1 {
        return (0..games).map(run).collect();
    }
    let mut scores = vec![0i64; games];
    let chunk = games.div_ceil(threads);
    std::thread::scope(|scope| {
        for (c, out) in scores.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                for (k, slot) in out.iter_mut().enumerate() {
                    *slot = run(c * chunk + k);
                }
            });
        }
    });
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_strategy_completes_a_game() {
        for name in Strategy::NAMES {
            let s = Strategy::from_name(name).unwrap();
            let board = play_game(&s, &mut Rng::new(3));
            assert_eq!(board.pieces().len(), 20, "{name}");
        }
    }

    #[test]
    fn threading_does_not_change_results() {
        let s = Strategy::EdgesThenUp;
        assert_eq!(play_games(&s, 12, 7, 1), play_games(&s, 12, 7, 4));
    }
}
