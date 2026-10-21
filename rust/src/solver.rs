//! Finding the best play of a known deal.
//!
//! With the order of the tiles fixed in advance there is no chance left: the
//! game becomes a single-player puzzle with roughly 10^45 ways to play it.
//! This module searches it with Nested Monte Carlo Search (Cazenave, 2009):
//!
//! - level 0 plays the rest of the game with a fast policy (a "playout");
//! - level n, at each step, tries every candidate move, judges it by a
//!   level n-1 search from the position after it, and then follows the best
//!   complete sequence found so far (so the result only ever improves).
//!
//! Because the tiles to come are known, playouts use the real remaining
//! order: each one is an exact, achievable game.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crate::board::{Board, Move};
use crate::rng::Rng;
use crate::rollout::candidate_moves;
use crate::strategies::Strategy;

/// A complete play of a deal: one move per tile, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    pub score: i64,
    pub moves: Vec<Move>,
}

impl Solution {
    fn none() -> Solution {
        Solution {
            score: i64::MIN,
            moves: Vec::new(),
        }
    }
}

/// Settings for nested Monte Carlo search.
#[derive(Clone, Debug)]
pub struct Nested {
    /// Nesting level (0 = a single playout).
    pub level: usize,
    /// Candidate moves considered per level of the board at each step (those
    /// touching the most edges; see [`candidate_moves`]).
    pub per_layer: usize,
    /// Policy for the playouts.
    pub policy: Strategy,
    /// Worker threads for the top level.
    pub threads: usize,
}

impl Default for Nested {
    fn default() -> Self {
        Nested {
            level: 2,
            per_layer: 4,
            policy: Strategy::LowToBase {
                max_low: 4,
                endgame: 4,
            },
            threads: thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }
}

/// Replay `moves` for `tiles`, checking each is legal; returns the score.
pub fn verify(tiles: &[u8], moves: &[Move]) -> Result<i64, String> {
    if tiles.len() != moves.len() {
        return Err(format!("{} tiles but {} moves", tiles.len(), moves.len()));
    }
    let mut board = Board::new();
    for (k, (&v, &mv)) in tiles.iter().zip(moves).enumerate() {
        if !board.is_valid_move(v, mv) {
            return Err(format!("move {k} ({v} at {mv:?}) is not legal"));
        }
        board.place(v, mv).map_err(|e| e.to_string())?;
    }
    Ok(board.score())
}

/// Play `tiles` on `board` with `policy`; the board is left as it was.
fn playout(board: &mut Board, tiles: &[u8], policy: &Strategy, rng: &mut Rng) -> Solution {
    let mut moves = Vec::with_capacity(tiles.len());
    for &v in tiles {
        let mv = policy.choose(board, v, rng);
        board.place(v, mv).expect("policies choose legal moves");
        moves.push(mv);
    }
    let score = board.score();
    for _ in tiles {
        board.undo();
    }
    Solution { score, moves }
}

/// Mix a step and candidate index into a seed, so results don't depend on
/// how work is split across threads.
fn mix(seed: u64, a: usize, b: usize) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    for x in [a as u64, b as u64] {
        h = (h ^ x).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    h
}

impl Nested {
    fn candidates(&self, board: &Board, value: u8) -> Vec<Move> {
        let all = board.find_valid_moves(value);
        candidate_moves(board, value, &all, self.per_layer)
            .into_iter()
            .map(|k| all[k])
            .collect()
    }

    /// Level-`level` search playing `tiles` from `board` (left unchanged).
    /// Returns the best complete continuation found.
    fn search(&self, board: &mut Board, tiles: &[u8], level: usize, seed: u64) -> Solution {
        if level == 0 || tiles.is_empty() {
            return playout(board, tiles, &self.policy, &mut Rng::new(seed));
        }
        let mut best = Solution::none();
        let mut played: Vec<Move> = Vec::with_capacity(tiles.len());
        for (step, &value) in tiles.iter().enumerate() {
            for (c, mv) in self.candidates(board, value).into_iter().enumerate() {
                board.place(value, mv).expect("legal move");
                let sub = self.search(board, &tiles[step + 1..], level - 1, mix(seed, step, c));
                board.undo();
                if sub.score > best.score {
                    let mut moves = played.clone();
                    moves.push(mv);
                    moves.extend(sub.moves);
                    best = Solution {
                        score: sub.score,
                        moves,
                    };
                }
            }
            // Follow the best sequence known so far.
            let next = best.moves[played.len()];
            board.place(value, next).expect("legal move");
            played.push(next);
        }
        for _ in &played {
            board.undo();
        }
        best
    }

    /// Search the whole deal. The top level spreads each step's candidates
    /// over threads; `progress(step, best)` is called after every step, and
    /// setting `stop` ends the search early (returning the best found).
    pub fn solve(
        &self,
        tiles: &[u8],
        seed: u64,
        stop: &AtomicBool,
        mut progress: impl FnMut(usize, &Solution),
    ) -> Solution {
        let mut board = Board::new();
        let mut best = Solution::none();
        let mut played: Vec<Move> = Vec::with_capacity(tiles.len());
        if self.level == 0 {
            return playout(&mut board, tiles, &self.policy, &mut Rng::new(seed));
        }
        for (step, &value) in tiles.iter().enumerate() {
            let cands = self.candidates(&board, value);
            let rest = &tiles[step + 1..];
            let results: Vec<Solution> = if stop.load(Ordering::Relaxed) {
                Vec::new()
            } else {
                let threads = self.threads.clamp(1, cands.len().max(1));
                let chunk = cands.len().div_ceil(threads).max(1);
                thread::scope(|s| {
                    let handles: Vec<_> = cands
                        .chunks(chunk)
                        .enumerate()
                        .map(|(t, part)| {
                            let mut b = board.clone();
                            s.spawn(move || {
                                part.iter()
                                    .enumerate()
                                    .map(|(k, &mv)| {
                                        b.place(value, mv).expect("legal move");
                                        let c = t * chunk + k;
                                        let sub = self.search(
                                            &mut b,
                                            rest,
                                            self.level - 1,
                                            mix(seed, step, c),
                                        );
                                        b.undo();
                                        sub
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap())
                        .collect()
                })
            };
            for (mv, sub) in cands.iter().zip(results) {
                if sub.score > best.score {
                    let mut moves = played.clone();
                    moves.push(*mv);
                    moves.extend(sub.moves);
                    best = Solution {
                        score: sub.score,
                        moves,
                    };
                }
            }
            if best.moves.len() <= played.len() {
                // Stopped before anything was found for this step: finish
                // with a playout so the solution is complete.
                let tail = playout(
                    &mut board,
                    &tiles[step..],
                    &self.policy,
                    &mut Rng::new(seed),
                );
                best.moves.truncate(played.len());
                best.moves.extend(tail.moves);
                best.score = verify(tiles, &best.moves).expect("playout is legal");
                return best;
            }
            let next = best.moves[played.len()];
            board.place(value, next).expect("legal move");
            played.push(next);
            progress(step, &best);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::shuffled_tiles;

    fn deal(seed: u64) -> Vec<u8> {
        shuffled_tiles(&mut Rng::new(seed))
    }

    #[test]
    fn solutions_are_legal_and_scored_correctly() {
        let tiles = deal(3);
        let stop = AtomicBool::new(false);
        for level in 0..=1 {
            let nested = Nested {
                level,
                threads: 2,
                ..Nested::default()
            };
            let sol = nested.solve(&tiles, 1, &stop, |_, _| {});
            assert_eq!(sol.moves.len(), tiles.len());
            assert_eq!(verify(&tiles, &sol.moves), Ok(sol.score));
        }
    }

    #[test]
    fn higher_levels_do_at_least_as_well_as_the_playouts_they_contain() {
        let tiles = deal(4);
        let stop = AtomicBool::new(false);
        let base = Nested {
            level: 0,
            ..Nested::default()
        };
        let one = Nested {
            level: 1,
            threads: 3,
            ..Nested::default()
        };
        let p = base.solve(&tiles, 9, &stop, |_, _| {});
        let mut steps = Vec::new();
        let s = one.solve(&tiles, 9, &stop, |step, best| {
            steps.push((step, best.score))
        });
        assert!(
            s.score >= p.score,
            "level 1 {} < playout {}",
            s.score,
            p.score
        );
        // The best score never decreases as the search goes on.
        assert!(steps.windows(2).all(|w| w[0].1 <= w[1].1));
    }

    #[test]
    fn deterministic_regardless_of_threads() {
        let tiles = deal(5);
        let stop = AtomicBool::new(false);
        let a = Nested {
            level: 1,
            threads: 1,
            ..Nested::default()
        }
        .solve(&tiles, 2, &stop, |_, _| {});
        let b = Nested {
            level: 1,
            threads: 4,
            ..Nested::default()
        }
        .solve(&tiles, 2, &stop, |_, _| {});
        assert_eq!(a, b);
    }

    #[test]
    fn stopping_early_still_gives_a_complete_legal_play() {
        let tiles = deal(6);
        let stop = AtomicBool::new(false);
        let nested = Nested {
            level: 1,
            threads: 2,
            ..Nested::default()
        };
        let sol = nested.solve(&tiles, 1, &stop, |step, _| {
            if step == 5 {
                stop.store(true, Ordering::Relaxed);
            }
        });
        assert_eq!(verify(&tiles, &sol.moves), Ok(sol.score));
    }

    #[test]
    fn deep_levels_handle_the_end_of_the_game() {
        // Searching the last few tiles at a level above the number left.
        let tiles = deal(8);
        let mut board = Board::new();
        let mut rng = Rng::new(1);
        for &v in &tiles[..17] {
            let mv = Strategy::EdgesThenUp.choose(&board, v, &mut rng);
            board.place(v, mv).unwrap();
        }
        let nested = Nested {
            level: 4,
            per_layer: 2,
            ..Nested::default()
        };
        let sol = nested.search(&mut board, &tiles[17..], 4, 3);
        assert_eq!(sol.moves.len(), 3);
        assert_eq!(board.pieces().len(), 17, "board restored");
        let mut prefix: Vec<Move> = board.pieces().iter().map(|p| p.mv).collect();
        prefix.extend(&sol.moves);
        assert_eq!(verify(&tiles, &prefix), Ok(sol.score));
    }

    #[test]
    fn verify_rejects_illegal_plays() {
        let tiles = deal(7);
        let mut moves = vec![Board::center_move(); tiles.len()];
        moves[1] = Move::new(0, 10, 10, 0);
        assert!(verify(&tiles, &moves).is_err());
        assert!(verify(&tiles, &moves[..3]).is_err());
    }
}
