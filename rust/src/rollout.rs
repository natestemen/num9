//! Monte Carlo rollouts: judge each candidate move by playing the rest of the
//! game out many times.
//!
//! The only hidden information in NMBR 9 is the order of the remaining tiles
//! (which tiles remain is known), so a rollout is an honest sample of how the
//! game could continue: shuffle the remaining tiles, finish the game with a
//! fast policy, and record the final score.

use crate::board::{Board, Move, DEPTH};
use crate::piece::shape;
use crate::rng::Rng;
use crate::strategies::Strategy;

/// Settings for [`Strategy::MonteCarlo`].
#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarlo {
    /// Candidate moves kept per layer (those touching the most edges).
    pub per_layer: usize,
    /// Simulated continuations per candidate.
    pub rollouts: usize,
    /// Policy that finishes each simulated game.
    pub policy: Box<Strategy>,
    /// Mixed into the per-position random seed.
    pub seed: u64,
}

impl Default for MonteCarlo {
    fn default() -> Self {
        MonteCarlo {
            per_layer: 4,
            rollouts: 64,
            policy: Box::new(Strategy::LowToBase {
                max_low: 4,
                endgame: 4,
            }),
            seed: 0,
        }
    }
}

/// Tiles still to be drawn after `value` is placed on `board`.
fn remaining_tiles(board: &Board, value: u8) -> Vec<u8> {
    let mut counts = [2u8; 10];
    for p in board.pieces() {
        counts[p.value as usize] -= 1;
    }
    counts[value as usize] -= 1;
    (0..10u8)
        .flat_map(|v| std::iter::repeat_n(v, counts[v as usize] as usize))
        .collect()
}

/// Indices of the moves worth simulating: on each layer, the `per_layer`
/// moves touching the most edges (earliest first on ties). Placements that
/// repeat an earlier one (half-turn-symmetric pieces) are skipped.
pub fn candidate_moves(board: &Board, value: u8, moves: &[Move], per_layer: usize) -> Vec<usize> {
    let half_turn = shape(value, 0).rows == shape(value, 2).rows;
    let mut by_layer: [Vec<(u32, usize)>; DEPTH] = Default::default();
    for (k, &mv) in moves.iter().enumerate() {
        if half_turn && mv.rot >= 2 {
            continue;
        }
        by_layer[mv.layer].push((board.num_edges_touching(value, mv), k));
    }
    let mut out = Vec::new();
    for layer in by_layer.iter_mut() {
        layer.sort_by_key(|&(edges, k)| (std::cmp::Reverse(edges), k));
        out.extend(layer.iter().take(per_layer).map(|&(_, k)| k));
    }
    out.sort_unstable();
    out
}

/// A seed determined by the position, so decisions are reproducible.
fn position_seed(board: &Board, value: u8, seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x0000_0100_0000_01b3);
    for p in board.pieces() {
        for x in [p.value as usize, p.mv.layer, p.mv.i, p.mv.j, p.mv.rot] {
            mix(x as u64);
        }
    }
    mix(value as u64);
    h
}

/// Candidates with the best total score over the rollouts (all of them on a tie).
pub fn monte_carlo(board: &Board, value: u8, moves: &[Move], mc: &MonteCarlo) -> Vec<usize> {
    let candidates = candidate_moves(board, value, moves, mc.per_layer);
    if candidates.len() <= 1 {
        return candidates;
    }
    let totals = rollout_totals(board, value, moves, &candidates, mc);
    let best = *totals.iter().max().unwrap();
    candidates
        .iter()
        .zip(&totals)
        .filter(|&(_, &t)| t == best)
        .map(|(&k, _)| k)
        .collect()
}

/// Sum of final scores over `mc.rollouts` simulated games for each candidate.
/// Every candidate sees the same tile orders and policy random numbers, so
/// differences between them come from the move, not from luck.
pub fn rollout_totals(
    board: &Board,
    value: u8,
    moves: &[Move],
    candidates: &[usize],
    mc: &MonteCarlo,
) -> Vec<i64> {
    let remaining = remaining_tiles(board, value);
    let mut rng = Rng::new(position_seed(board, value, mc.seed));
    let mut totals = vec![0i64; candidates.len()];
    let mut sim = board.clone();
    let depth = board.pieces().len();
    let mut order = remaining.clone();
    for _ in 0..mc.rollouts {
        order.copy_from_slice(&remaining);
        rng.shuffle(&mut order);
        let policy_seed = rng.next_u64();
        for (total, &k) in totals.iter_mut().zip(candidates) {
            sim.place(value, moves[k]).expect("legal move");
            let mut policy_rng = Rng::new(policy_seed);
            for &v in &order {
                let mv = mc.policy.choose(&sim, v, &mut policy_rng);
                sim.place(v, mv).expect("policies choose legal moves");
            }
            *total += sim.score();
            while sim.pieces().len() > depth {
                sim.undo();
            }
        }
    }
    totals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::shuffled_tiles;

    #[test]
    fn remaining_tiles_excludes_played_and_current() {
        let mut board = Board::new();
        board.place(9, Board::center_move()).unwrap();
        let rest = remaining_tiles(&board, 9);
        assert_eq!(rest.len(), 18);
        assert!(!rest.contains(&9));
        assert_eq!(rest.iter().filter(|&&v| v == 4).count(), 2);
    }

    #[test]
    fn candidates_cover_every_layer_and_skip_duplicates() {
        let mut rng = Rng::new(3);
        let mut board = Board::new();
        for value in shuffled_tiles(&mut rng).into_iter().take(12) {
            let mv = Strategy::LowToBase {
                max_low: 3,
                endgame: 0,
            }
            .choose(&board, value, &mut rng);
            board.place(value, mv).unwrap();
        }
        for value in 0..10u8 {
            let moves = board.find_valid_moves(value);
            let cands = candidate_moves(&board, value, &moves, 3);
            let layers = |ks: &[usize]| {
                let mut l: Vec<_> = ks.iter().map(|&k| moves[k].layer).collect();
                l.dedup();
                l
            };
            let all: Vec<usize> = (0..moves.len()).collect();
            assert_eq!(layers(&cands), layers(&all), "piece {value}");
            for &k in &cands {
                assert!(!(value == 0 || value == 8) || moves[k].rot < 2);
            }
        }
    }

    #[test]
    fn rollouts_restore_the_board_and_are_reproducible() {
        let mut rng = Rng::new(4);
        let mut board = Board::new();
        for value in shuffled_tiles(&mut rng).into_iter().take(8) {
            let mv = Strategy::EdgesThenUp.choose(&board, value, &mut rng);
            board.place(value, mv).unwrap();
        }
        let mc = MonteCarlo {
            rollouts: 8,
            ..Default::default()
        };
        let moves = board.find_valid_moves(6);
        let cands = candidate_moves(&board, 6, &moves, mc.per_layer);
        let a = rollout_totals(&board, 6, &moves, &cands, &mc);
        let b = rollout_totals(&board, 6, &moves, &cands, &mc);
        assert_eq!(a, b);
        assert!(a.iter().all(|&t| t >= board.score() * 8));
    }
}
