//! Ports of the strategies from the Python `num9/strategies.py`.
//!
//! Each strategy is split into a deterministic [`Strategy::decide`] step, which
//! returns the legal moves plus the indices the Python version would hand to
//! `random.choice`, and the random pick itself. That split is what lets
//! `tests/golden.rs` check the ports decision-by-decision.

use crate::board::{Board, Move};
use crate::game::TILES;
use crate::piece::shape;
use crate::rng::Rng;
use crate::rollout::{self, MonteCarlo};
use crate::support::{values_to_come, Support};

/// Weights for [`Strategy::SolidBaseHighTop`]; defaults match the Python ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidWeights {
    pub fill_weight: f64,
    pub hole_penalty: f64,
    pub support_weight: f64,
    pub edge_weight: f64,
    pub height_weight: f64,
    pub aspect_penalty_weight: f64,
    pub low_height_penalty_weight: f64,
}

impl Default for SolidWeights {
    fn default() -> Self {
        SolidWeights {
            fill_weight: 0.3866522473060556,
            hole_penalty: 0.2285685361703878,
            support_weight: 6.522728683569122,
            edge_weight: 2.209594713525341,
            height_weight: 13.939406411565585,
            aspect_penalty_weight: 2.3253084372479393,
            low_height_penalty_weight: 9.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Strategy {
    PlaceRandomly,
    GoUpRandomly,
    EdgesThenUp,
    SolidBaseHighTop(SolidWeights),
    MaximizeLookahead,
    /// `EdgesThenUp` with a tunable cut-off: pieces worth at most `max_low`
    /// go on the base layer, the rest as high as possible; ties broken by
    /// most edges touching. In the last `endgame` tiles of the game every
    /// piece goes as high as possible (nothing is left to build on).
    /// `EdgesThenUp` is `LowToBase { max_low: 1, endgame: 0 }`.
    LowToBase {
        max_low: u8,
        endgame: u8,
    },
    /// Pick the candidate move with the best average final score over
    /// simulated continuations (see [`crate::rollout`]).
    MonteCarlo(MonteCarlo),
    /// `LowToBase`, except that a tile going up also weighs the support it
    /// strands: cells of the level below that no tile still to come could
    /// rest on any more (see [`crate::support`]). Spots on the level are
    /// scored `edges touching - weight × cells stranded`.
    KeepSupport {
        max_low: u8,
        endgame: u8,
        weight: f64,
    },
}

/// A strategy's deterministic output: pick uniformly among `moves[candidates[k]]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub moves: Vec<Move>,
    pub candidates: Vec<usize>,
}

impl Decision {
    pub fn candidate_moves(&self) -> impl Iterator<Item = Move> + '_ {
        self.candidates.iter().map(|&k| self.moves[k])
    }
}

impl Strategy {
    pub const NAMES: [&'static str; 8] = [
        "place_randomly",
        "go_up_randomly",
        "edges_then_up",
        "solid_base_high_top",
        "maximize_lookahead",
        "low_to_base",
        "monte_carlo",
        "keep_support",
    ];

    /// Look a strategy up by the name used in `benchmark_strategies.py`
    /// (the Python function names are accepted too).
    pub fn from_name(name: &str) -> Option<Strategy> {
        Some(match name {
            "place_randomly" => Strategy::PlaceRandomly,
            "go_up_randomly" => Strategy::GoUpRandomly,
            "edges_then_up" | "choose_move_with_most_edges_touching" => Strategy::EdgesThenUp,
            "solid_base_high_top" | "choose_solid_base_high_top" => {
                Strategy::SolidBaseHighTop(SolidWeights::default())
            }
            "maximize_lookahead" => Strategy::MaximizeLookahead,
            "low_to_base" => Strategy::LowToBase {
                max_low: 4,
                endgame: 4,
            },
            "monte_carlo" => Strategy::MonteCarlo(MonteCarlo::default()),
            "keep_support" => Strategy::KeepSupport {
                max_low: 4,
                endgame: 4,
                weight: 1.0,
            },
            _ => return None,
        })
    }

    pub fn decide(&self, board: &Board, value: u8) -> Decision {
        let moves = board.find_valid_moves(value);
        let candidates = match self {
            Strategy::PlaceRandomly => (0..moves.len()).collect(),
            Strategy::GoUpRandomly => go_up_randomly(&moves),
            Strategy::EdgesThenUp => edges_then_up(board, value, &moves, Some(1)),
            Strategy::LowToBase { max_low, endgame } => {
                let tiles_left = TILES.saturating_sub(board.pieces().len());
                let max_low = (tiles_left > *endgame as usize).then_some(*max_low);
                edges_then_up(board, value, &moves, max_low)
            }
            Strategy::KeepSupport {
                max_low,
                endgame,
                weight,
            } => {
                let tiles_left = TILES.saturating_sub(board.pieces().len());
                let max_low = (tiles_left > *endgame as usize).then_some(*max_low);
                keep_support(board, value, &moves, max_low, *weight)
            }
            Strategy::MonteCarlo(mc) => rollout::monte_carlo(board, value, &moves, mc),
            Strategy::SolidBaseHighTop(w) => solid_base_high_top(board, value, &moves, w),
            Strategy::MaximizeLookahead => maximize_lookahead(board, value, &moves),
        };
        Decision { moves, candidates }
    }

    pub fn choose(&self, board: &Board, value: u8, rng: &mut Rng) -> Move {
        let d = self.decide(board, value);
        d.moves[*rng.choose(&d.candidates)]
    }
}

fn indices_on_layer(moves: &[Move], layer: usize) -> Vec<usize> {
    (0..moves.len())
        .filter(|&k| moves[k].layer == layer)
        .collect()
}

fn max_layer(moves: &[Move]) -> usize {
    moves.iter().map(|m| m.layer).max().unwrap()
}

fn go_up_randomly(moves: &[Move]) -> Vec<usize> {
    indices_on_layer(moves, max_layer(moves))
}

/// Pieces worth at most `max_low` (if any) on the base layer, the rest as
/// high as possible; then most edges touching.
fn edges_then_up(board: &Board, value: u8, moves: &[Move], max_low: Option<u8>) -> Vec<usize> {
    let options = if max_low.is_some_and(|m| value <= m) {
        indices_on_layer(moves, 0)
    } else {
        // When every move is on one layer this keeps them all, as in Python.
        indices_on_layer(moves, max_layer(moves))
    };
    let edges: Vec<u32> = options
        .iter()
        .map(|&k| board.num_edges_touching(value, moves[k]))
        .collect();
    let best = *edges.iter().max().unwrap();
    options
        .iter()
        .zip(&edges)
        .filter(|(_, &e)| e == best)
        .map(|(&k, _)| k)
        .collect()
}

/// `edges_then_up`, but spots above the ground lose `weight` per cell of
/// support they strand on the level below.
fn keep_support(
    board: &Board,
    value: u8,
    moves: &[Move],
    max_low: Option<u8>,
    weight: f64,
) -> Vec<usize> {
    let options = if max_low.is_some_and(|m| value <= m) {
        indices_on_layer(moves, 0)
    } else {
        indices_on_layer(moves, max_layer(moves))
    };
    let layer = moves[options[0]].layer;
    let values = values_to_come(board, value);
    let support = Support::new(board, layer);
    let before = support.as_ref().map(|s| s.usable(&values));
    let scores: Vec<f64> = options
        .iter()
        .map(|&k| {
            let mv = moves[k];
            let edges = board.num_edges_touching(value, mv) as f64;
            match (&support, &before) {
                (Some(s), Some(b)) => edges - weight * s.stranded_by(b, value, mv, &values) as f64,
                _ => edges,
            }
        })
        .collect();
    let best = scores.iter().cloned().fold(f64::MIN, f64::max);
    options
        .iter()
        .zip(&scores)
        .filter(|(_, &s)| s == best)
        .map(|(&k, _)| k)
        .collect()
}

/// Score of one move under `choose_solid_base_high_top`. The arithmetic is
/// done in the same order as the Python expression so results are bit-identical.
pub fn solid_score(board: &Board, value: u8, mv: Move, w: &SolidWeights) -> f64 {
    let s = shape(value, mv.rot);
    let piece_tiles = s.tiles;

    // Layer after placing: legal moves never overlap, so fill just adds up.
    let fill = board.layer_fill(mv.layer) + piece_tiles;
    let piece_bb = (mv.i, mv.i + s.h - 1, mv.j, mv.j + s.w - 1);
    let (min_i, max_i, min_j, max_j) = match board.layer_bbox(mv.layer) {
        Some((a, b, c, d)) => (
            a.min(piece_bb.0),
            b.max(piece_bb.1),
            c.min(piece_bb.2),
            d.max(piece_bb.3),
        ),
        None => piece_bb,
    };
    // Every rotation of every piece fills its first/last row and column, so the
    // piece's bounding box is exactly its array's extent.
    let height_span = (max_i - min_i + 1) as i64;
    let width_span = (max_j - min_j + 1) as i64;
    let holes = height_span * width_span - fill as i64;

    let edges = board.num_edges_touching(value, mv);

    let support_ratio = if mv.layer > 0 && piece_tiles > 0 {
        board.support_overlap(value, mv) as f64 / piece_tiles as f64
    } else {
        1.0
    };

    let v = value as i64;
    let layer = mv.layer as i64;
    let height_score = v * layer;
    let aspect_ratio = height_span.max(width_span) as f64 / height_span.min(width_span) as f64;
    let aspect_penalty = aspect_ratio - 1.0;
    let low_height_penalty = v * (2 - layer).max(0);

    w.fill_weight * fill as f64 - w.hole_penalty * holes as f64
        + w.support_weight * support_ratio
        + w.edge_weight * edges as f64
        + w.height_weight * height_score as f64
        - w.aspect_penalty_weight * aspect_penalty
        - w.low_height_penalty_weight * low_height_penalty as f64
}

fn solid_base_high_top(board: &Board, value: u8, moves: &[Move], w: &SolidWeights) -> Vec<usize> {
    let scores: Vec<f64> = moves
        .iter()
        .map(|&mv| solid_score(board, value, mv, w))
        .collect();
    // Python's max() keeps the first maximum; any maximum has the same value.
    let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let margin = best.abs() * 5e-2 + 1e-6;
    (0..moves.len())
        .filter(|&k| best - scores[k] <= margin)
        .collect()
}

/// Average future score of each move, as in `maximize_lookahead`.
pub fn lookahead_scores(board: &Board, value: u8, moves: &[Move]) -> Vec<f64> {
    // Leftover tiles: two of each value minus those already played. The piece
    // being placed is not yet on the board, so it still counts (as in Python).
    let mut counts = [2i64; 10];
    for p in board.pieces() {
        counts[p.value as usize] -= 1;
    }
    let current_score = board.score();
    let mut tmp = board.clone();
    moves
        .iter()
        .map(|&mv| {
            tmp.place(value, mv).expect("legal move");
            let base_score = current_score + value as i64 * mv.layer as i64;
            let mut total_moves: i64 = 0;
            let mut weighted_layer_sum: i64 = 0;
            for next in 0..10u8 {
                if counts[next as usize] == 0 {
                    continue;
                }
                let per_layer = tmp.count_valid_moves(next);
                let n: i64 = per_layer.iter().map(|&c| c as i64).sum();
                let layer_sum: i64 = (0..per_layer.len())
                    .map(|l| (l * per_layer[l]) as i64)
                    .sum();
                total_moves += counts[next as usize] * n;
                weighted_layer_sum += counts[next as usize] * next as i64 * layer_sum;
            }
            tmp.undo();
            if total_moves == 0 {
                base_score as f64
            } else {
                base_score as f64 + weighted_layer_sum as f64 / total_moves as f64
            }
        })
        .collect()
}

fn maximize_lookahead(board: &Board, value: u8, moves: &[Move]) -> Vec<usize> {
    let scores = lookahead_scores(board, value, moves);
    // First index attaining the maximum, like Python's max() over enumerate().
    let mut best = 0;
    for k in 1..scores.len() {
        if scores[k] > scores[best] {
            best = k;
        }
    }
    vec![best]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::shuffled_tiles;

    #[test]
    fn low_to_base_goes_up_in_the_endgame() {
        let mut rng = Rng::new(5);
        let s = Strategy::LowToBase {
            max_low: 4,
            endgame: 4,
        };
        for _ in 0..20 {
            let mut board = Board::new();
            for value in shuffled_tiles(&mut rng) {
                let d = s.decide(&board, value);
                let top = d.moves.iter().map(|m| m.layer).max().unwrap();
                let layers: Vec<usize> = d.candidate_moves().map(|m| m.layer).collect();
                if TILES - board.pieces().len() <= 4 || value > 4 {
                    assert!(layers.iter().all(|&l| l == top));
                } else {
                    assert!(layers.iter().all(|&l| l == 0));
                }
                board
                    .place(value, d.moves[*rng.choose(&d.candidates)])
                    .unwrap();
            }
        }
    }

    #[test]
    fn low_to_base_one_is_edges_then_up() {
        let mut rng = Rng::new(2);
        for _ in 0..20 {
            let mut board = Board::new();
            for value in shuffled_tiles(&mut rng) {
                let d = Strategy::EdgesThenUp.decide(&board, value);
                let low = Strategy::LowToBase {
                    max_low: 1,
                    endgame: 0,
                };
                assert_eq!(d, low.decide(&board, value));
                board
                    .place(value, d.moves[*rng.choose(&d.candidates)])
                    .unwrap();
            }
        }
    }
}
