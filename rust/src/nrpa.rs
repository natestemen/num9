//! Nested Rollout Policy Adaptation (Rosin, 2011) for finding high-scoring
//! plays, either of a given deal or with the order of the tiles chosen
//! freely (the "maximum possible score" problem).
//!
//! A policy gives every move a weight. A playout picks moves at random in
//! proportion to exp(weight) among all legal ones. Level n runs `iterations`
//! level n-1 searches, each starting from the current policy, keeps the best
//! sequence seen, and after each one shifts the policy towards that sequence
//! (raising its moves' weights and lowering the alternatives in proportion
//! to how likely they were). Moves are identified by tile value and absolute
//! placement; the first tile always goes in the middle, so a placement means
//! the same thing in every playout.

use std::collections::HashMap;

use crate::board::{Board, Move};
use crate::piece::shape;
use crate::rng::Rng;

/// A placement of a tile: its value and where it goes.
pub type Step = (u8, Move);

/// Settings for NRPA.
#[derive(Clone, Debug)]
pub struct Nrpa {
    pub level: usize,
    /// Level n-1 searches per level-n search.
    pub iterations: usize,
    /// Learning rate for the policy updates.
    pub alpha: f64,
    /// Prior added to every move's weight: this much per edge the tile
    /// touches (compact placements give better support)...
    pub edge_bias: f64,
    /// ...and this much per (level × value / 9) (high values high).
    pub height_bias: f64,
}

impl Default for Nrpa {
    fn default() -> Self {
        Nrpa {
            level: 3,
            iterations: 100,
            alpha: 1.0,
            edge_bias: 0.0,
            height_bias: 0.0,
        }
    }
}

/// What may be placed next: the deal's next tile, or (free order) any tile
/// not yet used.
#[derive(Clone, Debug)]
pub enum Order {
    /// Tiles in this order.
    Fixed(Vec<u8>),
    /// How many of each value to place, in any order.
    Free([u8; 10]),
}

impl Order {
    pub fn standard_free() -> Order {
        Order::Free([2; 10])
    }

    fn len(&self) -> usize {
        match self {
            Order::Fixed(t) => t.len(),
            Order::Free(c) => c.iter().map(|&n| n as usize).sum(),
        }
    }
}

/// The best play found: its score and placements (whose values give the
/// order in which the tiles were placed).
#[derive(Clone, Debug, PartialEq)]
pub struct Play {
    pub score: i64,
    pub steps: Vec<Step>,
}

impl Play {
    /// The tile order this play uses.
    pub fn order(&self) -> Vec<u8> {
        self.steps.iter().map(|s| s.0).collect()
    }

    pub fn moves(&self) -> Vec<Move> {
        self.steps.iter().map(|s| s.1).collect()
    }
}

type Policy = HashMap<u64, f64>;

/// The distinct legal placements of `value` (pieces 0 and 8 look the same
/// under rotations r and r + 2; only the first is kept).
fn distinct_moves(board: &Board, value: u8) -> Vec<Move> {
    let half_turn = shape(value, 0).rows == shape(value, 2).rows;
    let mut moves = board.find_valid_moves(value);
    if half_turn {
        moves.retain(|m| m.rot < 2);
    }
    moves
}

fn code((value, m): Step) -> u64 {
    (value as u64)
        | (m.layer as u64) << 4
        | (m.i as u64) << 8
        | (m.j as u64) << 16
        | (m.rot as u64) << 24
}

/// Legal steps after `placed` tiles, given what is left (`left[v]` copies
/// of each value).
fn legal(board: &Board, order: &Order, placed: usize, left: &[u8; 10]) -> Vec<Step> {
    match order {
        Order::Fixed(tiles) => {
            let v = tiles[placed];
            distinct_moves(board, v)
                .into_iter()
                .map(|m| (v, m))
                .collect()
        }
        Order::Free(_) => {
            let mut out = Vec::new();
            for v in (0..10u8).filter(|&v| left[v as usize] > 0) {
                if board.pieces().is_empty() {
                    // Every tile starts in the middle; only the value is a choice.
                    out.push((v, Board::center_move()));
                } else {
                    out.extend(distinct_moves(board, v).into_iter().map(|m| (v, m)));
                }
            }
            out
        }
    }
}

fn counts(order: &Order) -> [u8; 10] {
    match order {
        Order::Fixed(tiles) => {
            let mut c = [0u8; 10];
            for &v in tiles {
                c[v as usize] += 1;
            }
            c
        }
        Order::Free(c) => *c,
    }
}

impl Nrpa {
    /// exp(learned weight + prior) for each of `moves`.
    fn weights(&self, board: &Board, moves: &[Step], policy: &Policy, out: &mut Vec<f64>) {
        out.clear();
        out.extend(moves.iter().map(|&(v, m)| {
            let mut w = policy.get(&code((v, m))).copied().unwrap_or(0.0);
            if self.edge_bias != 0.0 {
                w += self.edge_bias * board.num_edges_touching(v, m) as f64;
            }
            w += self.height_bias * (m.layer * v as usize) as f64 / 9.0;
            w.exp()
        }));
    }

    fn playout(&self, order: &Order, policy: &Policy, rng: &mut Rng) -> Play {
        let mut board = Board::new();
        let mut left = counts(order);
        let mut steps = Vec::with_capacity(order.len());
        let mut weights = Vec::new();
        for placed in 0..order.len() {
            let moves = legal(&board, order, placed, &left);
            self.weights(&board, &moves, policy, &mut weights);
            let total: f64 = weights.iter().sum();
            let mut x = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
            let mut pick = moves.len() - 1;
            for (k, w) in weights.iter().enumerate() {
                if x < *w {
                    pick = k;
                    break;
                }
                x -= w;
            }
            let (v, m) = moves[pick];
            board.place(v, m).expect("legal move");
            left[v as usize] -= 1;
            steps.push((v, m));
        }
        Play {
            score: board.score(),
            steps,
        }
    }

    /// Shift `policy` towards `best`.
    fn adapt(&self, order: &Order, policy: &Policy, best: &Play) -> Policy {
        let mut next = policy.clone();
        let mut board = Board::new();
        let mut left = counts(order);
        for (placed, &step) in best.steps.iter().enumerate() {
            let moves = legal(&board, order, placed, &left);
            let mut weights = Vec::new();
            self.weights(&board, &moves, policy, &mut weights);
            let total: f64 = weights.iter().sum();
            for (&s, w) in moves.iter().zip(&weights) {
                *next.entry(code(s)).or_insert(0.0) -= self.alpha * w / total;
            }
            *next.entry(code(step)).or_insert(0.0) += self.alpha;
            board.place(step.0, step.1).expect("legal move");
            left[step.0 as usize] -= 1;
        }
        next
    }

    fn search(
        &self,
        order: &Order,
        level: usize,
        mut policy: Policy,
        rng: &mut Rng,
        report: &mut impl FnMut(&Play),
    ) -> Play {
        if level == 0 {
            return self.playout(order, &policy, rng);
        }
        let mut best: Option<Play> = None;
        for _ in 0..self.iterations {
            let play = self.search(order, level - 1, policy.clone(), rng, report);
            if best.as_ref().is_none_or(|b| play.score >= b.score) {
                if best.as_ref().is_none_or(|b| play.score > b.score) && level == self.level {
                    report(&play);
                }
                best = Some(play);
            }
            policy = self.adapt(order, &policy, best.as_ref().unwrap());
        }
        best.unwrap()
    }

    /// Run the search; `report` is called whenever the top level improves.
    pub fn solve(&self, order: &Order, seed: u64, mut report: impl FnMut(&Play)) -> Play {
        self.search(
            order,
            self.level,
            Policy::new(),
            &mut Rng::new(seed),
            &mut report,
        )
    }
}

/// Replay a play, checking every move is legal and the tiles match the
/// order (for free order, that each value is used as often as allowed).
pub fn verify(order: &Order, play: &Play) -> Result<i64, String> {
    if play.steps.len() != order.len() {
        return Err(format!(
            "{} steps for {} tiles",
            play.steps.len(),
            order.len()
        ));
    }
    let mut board = Board::new();
    let mut left = counts(order);
    for (k, &(v, m)) in play.steps.iter().enumerate() {
        if let Order::Fixed(tiles) = order {
            if tiles[k] != v {
                return Err(format!(
                    "step {k} places a {v}, the deal has a {}",
                    tiles[k]
                ));
            }
        }
        if left[v as usize] == 0 {
            return Err(format!("step {k} uses one {v} too many"));
        }
        if !board.is_valid_move(v, m) {
            return Err(format!("step {k} ({v} at {m:?}) is not legal"));
        }
        board.place(v, m).map_err(|e| e.to_string())?;
        left[v as usize] -= 1;
    }
    if board.score() != play.score {
        return Err(format!(
            "scores {} but claims {}",
            board.score(),
            play.score
        ));
    }
    Ok(play.score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::shuffled_tiles;

    #[test]
    fn plays_are_legal_for_fixed_and_free_orders() {
        let fixed = Order::Fixed(shuffled_tiles(&mut Rng::new(1)));
        let nrpa = Nrpa {
            level: 1,
            iterations: 10,
            ..Nrpa::default()
        };
        for order in [fixed, Order::standard_free()] {
            let play = nrpa.solve(&order, 3, |_| {});
            assert_eq!(verify(&order, &play), Ok(play.score));
        }
    }

    #[test]
    fn learning_improves_on_random_playouts() {
        let order = Order::Fixed((0..10u8).flat_map(|v| [v, v]).collect());
        let random = Nrpa {
            level: 0,
            iterations: 0,
            ..Nrpa::default()
        };
        let mut rng = Rng::new(5);
        let typical: f64 = (0..50)
            .map(|_| random.playout(&order, &Policy::new(), &mut rng).score as f64)
            .sum::<f64>()
            / 50.0;
        let learned = Nrpa {
            level: 2,
            iterations: 20,
            ..Nrpa::default()
        }
        .solve(&order, 5, |_| {});
        assert!(
            learned.score as f64 > typical + 10.0,
            "learned {} vs random {typical}",
            learned.score
        );
    }

    #[test]
    fn verify_catches_wrong_orders() {
        let order = Order::Fixed(vec![9, 8, 7]);
        let mut board = Board::new();
        board.place(9, Board::center_move()).unwrap();
        let m = distinct_moves(&board, 8)[0];
        let bad = Play {
            score: 0,
            steps: vec![(9, Board::center_move()), (7, m)],
        };
        assert!(verify(&order, &bad).is_err());
        let free = Order::Free([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]);
        let twice = Play {
            score: 0,
            steps: vec![(9, Board::center_move()), (9, m)],
        };
        assert!(verify(&free, &twice).is_err());
    }
}
