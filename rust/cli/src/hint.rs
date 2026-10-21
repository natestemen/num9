//! Background search for the best move, using the engine's Monte Carlo
//! rollouts: candidate moves are each played out many times with the
//! remaining tiles shuffled, and ranked by average final score.
//!
//! The work is split into small chunks of rollouts spread over worker
//! threads, so results stream in (the UI shows them improving) and the search
//! can be abandoned as soon as the board changes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

use num9_core::board::{Board, Move};
use num9_core::piece::shape;
use num9_core::rollout::{candidate_moves, rollout_totals, MonteCarlo};

/// Rollouts per chunk of work.
const CHUNK: usize = 8;
/// Candidate moves kept per level (more than the strategy's default: the
/// player may want to compare several options).
const PER_LAYER: usize = 6;

/// One move and how it fared.
#[derive(Clone, Copy, Debug)]
pub struct Rated {
    pub mv: Move,
    /// Average final score of the game over the rollouts so far.
    pub expected: f64,
}

pub struct Hint {
    /// Number of tiles on the board when the search started; the hint only
    /// applies to that position.
    pub placed: usize,
    pub value: u8,
    candidates: Vec<Move>,
    totals: Vec<i64>,
    /// Index in `candidates` of the move the player was considering.
    yours: Option<usize>,
    /// The player's move when the hint was asked for.
    pub asked_at: Option<Move>,
    pub done: usize,
    pub target: usize,
    rx: Receiver<Vec<i64>>,
    cancel: Arc<AtomicBool>,
}

impl Hint {
    /// Start searching for `value`'s best move on `board`, also rating the
    /// player's move `yours` if given. `rollouts` per candidate.
    pub fn start(board: &Board, value: u8, yours: Option<Move>, rollouts: usize) -> Hint {
        let moves = board.find_valid_moves(value);
        let mut picks = candidate_moves(board, value, &moves, PER_LAYER);
        // Rate the player's move too (or the identical placement under the
        // equivalent rotation of a symmetric piece).
        let same = |a: Move, b: Move| {
            (a.layer, a.i, a.j) == (b.layer, b.i, b.j)
                && shape(value, a.rot).rows == shape(value, b.rot).rows
        };
        let asked_at = yours;
        let yours = yours.and_then(|y| {
            if let Some(pos) = picks.iter().position(|&k| same(moves[k], y)) {
                return Some(pos);
            }
            let k = moves.iter().position(|&m| m == y)?;
            picks.push(k);
            Some(picks.len() - 1)
        });
        let candidates: Vec<Move> = picks.iter().map(|&k| moves[k]).collect();

        let chunks = rollouts.div_ceil(CHUNK).max(1);
        let workers = thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(chunks);
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let board = Arc::new(board.clone());
        let moves = Arc::new(moves);
        let picks = Arc::new(picks);
        for w in 0..workers {
            let (tx, cancel) = (tx.clone(), cancel.clone());
            let (board, moves, picks) = (board.clone(), moves.clone(), picks.clone());
            thread::spawn(move || {
                for chunk in (w..chunks).step_by(workers) {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let mc = MonteCarlo {
                        rollouts: CHUNK,
                        seed: chunk as u64,
                        ..MonteCarlo::default()
                    };
                    let totals = rollout_totals(&board, value, &moves, &picks, &mc);
                    if tx.send(totals).is_err() {
                        return;
                    }
                }
            });
        }
        Hint {
            placed: board.pieces().len(),
            value,
            totals: vec![0; candidates.len()],
            candidates,
            yours,
            asked_at,
            done: 0,
            target: chunks * CHUNK,
            rx,
            cancel,
        }
    }

    /// Collect finished chunks; returns whether anything changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(totals) = self.rx.try_recv() {
            for (t, x) in self.totals.iter_mut().zip(totals) {
                *t += x;
            }
            self.done += CHUNK;
            changed = true;
        }
        changed
    }

    pub fn finished(&self) -> bool {
        self.done >= self.target
    }

    pub fn progress(&self) -> f64 {
        self.done as f64 / self.target as f64
    }

    /// Whether there is only one way to place the tile (nothing to compare).
    pub fn forced(&self) -> bool {
        self.candidates.len() <= 1
    }

    fn rated(&self, k: usize) -> Rated {
        Rated {
            mv: self.candidates[k],
            expected: self.totals[k] as f64 / self.done.max(1) as f64,
        }
    }

    /// Candidates, best first (empty until the first results arrive).
    pub fn ranking(&self) -> Vec<Rated> {
        if self.done == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..self.candidates.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(self.totals[k]));
        order.into_iter().map(|k| self.rated(k)).collect()
    }

    /// The player's move, with its rank (1 = best).
    pub fn yours(&self) -> Option<(usize, Rated)> {
        let k = self.yours?;
        if self.done == 0 {
            return None;
        }
        let rank = 1 + self.totals.iter().filter(|&&t| t > self.totals[k]).count();
        Some((rank, self.rated(k)))
    }
}

impl Drop for Hint {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait(h: &mut Hint) {
        let start = Instant::now();
        while !h.finished() {
            h.poll();
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "hint never finished"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn position() -> Board {
        let mut b = Board::new();
        for v in [9u8, 8, 5, 0, 2, 6] {
            let mv = *b
                .find_valid_moves(v)
                .iter()
                .max_by_key(|m| (m.layer, m.j))
                .unwrap();
            b.place(v, mv).unwrap();
        }
        b
    }

    #[test]
    fn ranks_candidates_and_rates_the_players_move() {
        let board = position();
        let mine = board.find_valid_moves(7)[3];
        let mut h = Hint::start(&board, 7, Some(mine), 32);
        wait(&mut h);
        let ranking = h.ranking();
        assert!(ranking.len() > 1);
        assert!(ranking.windows(2).all(|w| w[0].expected >= w[1].expected));
        let (rank, rated) = h.yours().unwrap();
        assert_eq!(rated.mv, mine);
        assert_eq!(ranking[rank - 1].expected, rated.expected);
        // Expected final scores can't be below what's already on the board.
        assert!(ranking.iter().all(|r| r.expected >= board.score() as f64));
    }

    #[test]
    fn is_deterministic() {
        let board = position();
        let mut a = Hint::start(&board, 4, None, 24);
        let mut b = Hint::start(&board, 4, None, 24);
        wait(&mut a);
        wait(&mut b);
        let key = |h: &Hint| {
            h.ranking()
                .iter()
                .map(|r| (r.mv, r.expected.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&a), key(&b));
    }
}
