//! The highest score a small deck allows when its tiles may be played in any
//! order, proven by an exhaustive search, with a play that reaches it.
//!
//! [`crate::exact`] answers this too, but it visits every position because it
//! also solves the game with the order random. With the order free, two
//! things make the search far smaller:
//!
//! - Any finished stack can be built level by level (the ground first, then
//!   each level, each tile touching one already down), so the search never
//!   places a tile below the highest level in use.
//! - Branch and bound: a position is abandoned when even an optimistic
//!   ceiling on its final score can't beat the best play found so far. The
//!   ceiling puts the tiles left as high as the rule that every level above
//!   the ground rests on at least two tiles allows (see [`ceiling`]).
//!
//! Neither discards anything that could do better, so the best play found
//! is the best there is.

use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;

use crate::board::{Board, Move, DEPTH};
use crate::exact::{key, moves, remaining, Prehashed};

type Visited = HashSet<u128, BuildHasherDefault<Prehashed>>;

/// A play: the tiles in the order placed, from the empty board.
pub type Line = Vec<(u8, Move)>;

/// The most the tiles `left` could still add to a stack whose highest level
/// is `top` (holding `on_top` tiles), placed on `top` or above. Each level
/// in use above the ground needs at least two tiles on the level below it,
/// so for each final height the cheapest tiles fill those, and the rest go on
/// the top; area is ignored, so this never underestimates.
pub fn ceiling(left: &[u8; 10], top: usize, on_top: usize) -> i64 {
    let mut values: Vec<i64> = (0..10)
        .flat_map(|v| std::iter::repeat_n(v as i64, left[v] as usize))
        .collect();
    values.sort_unstable();
    let n = values.len();
    let mut best = values.iter().sum::<i64>() * top as i64;
    let first = 2usize.saturating_sub(on_top);
    for h in 1..DEPTH - top {
        // `first` more on `top`, two on each level up to `top + h - 1`,
        // at least one on `top + h`.
        let needed = first + 2 * (h - 1) + 1;
        if needed > n {
            break;
        }
        let mut total = 0;
        let mut k = 0;
        for _ in 0..first {
            total += values[k] * top as i64;
            k += 1;
        }
        for level in 1..h {
            for _ in 0..2 {
                total += values[k] * (top + level) as i64;
                k += 1;
            }
        }
        total += values[k..].iter().sum::<i64>() * (top + h) as i64;
        best = best.max(total);
    }
    best
}

/// The search's shared state: positions already explored, and the best
/// play so far.
struct Search {
    visited: Vec<Mutex<Visited>>,
    best: AtomicI64,
    line: Mutex<Line>,
    nodes: AtomicUsize,
}

impl Search {
    fn first_visit(&self, k: u128) -> bool {
        self.visited[k as usize % self.visited.len()]
            .lock()
            .unwrap()
            .insert(k)
    }

    fn found(&self, score: i64, path: &Line) {
        let mut line = self.line.lock().unwrap();
        if score > self.best.load(Relaxed) {
            self.best.store(score, Relaxed);
            *line = path.clone();
        }
    }

    fn dfs(&self, board: &mut Board, left: &mut [u8; 10], path: &mut Line) {
        self.nodes.fetch_add(1, Relaxed);
        let score = board.score();
        if remaining(left) == 0 {
            if score > self.best.load(Relaxed) {
                self.found(score, path);
            }
            return;
        }
        let top = (0..DEPTH)
            .rev()
            .find(|&l| !board.layer_is_empty(l))
            .unwrap_or(0);
        let on_top = board.pieces().iter().filter(|p| p.mv.layer == top).count();
        if score + ceiling(left, top, on_top) <= self.best.load(Relaxed) {
            return;
        }
        // Explored before (perhaps by another thread, perhaps still going):
        // whatever it can reach is, or will be, in `best` already.
        if !board.pieces().is_empty() && !self.first_visit(key(board)) {
            return;
        }
        // Higher placements of higher values first, to find good plays early.
        let mut options: Vec<(u8, Move)> = (0..10u8)
            .filter(|&v| left[v as usize] > 0)
            .flat_map(|v| moves(board, v).into_iter().map(move |m| (v, m)))
            .filter(|(_, m)| m.layer >= top)
            .collect();
        options.sort_by_key(|&(v, m)| std::cmp::Reverse((m.layer, v)));
        for (v, m) in options {
            board.place(v, m).expect("legal move");
            left[v as usize] -= 1;
            path.push((v, m));
            self.dfs(board, left, path);
            path.pop();
            left[v as usize] += 1;
            board.undo();
        }
    }
}

/// The best score for the deck `left` in free order, a play reaching it,
/// and how many positions were searched. `known` is a play to beat (its
/// score is a starting lower bound), if one is at hand.
pub fn best_free(left: [u8; 10], known: Option<(i64, Line)>, threads: usize) -> (i64, Line, usize) {
    let (best, line) = known.unwrap_or((-1, Vec::new()));
    let search = Search {
        visited: (0..256).map(|_| Mutex::default()).collect(),
        best: AtomicI64::new(best),
        line: Mutex::new(line),
        nodes: AtomicUsize::new(0),
    };
    // Split the work at the second tile: every pair of first two tiles,
    // shared out between the threads.
    let mut board = Board::new();
    let mut jobs: Vec<(u8, u8, Move)> = Vec::new();
    for a in (0..10u8).filter(|&v| left[v as usize] > 0) {
        board.place(a, Board::center_move()).expect("empty board");
        let mut rest = left;
        rest[a as usize] -= 1;
        for b in (0..10u8).filter(|&v| rest[v as usize] > 0) {
            jobs.extend(moves(&board, b).into_iter().map(|m| (a, b, m)));
        }
        board.undo();
    }
    if remaining(&left) < 2 {
        let mut path = Vec::new();
        search.dfs(&mut Board::new(), &mut left.clone(), &mut path);
    } else {
        let next = AtomicUsize::new(0);
        thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    while let Some(&(a, b, m)) = jobs.get(next.fetch_add(1, Relaxed)) {
                        let mut board = Board::new();
                        let mut left = left;
                        board.place(a, Board::center_move()).expect("empty board");
                        board.place(b, m).expect("legal move");
                        left[a as usize] -= 1;
                        left[b as usize] -= 1;
                        let mut path = vec![(a, Board::center_move()), (b, m)];
                        search.dfs(&mut board, &mut left, &mut path);
                    }
                });
            }
        });
    }
    let line = search.line.into_inner().unwrap();
    (search.best.load(Relaxed), line, search.nodes.load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::{solve_deck, Memo};

    #[test]
    fn agrees_with_the_full_solver() {
        for left in [
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 2],
        ] {
            let full = solve_deck(&Memo::new(), left, 2, 4).best;
            let (best, line, _) = best_free(left, None, 4);
            assert_eq!(best, full, "{left:?}");
            // The line is a legal play of exactly these tiles scoring that.
            let mut board = Board::new();
            let mut used = [0u8; 10];
            for &(v, m) in &line {
                assert!(
                    board.pieces().is_empty() || board.is_valid_move(v, m),
                    "{left:?}"
                );
                board.place(v, m).unwrap();
                used[v as usize] += 1;
            }
            assert_eq!(used, left);
            assert_eq!(board.score(), best);
        }
    }

    #[test]
    fn the_ceiling_is_never_below_a_real_play() {
        // Two of each 0-2 can make 7; from the empty board the ceiling must allow it.
        assert!(ceiling(&[2, 2, 2, 0, 0, 0, 0, 0, 0, 0], 0, 0) >= 7);
        // Nothing left: nothing to add above the ground.
        assert_eq!(ceiling(&[0; 10], 0, 0), 0);
        // One tile, one level up from a top with two tiles.
        assert_eq!(ceiling(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 1, 2), 18);
    }
}
