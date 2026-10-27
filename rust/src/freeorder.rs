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
//!   ceiling shares the tiles left out between the levels in the best way
//!   that keeps each level's area within the level below's and at least two
//!   tiles under every level in use (see [`ceiling`]).
//!
//! Neither discards anything that could do better, so the best play found
//! is the best there is.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;

use crate::board::{Board, Move, DEPTH};
use crate::exact::{key, moves, remaining, Prehashed};
use crate::piece::shape;

type Visited = HashSet<u128, BuildHasherDefault<Prehashed>>;

/// A play: the tiles in the order placed, from the empty board.
pub type Line = Vec<(u8, Move)>;

const NONE: i64 = i64::MIN / 4;

fn area(v: usize) -> u32 {
    shape(v as u8, 0).tiles
}

fn pack(left: &[u8; 10]) -> u64 {
    left.iter()
        .enumerate()
        .fold(0, |k, (v, &c)| k | (c as u64) << (4 * v))
}

/// Every way to take some of the tiles `left` (as counts per value).
fn subsets(left: &[u8; 10]) -> Vec<[u8; 10]> {
    let mut out = vec![[0u8; 10]];
    for v in 0..10 {
        out = out
            .into_iter()
            .flat_map(|s| {
                (0..=left[v]).map(move |c| {
                    let mut t = s;
                    t[v] = c;
                    t
                })
            })
            .collect();
    }
    out
}

thread_local! {
    /// Ceilings already worked out (they depend on few things, so repeat a lot).
    static CEILINGS: RefCell<HashMap<(u64, u8, u8, u16), i64>> = RefCell::new(HashMap::new());
}

fn cached(key: (u64, u8, u8, u16), f: impl FnOnce() -> i64) -> i64 {
    if let Some(v) = CEILINGS.with(|c| c.borrow().get(&key).copied()) {
        return v;
    }
    let v = f();
    CEILINGS.with(|c| c.borrow_mut().insert(key, v));
    v
}

/// The most the tiles `left` could still add to a stack whose highest level
/// is `top`, holding `on_top` tiles over `area_top` cells, placed on `top`
/// or above (or [`NONE`] if they can't all be placed). Every way of sharing
/// them out between the levels is tried, keeping only to two rules every
/// stack obeys: a level's area fits inside the level below's (its tiles lie
/// on it), and a level above the ground has at least two tiles under it.
/// Shapes are ignored, so this never underestimates.
pub fn ceiling(left: &[u8; 10], top: usize, on_top: usize, area_top: u32) -> i64 {
    cached(
        (
            pack(left),
            top as u8,
            100 + on_top.min(2) as u8,
            area_top as u16,
        ),
        || {
            let mut best = NONE;
            for s in subsets(left) {
                let rest: [u8; 10] = std::array::from_fn(|v| left[v] - s[v]);
                let count = on_top + s.iter().map(|&c| c as usize).sum::<usize>();
                let a = area_top + (0..10).map(|v| s[v] as u32 * area(v)).sum::<u32>();
                let gain: i64 = (0..10).map(|v| s[v] as i64 * v as i64).sum::<i64>() * top as i64;
                let above = above(top + 1, a, count >= 2, &rest);
                if above > NONE {
                    best = best.max(gain + above);
                }
            }
            best
        },
    )
}

/// The most the tiles `left` could add on `level` and above, the level below
/// covering `below` cells (with at least two tiles if `two`).
fn above(level: usize, below: u32, two: bool, left: &[u8; 10]) -> i64 {
    if left.iter().all(|&c| c == 0) {
        return 0;
    }
    if level >= DEPTH || !two {
        return NONE;
    }
    cached((pack(left), level as u8, 0, below as u16), || {
        let mut best = NONE;
        for s in subsets(left).into_iter().skip(1) {
            let a = (0..10).map(|v| s[v] as u32 * area(v)).sum::<u32>();
            if a > below {
                continue;
            }
            let rest: [u8; 10] = std::array::from_fn(|v| left[v] - s[v]);
            let count = s.iter().map(|&c| c as usize).sum::<usize>();
            let rest_gain = above(level + 1, a, count >= 2, &rest);
            if rest_gain > NONE {
                let gain: i64 = (0..10).map(|v| s[v] as i64 * v as i64).sum::<i64>() * level as i64;
                best = best.max(gain + rest_gain);
            }
        }
        best
    })
}

/// The search's shared state: positions already explored, and the best
/// play so far.
struct Search {
    visited: Vec<Mutex<Visited>>,
    best: AtomicI64,
    line: Mutex<Line>,
    nodes: AtomicUsize,
    /// Give up after this many positions (keeping the best found).
    budget: usize,
    /// Breaks ties in the order moves are tried (0: the engine's order), so
    /// different searches find different plays among equally good ones.
    salt: u64,
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
        if self.nodes.fetch_add(1, Relaxed) >= self.budget {
            return;
        }
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
        let on_top = board.pieces().iter().filter(|p| p.mv.layer == top);
        let (count, cells) = on_top.fold((0, 0), |(n, a), p| (n + 1, a + area(p.value as usize)));
        if score + ceiling(left, top, count, cells) <= self.best.load(Relaxed) {
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
        let jitter = |m: Move| {
            let x = (m.i as u64) << 24 ^ (m.j as u64) << 8 ^ m.rot as u64;
            (x ^ self.salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32
        };
        options.sort_by_key(|&(v, m)| {
            (
                std::cmp::Reverse((m.layer, v)),
                if self.salt == 0 { 0 } else { jitter(m) },
            )
        });
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
        budget: usize::MAX,
        salt: 0,
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

/// The best way to add the tiles `left` to the stack built by `start` (a
/// legal play), placing them on its highest level or above, if it beats
/// `to_beat`: the score and the whole play. Searches at most `budget`
/// positions, so a result is the best found within it rather than proven;
/// `salt` (non-zero) shuffles which of equally good plays turns up.
pub fn best_from(
    start: &Line,
    left: [u8; 10],
    to_beat: i64,
    budget: usize,
    salt: u64,
) -> Option<(i64, Line)> {
    let mut board = Board::new();
    for &(v, m) in start {
        board.place(v, m).ok()?;
    }
    let search = Search {
        visited: (0..16).map(|_| Mutex::default()).collect(),
        best: AtomicI64::new(to_beat),
        line: Mutex::new(Vec::new()),
        nodes: AtomicUsize::new(0),
        budget,
        salt,
    };
    let mut path = start.clone();
    search.dfs(&mut board, &mut left.clone(), &mut path);
    let best = search.best.load(Relaxed);
    (best > to_beat).then(|| (best, search.line.into_inner().unwrap()))
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
    fn the_ceiling_is_never_below_the_best_play() {
        for left in [
            [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
        ] {
            let (best, _, _) = best_free(left, None, 4);
            assert!(ceiling(&left, 0, 0, 0) >= best, "{left:?}");
        }
        // Nothing left: nothing to add.
        assert_eq!(ceiling(&[0; 10], 0, 0, 0), 0);
        // A 9 (10 cells) fits on a level of two tiles covering 20, not on 9.
        let nine = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(ceiling(&nine, 1, 2, 20), 18);
        assert_eq!(ceiling(&nine, 1, 2, 9), 9);
        // Two 7s and two 8s: by area alone a 7 and an 8 (15 cells) could sit
        // on the others (15 cells), so 15; the shapes allow only 8, but a
        // ceiling only has to stay above.
        assert_eq!(ceiling(&[0, 0, 0, 0, 0, 0, 0, 2, 2, 0], 0, 0, 0), 15);
    }

    #[test]
    fn rebuilding_the_top_finds_the_best_finish() {
        // The best play of two each of 0-2, cut back to its first three
        // tiles: the best way to add the other three is the best play's.
        let left = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0];
        let (best, line, _) = best_free(left, None, 4);
        let start: Line = line[..3].to_vec();
        let mut rest = left;
        for &(v, _) in &start {
            rest[v as usize] -= 1;
        }
        let (score, again) = best_from(&start, rest, -1, usize::MAX, 0).unwrap();
        assert!(score <= best);
        assert_eq!(again[..3], start[..]);
        // Asked to beat the best, it can't.
        assert!(best_from(&start, rest, best, usize::MAX, 0).is_none());
    }
}
