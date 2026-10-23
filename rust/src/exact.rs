//! Solving small decks exactly, by visiting every reachable position.
//!
//! A position is the tiles placed so far; positions that differ only by
//! moving or turning the whole stack are the same (see [`key`]). For each one
//! [`solve`] computes the best expected final score when every next tile is
//! drawn at random from those left (the game solved), the best score if the
//! order could be chosen freely, and, when counting, how many distinct games
//! continue from it. Only decks of up to six tiles or so are feasible: the
//! number of positions grows a hundredfold or more with every tile.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;

use crate::board::{Board, Move, DEPTH};
use crate::piece::shape;

/// What a position is worth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    /// Best expected final score with the rest of the deck drawn at random.
    pub expected: f64,
    /// Best final score with the rest placed in any order.
    pub best: i64,
    /// Distinct games (sequences of tile and placement) from here, when
    /// counting (otherwise 0).
    pub games: u128,
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

type Turn = fn((i64, i64)) -> (i64, i64);

/// A 128-bit hash of the tiles' cells, the same for every translation and
/// quarter turn of the stack.
pub fn key(board: &Board) -> u128 {
    let tiles: Vec<(u64, Vec<(i64, i64)>)> = board
        .pieces()
        .iter()
        .map(|p| {
            let cells = shape(p.value, p.mv.rot)
                .cells()
                .map(|(r, c)| ((p.mv.i + r) as i64, (p.mv.j + c) as i64))
                .collect();
            ((p.mv.layer as u64) << 8 | p.value as u64, cells)
        })
        .collect();
    let turns: [Turn; 4] = [
        |(r, c)| (r, c),
        |(r, c)| (c, -r),
        |(r, c)| (-r, -c),
        |(r, c)| (-c, r),
    ];
    turns
        .iter()
        .map(|turn| {
            let (mut r0, mut c0) = (i64::MAX, i64::MAX);
            for (_, cells) in &tiles {
                for &cell in cells {
                    let (r, c) = turn(cell);
                    r0 = r0.min(r);
                    c0 = c0.min(c);
                }
            }
            // A set of tiles hashes to the sum of its tiles' hashes, each
            // the hash of its kind and the sum of its cells' hashes.
            let (mut h1, mut h2) = (0u64, 0u64);
            for (kind, cells) in &tiles {
                let mut s = 0u64;
                for &cell in cells {
                    let (r, c) = turn(cell);
                    s = s.wrapping_add(mix(((r - r0) as u64) << 32 | (c - c0) as u64));
                }
                h1 = h1.wrapping_add(mix(s ^ mix(*kind)));
                h2 = h2.wrapping_add(mix(s.rotate_left(17) ^ mix(*kind ^ 0x5bd1e995)));
            }
            (h1 as u128) << 64 | h2 as u128
        })
        .min()
        .unwrap()
}

/// Keys are already hashes: use their high half as is.
#[derive(Default)]
struct Prehashed(u64);

impl Hasher for Prehashed {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u128 keys")
    }
    fn write_u128(&mut self, k: u128) {
        self.0 = (k >> 64) as u64;
    }
}

type Table<V> = HashMap<u128, V, BuildHasherDefault<Prehashed>>;

/// A stored position: expected and best scores, and tiles placed.
type Stored = (f64, i16, u8);

/// Solved positions, safe to share between threads (in shards, picked by
/// the low half of the key, to keep locks short). Kept small: games are
/// stored apart, and only when counting.
pub struct Memo {
    shards: Vec<Mutex<Table<Stored>>>,
    games: Vec<Mutex<Table<u128>>>,
    /// Positions solved, by tiles placed.
    found: Vec<AtomicUsize>,
    total: AtomicUsize,
    /// Off only to test that merging positions changes nothing.
    pub merge: bool,
    /// Count every game (slow: no pruning).
    pub count: bool,
    /// Report progress on stderr every million positions.
    pub verbose: bool,
}

/// Bytes per entry in [`Memo::export`]'s format.
pub const ENTRY_BYTES: usize = 27;

impl Memo {
    pub fn new() -> Memo {
        Memo {
            shards: (0..256).map(|_| Mutex::default()).collect(),
            games: (0..256).map(|_| Mutex::default()).collect(),
            found: (0..=DEPTH * 4).map(|_| AtomicUsize::new(0)).collect(),
            total: AtomicUsize::new(0),
            merge: true,
            count: false,
            verbose: false,
        }
    }

    fn shard(k: u128) -> usize {
        k as usize % 256
    }

    pub fn get(&self, k: u128) -> Option<Node> {
        if !self.merge {
            return None;
        }
        let s = Self::shard(k);
        let (expected, best, _) = *self.shards[s].lock().unwrap().get(&k)?;
        let games = if self.count {
            self.games[s].lock().unwrap()[&k]
        } else {
            0
        };
        Some(Node {
            expected,
            best: best as i64,
            games,
        })
    }

    fn insert(&self, k: u128, placed: usize, node: Node) {
        if self.merge {
            let s = Self::shard(k);
            if self.count {
                self.games[s].lock().unwrap().insert(k, node.games);
            }
            let value = (node.expected, node.best as i16, placed as u8);
            if self.shards[s].lock().unwrap().insert(k, value).is_some() {
                return; // another thread got here first
            }
        }
        self.found[placed].fetch_add(1, Relaxed);
        let total = self.total.fetch_add(1, Relaxed) + 1;
        if self.verbose && total.is_multiple_of(1_000_000) {
            eprintln!("{total} positions {:?}", self.found());
        }
    }

    /// Positions solved, by tiles placed (up to the most placed).
    pub fn found(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.found.iter().map(|f| f.load(Relaxed)).collect();
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }

    pub fn len(&self) -> usize {
        self.total.load(Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The positions with at most `placed` tiles, as bytes: per position the
    /// key, expected score (f64) and best score (i16), little-endian, then
    /// the tiles placed (u8). Sorted, so the output is reproducible.
    pub fn export(&self, placed: usize) -> Vec<u8> {
        let mut rows: Vec<(u128, Stored)> = Vec::new();
        for shard in &self.shards {
            let shard = shard.lock().unwrap();
            rows.extend(
                shard
                    .iter()
                    .filter(|(_, v)| v.2 as usize <= placed)
                    .map(|(&k, &v)| (k, v)),
            );
        }
        rows.sort_by_key(|r| r.0);
        let mut out = Vec::with_capacity(rows.len() * ENTRY_BYTES);
        for (k, (expected, best, placed)) in rows {
            out.extend(k.to_le_bytes());
            out.extend(expected.to_le_bytes());
            out.extend(best.to_le_bytes());
            out.push(placed);
        }
        out
    }

    /// Add positions solved earlier, in [`Memo::export`]'s format.
    pub fn import(&self, bytes: &[u8]) {
        for row in bytes.chunks_exact(ENTRY_BYTES) {
            let k = u128::from_le_bytes(row[..16].try_into().unwrap());
            let expected = f64::from_le_bytes(row[16..24].try_into().unwrap());
            let best = i16::from_le_bytes(row[24..26].try_into().unwrap());
            let node = Node {
                expected,
                best: best as i64,
                games: 0,
            };
            self.insert(k, row[26] as usize, node);
        }
    }
}

impl Default for Memo {
    fn default() -> Self {
        Memo::new()
    }
}

/// The distinct placements of `value` (pieces 0 and 8 look the same after a
/// half turn; only one of each pair is kept). The first tile goes in the
/// middle.
pub fn moves(board: &Board, value: u8) -> Vec<Move> {
    if board.pieces().is_empty() {
        return vec![Board::center_move()];
    }
    let mut moves = board.find_valid_moves(value);
    if shape(value, 0).rows == shape(value, 2).rows {
        moves.retain(|m| m.rot < 2);
    }
    moves
}

/// The tiles left in a deck of `left[v]` copies of each value.
pub fn remaining(left: &[u8; 10]) -> u32 {
    left.iter().map(|&c| c as u32).sum()
}

/// Solve the position on `board` with `left[v]` copies of each value still
/// to be drawn.
pub fn solve(memo: &Memo, board: &mut Board, left: &mut [u8; 10]) -> Node {
    let rem = remaining(left);
    if rem == 0 {
        let s = board.score();
        return Node {
            expected: s as f64,
            best: s,
            games: 1,
        };
    }
    if rem == 1 {
        // The last tile goes as high as it can; no need to place it.
        let v = left.iter().position(|&c| c > 0).unwrap() as u8;
        let counts = board.count_valid_moves(v);
        let top = (0..counts.len()).rev().find(|&l| counts[l] > 0).unwrap();
        let s = board.score() + v as i64 * top as i64;
        let symmetric = shape(v, 0).rows == shape(v, 2).rows && !board.pieces().is_empty();
        let games: usize = counts.iter().sum();
        return Node {
            expected: s as f64,
            best: s,
            games: if symmetric { games / 2 } else { games } as u128,
        };
    }
    let k = key(board);
    if let Some(node) = memo.get(k) {
        return node;
    }
    if rem == 2 && !memo.count {
        let node = last_two(board, left);
        memo.insert(k, board.pieces().len(), node);
        return node;
    }
    let mut node = Node {
        expected: 0.0,
        best: i64::MIN,
        games: 0,
    };
    for v in 0..10u8 {
        let copies = left[v as usize];
        if copies == 0 {
            continue;
        }
        let mut top = f64::MIN;
        for m in moves(board, v) {
            board.place(v, m).expect("legal move");
            left[v as usize] -= 1;
            let child = solve(memo, board, left);
            left[v as usize] += 1;
            board.undo();
            top = top.max(child.expected);
            node.best = node.best.max(child.best);
            node.games += child.games;
        }
        node.expected += copies as f64 / rem as f64 * top;
    }
    memo.insert(k, board.pieces().len(), node);
    node
}

/// Each distinct placement of `value` (just drawn; `left` holds what comes
/// after it) and what the position after it is worth.
pub fn placements(
    memo: &Memo,
    board: &mut Board,
    value: u8,
    left: &mut [u8; 10],
) -> Vec<(Move, Node)> {
    moves(board, value)
        .into_iter()
        .map(|m| {
            board.place(value, m).expect("legal move");
            let node = solve(memo, board, left);
            board.undo();
            (m, node)
        })
        .collect()
}

/// The highest layer `value` can go on.
fn reach(board: &Board, value: u8) -> i64 {
    let counts = board.count_valid_moves(value);
    (0..counts.len()).rev().find(|&l| counts[l] > 0).unwrap() as i64
}

/// The two tiles left, in either order, with the second as high as it can
/// go. Placements of the first are tried in order of an optimistic score
/// (the second lands at most one layer above the stack), stopping once
/// that cannot beat the best found.
fn last_two(board: &mut Board, left: &[u8; 10]) -> Node {
    let tiles: Vec<u8> = (0..10u8)
        .flat_map(|v| std::iter::repeat_n(v, left[v as usize] as usize))
        .collect();
    let (a, b) = (tiles[0], tiles[1]);
    let orders = if a == b {
        vec![(a, b)]
    } else {
        vec![(a, b), (b, a)]
    };
    let top = (0..DEPTH)
        .rev()
        .find(|&l| !board.layer_is_empty(l))
        .map_or(-1, |l| l as i64);
    let mut node = Node {
        expected: 0.0,
        best: i64::MIN,
        games: 0,
    };
    for &(first, second) in &orders {
        let base = board.score();
        let mut options: Vec<(i64, Move)> = moves(board, first)
            .into_iter()
            .map(|m| {
                let l = m.layer as i64;
                let above = (top.max(l) + 1).min(DEPTH as i64 - 1);
                (base + first as i64 * l + second as i64 * above, m)
            })
            .collect();
        options.sort_by_key(|&(bound, _)| std::cmp::Reverse(bound));
        let mut got = i64::MIN;
        for (bound, m) in options {
            if bound <= got {
                break;
            }
            board.place(first, m).expect("legal move");
            got = got.max(board.score() + second as i64 * reach(board, second));
            board.undo();
        }
        node.expected += got as f64 / orders.len() as f64;
        node.best = node.best.max(got);
    }
    node
}

/// Solve a whole deck from the empty board: first the positions after
/// `split` tiles, spread over `threads`, then the rest (which finds those in
/// the memo).
pub fn solve_deck(memo: &Memo, left: [u8; 10], split: usize, threads: usize) -> Node {
    type Job = (Vec<(u8, Move)>, [u8; 10]);
    fn expand(board: &mut Board, left: &mut [u8; 10], depth: usize, jobs: &mut HashMap<u128, Job>) {
        if depth == 0 {
            let steps = board.pieces().iter().map(|p| (p.value, p.mv)).collect();
            jobs.entry(key(board)).or_insert((steps, *left));
            return;
        }
        for v in 0..10u8 {
            if left[v as usize] == 0 {
                continue;
            }
            for m in moves(board, v) {
                board.place(v, m).expect("legal move");
                left[v as usize] -= 1;
                expand(board, left, depth - 1, jobs);
                left[v as usize] += 1;
                board.undo();
            }
        }
    }
    if threads > 1 {
        let mut jobs = HashMap::new();
        expand(&mut Board::new(), &mut left.clone(), split, &mut jobs);
        let jobs: Vec<Job> = jobs.into_values().collect();
        let next = AtomicUsize::new(0);
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    while let Some((steps, left)) = jobs.get(next.fetch_add(1, Relaxed)) {
                        let mut board = Board::new();
                        for &(v, m) in steps {
                            board.place(v, m).expect("legal move");
                        }
                        solve(memo, &mut board, &mut left.clone());
                    }
                });
            }
        });
    }
    solve(memo, &mut Board::new(), &mut left.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(left: [u8; 10], merge: bool, count: bool) -> (Node, usize) {
        let memo = Memo {
            merge,
            count,
            ..Memo::new()
        };
        let node = solve_deck(&memo, left, 2, 4);
        (node, memo.found().iter().sum())
    }

    #[test]
    fn merging_symmetric_positions_changes_nothing() {
        for left in [
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        ] {
            let (merged, positions) = run(left, true, true);
            let (plain, visits) = run(left, false, true);
            assert!(positions < visits, "{positions} positions, {visits} visits");
            assert_eq!(merged.games, plain.games);
            assert_eq!(merged.best, plain.best);
            assert!((merged.expected - plain.expected).abs() < 1e-9);
        }
    }

    #[test]
    fn pruning_the_last_two_tiles_changes_nothing() {
        for left in [
            [2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
        ] {
            let (full, _) = run(left, true, true);
            let (pruned, _) = run(left, true, false);
            assert_eq!(full.best, pruned.best, "{left:?}");
            assert!((full.expected - pruned.expected).abs() < 1e-9, "{left:?}");
        }
    }

    #[test]
    fn known_values() {
        let (node, _) = run([2, 2, 0, 0, 0, 0, 0, 0, 0, 0], true, true);
        assert_eq!(node.games, 3_100_765);
        assert_eq!(node.best, 2);
        assert!((node.expected - 1.0).abs() < 1e-9);
        let (node, _) = run([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], true, false);
        assert!((node.expected - 7.0 / 3.0).abs() < 1e-9);
        assert_eq!(node.best, 3);
    }

    #[test]
    fn threads_and_exported_tables_change_nothing() {
        let left = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let memo = Memo::new();
        let one = solve_deck(&memo, left, 2, 1);
        let table = memo.export(2);
        assert_eq!(table.len() % ENTRY_BYTES, 0);
        assert!(!table.is_empty());
        // A fresh memo seeded with the table agrees, everywhere.
        let seeded = Memo::new();
        seeded.import(&table);
        assert_eq!(seeded.len(), table.len() / ENTRY_BYTES);
        assert_eq!(solve_deck(&seeded, left, 2, 1), one);
        let mut board = Board::new();
        board.place(2, Board::center_move()).unwrap();
        let mut rest = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let a = placements(&memo, &mut board, 3, &mut rest.clone());
        let b = placements(&seeded, &mut board, 3, &mut rest);
        assert_eq!(a, b);
        assert_eq!(solve_deck(&Memo::new(), left, 2, 4), one);
    }

    #[test]
    fn key_ignores_moving_and_turning_the_stack() {
        let mut a = Board::new();
        a.place(3, Board::center_move()).unwrap();
        let mut keys = Vec::new();
        for m in moves(&a, 5) {
            a.place(5, m).unwrap();
            keys.push(key(&a));
            a.undo();
        }
        let mut b = Board::new();
        b.place(5, Board::center_move()).unwrap();
        for m in moves(&b, 3).into_iter().filter(|m| m.rot == 1) {
            b.place(3, m).unwrap();
            assert!(keys.contains(&key(&b)), "{m:?}");
            b.undo();
        }
    }
}
