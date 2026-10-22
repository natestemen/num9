//! Solve small versions of the game exactly by visiting every reachable
//! position: the deck is two (or `copies`) of each value below `n`, dealt
//! in random order.
//!
//!     cargo run --release --bin brute -- <n> [copies] [max positions]
//!
//! For each position (the tiles placed so far, up to moving and turning the
//! whole stack) it computes the best expected score when each next tile is
//! drawn at random from those left (the game solved), the best score if the
//! order could be chosen freely, and (with `BRUTE_COUNT=1`, which turns off
//! pruning) how many distinct games continue from it. Positions are
//! identified by a 128-bit hash of their tiles' cells.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use num9_core::board::{Board, Move, DEPTH};
use num9_core::piece::shape;

#[derive(Clone, Copy)]
struct Node {
    /// Best expected final score with the rest of the deck drawn at random.
    expected: f64,
    /// Best final score with the rest placed in any order.
    best: i64,
    /// Distinct games (sequences of tile and placement) from here.
    games: u128,
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

type Turn = fn((i64, i64)) -> (i64, i64);

/// The same key for every translation and quarter turn of the stack.
fn key(board: &Board) -> u128 {
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

/// Solved positions, shared by every thread (in shards to keep locks short).
struct Memo {
    shards: Vec<Mutex<HashMap<u128, Node>>>,
    /// Positions solved, by tiles placed.
    found: Vec<AtomicUsize>,
    total: AtomicUsize,
    limit: usize,
    /// Off only to test that merging positions changes nothing.
    merge: bool,
    /// Count every game (slow: no pruning); otherwise `games` stays 0.
    count: bool,
    start: Instant,
}

impl Memo {
    fn new(tiles: usize, limit: usize, merge: bool, count: bool) -> Memo {
        Memo {
            shards: (0..256).map(|_| Mutex::new(HashMap::new())).collect(),
            found: (0..tiles).map(|_| AtomicUsize::new(0)).collect(),
            total: AtomicUsize::new(0),
            limit,
            merge,
            count,
            start: Instant::now(),
        }
    }

    fn shard(&self, k: u128) -> &Mutex<HashMap<u128, Node>> {
        &self.shards[k as usize % self.shards.len()]
    }

    fn get(&self, k: u128) -> Option<Node> {
        if !self.merge {
            return None;
        }
        self.shard(k).lock().unwrap().get(&k).copied()
    }

    fn insert(&self, k: u128, placed: usize, node: Node) {
        if self.merge && self.shard(k).lock().unwrap().insert(k, node).is_some() {
            return; // another thread got here first
        }
        self.found[placed].fetch_add(1, Relaxed);
        let total = self.total.fetch_add(1, Relaxed) + 1;
        if total.is_multiple_of(1_000_000) {
            eprintln!(
                "[{:.0}s] {total} positions {:?}",
                self.start.elapsed().as_secs_f64(),
                self.found()
            );
        }
        if total >= self.limit {
            eprintln!("stopping: more than {} positions", self.limit);
            std::process::exit(2);
        }
    }

    fn found(&self) -> Vec<usize> {
        self.found.iter().map(|f| f.load(Relaxed)).collect()
    }
}

fn moves(board: &Board, value: u8) -> Vec<Move> {
    if board.pieces().is_empty() {
        return vec![Board::center_move()];
    }
    let mut moves = board.find_valid_moves(value);
    if shape(value, 0).rows == shape(value, 2).rows {
        moves.retain(|m| m.rot < 2);
    }
    moves
}

fn solve(memo: &Memo, board: &mut Board, left: &mut [u8; 10], rem: u32) -> Node {
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
            let child = solve(memo, board, left, rem - 1);
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

/// Solve the positions after `split` tiles on every core, then the rest
/// (which finds those in the memo).
fn solve_parallel(memo: &Memo, left: [u8; 10], split: usize) -> Node {
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
    let rem: u32 = left.iter().map(|&c| c as u32).sum();
    let mut jobs = HashMap::new();
    expand(&mut Board::new(), &mut left.clone(), split, &mut jobs);
    let jobs: Vec<Job> = jobs.into_values().collect();
    let next = AtomicUsize::new(0);
    thread::scope(|s| {
        for _ in 0..thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| {
                while let Some((steps, left)) = jobs.get(next.fetch_add(1, Relaxed)) {
                    let mut board = Board::new();
                    for &(v, m) in steps {
                        board.place(v, m).expect("legal move");
                    }
                    solve(memo, &mut board, &mut left.clone(), rem - split as u32);
                }
            });
        }
    });
    solve(memo, &mut Board::new(), &mut left.clone(), rem)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |k: usize, default: usize| args.get(k).and_then(|a| a.parse().ok()).unwrap_or(default);
    let (n, copies, limit) = (arg(1, 2), arg(2, 2), arg(3, 200_000_000));
    let count = std::env::var_os("BRUTE_COUNT").is_some();
    let mut left = [0u8; 10];
    left[..n].fill(copies as u8);
    let tiles = n * copies;
    let memo = Memo::new(tiles, limit, true, count);
    let root = solve_parallel(&memo, left, tiles.saturating_sub(2).min(3));
    println!("values 0..{}, {copies} of each ({tiles} tiles)", n - 1);
    println!("positions (by tiles placed): {:?}", memo.found());
    println!("positions in all: {}", memo.total.load(Relaxed));
    if count {
        println!("distinct games: {}", root.games);
    }
    println!("best expected score (random order): {:.4}", root.expected);
    println!("best score (order chosen freely): {}", root.best);
    println!("time: {:.1}s", memo.start.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(left: [u8; 10], merge: bool) -> (Node, usize) {
        let memo = Memo::new(20, usize::MAX, merge, true);
        let node = solve_parallel(&memo, left, 2);
        (node, memo.found().iter().sum())
    }

    #[test]
    fn merging_symmetric_positions_changes_nothing() {
        for left in [
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        ] {
            let (merged, positions) = run(left, true);
            let (plain, visits) = run(left, false);
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
            let (full, _) = run(left, true);
            let pruned = solve_parallel(&Memo::new(20, usize::MAX, true, false), left, 2);
            assert_eq!(full.best, pruned.best, "{left:?}");
            assert!((full.expected - pruned.expected).abs() < 1e-9, "{left:?}");
        }
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
