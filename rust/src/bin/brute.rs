//! Solve small versions of the game exactly by visiting every reachable
//! position: the deck is two of each value below `n`, dealt in random order.
//!
//!     cargo run --release --bin brute -- <n> [max positions]
//!
//! For each position (the tiles placed so far, up to moving and turning the
//! whole stack) it computes the best expected score when each next tile is
//! drawn at random from those left (the game solved), the best score if the
//! order could be chosen freely, and how many distinct games continue from
//! it. Positions are identified by a 128-bit hash of their tiles' cells.

use std::collections::HashMap;
use std::time::Instant;

use num9_core::board::{Board, Move};
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
    let turns: [fn((i64, i64)) -> (i64, i64); 4] = [
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

struct Solver {
    memo: HashMap<u128, Node>,
    /// Positions found, by tiles placed.
    found: Vec<usize>,
    limit: usize,
    /// Off only to test that merging positions changes nothing.
    merge: bool,
    start: Instant,
}

impl Solver {
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

    fn solve(&mut self, board: &mut Board, left: &mut [u8; 10], rem: u32) -> Node {
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
            let symmetric = shape(v, 0).rows == shape(v, 2).rows;
            let games: usize = counts.iter().sum();
            return Node {
                expected: s as f64,
                best: s,
                games: if symmetric { games / 2 } else { games } as u128,
            };
        }
        let k = key(board);
        if let Some(&node) = self.memo.get(&k).filter(|_| self.merge) {
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
            for m in Self::moves(board, v) {
                board.place(v, m).expect("legal move");
                left[v as usize] -= 1;
                let child = self.solve(board, left, rem - 1);
                left[v as usize] += 1;
                board.undo();
                top = top.max(child.expected);
                node.best = node.best.max(child.best);
                node.games += child.games;
            }
            node.expected += copies as f64 / rem as f64 * top;
        }
        self.memo.insert(k, node);
        let placed = board.pieces().len();
        self.found[placed] += 1;
        if self.memo.len() % 1_000_000 == 0 {
            eprintln!(
                "[{:.0}s] {} positions {:?}",
                self.start.elapsed().as_secs_f64(),
                self.memo.len(),
                self.found
            );
        }
        if self.memo.len() >= self.limit {
            eprintln!("stopping: more than {} positions", self.limit);
            std::process::exit(2);
        }
        node
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(2);
    let limit = args
        .get(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100_000_000);
    let mut left = [0u8; 10];
    left[..n].fill(2);
    let tiles = 2 * n as u32;
    let mut solver = Solver {
        memo: HashMap::new(),
        found: vec![0; tiles as usize],
        limit,
        merge: true,
        start: Instant::now(),
    };
    let root = solver.solve(&mut Board::new(), &mut left, tiles);
    println!("values 0..{}, two of each ({tiles} tiles)", n - 1);
    println!("positions (by tiles placed): {:?}", solver.found);
    println!("positions in all: {}", solver.memo.len());
    println!("distinct games: {}", root.games);
    println!("best expected score (random order): {:.4}", root.expected);
    println!("best score (order chosen freely): {}", root.best);
    println!("time: {:.1}s", solver.start.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(left: [u8; 10], merge: bool) -> (Node, usize) {
        let mut solver = Solver {
            memo: HashMap::new(),
            found: vec![0; 20],
            limit: usize::MAX,
            merge,
            start: Instant::now(),
        };
        let rem = left.iter().map(|&c| c as u32).sum();
        let node = solver.solve(&mut Board::new(), &mut left.clone(), rem);
        (node, solver.found.iter().sum())
    }

    #[test]
    fn merging_symmetric_positions_changes_nothing() {
        for left in [
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        ] {
            let (merged, positions) = solve(left, true);
            let (plain, visits) = solve(left, false);
            assert!(positions < visits, "{positions} positions, {visits} visits");
            assert_eq!(merged.games, plain.games);
            assert_eq!(merged.best, plain.best);
            assert!((merged.expected - plain.expected).abs() < 1e-9);
        }
    }

    #[test]
    fn key_ignores_moving_and_turning_the_stack() {
        let mut a = Board::new();
        a.place(3, Board::center_move()).unwrap();
        let mut keys = Vec::new();
        for m in Solver::moves(&a, 5) {
            a.place(5, m).unwrap();
            keys.push(key(&a));
            a.undo();
        }
        let mut b = Board::new();
        b.place(5, Board::center_move()).unwrap();
        for m in Solver::moves(&b, 3).into_iter().filter(|m| m.rot == 1) {
            b.place(3, m).unwrap();
            assert!(keys.contains(&key(&b)), "{m:?}");
            b.undo();
        }
    }
}
