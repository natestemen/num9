//! Solve small versions of the game exactly by visiting every reachable
//! position (see `num9_core::exact`): the deck is two (or `copies`) of each
//! value below `n`, dealt in random order.
//!
//!     cargo run --release --bin brute -- <n> [copies] [--table FILE] [--line]
//!
//! Prints the best expected score (the game solved), the best score if the
//! order could be chosen freely, and (with `BRUTE_COUNT=1`, which turns off
//! pruning) how many distinct games there are. `--table` saves every
//! position with three or more tiles still to come, so that what remains can
//! be solved quickly later. `--line` prints a play reaching the best score
//! with the order free, in the records' value@level:row:col:rotation form.

use std::time::Instant;

use num9_core::exact::{solve_deck, Memo};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let table = args
        .iter()
        .position(|a| a == "--table")
        .map(|k| args.get(k + 1).expect("--table needs a file").clone());
    let numbers: Vec<usize> = args[1..]
        .iter()
        .take_while(|a| a.parse::<usize>().is_ok())
        .map(|a| a.parse().expect("usage: brute <n> [copies] [--table FILE]"))
        .collect();
    let n = numbers.first().copied().unwrap_or(2);
    let copies = numbers.get(1).copied().unwrap_or(2);
    let mut left = [0u8; 10];
    left[..n].fill(copies as u8);
    let tiles = n * copies;
    let mut memo = Memo::new();
    memo.count = std::env::var_os("BRUTE_COUNT").is_some();
    memo.verbose = true;
    let start = Instant::now();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let root = solve_deck(&memo, left, tiles.saturating_sub(2).min(3), threads);
    println!("values 0..{}, {copies} of each ({tiles} tiles)", n - 1);
    println!("positions (by tiles placed): {:?}", memo.found());
    println!("positions in all: {}", memo.len());
    if memo.count {
        println!("distinct games: {}", root.games);
    }
    println!("best expected score (random order): {:.4}", root.expected);
    println!("best score (order chosen freely): {}", root.best);
    println!("time: {:.1}s", start.elapsed().as_secs_f64());
    if args.iter().any(|a| a == "--line") {
        let line = num9_core::exact::best_line(&memo, left);
        let moves: Vec<String> = line
            .iter()
            .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
            .collect();
        println!("best line: {}", moves.join(" "));
    }
    if let Some(path) = table {
        let bytes = memo.export(tiles.saturating_sub(3));
        std::fs::write(&path, &bytes).expect("can't write the table");
        println!(
            "table: {} positions in {path}",
            bytes.len() / num9_core::exact::ENTRY_BYTES
        );
    }
}
