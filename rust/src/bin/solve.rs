//! Find a high-scoring play of a known deal:
//!
//!     cargo run --release --bin solve -- [--seed N | --tiles 8,3,7,...]
//!         [--level 2] [--per-layer 4] [--restarts 1] [--threads N]
//!
//! Prints progress, the best score, its points per level, and the moves.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use num9_core::game::shuffled_tiles;
use num9_core::solver::{verify, Nested, Solution};
use num9_core::{Rng, Strategy};

fn usage() -> ! {
    eprintln!("usage: solve [--seed N | --tiles a,b,...] [--level L] [--per-layer K] [--restarts R] [--threads T]");
    std::process::exit(2)
}

fn main() {
    let mut tiles: Option<Vec<u8>> = None;
    let mut seed = 0u64;
    let mut nested = Nested::default();
    let mut restarts = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| usage());
        let num = || value.parse::<usize>().unwrap_or_else(|_| usage());
        match flag.as_str() {
            "--seed" => seed = num() as u64,
            "--tiles" => {
                tiles = Some(
                    value
                        .split(',')
                        .map(|t| t.trim().parse().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            "--level" => nested.level = num(),
            "--per-layer" => nested.per_layer = num(),
            "--restarts" => restarts = num().max(1),
            "--threads" => nested.threads = num().max(1),
            _ => usage(),
        }
    }
    let tiles = tiles.unwrap_or_else(|| shuffled_tiles(&mut Rng::new(seed)));
    let mut sorted = tiles.clone();
    sorted.sort_unstable();
    if sorted != (0..10).flat_map(|v| [v, v]).collect::<Vec<u8>>() {
        eprintln!("--tiles must list two of each value 0-9");
        std::process::exit(2);
    }
    let list = |t: &[u8]| {
        t.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    println!("deal: {}", list(&tiles));

    // For reference: the order-unaware strategies on the same deal.
    for name in ["low_to_base", "monte_carlo"] {
        let s = Strategy::from_name(name).unwrap();
        let score = num9_core::game::play_tiles(&s, &tiles, &mut Rng::new(seed)).score();
        println!("{name:>12}: {score}");
    }

    let stop = AtomicBool::new(false);
    let start = Instant::now();
    let mut best = Solution {
        score: i64::MIN,
        moves: Vec::new(),
    };
    for r in 0..restarts {
        let sol = nested.solve(&tiles, r as u64, &stop, |step, b| {
            eprint!(
                "\r  restart {}/{restarts}: tile {:2}/{}  best {:3}  ({:.1}s)   ",
                r + 1,
                step + 1,
                tiles.len(),
                b.score.max(best.score),
                start.elapsed().as_secs_f64()
            );
        });
        if sol.score > best.score {
            best = sol;
        }
    }
    eprintln!();
    assert_eq!(verify(&tiles, &best.moves), Ok(best.score));
    println!(
        "nested level {} (per_layer {}, {} restart{}): {}  in {:.1}s",
        nested.level,
        nested.per_layer,
        restarts,
        if restarts == 1 { "" } else { "s" },
        best.score,
        start.elapsed().as_secs_f64()
    );
    let mut per_level = [(0usize, 0i64); 6];
    for (&v, m) in tiles.iter().zip(&best.moves) {
        per_level[m.layer].0 += 1;
        per_level[m.layer].1 += v as i64 * m.layer as i64;
    }
    for (l, (n, pts)) in per_level.iter().enumerate().filter(|(_, (n, _))| *n > 0) {
        println!("  level {l}: {n:2} tiles, {pts:3} points");
    }
    let moves: Vec<String> = best
        .moves
        .iter()
        .map(|m| format!("{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
        .collect();
    println!("moves (layer:row:col:rotation): {}", moves.join(" "));
}
