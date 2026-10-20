//! Native benchmark: `cargo run --release --bin bench -- [games] [seed] [threads]`.
//!
//! Prints a markdown table of per-strategy timings (single-threaded and, when
//! `threads > 1`, parallel throughput) and a move-generation micro-benchmark.

use std::hint::black_box;
use std::time::Instant;

use num9_core::game::{play_games, shuffled_tiles};
use num9_core::{Board, Rng, Strategy};

fn arg(n: usize, default: u64) -> u64 {
    std::env::args()
        .nth(n)
        .map(|a| a.parse().expect("numeric argument"))
        .unwrap_or(default)
}

fn main() {
    let games = arg(1, 1000) as usize;
    let seed = arg(2, 1);
    let threads = arg(
        3,
        std::thread::available_parallelism().map_or(1, |n| n.get()) as u64,
    ) as usize;

    println!("## Full games ({games} games per strategy, seed {seed})\n");
    println!("| strategy | games | avg | median | min | max | time 1 thread (s) | games/s 1 thread | games/s {threads} threads |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for name in Strategy::NAMES {
        let strategy = Strategy::from_name(name).unwrap();
        // The lookahead strategy is far more expensive; keep its run short.
        let n = if name == "maximize_lookahead" {
            games.min(100)
        } else {
            games
        };

        let start = Instant::now();
        let mut scores = play_games(&strategy, n, seed, 1);
        let single = start.elapsed().as_secs_f64();

        let start = Instant::now();
        black_box(play_games(&strategy, n, seed, threads));
        let multi = start.elapsed().as_secs_f64();

        scores.sort_unstable();
        let avg = scores.iter().sum::<i64>() as f64 / n as f64;
        let median = if n % 2 == 1 {
            scores[n / 2] as f64
        } else {
            (scores[n / 2 - 1] + scores[n / 2]) as f64 / 2.0
        };
        println!(
            "| {name} | {n} | {avg:.2} | {median:.1} | {} | {} | {single:.3} | {:.1} | {:.1} |",
            scores[0],
            scores[n - 1],
            n as f64 / single,
            n as f64 / multi,
        );
    }

    // Move generation on positions from real games (all ten piece values each).
    let mut rng = Rng::new(seed);
    let mut positions: Vec<Board> = Vec::new();
    for _ in 0..50 {
        let tiles = shuffled_tiles(&mut rng);
        let mut board = Board::new();
        for &v in &tiles {
            positions.push(board.clone());
            let mv = Strategy::EdgesThenUp.choose(&board, v, &mut rng);
            board.place(v, mv).unwrap();
        }
    }
    let reps = 5;
    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..reps {
        for b in &positions {
            for v in 0..10u8 {
                total += black_box(b.find_valid_moves(v)).len();
            }
        }
    }
    let calls = reps * positions.len() * 10;
    let elapsed = start.elapsed().as_secs_f64();
    println!("\n## Move generation\n");
    println!(
        "{calls} `find_valid_moves` calls on {} positions: {:.2} µs/call ({} moves found)",
        positions.len(),
        elapsed / calls as f64 * 1e6,
        total
    );
}
