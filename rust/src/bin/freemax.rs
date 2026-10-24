//! The highest score a small deck allows with its tiles in any order, proven
//! by exhaustive search (see `num9_core::freeorder`), and a play reaching it:
//!
//!     cargo run --release --bin freemax -- <counts, e.g. 1,1,1,1,1,1>
//!
//! The counts are how many tiles of each value from 0 up. The play is
//! printed in the records' value@level:row:col:rotation form.

use std::time::Instant;

use num9_core::freeorder::best_free;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: freemax <counts, e.g. 2,2,1>");
        std::process::exit(2)
    });
    let mut left = [0u8; 10];
    for (v, c) in arg.split(',').enumerate().take(10) {
        left[v] = c.trim().parse().expect("counts are numbers");
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let start = Instant::now();
    let (best, line, nodes) = best_free(left, None, threads);
    let moves: Vec<String> = line
        .iter()
        .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
        .collect();
    println!("counts {arg}: best {best}");
    println!("positions searched: {nodes}");
    println!("time: {:.1}s", start.elapsed().as_secs_f64());
    println!("best line: {}", moves.join(" "));
}
