//! Search for the highest possible score (free tile order) by designing
//! towers top-down with a beam search (see `num9_core::tower`):
//!
//!     cargo run --release --bin tower -- [--tops 4,5,6] [--width 60]
//!         [--per-design 8] [--nodes 100000] [--area-cost 8] [--growth 4]
//!         [--out tower.txt]
//!
//! Each top height runs on its own thread. Prints every new best as it is
//! found, checks the best play move by move, and writes it to --out.

use std::sync::mpsc::channel;
use std::thread;
use std::time::Instant;

use num9_core::nrpa::{verify, Order, Play};
use num9_core::tower::TowerSearch;

fn usage() -> ! {
    eprintln!(
        "usage: tower [--tops a,b,...] [--width W] [--per-design K] [--nodes N] \
         [--area-cost C] [--growth G] [--out FILE]"
    );
    std::process::exit(2)
}

fn describe(play: &Play) -> String {
    let mut per_level: [(usize, i64, Vec<u8>); num9_core::DEPTH] =
        std::array::from_fn(|_| (0, 0, Vec::new()));
    for &(v, m) in &play.steps {
        per_level[m.layer].0 += 1;
        per_level[m.layer].1 += v as i64 * m.layer as i64;
        per_level[m.layer].2.push(v);
    }
    let order: Vec<String> = play.order().iter().map(|v| v.to_string()).collect();
    let moves: Vec<String> = play
        .steps
        .iter()
        .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
        .collect();
    let mut out = format!("score {}\norder {}\n", play.score, order.join(","));
    for (l, (n, pts, values)) in per_level.iter().enumerate().filter(|(_, (n, ..))| *n > 0) {
        out += &format!("level {l}: {n:2} tiles, {pts:3} points  {values:?}\n");
    }
    out + &format!("moves (value@layer:row:col:rotation) {}\n", moves.join(" "))
}

fn main() {
    let mut tops = vec![4, 5, 6];
    let (mut width, mut per_design) = (60, 8);
    let mut search = TowerSearch {
        area_cost: 8,
        nodes: 100_000,
        growth: 4,
        top: 0,
        tiles: [2; 10],
    };
    let mut out = String::from("tower.txt");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| usage());
        let num = || value.parse::<usize>().unwrap_or_else(|_| usage());
        match flag.as_str() {
            "--tops" => {
                tops = value
                    .split(',')
                    .map(|t| t.parse().unwrap_or_else(|_| usage()))
                    .collect()
            }
            "--width" => width = num(),
            "--per-design" => per_design = num(),
            "--nodes" => search.nodes = num(),
            "--area-cost" => search.area_cost = num() as i64,
            "--growth" => search.growth = num(),
            "--out" => out = value,
            _ => usage(),
        }
    }
    println!(
        "free order, towers topped at levels {tops:?}: beam width {width}, {per_design} coverings per design, \
         {} nodes, area cost {}, growth {}",
        search.nodes, search.area_cost, search.growth
    );
    let start = Instant::now();
    let (tx, rx) = channel::<(usize, Play)>();
    let handles: Vec<_> = tops
        .iter()
        .map(|&top| {
            let tx = tx.clone();
            let s = TowerSearch {
                top,
                ..search.clone()
            };
            thread::spawn(move || {
                s.beam(width, per_design, |f| {
                    let _ = tx.send((
                        top,
                        Play {
                            score: f.score,
                            steps: f.steps.clone(),
                        },
                    ));
                });
            })
        })
        .collect();
    drop(tx);
    let mut best: Option<Play> = None;
    for (top, play) in rx {
        if best.as_ref().is_none_or(|b| play.score > b.score) {
            println!(
                "[{:7.1}s] top level {top}: new best {}",
                start.elapsed().as_secs_f64(),
                play.score
            );
            best = Some(play);
        }
    }
    for h in handles {
        h.join().unwrap();
    }
    let Some(best) = best else {
        println!("no design found");
        return;
    };
    assert_eq!(
        verify(&Order::standard_free(), &best),
        Ok(best.score),
        "best play must be legal"
    );
    let text = describe(&best);
    println!(
        "\nbest after {:.1}s (checked move by move):\n{text}",
        start.elapsed().as_secs_f64()
    );
    std::fs::write(&out, text).expect("write --out");
    println!("saved to {out}");
}
