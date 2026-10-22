//! Search for the highest-scoring play with NRPA:
//!
//!     cargo run --release --bin maxscore -- [--free | --tiles 0,0,1,1,...]
//!         [--level 3] [--iterations 100] [--alpha 1.0] [--threads N]
//!         [--seed S] [--out best.txt]
//!
//! With --free the order of the tiles is chosen too (the "maximum possible
//! score" question). Runs one independent search per thread, prints every
//! new best as it is found, and writes the best play to --out.

use std::sync::mpsc::channel;
use std::thread;
use std::time::Instant;

use num9_core::nrpa::{verify, Nrpa, Order, Play};

fn usage() -> ! {
    eprintln!(
        "usage: maxscore [--free | --tiles a,b,...] [--level L] [--iterations N] [--alpha A] [--edge-bias E] [--height-bias H] \
         [--threads T] [--seed S] [--out FILE]"
    );
    std::process::exit(2)
}

fn describe(play: &Play) -> String {
    let mut per_level = [(0usize, 0i64); num9_core::DEPTH];
    for &(v, m) in &play.steps {
        per_level[m.layer].0 += 1;
        per_level[m.layer].1 += v as i64 * m.layer as i64;
    }
    let order: Vec<String> = play.order().iter().map(|v| v.to_string()).collect();
    let moves: Vec<String> = play
        .steps
        .iter()
        .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
        .collect();
    let mut out = format!("score {}\norder {}\n", play.score, order.join(","));
    for (l, (n, pts)) in per_level.iter().enumerate().filter(|(_, (n, _))| *n > 0) {
        out += &format!("level {l}: {n:2} tiles, {pts:3} points\n");
    }
    out + &format!("moves (value@layer:row:col:rotation) {}\n", moves.join(" "))
}

fn main() {
    let mut order = Order::Fixed((0..10u8).flat_map(|v| [v, v]).collect());
    let mut nrpa = Nrpa::default();
    let mut threads = thread::available_parallelism().map_or(1, |n| n.get());
    let mut seed = 0u64;
    let mut out = String::from("best.txt");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--free" {
            order = Order::standard_free();
            continue;
        }
        let value = args.next().unwrap_or_else(|| usage());
        let num = || value.parse::<usize>().unwrap_or_else(|_| usage());
        match flag.as_str() {
            "--tiles" => {
                order = Order::Fixed(
                    value
                        .split(',')
                        .map(|t| t.trim().parse().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            "--level" => nrpa.level = num(),
            "--iterations" => nrpa.iterations = num(),
            "--alpha" => nrpa.alpha = value.parse().unwrap_or_else(|_| usage()),
            "--edge-bias" => nrpa.edge_bias = value.parse().unwrap_or_else(|_| usage()),
            "--height-bias" => nrpa.height_bias = value.parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = num().max(1),
            "--seed" => seed = num() as u64,
            "--out" => out = value,
            _ => usage(),
        }
    }
    println!(
        "{} order, NRPA level {} × {} iterations (alpha {}), {threads} independent runs",
        if matches!(order, Order::Free(_)) {
            "free"
        } else {
            "fixed"
        },
        nrpa.level,
        nrpa.iterations,
        nrpa.alpha
    );

    let start = Instant::now();
    let (tx, rx) = channel::<(usize, Play)>();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (tx, order, nrpa) = (tx.clone(), order.clone(), nrpa.clone());
            thread::spawn(move || {
                let play = nrpa.solve(&order, seed + t as u64, |p| {
                    let _ = tx.send((t, p.clone()));
                });
                let _ = tx.send((t, play));
            })
        })
        .collect();
    drop(tx);
    let mut best: Option<Play> = None;
    for (t, play) in rx {
        if best.as_ref().is_none_or(|b| play.score > b.score) {
            println!(
                "[{:7.1}s] run {t}: new best {}",
                start.elapsed().as_secs_f64(),
                play.score
            );
            best = Some(play);
        }
    }
    for h in handles {
        h.join().unwrap();
    }
    let best = best.expect("at least one run");
    assert_eq!(
        verify(&order, &best),
        Ok(best.score),
        "best play must be legal"
    );
    let text = describe(&best);
    println!(
        "\nbest after {:.1}s:\n{text}",
        start.elapsed().as_secs_f64()
    );
    std::fs::write(&out, text).expect("write --out");
    println!("saved to {out}");
}
