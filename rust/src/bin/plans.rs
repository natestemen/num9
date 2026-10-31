//! Plan first, then build: look for plans (which tiles go on which level)
//! that would beat a score, near the plans of the best stacks found, and try
//! to build each one.
//!
//!     cargo run --release --bin plans -- [--beat SCORE] [--edits N]
//!         [--budget NODES] [--out DIR] RECORD_FILES...
//!
//! Each record's plan is edited up to N times (by default 2): two tiles on
//! different levels swapped, or one tile moved to another level (a new one
//! on top included). Plans scoring more than SCORE (by default the best
//! record) that pass quick checks every stack obeys (one tile on top, every
//! level in use, at least two tiles under each level, area shrinking going
//! up) are tried, the lowest score first (those are likeliest to be
//! buildable), by `num9_core::tower::realize` within a budget of steps each.
//! Stacks found are checked move by move and saved to DIR.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use num9_core::nrpa::{verify, Order, Play};
use num9_core::piece::shape;
use num9_core::tower::realize;

/// Tiles per level, level 0 (the ground) first.
type Plan = Vec<[u8; 10]>;

fn usage() -> ! {
    eprintln!(
        "usage: plans [--beat SCORE] [--edits N] [--budget NODES] [--out DIR] RECORD_FILES..."
    );
    std::process::exit(2)
}

fn plan_of(text: &str) -> Plan {
    let mut plan: Plan = Vec::new();
    for t in text.split_whitespace() {
        if !t.contains('@') || !t.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let (v, rest) = t.split_once('@').unwrap();
        let (Ok(v), Some(Ok(level))) = (
            v.parse::<usize>(),
            rest.split(':').next().map(str::parse::<usize>),
        ) else {
            continue;
        };
        if plan.len() <= level {
            plan.resize(level + 1, [0; 10]);
        }
        plan[level][v] += 1;
    }
    plan
}

fn score(plan: &Plan) -> i64 {
    plan.iter()
        .enumerate()
        .map(|(l, c)| l as i64 * (0..10).map(|v| v as i64 * c[v] as i64).sum::<i64>())
        .sum()
}

fn area(c: &[u8; 10]) -> u32 {
    (0..10).map(|v| c[v] as u32 * shape(v as u8, 0).tiles).sum()
}

fn count(c: &[u8; 10]) -> u8 {
    c.iter().sum()
}

/// Checks every finished stack passes.
fn plausible(plan: &Plan) -> bool {
    let top = plan.len() - 1;
    top >= 1
        && count(&plan[top]) == 1
        && (1..=top).all(|l| count(&plan[l]) >= 1)
        && (1..top).all(|l| count(&plan[l]) >= 2)
        && (2..=top).all(|l| area(&plan[l]) <= area(&plan[l - 1]))
}

/// Trailing empty levels off.
fn trim(mut plan: Plan) -> Plan {
    while plan.len() > 1 && count(plan.last().unwrap()) == 0 {
        plan.pop();
    }
    plan
}

/// Every plan one edit away: a swap of two tiles between levels, or a tile
/// moved to another level (or to a new level on top).
fn neighbours(plan: &Plan) -> Vec<Plan> {
    let mut out = Vec::new();
    let levels = plan.len();
    for a in 0..levels {
        for v in 0..10 {
            if plan[a][v] == 0 {
                continue;
            }
            for b in 0..=levels {
                if b == a {
                    continue;
                }
                // Move v from level a to level b.
                let mut p = plan.clone();
                if b == levels {
                    p.push([0; 10]);
                }
                p[a][v] -= 1;
                p[b][v] += 1;
                out.push(trim(p));
                // Swap v (on a) with some w on b.
                if b < levels {
                    for w in 0..10 {
                        if w != v && plan[b][w] > 0 {
                            let mut p = plan.clone();
                            p[a][v] -= 1;
                            p[b][v] += 1;
                            p[b][w] -= 1;
                            p[a][w] += 1;
                            out.push(trim(p));
                        }
                    }
                }
            }
        }
    }
    out
}

fn main() {
    let mut beat: Option<i64> = None;
    let mut edits = 2;
    let mut budget = 5_000_000usize;
    let mut dir = String::from("runs/plans");
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--beat" => beat = Some(value().parse().unwrap_or_else(|_| usage())),
            "--edits" => edits = value().parse().unwrap_or_else(|_| usage()),
            "--budget" => budget = value().parse().unwrap_or_else(|_| usage()),
            "--out" => dir = value(),
            _ if a.starts_with("--") => usage(),
            _ => files.push(a),
        }
    }
    std::fs::create_dir_all(&dir).expect("make the output directory");
    let seeds: Vec<Plan> = files
        .iter()
        .map(|f| plan_of(&std::fs::read_to_string(f).expect("read a record")))
        .collect();
    let beat = beat.unwrap_or_else(|| seeds.iter().map(score).max().unwrap_or(0));
    // Plans within `edits` edits of a seed.
    let mut seen: HashSet<Plan> = seeds.iter().cloned().collect();
    let mut frontier: Vec<Plan> = seeds.clone();
    for _ in 0..edits {
        let mut next = Vec::new();
        for p in &frontier {
            for q in neighbours(p) {
                if seen.insert(q.clone()) {
                    next.push(q);
                }
            }
        }
        frontier = next;
    }
    let mut candidates: Vec<Plan> = seen
        .into_iter()
        .filter(|p| score(p) > beat && plausible(p))
        .collect();
    candidates.sort_by_key(|p| (score(p), p.len()));
    println!(
        "{} seed plans, {} plans within {edits} edits scoring above {beat} that pass the checks; budget {budget} steps each",
        seeds.len(),
        candidates.len()
    );
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let found = Mutex::new(0usize);
    let (impossible, out_of_budget) = (AtomicUsize::new(0), AtomicUsize::new(0));
    thread::scope(|s| {
        for _ in 0..thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| {
                while let Some(plan) = candidates.get(next.fetch_add(1, Relaxed)) {
                    let (got, steps) = match realize(plan, [2; 10], budget) {
                        Ok(found) => found,
                        Err(ran_out) => {
                            if ran_out { &out_of_budget } else { &impossible }.fetch_add(1, Relaxed);
                            continue;
                        }
                    };
                    let play = Play { score: got, steps };
                    if verify(&Order::standard_free(), &play) != Ok(got) || got <= beat {
                        continue;
                    }
                    let moves: Vec<String> = play
                        .steps
                        .iter()
                        .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
                        .collect();
                    let k = next.load(Relaxed);
                    std::fs::write(
                        format!("{dir}/{got}-plan-{k}.txt"),
                        format!("score {got}\nfound by: plans (built to plan {plan:?})\n\nmoves (value@layer:row:col:rotation) {}\n", moves.join(" ")),
                    )
                    .expect("write a play");
                    *found.lock().unwrap() += 1;
                    println!("[{:6.0}s] built a {got} (plan scoring {})", start.elapsed().as_secs_f64(), score(plan));
                }
            });
        }
    });
    println!(
        "tried {} plans in {:.0}s: built {} beating {beat}; {} can't be built (searched through); {} ran out of budget",
        candidates.len(),
        start.elapsed().as_secs_f64(),
        found.into_inner().unwrap(),
        impossible.load(Relaxed),
        out_of_budget.load(Relaxed)
    );
}
