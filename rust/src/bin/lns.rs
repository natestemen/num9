//! Large neighbourhood search for the highest free-order score:
//!
//!     cargo run --release --bin lns -- [--minutes M] [--budget NODES]
//!         [--lift MIN,MAX] [--sweep TRIES] [--out DIR] RECORD_FILES...
//!
//! Starts from recorded plays (the best stacks found so far) and improves
//! them by destroying and repairing: lift off the top of a stack (every tile
//! above some level, and a few on it: MIN to MAX tiles, 6 to 12 by
//! default), then put them
//! back in the best way there is, by the exhaustive free-order search started
//! from what's left (`num9_core::freeorder::best_from`). A repair can always
//! put the stack back as it was, so stacks never get worse; when the best
//! rebuild of a top scores more, the stack has improved. A pool of the best
//! distinct stacks (equal scores included, so the search can drift along
//! plateaus) supplies each step. Improvements are written to DIR.
//!
//! With --sweep, every stack instead gets a turn: each is improved on its
//! own, step after step, until TRIES steps in a row fail to improve it, and
//! what each became is written to DIR (when better) and summed up at the
//! end.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use num9_core::board::{Board, Move};
use num9_core::exact::key;
use num9_core::freeorder::{best_from, Line};
use num9_core::piece::shape;
use num9_core::Rng;

fn usage() -> ! {
    eprintln!(
        "usage: lns [--minutes M] [--budget NODES] [--lift MIN,MAX] [--sweep TRIES] [--out DIR] RECORD_FILES..."
    );
    std::process::exit(2)
}

/// The moves of a recorded play (`value@level:row:col:rotation` tokens).
fn parse(text: &str) -> Line {
    text.split_whitespace()
        .filter(|t| t.contains('@') && t.starts_with(|c: char| c.is_ascii_digit()))
        .filter_map(|t| {
            let (v, rest) = t.split_once('@')?;
            let n: Vec<usize> = rest.split(':').filter_map(|x| x.parse().ok()).collect();
            if n.len() == 4 {
                Some((v.parse().ok()?, Move::new(n[0], n[1], n[2], n[3])))
            } else {
                None
            }
        })
        .collect()
}

fn score(line: &Line) -> i64 {
    line.iter().map(|(v, m)| *v as i64 * m.layer as i64).sum()
}

fn cells(v: u8, m: Move) -> Vec<(usize, usize)> {
    shape(v, m.rot)
        .cells()
        .map(|(r, c)| (m.i + r, m.j + c))
        .collect()
}

/// An order to build a finished stack in, if it can be built: level by
/// level, each tile touching one already down (the first tile first).
fn order(pieces: &[(u8, Move)]) -> Option<Line> {
    let top = pieces.iter().map(|p| p.1.layer).max()?;
    let mut out: Line = Vec::with_capacity(pieces.len());
    for level in 0..=top {
        let here: Vec<usize> = (0..pieces.len())
            .filter(|&k| pieces[k].1.layer == level)
            .collect();
        let Some(&start) = here.first() else { continue };
        // Whether tiles a and b share an edge.
        let touch = |a: usize, b: usize| {
            let theirs: HashSet<(usize, usize)> =
                cells(pieces[b].0, pieces[b].1).into_iter().collect();
            cells(pieces[a].0, pieces[a].1).iter().any(|&(i, j)| {
                theirs.contains(&(i + 1, j))
                    || theirs.contains(&(i, j + 1))
                    || (i > 0 && theirs.contains(&(i - 1, j)))
                    || (j > 0 && theirs.contains(&(i, j - 1)))
            })
        };
        let mut seen = vec![start];
        let mut k = 0;
        while k < seen.len() {
            let a = seen[k];
            for &b in &here {
                if !seen.contains(&b) && touch(a, b) {
                    seen.push(b);
                }
            }
            k += 1;
        }
        if seen.len() != here.len() {
            return None;
        }
        out.extend(seen.iter().map(|&k| pieces[k]));
    }
    // Check it on a board, move by move.
    let mut board = Board::new();
    for (n, &(v, m)) in out.iter().enumerate() {
        if n > 0 && !board.is_valid_move(v, m) {
            return None;
        }
        board.place(v, m).ok()?;
    }
    Some(out)
}

/// Lift off the top of `line`: every tile above a level, and some on it
/// (at most `lift.1` in all). Returns what's left, in an order to
/// build it, and the tiles lifted.
fn destroy(line: &Line, lift: (usize, usize), rng: &mut Rng) -> Option<(Line, [u8; 10])> {
    let top = line.iter().map(|p| p.1.layer).max()?;
    let most = lift.0 + rng.below(lift.1 - lift.0 + 1);
    let levels: Vec<usize> = (1..=top)
        .filter(|&m| {
            let above = line.iter().filter(|p| p.1.layer > m).count();
            let on = line.iter().filter(|p| p.1.layer == m).count();
            above < most && on >= 2
        })
        .collect();
    let &m = levels.get(rng.below(levels.len().max(1)))?;
    let above = line.iter().filter(|p| p.1.layer > m).count();
    let mut on: Vec<usize> = (0..line.len()).filter(|&k| line[k].1.layer == m).collect();
    rng.shuffle(&mut on);
    let lift = 1 + rng.below((most - above).min(on.len() - 1));
    let lifted: HashSet<usize> = on[..lift].iter().copied().collect();
    let mut removed = [0u8; 10];
    let mut kept = Vec::new();
    for (k, &p) in line.iter().enumerate() {
        if p.1.layer > m || lifted.contains(&k) {
            removed[p.0 as usize] += 1;
        } else {
            kept.push(p);
        }
    }
    Some((order(&kept)?, removed))
}

fn fingerprint(line: &Line) -> u128 {
    let mut board = Board::new();
    for &(v, m) in line {
        board.place(v, m).expect("a legal play");
    }
    key(&board)
}

fn write(dir: &str, line: &Line) {
    let moves: Vec<String> = line
        .iter()
        .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
        .collect();
    let s = score(line);
    let top = line.iter().map(|p| p.1.layer).max().unwrap_or(0);
    let path = format!("{dir}/{s}-L{top}-{:016x}.txt", fingerprint(line) as u64);
    let text = format!(
        "score {s}\nfound by: lns (large neighbourhood search)\n\nmoves (value@layer:row:col:rotation) {}\n",
        moves.join(" ")
    );
    std::fs::write(path, text).expect("write a play");
}

struct Pool {
    /// (score, play), best first.
    stacks: Vec<(i64, Line)>,
    seen: HashSet<u128>,
}

fn main() {
    let mut minutes = 60u64;
    let mut budget = 20_000_000usize;
    let mut dir = String::from("runs/lns");
    let mut lift = (6, 12);
    let mut sweep_tries = 0usize;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--minutes" => minutes = value().parse().unwrap_or_else(|_| usage()),
            "--budget" => budget = value().parse().unwrap_or_else(|_| usage()),
            "--out" => dir = value(),
            "--sweep" => sweep_tries = value().parse().unwrap_or_else(|_| usage()),
            "--lift" => {
                let v = value();
                let (a, b) = v.split_once(',').unwrap_or_else(|| usage());
                lift = (
                    a.parse().unwrap_or_else(|_| usage()),
                    b.parse().unwrap_or_else(|_| usage()),
                );
            }
            _ if a.starts_with("--") => usage(),
            _ => files.push(a),
        }
    }
    std::fs::create_dir_all(&dir).expect("make the output directory");
    let mut pool = Pool {
        stacks: Vec::new(),
        seen: HashSet::new(),
    };
    for f in &files {
        let line = parse(&std::fs::read_to_string(f).expect("read a record"));
        if let Some(line) = order(&line) {
            if pool.seen.insert(fingerprint(&line)) {
                pool.stacks.push((score(&line), line));
            }
        }
    }
    pool.stacks.sort_by_key(|s| std::cmp::Reverse(s.0));
    let start_best = pool.stacks.first().map_or(0, |s| s.0);
    println!(
        "lns: {} distinct stacks to start from, best {start_best}; {minutes} minutes, lifting {} to {} tiles, budget {budget} positions a repair",
        pool.stacks.len(),
        lift.0,
        lift.1
    );
    if sweep_tries > 0 {
        sweep(pool.stacks, sweep_tries, lift, budget, minutes, &dir);
        return;
    }
    let pool = Mutex::new(pool);
    let (steps, better, plateau, gave_up) = (
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    );
    let start = Instant::now();
    let end = start + Duration::from_secs(minutes * 60);
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    thread::scope(|s| {
        for t in 0..threads {
            let (pool, dir) = (&pool, &dir);
            let (steps, better, plateau, gave_up) = (&steps, &better, &plateau, &gave_up);
            s.spawn(move || {
                let mut rng = Rng::new(start.elapsed().as_nanos() as u64 ^ (t as u64) << 40);
                while Instant::now() < end {
                    // One of the best fifty stacks.
                    let (from, line) = {
                        let p = pool.lock().unwrap();
                        p.stacks[rng.below(p.stacks.len().min(50))].clone()
                    };
                    let Some((kept, removed)) = destroy(&line, lift, &mut rng) else {
                        continue;
                    };
                    steps.fetch_add(1, Relaxed);
                    let Some((got, new)) =
                        best_from(&kept, removed, from - 1, budget, rng.next_u64() | 1)
                    else {
                        gave_up.fetch_add(1, Relaxed);
                        continue;
                    };
                    let mut p = pool.lock().unwrap();
                    if !p.seen.insert(fingerprint(&new)) {
                        continue;
                    }
                    let best = p.stacks[0].0;
                    if got > from {
                        better.fetch_add(1, Relaxed);
                        write(dir, &new);
                        println!(
                            "[{:6.0}s] {from} -> {got}{}",
                            start.elapsed().as_secs_f64(),
                            if got > best { "  NEW BEST" } else { "" }
                        );
                    } else {
                        plateau.fetch_add(1, Relaxed);
                    }
                    p.stacks.push((got, new));
                    p.stacks.sort_by_key(|s| std::cmp::Reverse(s.0));
                    p.stacks.truncate(500);
                }
            });
        }
        // A heartbeat every ten minutes.
        s.spawn(|| {
            while Instant::now() < end {
                thread::sleep(Duration::from_secs(600).min(end.saturating_duration_since(Instant::now())));
                let p = pool.lock().unwrap();
                println!(
                    "[{:6.0}s] {} repairs: {} better, {} new stacks as good, {} out of budget; best {}",
                    start.elapsed().as_secs_f64(),
                    steps.load(Relaxed),
                    better.load(Relaxed),
                    plateau.load(Relaxed),
                    gave_up.load(Relaxed),
                    p.stacks[0].0
                );
            }
        });
    });
    let p = pool.into_inner().unwrap();
    println!("done: best {} (started from {start_best})", p.stacks[0].0);
}

/// Improve every stack on its own (see --sweep), and sum up what each became.
fn sweep(
    stacks: Vec<(i64, Line)>,
    tries: usize,
    lift: (usize, usize),
    budget: usize,
    minutes: u64,
    dir: &str,
) {
    let start = Instant::now();
    let end = start + Duration::from_secs(minutes * 60);
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(i64, i64)>> = Mutex::new(Vec::new());
    let best = Mutex::new(stacks.first().map_or(0, |s| s.0));
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    thread::scope(|s| {
        for t in 0..threads {
            let (stacks, next, results, best) = (&stacks, &next, &results, &best);
            s.spawn(move || {
                let mut rng = Rng::new(start.elapsed().as_nanos() as u64 ^ (t as u64) << 40);
                while Instant::now() < end {
                    let Some((from, line)) = stacks.get(next.fetch_add(1, Relaxed)).cloned() else {
                        break;
                    };
                    let (mut score_now, mut now, mut fails) = (from, line, 0);
                    while fails < tries && Instant::now() < end {
                        let Some((kept, removed)) = destroy(&now, lift, &mut rng) else {
                            break;
                        };
                        match best_from(&kept, removed, score_now, budget, rng.next_u64() | 1) {
                            Some((got, new)) => (score_now, now, fails) = (got, new, 0),
                            None => fails += 1,
                        }
                    }
                    if score_now > from {
                        write(dir, &now);
                        let mut b = best.lock().unwrap();
                        let note = if score_now > *b { "  NEW BEST" } else { "" };
                        *b = (*b).max(score_now);
                        println!(
                            "[{:6.0}s] {from} -> {score_now}{note}",
                            start.elapsed().as_secs_f64()
                        );
                    }
                    results.lock().unwrap().push((from, score_now));
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let improved = results.iter().filter(|(a, b)| b > a).count();
    let gain: i64 = results.iter().map(|(a, b)| b - a).sum();
    let mut finals: Vec<i64> = results.iter().map(|r| r.1).collect();
    finals.sort_unstable_by(|a, b| b.cmp(a));
    println!(
        "swept {} of {} stacks in {:.0}s: {improved} improved, by {:.1} on average when they were; best now {}; top ten {:?}",
        results.len(),
        stacks.len(),
        start.elapsed().as_secs_f64(),
        gain as f64 / improved.max(1) as f64,
        finals.first().copied().unwrap_or(0),
        &finals[..finals.len().min(10)]
    );
}
