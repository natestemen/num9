//! Rearrange a band of levels inside a finished stack:
//!
//!     cargo run --release --bin window -- [--budget NODES] [--ground] [--out DIR] RECORD_FILES...
//!
//! For each stack, and each band of two or three consecutive levels (from the
//! ground up), everything below and above the band stays exactly where it
//! is; the band's tiles are taken out and put back in the best possible way
//! on the band's levels. Each band level rests on the one below as the rules
//! require, and the band's top level must still hold up the unchanged level
//! above it (every tile there fully supported, on at least two tiles). The
//! search is exhaustive (within a budget of positions per band), so a band
//! that can't do better than it is really can't. Better stacks go to DIR.
//! --ground searches only the bands that include the ground (the hardest).

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use num9_core::board::{Board, Move};
use num9_core::exact::moves;
use num9_core::freeorder::{ceiling, Line};
use num9_core::piece::shape;

fn usage() -> ! {
    eprintln!("usage: window [--budget NODES] [--ground] [--out DIR] RECORD_FILES...");
    std::process::exit(2)
}

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

/// The board's tiles where they are (no allowance for moving or turning:
/// the fixed tiles pin everything in place).
fn position_key(board: &Board) -> u128 {
    let mut ts: Vec<(u8, usize, usize, usize, usize)> = board
        .pieces()
        .iter()
        .map(|p| (p.value, p.mv.layer, p.mv.i, p.mv.j, p.mv.rot))
        .collect();
    ts.sort_unstable();
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    for t in ts {
        for x in [
            t.0 as u128,
            t.1 as u128,
            t.2 as u128,
            t.3 as u128,
            t.4 as u128,
        ] {
            h = (h ^ x).wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    }
    h
}

/// The search for one band: tiles `left` to place on levels `lo..=hi`,
/// with `above` (the stack above the band, in a buildable order) to fit on
/// top afterwards.
struct Band<'a> {
    lo: usize,
    hi: usize,
    above: &'a Line,
    above_score: i64,
    /// A cell the band's ground must cover (when the band includes the
    /// ground and nothing below pins it).
    anchor: Option<(usize, usize)>,
    best: i64,
    line: Line,
    nodes: usize,
    budget: usize,
    seen: HashSet<u128>,
}

impl Band<'_> {
    fn dfs(&mut self, board: &mut Board, left: &mut [u8; 10], path: &mut Line) {
        self.nodes += 1;
        if self.nodes > self.budget {
            return;
        }
        let here = board.score();
        if left.iter().all(|&c| c == 0) {
            // Does the unchanged stack above still fit?
            let mut placed = 0;
            for &(v, m) in self.above {
                if !board.is_valid_move(v, m) {
                    break;
                }
                board.place(v, m).expect("legal move");
                placed += 1;
            }
            if placed == self.above.len() && here + self.above_score > self.best {
                self.best = here + self.above_score;
                self.line = path.iter().chain(self.above.iter()).copied().collect();
            }
            for _ in 0..placed {
                board.undo();
            }
            return;
        }
        let top = board.pieces().iter().map(|p| p.mv.layer).max();
        let (count, area) = board
            .pieces()
            .iter()
            .filter(|p| Some(p.mv.layer) == top)
            .fold((0, 0), |(n, a), p| (n + 1, a + shape(p.value, 0).tiles));
        let all_high: i64 = (0..10)
            .map(|v| left[v] as i64 * v as i64 * self.hi as i64)
            .sum();
        let roof = all_high.min(ceiling(left, top.unwrap_or(0), count, area));
        if here + roof + self.above_score <= self.best {
            return;
        }
        // Positions already searched (remembered up to a limit, to bound
        // memory: past it, some are searched again, never wrongly skipped).
        let k = position_key(board);
        if self.seen.contains(&k) {
            return;
        }
        if self.seen.len() < 10_000_000 {
            self.seen.insert(k);
        }
        let floor = top.map_or(self.lo, |t| t.max(self.lo));
        let mut options: Vec<(u8, Move)> = Vec::new();
        for v in (0..10u8).filter(|&v| left[v as usize] > 0) {
            if board.pieces().is_empty() {
                // The band's first ground tile: every placement covering the anchor.
                let (ai, aj) = self.anchor.expect("an anchor for a band on the ground");
                for rot in 0..4 {
                    for (r, c) in shape(v, rot).cells() {
                        if let (Some(i), Some(j)) = (ai.checked_sub(r), aj.checked_sub(c)) {
                            options.push((v, Move::new(0, i, j, rot)));
                        }
                    }
                }
            } else {
                options.extend(
                    moves(board, v)
                        .into_iter()
                        .filter(|m| m.layer >= floor && m.layer <= self.hi)
                        .map(|m| (v, m)),
                );
            }
        }
        options.sort_by_key(|&(v, m)| std::cmp::Reverse((m.layer, v)));
        for (v, m) in options {
            if board.place(v, m).is_err() {
                continue;
            }
            left[v as usize] -= 1;
            path.push((v, m));
            self.dfs(board, left, path);
            path.pop();
            left[v as usize] += 1;
            board.undo();
        }
    }
}

/// The best way to rearrange levels `lo..=hi` of `line` (a play built level
/// by level), and whether the search finished within the budget.
fn rearrange(line: &Line, lo: usize, hi: usize, budget: usize) -> (i64, Line, bool) {
    let below: Line = line.iter().filter(|p| p.1.layer < lo).copied().collect();
    let above: Line = line.iter().filter(|p| p.1.layer > hi).copied().collect();
    let mut left = [0u8; 10];
    for p in line.iter().filter(|p| (lo..=hi).contains(&p.1.layer)) {
        left[p.0 as usize] += 1;
    }
    let anchor = (lo == 0)
        .then(|| above.first().map(|&(v, m)| cells(v, m)[0]))
        .flatten();
    let mut board = Board::new();
    for &(v, m) in &below {
        board.place(v, m).expect("a legal play");
    }
    let mut band = Band {
        lo,
        hi,
        above: &above,
        above_score: score(&above),
        anchor,
        best: score(line) - 1,
        line: line.clone(),
        nodes: 0,
        budget,
        seen: HashSet::new(),
    };
    let mut path = below.clone();
    band.dfs(&mut board, &mut left, &mut path);
    let finished = band.nodes <= budget;
    (band.best.max(score(line)), band.line, finished)
}

fn main() {
    let mut budget = 300_000_000usize;
    let mut dir = String::from("runs/window");
    let mut ground = false;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--budget" => budget = value().parse().unwrap_or_else(|_| usage()),
            "--out" => dir = value(),
            "--ground" => ground = true,
            _ if a.starts_with("--") => usage(),
            _ => files.push(a),
        }
    }
    std::fs::create_dir_all(&dir).expect("make the output directory");
    // Every (stack, band) pair is a job.
    let mut jobs = Vec::new();
    for f in &files {
        let line = parse(&std::fs::read_to_string(f).expect("read a record"));
        let top = line.iter().map(|p| p.1.layer).max().unwrap_or(0);
        for size in [2, 3] {
            for lo in 0..=if ground {
                0
            } else {
                top.saturating_sub(size - 1)
            } {
                jobs.push((f.clone(), line.clone(), lo, lo + size - 1));
            }
        }
    }
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    thread::scope(|s| {
        for _ in 0..thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| {
                while let Some((f, line, lo, hi)) = jobs.get(next.fetch_add(1, Relaxed)) {
                    let t = Instant::now();
                    let (got, new, finished) = rearrange(line, *lo, *hi, budget);
                    let from = score(line);
                    let name = std::path::Path::new(f).file_stem().unwrap().to_string_lossy().into_owned();
                    let note = if got > from {
                        let moves: Vec<String> = new
                            .iter()
                            .map(|(v, m)| format!("{v}@{}:{}:{}:{}", m.layer, m.i, m.j, m.rot))
                            .collect();
                        let text = format!(
                            "score {got}\nfound by: window (levels {lo} to {hi} of {name} rearranged)\n\nmoves (value@layer:row:col:rotation) {}\n",
                            moves.join(" ")
                        );
                        std::fs::write(format!("{dir}/{got}-{name}-levels-{lo}-{hi}.txt"), text).expect("write a play");
                        format!("  -> {got}  BETTER")
                    } else {
                        String::new()
                    };
                    let line_out = format!(
                        "{name}  levels {lo}-{hi}: {from}{note}{}  ({:.1}s)",
                        if finished { "" } else { "  (budget ran out: not exhaustive)" },
                        t.elapsed().as_secs_f64()
                    );
                    println!("{line_out}");
                    out.lock().unwrap().push(line_out);
                }
            });
        }
    });
    println!(
        "{} bands in {:.0}s",
        jobs.len(),
        start.elapsed().as_secs_f64()
    );
}
