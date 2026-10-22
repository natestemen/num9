//! Exhaustive "widening" facts for the upper bound on the best score.
//!
//! Every cell of a level sits on a cell of the level below, so the level
//! below covers at least the same area. This computes, exactly, how much
//! more it must cover when a level holds one tile, or two touching tiles:
//! the minimum total area of any set of non-overlapping tiles covering
//! them such that each of them rests on at least two of those tiles.
//! Copies of each value are not limited and the covering need not be
//! connected, which can only make the minimum smaller, so the results are
//! safe lower bounds.
//!
//!     cargo run --release --bin widening > widening.tsv
//!
//! Output (tab separated): `single <value> <min extra cells>` for one tile,
//! and `pair <value> <value> <min extra cells>` for two touching tiles (the
//! minimum over every way they can touch); and `supports <value> <value>
//! <max area>`, the most area a level can hold when the level below it is
//! exactly those two touching tiles (every tile above must lie within them
//! and rest on both; the maximum over every way they can touch); and
//! `supports3 <value> <value> <value> <max area>`, the same for a level of
//! exactly three connected tiles.

use std::sync::Mutex;
use std::thread;

use num9_core::piece::shape;

const G: usize = 24;

type Cells = Vec<(usize, usize)>;

fn rotations(v: u8) -> Vec<usize> {
    if shape(v, 0).rows == shape(v, 2).rows {
        vec![0, 1]
    } else {
        vec![0, 1, 2, 3]
    }
}

fn cells(v: u8, rot: usize, i: usize, j: usize) -> Cells {
    shape(v, rot).cells().map(|(r, c)| (i + r, j + c)).collect()
}

/// Every placement of every tile, as cell offsets, for the covering search.
fn all_shapes() -> Vec<(u8, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    for v in 0..10u8 {
        for rot in rotations(v) {
            out.push((v, shape(v, rot).cells().collect()));
        }
    }
    out
}

struct Cover<'a> {
    /// Which upper tile (1-based) is over each cell, 0 if none.
    upper: Vec<u8>,
    upper_cells: Vec<Cells>,
    /// Lower tile (1-based) covering each cell, 0 if none.
    lower: Vec<u8>,
    lower_cells: Vec<Cells>,
    /// Lower cells not under an upper tile so far, and the most allowed.
    waste: usize,
    budget: usize,
    shapes: &'a [(u8, Vec<(usize, usize)>)],
}

impl Cover<'_> {
    fn first_uncovered(&self) -> Option<(usize, usize)> {
        (0..G * G)
            .find(|&k| self.upper[k] != 0 && self.lower[k] == 0)
            .map(|k| (k / G, k % G))
    }

    /// An upper tile lying entirely on one lower tile breaks the rule.
    fn rests_on_one(&self, lower_id: usize) -> bool {
        let mine = &self.lower_cells[lower_id];
        self.upper_cells
            .iter()
            .any(|u| u.iter().all(|c| mine.contains(c)))
    }

    /// Whether the upper tiles can be covered with at most `budget` extra
    /// cells. Every uncovered upper cell must be covered by some tile, so
    /// trying every tile over the first one is exhaustive.
    fn feasible(&mut self) -> bool {
        let Some((ti, tj)) = self.first_uncovered() else {
            // Covered: every upper tile must touch at least two lower tiles.
            return self.upper_cells.iter().all(|u| {
                let mut ids: Vec<u8> = u.iter().map(|&(a, b)| self.lower[a * G + b]).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.len() >= 2
            });
        };
        let shapes = self.shapes;
        for (_, offsets) in shapes {
            for &(r, c) in offsets {
                let (Some(i), Some(j)) = (ti.checked_sub(r), tj.checked_sub(c)) else {
                    continue;
                };
                let placed: Cells = offsets.iter().map(|&(a, b)| (i + a, j + b)).collect();
                if placed
                    .iter()
                    .any(|&(a, b)| a >= G || b >= G || self.lower[a * G + b] != 0)
                {
                    continue;
                }
                let waste = placed
                    .iter()
                    .filter(|&&(a, b)| self.upper[a * G + b] == 0)
                    .count();
                if self.waste + waste > self.budget {
                    continue;
                }
                let id = self.lower_cells.len() as u8 + 1;
                for &(a, b) in &placed {
                    self.lower[a * G + b] = id;
                }
                self.lower_cells.push(placed);
                self.waste += waste;
                let found = !self.rests_on_one(id as usize - 1) && self.feasible();
                self.waste -= waste;
                let placed = self.lower_cells.pop().unwrap();
                for (a, b) in placed {
                    self.lower[a * G + b] = 0;
                }
                if found {
                    return true;
                }
            }
        }
        false
    }
}

/// Minimum extra cells to support `upper` tiles (already placed cells):
/// the smallest budget for which a covering exists.
fn min_widening(upper: &[Cells], shapes: &[(u8, Vec<(usize, usize)>)]) -> usize {
    let mut grid = vec![0u8; G * G];
    for (k, u) in upper.iter().enumerate() {
        for &(a, b) in u {
            grid[a * G + b] = k as u8 + 1;
        }
    }
    (0..)
        .find(|&budget| {
            Cover {
                upper: grid.clone(),
                upper_cells: upper.to_vec(),
                lower: vec![0; G * G],
                lower_cells: Vec::new(),
                waste: 0,
                budget,
                shapes,
            }
            .feasible()
        })
        .unwrap()
}

/// The most area that tiles on the level above can cover when the level
/// below is exactly `lower` (the cells of two tiles): every tile above lies
/// within them and rests on both. Exhaustive over packings; copies are not
/// limited, so this can only overestimate.
fn max_supported(lower: &[Cells], shapes: &[(u8, Vec<(usize, usize)>)]) -> usize {
    let mut owner = vec![0u8; G * G];
    for (k, t) in lower.iter().enumerate() {
        for &(a, b) in t {
            owner[a * G + b] = k as u8 + 1;
        }
    }
    // Every placement inside the lower tiles that touches at least two of them.
    let mut options: Vec<Cells> = Vec::new();
    for (_, offsets) in shapes {
        for i in 0..G - 4 {
            for j in 0..G - 4 {
                let placed: Cells = offsets.iter().map(|&(a, b)| (i + a, j + b)).collect();
                if placed.iter().any(|&(a, b)| owner[a * G + b] == 0) {
                    continue;
                }
                let mut ids: Vec<u8> = placed.iter().map(|&(a, b)| owner[a * G + b]).collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.len() >= 2 {
                    options.push(placed);
                }
            }
        }
    }
    fn best(options: &[Cells], used: &mut Vec<bool>, from: usize, area: usize, top: &mut usize) {
        *top = (*top).max(area);
        for k in from..options.len() {
            if options[k].iter().any(|&(a, b)| used[a * G + b]) {
                continue;
            }
            for &(a, b) in &options[k] {
                used[a * G + b] = true;
            }
            best(options, used, k + 1, area + options[k].len(), top);
            for &(a, b) in &options[k] {
                used[a * G + b] = false;
            }
        }
    }
    let mut top = 0;
    best(&options, &mut vec![false; G * G], 0, 0, &mut top);
    top
}

fn touches(a: &Cells, b: &Cells) -> bool {
    a.iter().any(|&(x, y)| {
        b.iter()
            .any(|&(p, q)| (x == p && y.abs_diff(q) == 1) || (y == q && x.abs_diff(p) == 1))
    })
}

fn main() {
    let shapes = all_shapes();
    let o = 9; // place the upper tiles away from the grid edge
    for v in 0..10u8 {
        let w = min_widening(&[cells(v, 0, o, o)], &shapes);
        println!("single\t{v}\t{w}");
    }
    // Pairs: the first tile unrotated (turning the whole level changes
    // nothing), the second in every rotation and touching position.
    let jobs: Vec<(u8, u8)> = (0..10u8)
        .flat_map(|a| (a..10u8).map(move |b| (a, b)))
        .collect();
    let results = Mutex::new(Vec::new());
    let next = Mutex::new(0usize);
    thread::scope(|s| {
        for _ in 0..thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| loop {
                let k = {
                    let mut n = next.lock().unwrap();
                    *n += 1;
                    *n - 1
                };
                let Some(&(a, b)) = jobs.get(k) else { break };
                let first = cells(a, 0, o, o);
                let mut best = usize::MAX;
                for rb in rotations(b) {
                    for i in o - 4..=o + 4 {
                        for j in o - 4..=o + 4 {
                            let second = cells(b, rb, i, j);
                            if second.iter().any(|c| first.contains(c)) || !touches(&first, &second)
                            {
                                continue;
                            }
                            best = best.min(min_widening(&[first.clone(), second], &shapes));
                        }
                    }
                }
                results.lock().unwrap().push((a, b, best));
                eprintln!("pair {a} {b}: {best}");
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort();
    for (a, b, w) in r {
        println!("pair\t{a}\t{b}\t{w}");
    }
    // A level of exactly two touching tiles: the most it can support above.
    for a in 0..10u8 {
        for b in a..10u8 {
            let first = cells(a, 0, o, o);
            let mut most = 0;
            for rb in rotations(b) {
                for i in o - 4..=o + 4 {
                    for j in o - 4..=o + 4 {
                        let second = cells(b, rb, i, j);
                        if second.iter().any(|c| first.contains(c)) || !touches(&first, &second) {
                            continue;
                        }
                        most = most.max(max_supported(&[first.clone(), second], &shapes));
                    }
                }
            }
            println!("supports\t{a}\t{b}\t{most}");
        }
    }
    // A level of exactly three connected tiles: the most it can support.
    let triples: Vec<(u8, u8, u8)> = (0..10u8)
        .flat_map(|a| (a..10u8).flat_map(move |b| (b..10u8).map(move |c| (a, b, c))))
        .collect();
    let results = Mutex::new(Vec::new());
    let next = Mutex::new(0usize);
    thread::scope(|s| {
        for _ in 0..thread::available_parallelism().map_or(1, |n| n.get()) {
            s.spawn(|| loop {
                let k = {
                    let mut n = next.lock().unwrap();
                    *n += 1;
                    *n - 1
                };
                let Some(&(a, b, c)) = triples.get(k) else {
                    break;
                };
                let first = cells(a, 0, o, o);
                let mut most = 0;
                // The second tile touches the first; the third touches either.
                // Both orders of the other two values cover every connected
                // arrangement (rotating the whole level fixes the first).
                for (x, y) in [(b, c), (c, b)] {
                    for rx in rotations(x) {
                        for i in o - 4..=o + 4 {
                            for j in o - 4..=o + 4 {
                                let second = cells(x, rx, i, j);
                                if second.iter().any(|q| first.contains(q))
                                    || !touches(&first, &second)
                                {
                                    continue;
                                }
                                let both: Cells = first.iter().chain(&second).copied().collect();
                                for ry in rotations(y) {
                                    for p in o - 8..=o + 8 {
                                        for q in o - 8..=o + 8 {
                                            let third = cells(y, ry, p, q);
                                            if third.iter().any(|z| both.contains(z))
                                                || !touches(&both, &third)
                                            {
                                                continue;
                                            }
                                            let lower = [first.clone(), second.clone(), third];
                                            most = most.max(max_supported(&lower, &shapes));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                eprintln!("supports3 {a} {b} {c}: {most}");
                results.lock().unwrap().push((a, b, c, most));
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort();
    for (a, b, c, m) in r {
        println!("supports3\t{a}\t{b}\t{c}\t{m}");
    }
}
