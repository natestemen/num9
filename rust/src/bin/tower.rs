//! Search for the highest possible score (free tile order) by designing
//! towers top-down with a beam search (see `num9_core::tower`):
//!
//!     cargo run --release --bin tower -- [--tops 4,5,6] [--width 60]
//!         [--per-design 8] [--nodes 100000] [--area-cost 8] [--growth 4]
//!         [--noise 0] [--seed 0] [--out tower.txt]
//!         [--record-above 200] [--record-dir records/above-200]
//!
//! Each top height runs on its own thread. Prints every new best as it is
//! found, checks the best play move by move, and writes it to --out. With
//! --record-above, every distinct play scoring more than that is checked
//! and saved to --record-dir (one file per tower, ignoring rotations and
//! shifts of the same tower; several searches can share a directory), and
//! the directory's INDEX.md is updated.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Instant;

use num9_core::nrpa::{verify, Order, Play, Step};
use num9_core::piece::shape;
use num9_core::tower::TowerSearch;

fn usage() -> ! {
    eprintln!(
        "usage: tower [--tops a,b,...] [--counts a,b,...] [--width W] [--per-design K] [--nodes N] [--area-cost C] \
         [--growth G] [--noise J] [--seed S] [--out FILE] [--record-above SCORE] [--record-dir DIR]"
    );
    std::process::exit(2)
}

/// A tile's level, value and cells.
type TileCells = (usize, u8, Vec<(i64, i64)>);

/// Cells of each step, with its level and value.
fn tiles(steps: &[Step]) -> Vec<TileCells> {
    steps
        .iter()
        .map(|&(v, m)| {
            let cells = shape(v, m.rot)
                .cells()
                .map(|(r, c)| ((m.i + r) as i64, (m.j + c) as i64))
                .collect();
            (m.layer, v, cells)
        })
        .collect()
}

/// A name for the tower that ignores how it is rotated or shifted (tiles
/// can't be flipped, so mirror images count as different).
fn fingerprint(steps: &[Step]) -> u64 {
    let base = tiles(steps);
    let mut best: Option<Vec<TileCells>> = None;
    for turns in 0..4 {
        let mut t = base.clone();
        for _ in 0..turns {
            for (_, _, cells) in t.iter_mut() {
                for c in cells.iter_mut() {
                    *c = (c.1, -c.0);
                }
            }
        }
        let min_i = t
            .iter()
            .flat_map(|x| x.2.iter().map(|c| c.0))
            .min()
            .unwrap();
        let min_j = t
            .iter()
            .flat_map(|x| x.2.iter().map(|c| c.1))
            .min()
            .unwrap();
        for (_, _, cells) in t.iter_mut() {
            for c in cells.iter_mut() {
                *c = (c.0 - min_i, c.1 - min_j);
            }
            cells.sort_unstable();
        }
        t.sort();
        if best.as_ref().is_none_or(|b| &t < b) {
            best = Some(t);
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (layer, v, cells) in best.unwrap() {
        let nums = [layer as i64, v as i64]
            .into_iter()
            .chain(cells.into_iter().flat_map(|c| [c.0, c.1]));
        for x in nums {
            h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The tower level by level, top first; each cell shows its tile's value.
fn picture(steps: &[Step]) -> String {
    let cells: Vec<(usize, i64, i64, u8)> = tiles(steps)
        .into_iter()
        .flat_map(|(l, v, cs)| cs.into_iter().map(move |(i, j)| (l, i, j, v)))
        .collect();
    let (i0, i1) = (
        cells.iter().map(|c| c.1).min().unwrap(),
        cells.iter().map(|c| c.1).max().unwrap(),
    );
    let (j0, j1) = (
        cells.iter().map(|c| c.2).min().unwrap(),
        cells.iter().map(|c| c.2).max().unwrap(),
    );
    let top = cells.iter().map(|c| c.0).max().unwrap();
    let mut out = String::new();
    for l in (0..=top).rev() {
        let pts: i64 = steps
            .iter()
            .filter(|s| s.1.layer == l)
            .map(|s| s.0 as i64 * l as i64)
            .sum();
        out += &format!("level {l}  ({pts} points)\n");
        for i in i0..=i1 {
            let row: Vec<String> = (j0..=j1)
                .map(|j| {
                    cells
                        .iter()
                        .find(|c| c.0 == l && c.1 == i && c.2 == j)
                        .map_or("·".to_string(), |c| c.3.to_string())
                })
                .collect();
            out += &format!("   {}\n", row.join(" "));
        }
        out += "\n";
    }
    out
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

/// Saves every distinct play above a score, one file per tower named
/// `<score>-L<top level>-<fingerprint>.txt`, and keeps an index. Everything
/// needed for deduplication and the index is in the file names, so several
/// searches can record into the same directory.
struct Recorder {
    dir: PathBuf,
    above: i64,
    known: HashSet<u64>,
    how: String,
}

/// (score, top level, fingerprint) from a record file name.
fn parse_name(name: &str) -> Option<(i64, usize, u64)> {
    let stem = name.strip_suffix(".txt")?;
    let mut parts = stem.split('-');
    let score = parts.next()?.parse().ok()?;
    let top = parts.next()?.strip_prefix('L')?.parse().ok()?;
    let key = u64::from_str_radix(parts.next()?, 16).ok()?;
    parts.next().is_none().then_some((score, top, key))
}

fn record_names(dir: &Path) -> Vec<(i64, usize, u64, String)> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter_map(|n| parse_name(&n).map(|(s, t, k)| (s, t, k, n)))
                .collect()
        })
        .unwrap_or_default()
}

impl Recorder {
    fn new(dir: PathBuf, above: i64, how: String) -> Recorder {
        std::fs::create_dir_all(&dir).expect("create the record directory");
        let known = record_names(&dir).into_iter().map(|r| r.2).collect();
        Recorder {
            dir,
            above,
            known,
            how,
        }
    }

    /// Save `play` if it scores enough and is a new tower; returns the file.
    fn offer(&mut self, play: &Play) -> Option<PathBuf> {
        if play.score <= self.above {
            return None;
        }
        let key = fingerprint(&play.steps);
        if !self.known.insert(key) {
            return None;
        }
        // Another search may have recorded it since we started.
        let suffix = format!("-{key:016x}.txt");
        if record_names(&self.dir)
            .iter()
            .any(|r| r.3.ends_with(&suffix))
        {
            return None;
        }
        let mut tiles = [0u8; 10];
        for s in &play.steps {
            tiles[s.0 as usize] += 1;
        }
        verify(&Order::Free(tiles), play).expect("plays are checked before recording");
        let top = play.steps.iter().map(|s| s.1.layer).max().unwrap_or(0);
        let path = self.dir.join(format!("{}-L{top}{suffix}", play.score));
        let text = format!(
            "{}\nfound by: {}\ncheck:    python records/check_play.py {}\n\n\
             The tower, top level first (each cell shows its tile's value):\n\n{}",
            describe(play),
            self.how,
            path.display(),
            picture(&play.steps)
        );
        std::fs::write(&path, text).expect("write a record");
        write_index(&self.dir, self.above);
        Some(path)
    }
}

/// INDEX.md: every recorded play, best first.
fn write_index(dir: &Path, above: i64) {
    let mut rows = record_names(dir);
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.3.cmp(&b.3)));
    let mut out = format!(
        "# Plays scoring above {above} (free tile order)\n\n\
         {} distinct towers, best first. Rotations and shifts of the same tower count once; \
         towers differing only in where a leftover tile sits count separately. Check any of \
         them with `python records/check_play.py <file>` or step through it with \
         `num9 --replay <file>`.\n\n| score | top level | file |\n| --- | --- | --- |\n",
        rows.len()
    );
    for (score, top, _, name) in rows {
        out += &format!("| {score} | {top} | [{name}]({name}) |\n");
    }
    std::fs::write(dir.join("INDEX.md"), out).expect("write the index");
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
        noise: 0,
        seed: 0,
    };
    let mut out = String::from("tower.txt");
    let mut record: Option<i64> = None;
    let mut record_dir = PathBuf::from("records/above-200");
    let how = format!(
        "tower {}",
        std::env::args().skip(1).collect::<Vec<_>>().join(" ")
    );
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
            // A smaller deck: this many tiles of each value from 0 up.
            "--counts" => {
                search.tiles = [0; 10];
                for (v, c) in value.split(',').enumerate() {
                    if v >= 10 {
                        usage();
                    }
                    search.tiles[v] = c.trim().parse().unwrap_or_else(|_| usage());
                }
            }
            "--width" => width = num(),
            "--per-design" => per_design = num(),
            "--nodes" => search.nodes = num(),
            "--area-cost" => search.area_cost = num() as i64,
            "--growth" => search.growth = num(),
            "--noise" => search.noise = num() as i64,
            "--seed" => search.seed = num() as u64,
            "--out" => out = value,
            "--record-above" => record = Some(num() as i64),
            "--record-dir" => record_dir = PathBuf::from(value),
            _ => usage(),
        }
    }
    println!(
        "free order, towers topped at levels {tops:?}: beam width {width}, {per_design} coverings per \
         design, {} nodes, area cost {}, growth {}, noise {}, seed {}",
        search.nodes, search.area_cost, search.growth, search.noise, search.seed
    );
    let mut recorder = record.map(|above| Recorder::new(record_dir, above, how));
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
        if let Some(path) = recorder.as_mut().and_then(|r| r.offer(&play)) {
            println!(
                "[{:7.1}s] recorded {} ({})",
                start.elapsed().as_secs_f64(),
                play.score,
                path.display()
            );
        }
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
        verify(&Order::Free(search.tiles), &best),
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

#[cfg(test)]
mod tests {
    use super::*;
    use num9_core::board::Move;

    /// Moves written as value@level:row:col:rotation in a record file.
    fn read_steps(text: &str) -> Vec<Step> {
        text.split_whitespace()
            .filter(|t| t.contains('@') && t.starts_with(|c: char| c.is_ascii_digit()))
            .filter_map(|t| {
                let (v, rest) = t.split_once('@')?;
                let n: Vec<usize> = rest
                    .split(':')
                    .map(|x| x.parse().ok())
                    .collect::<Option<_>>()?;
                if n.len() != 4 {
                    return None;
                }
                Some((v.parse().ok()?, Move::new(n[0], n[1], n[2], n[3])))
            })
            .collect()
    }

    fn sample() -> Vec<Step> {
        let mut b = num9_core::Board::new();
        let mut steps = Vec::new();
        for v in [9u8, 8, 5, 3] {
            let m = b.find_valid_moves(v)[0];
            b.place(v, m).unwrap();
            steps.push((v, m));
        }
        steps
    }

    #[test]
    fn fingerprints_ignore_rotation_and_shift_but_not_changes() {
        let steps = sample();
        let shifted: Vec<Step> = steps
            .iter()
            .map(|&(v, m)| (v, Move::new(m.layer, m.i + 3, m.j + 7, m.rot)))
            .collect();
        assert_eq!(fingerprint(&steps), fingerprint(&shifted));
        // Every tile turned a quarter turn about a common point.
        let rotated: Vec<Step> = steps
            .iter()
            .map(|&(v, m)| {
                let turned: Vec<(i64, i64)> = shape(v, m.rot)
                    .cells()
                    .map(|(r, c)| ((m.j + c) as i64, 200 - (m.i + r) as i64))
                    .collect();
                let i = turned.iter().map(|c| c.0).min().unwrap() as usize;
                let j = turned.iter().map(|c| c.1).min().unwrap() as usize;
                (v, Move::new(m.layer, i, j, (m.rot + 1) % 4))
            })
            .collect();
        assert_eq!(fingerprint(&steps), fingerprint(&rotated));
        let mut changed = steps.clone();
        changed[3].1.j += 1;
        assert_ne!(fingerprint(&steps), fingerprint(&changed));
    }

    #[test]
    fn record_names_round_trip() {
        assert_eq!(
            parse_name("236-L6-00000000000000ff.txt"),
            Some((236, 6, 255))
        );
        assert_eq!(parse_name("INDEX.md"), None);
        assert_eq!(parse_name("free-order-236.txt"), None);
    }

    #[test]
    fn record_files_round_trip() {
        let play = Play {
            score: 0,
            steps: sample(),
        };
        assert_eq!(read_steps(&describe(&play)), play.steps);
        assert!(picture(&play.steps).starts_with("level 0"));
    }
}
