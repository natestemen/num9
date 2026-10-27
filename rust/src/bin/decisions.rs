//! How good each strategy's decisions are, one move at a time:
//!
//!     cargo run --release --bin decisions -- [--positions N] [--seed S]
//!         [--reference R] SPEC...
//!
//! Positions come from games played by keep_support, stopped at a random
//! tile. At each, every strategy picks a move; then a reference search, R
//! simulated games per move (on its own random numbers, shared by every
//! move) over a wide shortlist plus every move a strategy picked, says how
//! many expected points each pick gives away against the best of them.
//! That measures decisions directly, so differences show up with far fewer
//! games than whole-game averages need. A SPEC is a strategy with optional
//! settings (see `Strategy::from_spec`).

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use num9_core::board::{Board, Move};
use num9_core::game::shuffled_tiles;
use num9_core::piece::shape;
use num9_core::rollout::{candidate_moves, rollout_totals, MonteCarlo};
use num9_core::{Rng, Strategy};

/// For each strategy: (points lost, picked the best, seconds deciding).
type Verdicts = Vec<(f64, f64, f64)>;

fn usage() -> ! {
    eprintln!("usage: decisions [--positions N] [--seed S] [--reference R] SPEC...");
    std::process::exit(2)
}

/// A position from a keep_support game stopped at a random tile: the
/// board, and the tile to place.
fn position(seed: u64) -> (Board, u8) {
    let mut rng = Rng::new(seed);
    let tiles = shuffled_tiles(&mut rng);
    let stop = 2 + rng.below(17);
    let player = Strategy::from_name("keep_support").unwrap();
    let mut board = Board::new();
    for &v in &tiles[..stop] {
        let mv = player.choose(&board, v, &mut rng);
        board.place(v, mv).expect("legal move");
    }
    (board, tiles[stop])
}

/// The move index a pick stands for (for 0 and 8, rotations r and r + 2
/// are the same placement: use the first).
fn canonical(moves: &[Move], value: u8, k: usize) -> usize {
    let m = moves[k];
    if m.rot >= 2 && shape(value, 0).rows == shape(value, 2).rows {
        let twin = Move::new(m.layer, m.i, m.j, m.rot - 2);
        moves.iter().position(|&x| x == twin).unwrap_or(k)
    } else {
        k
    }
}

fn judge(seed: u64, strategies: &[Strategy], reference: usize) -> Option<Verdicts> {
    let (board, value) = position(seed);
    let moves = board.find_valid_moves(value);
    let mut picks: Vec<(Vec<usize>, f64)> = Vec::new();
    for s in strategies {
        let start = Instant::now();
        let d = s.decide(&board, value);
        let secs = start.elapsed().as_secs_f64();
        picks.push((
            d.candidates
                .iter()
                .map(|&k| canonical(&moves, value, k))
                .collect(),
            secs,
        ));
    }
    // The reference shortlist: wide, plus everything anyone picked.
    let mut shortlist = candidate_moves(&board, value, &moves, 10);
    for (p, _) in &picks {
        for &k in p {
            if !shortlist.contains(&k) {
                shortlist.push(k);
            }
        }
    }
    if shortlist.len() <= 1 {
        return None; // nothing to decide
    }
    let mc = MonteCarlo {
        rollouts: reference,
        seed: 0x5eed_0f0e_fe4e_ce00,
        ..MonteCarlo::default()
    };
    let totals = rollout_totals(&board, value, &moves, &shortlist, &mc);
    let worth = |k: usize| {
        totals[shortlist.iter().position(|&s| s == k).unwrap()] as f64 / reference as f64
    };
    let best = shortlist.iter().map(|&k| worth(k)).fold(f64::MIN, f64::max);
    Some(
        picks
            .iter()
            .map(|(p, secs)| {
                // Ties are broken at random in play: average over them.
                let lost = p.iter().map(|&k| best - worth(k)).sum::<f64>() / p.len() as f64;
                let hit = p.iter().filter(|&&k| worth(k) >= best).count() as f64 / p.len() as f64;
                (lost, hit, *secs)
            })
            .collect(),
    )
}

fn main() {
    let mut positions = 300;
    let mut seed = 1u64;
    let mut reference = 4096;
    let mut specs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--positions" => positions = value().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = value().parse().unwrap_or_else(|_| usage()),
            "--reference" => reference = value().parse().unwrap_or_else(|_| usage()),
            _ if a.starts_with("--") => usage(),
            _ => specs.push(a),
        }
    }
    let strategies: Vec<Strategy> = specs
        .iter()
        .map(|s| {
            Strategy::from_spec(s).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2)
            })
        })
        .collect();
    let results: Mutex<Vec<Verdicts>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Relaxed);
                if k >= positions {
                    break;
                }
                if let Some(v) = judge(seed + k as u64, &strategies, reference) {
                    let mut r = results.lock().unwrap();
                    r.push(v);
                    if r.len().is_multiple_of(50) {
                        eprintln!("{} positions judged", r.len());
                    }
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let n = results.len() as f64;
    println!(
        "`decisions --positions {positions} --seed {seed} --reference {reference}`: {} positions from keep_support games, \
         each move judged by {reference} simulated games; ± is one standard error.\n",
        results.len()
    );
    println!("| strategy | points lost a move | picks the best | ms a move |");
    println!("| --- | --- | --- | --- |");
    for (i, spec) in specs.iter().enumerate() {
        let lost: Vec<f64> = results.iter().map(|r| r[i].0).collect();
        let mean = lost.iter().sum::<f64>() / n;
        let se = (lost.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        let hits = results.iter().map(|r| r[i].1).sum::<f64>() / n;
        let ms = results.iter().map(|r| r[i].2).sum::<f64>() / n * 1000.0;
        println!(
            "| `{spec}` | {mean:.3} ± {se:.3} | {:.0}% | {ms:.1} |",
            hits * 100.0
        );
    }
}
