//! Strategies head to head on the same deals:
//!
//!     cargo run --release --bin arena -- [--games N] [--seed S] [--threads T]
//!         [--baseline SPEC] [--readme] SPEC...
//!
//! Every strategy plays deals seed, seed + 1, ... (the same tiles in the same
//! order for all of them), so the difference between two strategies on the
//! same deals is measured far more precisely than their separate averages.
//! Prints a markdown table: each strategy's average (± standard error), and
//! its average difference from the baseline on the same deals (± standard
//! error) with how often it won, tied and lost. A SPEC is a strategy name with
//! optional settings, e.g. `monte_carlo:policy=keep_support,rollouts=128`
//! (see `Strategy::from_spec`). With --readme the table also replaces the one
//! between the arena markers in README.md.

use std::time::Instant;

use num9_core::game::play_games;
use num9_core::Strategy;

struct Run {
    spec: String,
    scores: Vec<i64>,
    seconds: f64,
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

fn usage() -> ! {
    eprintln!(
        "usage: arena [--games N] [--seed S] [--threads T] [--baseline SPEC] [--readme] SPEC..."
    );
    std::process::exit(2)
}

fn main() {
    let mut games = 1000;
    let mut seed = 1;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut baseline = String::from("low_to_base");
    let mut readme = false;
    let mut specs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--games" => games = value().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = value().parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = value().parse().unwrap_or_else(|_| usage()),
            "--baseline" => baseline = value(),
            "--readme" => readme = true,
            _ if a.starts_with("--") => usage(),
            _ => specs.push(a),
        }
    }
    if !specs.contains(&baseline) {
        specs.insert(0, baseline.clone());
    }
    let runs: Vec<Run> = specs
        .iter()
        .map(|spec| {
            let strategy = Strategy::from_spec(spec).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2)
            });
            let start = Instant::now();
            let scores = play_games(&strategy, games, seed, threads);
            let seconds = start.elapsed().as_secs_f64();
            eprintln!("{spec}: done in {seconds:.1}s");
            Run {
                spec: spec.clone(),
                scores,
                seconds,
            }
        })
        .collect();
    let base = runs.iter().find(|r| r.spec == baseline).unwrap();
    let mut table = format!(
        "`arena --games {games} --seed {seed}`: {games} deals, the same for every strategy; ± is one standard error.\n\n\
         | strategy | avg | vs {baseline} on the same deals | wins / ties / losses | games/s ({threads} threads) |\n\
         | --- | --- | --- | --- | --- |\n"
    );
    for r in &runs {
        let (avg, se) = mean_se(&r.scores.iter().map(|&s| s as f64).collect::<Vec<_>>());
        let versus = if r.spec == baseline {
            ("".to_string(), "".to_string())
        } else {
            let d: Vec<f64> = r
                .scores
                .iter()
                .zip(&base.scores)
                .map(|(a, b)| (a - b) as f64)
                .collect();
            let (m, e) = mean_se(&d);
            let wins = d.iter().filter(|&&x| x > 0.0).count();
            let ties = d.iter().filter(|&&x| x == 0.0).count();
            (
                format!("{m:+.2} ± {e:.2}"),
                format!("{wins} / {ties} / {}", d.len() - wins - ties),
            )
        };
        table += &format!(
            "| `{}` | {avg:.2} ± {se:.2} | {} | {} | {:.1} |\n",
            r.spec,
            versus.0,
            versus.1,
            games as f64 / r.seconds
        );
    }
    print!("{table}");
    if readme {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../README.md");
        let text = std::fs::read_to_string(path).expect("read README.md");
        let (start, end) = ("<!-- arena:start -->\n", "<!-- arena:end -->");
        let (a, b) = (text.find(start), text.find(end));
        let (Some(a), Some(b)) = (a, b) else {
            eprintln!("README.md has no arena markers");
            std::process::exit(1)
        };
        let updated = format!("{}{start}{table}{}", &text[..a], &text[b..]);
        std::fs::write(path, updated).expect("write README.md");
        eprintln!("README.md updated");
    }
}
