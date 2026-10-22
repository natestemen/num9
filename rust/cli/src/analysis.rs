//! After a game: search for the best play of the deal just played.
//!
//! Once the game is over the tile order is no longer a secret, so the
//! engine's solver (nested Monte Carlo search, see `num9_core::solver`) can
//! look for the highest-scoring play of exactly this deal. It runs in stages
//! of increasing depth on a background thread, reporting every improvement,
//! and stops as soon as the analysis is dropped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

use num9_core::solver::{Nested, Solution};

/// Search stages as (nesting level, candidates per board level): quick
/// passes first so there is an answer within a second, then deeper ones.
pub const STAGES: &[(usize, usize)] = &[(1, 4), (2, 4), (2, 8), (3, 8)];

enum Update {
    Improved(Solution),
    StageDone,
}

pub struct Analysis {
    rx: Receiver<Update>,
    stop: Arc<AtomicBool>,
    /// The best play found so far.
    pub best: Option<Solution>,
    pub stages_done: usize,
    pub stages: usize,
}

impl Analysis {
    pub fn start(tiles: Vec<u8>, stages: &[(usize, usize)]) -> Analysis {
        let (tx, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stages_list = stages.to_vec();
        let flag = stop.clone();
        thread::spawn(move || {
            // Leave a core for the interface.
            let threads = thread::available_parallelism()
                .map_or(1, |n| n.get())
                .saturating_sub(1)
                .max(1);
            let mut best = i64::MIN;
            for (k, &(level, per_layer)) in stages_list.iter().enumerate() {
                if flag.load(Ordering::Relaxed) {
                    return;
                }
                let nested = Nested {
                    level,
                    per_layer,
                    threads,
                    ..Nested::default()
                };
                let sol = nested.solve(&tiles, k as u64, &flag, |_, partial| {
                    // Each partial result is already a complete, legal play.
                    if partial.score > best {
                        best = partial.score;
                        let _ = tx.send(Update::Improved(partial.clone()));
                    }
                });
                if sol.score > best {
                    best = sol.score;
                    let _ = tx.send(Update::Improved(sol));
                }
                if flag.load(Ordering::Relaxed) || tx.send(Update::StageDone).is_err() {
                    return;
                }
            }
        });
        Analysis {
            rx,
            stop,
            best: None,
            stages_done: 0,
            stages: stages.len(),
        }
    }

    /// Collect progress; returns whether anything changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(update) = self.rx.try_recv() {
            match update {
                Update::Improved(sol) => {
                    if self.best.as_ref().is_none_or(|b| sol.score > b.score) {
                        self.best = Some(sol);
                    }
                }
                Update::StageDone => self.stages_done += 1,
            }
            changed = true;
        }
        changed
    }

    pub fn finished(&self) -> bool {
        self.stages_done >= self.stages
    }
}

impl Drop for Analysis {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num9_core::game::shuffled_tiles;
    use num9_core::solver::verify;
    use num9_core::Rng;
    use std::time::{Duration, Instant};

    #[test]
    fn finds_a_legal_play_that_improves_with_each_stage() {
        let tiles = shuffled_tiles(&mut Rng::new(42));
        let mut a = Analysis::start(tiles.clone(), &[(1, 2), (1, 4), (2, 4)]);
        let start = Instant::now();
        let mut scores = Vec::new();
        while !a.finished() {
            if a.poll() {
                scores.push(a.best.as_ref().unwrap().score);
            }
            assert!(start.elapsed() < Duration::from_secs(60));
            thread::sleep(Duration::from_millis(5));
        }
        a.poll();
        let best = a.best.as_ref().unwrap();
        assert_eq!(verify(&tiles, &best.moves), Ok(best.score));
        assert!(scores.windows(2).all(|w| w[0] <= w[1]), "{scores:?}");
    }
}
