//! Searching for the highest possible score by building towers top-down.
//!
//! When the order of the tiles is free, any finished stack in which every
//! tile above the ground is fully supported, rests on at least two tiles,
//! and every level is one connected group can be built (ground first, then
//! each level, each grown outward from one tile). So the "maximum possible
//! score" is a question about stacks, and a stack can be designed from the
//! top down: put a tile on top, then choose tiles on the level below that
//! cover it exactly (on at least two of them), then cover those, and so on
//! to the ground. Tiles left over go wherever they score most.
//!
//! The search is a beam over designs, one level at a time: each level's
//! coverings are found by branch and bound (keeping those adding the least
//! area, since a tower that widens too fast runs out of tiles before the
//! ground), and designs are ranked by points so far against the area still
//! to cover and the tiles still needed. Finished designs are rebuilt as a
//! normal game and checked move by move with the engine's rules.

use crate::board::{Board, Move, HEIGHT, WIDTH};
use crate::nrpa::Step;
use crate::piece::shape;
use crate::rng::Rng;

/// Side of the design grid (cells); the tower is built around its centre.
const G: usize = 48;

/// A tile in a design: value, level, top-left corner on the grid, rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub value: u8,
    pub layer: usize,
    pub i: usize,
    pub j: usize,
    pub rot: usize,
}

/// One design decision: a tile to add (its level is the one being built).
pub type Choice = (u8, usize, usize, usize); // value, rot, i, j

#[derive(Clone)]
struct Design {
    /// Per level, which tile (index + 1) covers each cell (0 = empty).
    owner: Vec<Vec<u8>>,
    tiles: Vec<Tile>,
    left: [u8; 10],
    /// The level being built (covering the level above it).
    layer: usize,
    done: bool,
    dead: bool,
}

fn cells(t: &Tile) -> impl Iterator<Item = (usize, usize)> + '_ {
    shape(t.value, t.rot)
        .cells()
        .map(move |(r, c)| (t.i + r, t.j + c))
}

impl Design {
    fn new(top: usize, left: [u8; 10]) -> Design {
        Design {
            owner: vec![vec![0; G * G]; top + 1],
            tiles: Vec::new(),
            left,
            layer: top,
            done: false,
            dead: false,
        }
    }

    /// The first cell of the level above that isn't covered yet.
    /// The uncovered cell of the level above to cover next: the most hemmed
    /// in (fewest free neighbours on this level), so gaps that only a few
    /// tiles could fill are dealt with before they become impossible.
    fn target(&self) -> Option<(usize, usize)> {
        let above = &self.owner[self.layer + 1];
        let here = &self.owner[self.layer];
        let free = |a: usize, b: usize| a < G && b < G && here[a * G + b] == 0;
        (0..G * G)
            .filter(|&k| above[k] != 0 && here[k] == 0)
            .map(|k| (k / G, k % G))
            .min_by_key(|&(a, b)| {
                [
                    (a.wrapping_sub(1), b),
                    (a + 1, b),
                    (a, b.wrapping_sub(1)),
                    (a, b + 1),
                ]
                .iter()
                .filter(|&&(x, y)| free(x, y))
                .count()
            })
    }

    /// How many cells of the level above a choice would cover.
    fn covers(&self, (v, rot, i, j): Choice) -> usize {
        if self.tiles.is_empty() {
            return 1;
        }
        let above = &self.owner[self.layer + 1];
        let t = Tile {
            value: v,
            layer: self.layer,
            i,
            j,
            rot,
        };
        cells(&t).filter(|&(a, b)| above[a * G + b] != 0).count()
    }

    fn fits(&self, t: &Tile) -> bool {
        let s = shape(t.value, t.rot);
        if t.i + s.h > G || t.j + s.w > G {
            return false;
        }
        let here = &self.owner[t.layer];
        if cells(t).any(|(a, b)| here[a * G + b] != 0) {
            return false;
        }
        // A tile above must not end up resting on this tile alone.
        if t.layer + 1 < self.owner.len() {
            let above = &self.owner[t.layer + 1];
            let mine: Vec<(usize, usize)> = cells(t).collect();
            let mut seen = Vec::new();
            for &(a, b) in &mine {
                let id = above[a * G + b];
                if id != 0 && !seen.contains(&id) {
                    seen.push(id);
                    let other = &self.tiles[id as usize - 1];
                    if cells(other).all(|c| mine.contains(&c)) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn choices(&self) -> Vec<Choice> {
        let mut out = Vec::new();
        if self.done || self.dead {
            return out;
        }
        let values = (0..10u8).filter(|&v| self.left[v as usize] > 0);
        if self.tiles.is_empty() {
            for v in values {
                for rot in rotations(v) {
                    out.push((v, rot, G / 2 - 2, G / 2 - 2));
                }
            }
            return out;
        }
        let Some((ti, tj)) = self.target() else {
            return out;
        };
        for v in values {
            for rot in rotations(v) {
                for (r, c) in shape(v, rot).cells() {
                    let (Some(i), Some(j)) = (ti.checked_sub(r), tj.checked_sub(c)) else {
                        continue;
                    };
                    let t = Tile {
                        value: v,
                        layer: self.layer,
                        i,
                        j,
                        rot,
                    };
                    if self.fits(&t) {
                        out.push((v, rot, i, j));
                    }
                }
            }
        }
        out
    }

    fn connected(&self, layer: usize) -> bool {
        let ids: Vec<usize> = (0..self.tiles.len())
            .filter(|&k| self.tiles[k].layer == layer)
            .collect();
        if ids.len() <= 1 {
            return true;
        }
        let owner = &self.owner[layer];
        let mut seen = vec![ids[0]];
        let mut k = 0;
        while k < seen.len() {
            let t = self.tiles[seen[k]];
            for (a, b) in cells(&t) {
                for (x, y) in [
                    (a.wrapping_sub(1), b),
                    (a + 1, b),
                    (a, b.wrapping_sub(1)),
                    (a, b + 1),
                ] {
                    if x < G && y < G {
                        let id = owner[x * G + y];
                        if id != 0 && !seen.contains(&(id as usize - 1)) {
                            seen.push(id as usize - 1);
                        }
                    }
                }
            }
            k += 1;
        }
        seen.len() == ids.len()
    }

    fn apply(&mut self, (v, rot, i, j): Choice) {
        let t = Tile {
            value: v,
            layer: self.layer,
            i,
            j,
            rot,
        };
        let id = self.tiles.len() as u8 + 1;
        for (a, b) in cells(&t) {
            self.owner[t.layer][a * G + b] = id;
        }
        self.tiles.push(t);
        self.left[v as usize] -= 1;
        if self.tiles.len() == 1 {
            // The top tile: start building the level below it.
            if self.layer == 0 {
                self.done = true;
            } else {
                self.layer -= 1;
            }
            return;
        }
        if self.target().is_none() {
            if !self.connected(self.layer) {
                self.dead = true;
            } else if self.layer == 0 {
                self.done = true;
            } else {
                self.layer -= 1;
            }
        }
    }
}

/// Rotations that give distinct shapes.
fn rotations(v: u8) -> std::ops::Range<usize> {
    if shape(v, 0).rows == shape(v, 2).rows {
        0..2
    } else {
        0..4
    }
}

/// Rotate tiles a quarter turn clockwise (cells (r, c) -> (c, -r)), then
/// shift so the top-left of everything is at `origin`.
fn rotate_and_place(tiles: &[Tile], quarter_turns: usize, origin: (usize, usize)) -> Vec<Tile> {
    let mut out: Vec<(Tile, Vec<(i64, i64)>)> = tiles
        .iter()
        .map(|t| (*t, cells(t).map(|(a, b)| (a as i64, b as i64)).collect()))
        .collect();
    for _ in 0..quarter_turns {
        for (t, cs) in out.iter_mut() {
            for c in cs.iter_mut() {
                *c = (c.1, -c.0);
            }
            t.rot = (t.rot + 1) % 4;
        }
    }
    let min_i = out
        .iter()
        .flat_map(|(_, cs)| cs.iter().map(|c| c.0))
        .min()
        .unwrap();
    let min_j = out
        .iter()
        .flat_map(|(_, cs)| cs.iter().map(|c| c.1))
        .min()
        .unwrap();
    out.into_iter()
        .map(|(mut t, cs)| {
            t.i = (cs.iter().map(|c| c.0).min().unwrap() - min_i) as usize + origin.0;
            t.j = (cs.iter().map(|c| c.1).min().unwrap() - min_j) as usize + origin.1;
            t
        })
        .collect()
}

/// Turn a finished design into a legal game: ground tiles first, each level
/// grown outward from one tile, then leftover tiles wherever they score
/// most. Returns the steps, or why the design can't be built.
pub fn assemble(tiles: &[Tile], left: [u8; 10]) -> Result<Vec<Step>, String> {
    // The engine starts every game with a rotation-0 tile in the middle: turn
    // the design so some ground tile has rotation 0.
    let turns = (0..4)
        .find(|&q| tiles.iter().any(|t| t.layer == 0 && (t.rot + q) % 4 == 0))
        .ok_or("no ground tile")?;
    let mut placed = rotate_and_place(tiles, turns, (40, 40));
    let first = placed
        .iter()
        .position(|t| t.layer == 0 && t.rot == 0)
        .unwrap();
    let centre = Board::center_move();
    let (di, dj) = (
        centre.i as i64 - placed[first].i as i64,
        centre.j as i64 - placed[first].j as i64,
    );
    for t in placed.iter_mut() {
        t.i = (t.i as i64 + di) as usize;
        t.j = (t.j as i64 + dj) as usize;
        if t.i + 4 > HEIGHT || t.j + 4 > WIDTH {
            return Err("design too large".into());
        }
    }
    // Build order: by level; within a level, outward from a tile that is
    // adjacent to earlier ones (breadth first), the very first being `first`.
    let mut board = Board::new();
    let mut steps = Vec::new();
    let mut todo: Vec<usize> = (0..placed.len()).collect();
    todo.retain(|&k| k != first);
    let mv = |t: &Tile| Move::new(t.layer, t.i, t.j, t.rot);
    board
        .place(placed[first].value, mv(&placed[first]))
        .map_err(|e| e.to_string())?;
    if mv(&placed[first]) != centre {
        return Err("first tile not centred".into());
    }
    steps.push((placed[first].value, mv(&placed[first])));
    let top = placed.iter().map(|t| t.layer).max().unwrap();
    for layer in 0..=top {
        loop {
            let next = todo.iter().position(|&k| {
                placed[k].layer == layer && board.is_valid_move(placed[k].value, mv(&placed[k]))
            });
            let Some(p) = next else { break };
            let k = todo.remove(p);
            board
                .place(placed[k].value, mv(&placed[k]))
                .map_err(|e| e.to_string())?;
            steps.push((placed[k].value, mv(&placed[k])));
        }
    }
    if !todo.is_empty() {
        return Err(format!("{} tiles could not be placed in order", todo.len()));
    }
    // Leftovers, highest first, wherever they score most (then most edges).
    let mut rest: Vec<u8> = (0..10u8)
        .flat_map(|v| std::iter::repeat_n(v, left[v as usize] as usize))
        .collect();
    rest.sort_unstable_by(|a, b| b.cmp(a));
    for v in rest {
        let moves = board.find_valid_moves(v);
        let m = *moves
            .iter()
            .max_by_key(|m| (m.layer, board.num_edges_touching(v, **m)))
            .ok_or("no legal move for a leftover tile")?;
        board.place(v, m).map_err(|e| e.to_string())?;
        steps.push((v, m));
    }
    Ok(steps)
}

/// Score of a finished design once assembled (None if it can't be built).
fn evaluate(d: &Design) -> Option<(i64, Vec<Step>)> {
    let steps = assemble(&d.tiles, d.left).ok()?;
    let mut board = Board::new();
    for &(v, m) in &steps {
        board.place(v, m).ok()?;
    }
    Some((board.score(), steps))
}

impl Design {
    /// Rough number of tiles the levels still to build will need, if each
    /// is `growth` cells wider than the one above and tiles average 8.5.
    fn tiles_needed(&self, growth: usize) -> i64 {
        let mut area = self.bottom_area();
        let mut need = 0.0;
        for _ in 0..=self.layer {
            area += growth;
            need += (area as f64 / 8.5).ceil();
        }
        need as i64
    }

    /// Cells covered on `layer`.
    fn level_area(&self, layer: usize) -> usize {
        self.owner[layer].iter().filter(|&&x| x != 0).count()
    }

    /// Points of the tiles designed so far.
    fn points(&self) -> i64 {
        self.tiles
            .iter()
            .map(|t| t.value as i64 * t.layer as i64)
            .sum()
    }

    /// Cells on the level currently being built's level above (what must be
    /// covered next), or on the bottom level once a level is finished.
    fn bottom_area(&self) -> usize {
        self.owner[(self.layer + 1).min(self.owner.len() - 1)]
            .iter()
            .filter(|&&x| x != 0)
            .count()
    }

    /// Take back the last tile added, restoring the state saved before it.
    fn undo(&mut self, saved: (usize, bool, bool)) {
        let t = self.tiles.pop().expect("a tile to undo");
        for (a, b) in cells(&t) {
            self.owner[t.layer][a * G + b] = 0;
        }
        self.left[t.value as usize] += 1;
        (self.layer, self.done, self.dead) = saved;
    }

    /// The (up to) `keep` ways to finish covering the current level that
    /// use the least area on it, each as the resulting design.
    /// Branch and bound: a partial covering is abandoned once its area plus
    /// the cells still uncovered can't beat the `keep`-th best found.
    fn coverings(&self, keep: usize, nodes: &mut usize) -> Vec<Design> {
        let mut best: Vec<(usize, Design)> = Vec::new();
        let layer = self.layer;
        self.clone().cover(layer, keep, nodes, &mut best);
        best.into_iter().map(|(_, d)| d).collect()
    }

    /// Cells of the level above not yet covered on the current level.
    fn uncovered(&self) -> usize {
        let above = &self.owner[self.layer + 1];
        let here = &self.owner[self.layer];
        (0..G * G)
            .filter(|&k| above[k] != 0 && here[k] == 0)
            .count()
    }

    fn cover(
        &mut self,
        layer: usize,
        keep: usize,
        nodes: &mut usize,
        best: &mut Vec<(usize, Design)>,
    ) {
        if *nodes == 0 {
            return;
        }
        *nodes -= 1;
        let bound = |best: &Vec<(usize, Design)>| {
            if best.len() < keep {
                usize::MAX
            } else {
                best.last().unwrap().0
            }
        };
        if self.level_area(layer) + self.uncovered() >= bound(best) {
            return;
        }
        // Snug tiles first (least overhang beyond the level above), then
        // high values.
        let mut options = self.choices();
        options.sort_by_key(|&c| {
            let waste = shape(c.0, c.1).tiles as usize - self.covers(c);
            (waste, std::cmp::Reverse(c.0))
        });
        for c in options {
            let saved = (self.layer, self.done, self.dead);
            self.apply(c);
            if !self.dead {
                if self.done || self.layer != layer {
                    let a = self.level_area(layer);
                    if a < bound(best) {
                        let pos = best.partition_point(|(x, _)| *x <= a);
                        best.insert(pos, (a, self.clone()));
                        best.truncate(keep);
                    }
                } else {
                    self.cover(layer, keep, nodes, best);
                }
            }
            self.undo(saved);
            if *nodes == 0 {
                return;
            }
        }
    }
}

/// Settings for the top-down search.
#[derive(Clone, Debug)]
pub struct TowerSearch {
    /// Beam ranking: half a point lost per this many cells left to cover.
    pub area_cost: i64,
    /// Search budget (nodes) for the coverings of each design's level.
    pub nodes: usize,
    /// Expected widening per level, for estimating tiles still needed.
    pub growth: usize,
    /// Height of the top tile (its level).
    pub top: usize,
    /// How many of each value are available.
    pub tiles: [u8; 10],
    /// Random jitter (0 = none) added to the beam ranking, so repeated
    /// runs with different seeds explore different designs.
    pub noise: i64,
    pub seed: u64,
}

/// A designed play: its score and the steps of the game that builds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub score: i64,
    pub steps: Vec<Step>,
    pub choices: Vec<Choice>,
}

impl TowerSearch {
    /// Beam search: start from every top tile, repeatedly replace each
    /// design by up to `per_design` ways of covering its bottom level, and
    /// keep the `width` most promising (points so far, with designs that
    /// can't reach the ground with the tiles left dropped). Every finished
    /// design is assembled and scored exactly; returns the best.
    /// `report` is called for every finished design (not only improvements).
    pub fn beam(
        &self,
        width: usize,
        per_design: usize,
        mut report: impl FnMut(&Found),
    ) -> Option<Found> {
        let mut rng = Rng::new(self.seed);
        let mut beam: Vec<(Design, Vec<Choice>)> = Vec::new();
        let root = Design::new(self.top, self.tiles);
        for c in root.choices() {
            let mut d = root.clone();
            d.apply(c);
            beam.push((d, vec![c]));
        }
        let mut best: Option<Found> = None;
        while !beam.is_empty() {
            let mut children: Vec<(Design, Vec<Choice>)> = Vec::new();
            for (d, made) in &beam {
                let mut nodes = self.nodes;
                // The coverings adding the least area (narrow towers reach
                // higher), most points first among equals.
                let mut found = d.coverings(per_design, &mut nodes);
                found.sort_by_key(|c| (c.level_area(d.layer), std::cmp::Reverse(c.points())));
                for child in found {
                    let mut choices = made.clone();
                    choices.extend(
                        child.tiles[d.tiles.len()..]
                            .iter()
                            .map(|t| (t.value, t.rot, t.i, t.j)),
                    );
                    if child.done {
                        if let Some((score, steps)) = evaluate(&child) {
                            let f = Found {
                                score,
                                steps,
                                choices,
                            };
                            report(&f);
                            if best.as_ref().is_none_or(|b| score > b.score) {
                                best = Some(f);
                            }
                        }
                        continue;
                    }
                    // Each level below needs at least this many tiles.
                    let per_level = child.bottom_area().div_ceil(10).max(2);
                    let left: usize = child.left.iter().map(|&n| n as usize).sum();
                    if left < per_level * (child.layer + 1) {
                        continue;
                    }
                    children.push((child, choices));
                }
            }
            // Most points, less a cost for the area still to be covered and
            // a heavy one for needing more tiles than are left.
            let mut keyed: Vec<(i64, (Design, Vec<Choice>))> = children
                .into_iter()
                .map(|c| {
                    let d = &c.0;
                    let left: i64 = d.left.iter().map(|&n| n as i64).sum();
                    let short = (d.tiles_needed(self.growth) - left).max(0);
                    let jitter = if self.noise > 0 {
                        rng.below(self.noise as usize + 1) as i64
                    } else {
                        0
                    };
                    let key = d.points() * 2 - self.area_cost * d.bottom_area() as i64 - 40 * short
                        + jitter;
                    (key, c)
                })
                .collect();
            keyed.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
            let mut children: Vec<(Design, Vec<Choice>)> =
                keyed.into_iter().map(|(_, c)| c).collect();
            children.truncate(width);
            beam = children;
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nrpa::{verify, Order, Play};

    #[test]
    fn beam_search_builds_tall_legal_towers() {
        let s = TowerSearch {
            area_cost: 8,
            nodes: 20_000,
            growth: 4,
            top: 4,
            tiles: [2; 10],
            noise: 0,
            seed: 0,
        };
        let f = s.beam(10, 4, |_| {}).expect("a design");
        let play = Play {
            score: f.score,
            steps: f.steps.clone(),
        };
        assert_eq!(verify(&Order::standard_free(), &play), Ok(f.score));
        assert!(f.steps.iter().any(|s| s.1.layer == 4), "reaches level 4");
    }

    #[test]
    fn rotating_a_design_keeps_it_the_same_shape() {
        let t = Tile {
            value: 7,
            layer: 0,
            i: 5,
            j: 9,
            rot: 1,
        };
        let mine: Vec<_> = cells(&t).collect();
        for q in 0..4 {
            let r = rotate_and_place(&[t], q, (0, 0))[0];
            let theirs: Vec<_> = cells(&r).collect();
            assert_eq!(theirs.len(), mine.len());
            assert_eq!(r.rot, (t.rot + q) % 4);
        }
        // Four quarter turns give back the original placement (up to shift).
        assert_eq!(rotate_and_place(&[t], 4, (5, 9))[0], t);
    }
}
