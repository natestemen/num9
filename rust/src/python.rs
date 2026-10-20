//! Python bindings (`import num9_core`).

use std::collections::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::board::{Board, Move, DEPTH, HEIGHT, WIDTH};
use crate::game;
use crate::piece::shape;
use crate::rng::Rng;
use crate::strategies::Strategy;

create_exception!(num9_core, InvalidPlacementError, PyValueError);

type MoveTuple = (usize, usize, usize, usize);

fn to_tuple(m: Move) -> MoveTuple {
    (m.layer, m.i, m.j, m.rot)
}

fn check_value(value: u8) -> PyResult<u8> {
    if value > 9 {
        return Err(PyValueError::new_err(format!(
            "piece value must be in 0..=9, got {value}"
        )));
    }
    Ok(value)
}

/// NMBR 9 board. Mirrors `num9.Board`, but takes piece values (ints) rather
/// than `Piece` objects and explicit rotations.
#[pyclass(name = "Board", module = "num9_core")]
#[derive(Clone)]
struct PyBoard {
    inner: Board,
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        PyBoard {
            inner: Board::new(),
        }
    }

    /// All legal `(layer, i, j, rotation)` placements, in the same order as
    /// `num9.Board.find_valid_moves`.
    fn find_valid_moves(&self, value: u8) -> PyResult<Vec<MoveTuple>> {
        let value = check_value(value)?;
        Ok(self
            .inner
            .find_valid_moves(value)
            .into_iter()
            .map(to_tuple)
            .collect())
    }

    #[pyo3(signature = (value, location, rotation=0))]
    fn is_valid_move(
        &self,
        value: u8,
        location: (usize, usize, usize),
        rotation: usize,
    ) -> PyResult<bool> {
        let value = check_value(value)?;
        let (layer, i, j) = location;
        Ok(self
            .inner
            .is_valid_move(value, Move::new(layer, i, j, rotation)))
    }

    /// Place a piece. Raises `InvalidPlacementError` on overlap or if the piece
    /// would leave the board; does not otherwise check the rules.
    #[pyo3(signature = (value, location, rotation=0))]
    fn place(
        &mut self,
        value: u8,
        location: (usize, usize, usize),
        rotation: usize,
    ) -> PyResult<()> {
        let value = check_value(value)?;
        let (layer, i, j) = location;
        self.inner
            .place(value, Move::new(layer, i, j, rotation % 4))
            .map_err(|e| InvalidPlacementError::new_err(format!("cannot place a '{value}': {e}")))
    }

    #[pyo3(signature = (value, location, rotation=0))]
    fn num_edges_touching(
        &self,
        value: u8,
        location: (usize, usize, usize),
        rotation: usize,
    ) -> PyResult<u32> {
        let value = check_value(value)?;
        let (layer, i, j) = location;
        let s = shape(value, rotation);
        if layer >= DEPTH || i + s.h > HEIGHT || j + s.w > WIDTH {
            return Err(PyValueError::new_err("location is off the board"));
        }
        Ok(self
            .inner
            .num_edges_touching(value, Move::new(layer, i, j, rotation % 4)))
    }

    fn score(&self) -> i64 {
        self.inner.score()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    /// Cell values as a flat list of length `DEPTH * HEIGHT * WIDTH` (reshape
    /// to `(DEPTH, HEIGHT, WIDTH)` to compare with `num9.Board.board`).
    fn values(&self) -> Vec<i32> {
        self.inner.values()
    }

    /// Placed pieces as `(value, layer, i, j, rotation)`, in placement order.
    #[getter]
    fn pieces(&self) -> Vec<(u8, usize, usize, usize, usize)> {
        self.inner
            .pieces()
            .iter()
            .map(|p| (p.value, p.mv.layer, p.mv.i, p.mv.j, p.mv.rot))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.pieces().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<num9_core.Board pieces={} score={}>",
            self.inner.pieces().len(),
            self.inner.score()
        )
    }
}

fn strategy(name: &str, weights: Option<HashMap<String, f64>>) -> PyResult<Strategy> {
    let mut s = Strategy::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown strategy {name:?}; expected one of {:?}",
            Strategy::NAMES
        ))
    })?;
    if let Some(weights) = weights {
        let Strategy::SolidBaseHighTop(w) = &mut s else {
            return Err(PyValueError::new_err(
                "weights only apply to solid_base_high_top",
            ));
        };
        for (key, value) in weights {
            let slot = match key.as_str() {
                "fill_weight" => &mut w.fill_weight,
                "hole_penalty" => &mut w.hole_penalty,
                "support_weight" => &mut w.support_weight,
                "edge_weight" => &mut w.edge_weight,
                "height_weight" => &mut w.height_weight,
                "aspect_penalty_weight" => &mut w.aspect_penalty_weight,
                "low_height_penalty_weight" => &mut w.low_height_penalty_weight,
                _ => return Err(PyValueError::new_err(format!("unknown weight {key:?}"))),
            };
            *slot = value;
        }
    }
    Ok(s)
}

/// A strategy's deterministic step: `(moves, candidates)`, where the Python
/// strategy would `random.choice` among `moves[k] for k in candidates`.
#[pyfunction]
#[pyo3(signature = (name, board, value, weights=None))]
fn decide(
    py: Python<'_>,
    name: &str,
    board: &PyBoard,
    value: u8,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<(Vec<MoveTuple>, Vec<usize>)> {
    let value = check_value(value)?;
    let s = strategy(name, weights)?;
    let b = board.inner.clone();
    let d = py.detach(move || s.decide(&b, value));
    Ok((d.moves.into_iter().map(to_tuple).collect(), d.candidates))
}

/// Pick a move with a strategy, breaking ties with a PRNG seeded by `seed`.
#[pyfunction]
#[pyo3(signature = (name, board, value, seed=0, weights=None))]
fn choose_move(
    py: Python<'_>,
    name: &str,
    board: &PyBoard,
    value: u8,
    seed: u64,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<MoveTuple> {
    let value = check_value(value)?;
    let s = strategy(name, weights)?;
    let b = board.inner.clone();
    Ok(to_tuple(
        py.detach(move || s.choose(&b, value, &mut Rng::new(seed))),
    ))
}

/// Play one game and return the final board. `tiles` defaults to a shuffle
/// of the twenty tiles drawn from `seed`.
#[pyfunction]
#[pyo3(signature = (name, seed=0, tiles=None, weights=None))]
fn play_game(
    py: Python<'_>,
    name: &str,
    seed: u64,
    tiles: Option<Vec<u8>>,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<PyBoard> {
    let s = strategy(name, weights)?;
    if let Some(t) = &tiles {
        for &v in t {
            check_value(v)?;
        }
    }
    let board = py.detach(move || {
        let mut rng = Rng::new(seed);
        match tiles {
            Some(t) => game::play_tiles(&s, &t, &mut rng),
            None => game::play_game(&s, &mut rng),
        }
    });
    Ok(PyBoard { inner: board })
}

/// Scores of `games` games (game `k` uses seed `seed + k`), run on `threads`
/// OS threads (default: all cores) with the GIL released.
#[pyfunction]
#[pyo3(signature = (name, games, seed=0, threads=None, weights=None))]
fn play_games(
    py: Python<'_>,
    name: &str,
    games: usize,
    seed: u64,
    threads: Option<usize>,
    weights: Option<HashMap<String, f64>>,
) -> PyResult<Vec<i64>> {
    let s = strategy(name, weights)?;
    let threads =
        threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    Ok(py.detach(move || game::play_games(&s, games, seed, threads)))
}

/// `Piece(value)` rotated `rotation` times, as `Piece.array` returns it.
#[pyfunction]
#[pyo3(signature = (value, rotation=0))]
fn piece_array(value: u8, rotation: usize) -> PyResult<Vec<Vec<i32>>> {
    let value = check_value(value)?;
    let fill = if value == 0 { 10 } else { value as i32 };
    Ok(shape(value, rotation)
        .grid()
        .into_iter()
        .map(|row| row.into_iter().map(|c| c as i32 * fill).collect())
        .collect())
}

#[pymodule]
fn num9_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add(
        "InvalidPlacementError",
        m.py().get_type::<InvalidPlacementError>(),
    )?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    m.add_function(wrap_pyfunction!(choose_move, m)?)?;
    m.add_function(wrap_pyfunction!(play_game, m)?)?;
    m.add_function(wrap_pyfunction!(play_games, m)?)?;
    m.add_function(wrap_pyfunction!(piece_array, m)?)?;
    m.add("STRATEGIES", Strategy::NAMES.to_vec())?;
    m.add("DEPTH", DEPTH)?;
    m.add("HEIGHT", HEIGHT)?;
    m.add("WIDTH", WIDTH)?;
    Ok(())
}
