//! Fast NMBR 9 engine: a Rust port of the `num9` Python package.
//!
//! The move generator reproduces `num9.Board.find_valid_moves` exactly (same
//! moves, same order) and the strategies reproduce `num9.strategies`; the
//! Python test-suite in `tests/test_rust_equivalence.py` checks both.

pub mod board;
pub mod game;
pub mod piece;
pub mod rng;
pub mod strategies;

#[cfg(feature = "python")]
mod python;

pub use board::{Board, Move, PlaceError, DEPTH, HEIGHT, WIDTH};
pub use rng::Rng;
pub use strategies::{Decision, SolidWeights, Strategy};
