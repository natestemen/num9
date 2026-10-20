//! Fast NMBR 9 engine, ported from the original Python `num9` package.
//!
//! The move generator reproduces the Python `Board.find_valid_moves` exactly
//! (same moves, same order) and the strategies make the same decisions as the
//! Python ones; `tests/golden.rs` checks both against fixtures recorded from
//! the Python engine before it was removed.

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
