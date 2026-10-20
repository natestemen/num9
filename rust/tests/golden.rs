//! Checks the engine against fixtures recorded from the original Python
//! implementation (see `tests/golden/generate.py` for how and the format).

use num9_core::{Board, Move, SolidWeights, Strategy};

const MOVEGEN: &str = include_str!("golden/movegen.txt");
const STRATEGIES: &str = include_str!("golden/strategies.txt");

/// 64-bit FNV-1a folded over whole integers, as in `generate.py`.
fn fnv(ints: impl IntoIterator<Item = u64>) -> u64 {
    ints.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        (h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn digest(moves: &[Move]) -> String {
    let h = fnv(moves
        .iter()
        .flat_map(|m| [m.layer, m.i, m.j, m.rot].map(|x| x as u64)));
    format!("{}:{h:016x}", moves.len())
}

fn board_hash(board: &Board) -> String {
    let h = fnv(board
        .values()
        .into_iter()
        .enumerate()
        .filter(|&(_, v)| v != 0)
        .flat_map(|(k, v)| [k as u64, v as u64]));
    format!("{h:016x}")
}

fn records(text: &str) -> impl Iterator<Item = (usize, &str, Vec<&str>)> {
    text.lines().enumerate().filter_map(|(n, line)| {
        let mut words = line.split_whitespace();
        let kind = words.next()?;
        (!kind.starts_with('#')).then(|| (n + 1, kind, words.collect()))
    })
}

fn nums<T: std::str::FromStr>(words: &[&str]) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    words.iter().map(|w| w.parse().unwrap()).collect()
}

fn as_move(words: &[&str]) -> Move {
    let n: Vec<usize> = nums(words);
    Move::new(n[0], n[1], n[2], n[3])
}

#[test]
fn move_generation_matches_python() {
    let mut board = Board::new();
    let mut tiles: Vec<u8> = Vec::new();
    let mut ply = 0;
    let (mut games, mut positions) = (0, 0);
    for (line, kind, words) in records(MOVEGEN) {
        match kind {
            "game" => {
                board = Board::new();
                tiles = nums(&words);
                ply = 0;
                games += 1;
            }
            "pos" => {
                for (value, expected) in words.iter().enumerate() {
                    let got = digest(&board.find_valid_moves(value as u8));
                    assert_eq!(&got, expected, "line {line}: moves for piece {value}");
                }
                positions += 1;
            }
            "edges" => {
                let value = tiles[ply];
                let moves = board.find_valid_moves(value);
                let edges: Vec<u64> = moves
                    .iter()
                    .map(|&m| board.num_edges_touching(value, m) as u64)
                    .collect();
                let got = format!("{}:{:016x}", edges.len(), fnv(edges));
                assert_eq!(got, words[0], "line {line}: edge counts");
                assert_is_valid_move_consistent(&board, value, &moves, line);
            }
            "place" => {
                let value: u8 = words[0].parse().unwrap();
                assert_eq!(value, tiles[ply], "line {line}");
                board.place(value, as_move(&words[1..5])).unwrap();
                assert_eq!(
                    board.score(),
                    words[5].parse::<i64>().unwrap(),
                    "line {line}"
                );
                assert_eq!(board_hash(&board), words[6], "line {line}: board");
                ply += 1;
            }
            other => panic!("line {line}: unknown record {other:?}"),
        }
    }
    assert!(games >= 20 && positions >= 400, "fixtures look truncated");
}

/// `is_valid_move` must accept exactly the moves `find_valid_moves` lists,
/// for every placement in a window around the pieces.
fn assert_is_valid_move_consistent(board: &Board, value: u8, moves: &[Move], line: usize) {
    if board.pieces().is_empty() {
        return;
    }
    let legal: std::collections::HashSet<Move> = moves.iter().copied().collect();
    let (min_i, max_i, min_j, max_j) = board.layer_bbox(0).unwrap();
    let top = moves.iter().map(|m| m.layer).max().unwrap();
    for layer in 0..=(top + 1).min(num9_core::DEPTH - 1) {
        for rot in 0..4 {
            for i in min_i - 5..=max_i + 1 {
                for j in min_j - 5..=max_j + 1 {
                    let m = Move::new(layer, i, j, rot);
                    assert_eq!(
                        board.is_valid_move(value, m),
                        legal.contains(&m),
                        "line {line}: is_valid_move{m:?}"
                    );
                }
            }
        }
    }
}

fn parse_strategy(words: &[&str]) -> Strategy {
    let mut strategy = Strategy::from_name(words[0]).expect("known strategy");
    if words.len() > 1 {
        let mut w = SolidWeights::default();
        for kv in &words[1..] {
            let (key, value) = kv.split_once('=').unwrap();
            let value: f64 = value.parse().unwrap();
            match key {
                "fill_weight" => w.fill_weight = value,
                "hole_penalty" => w.hole_penalty = value,
                "support_weight" => w.support_weight = value,
                "edge_weight" => w.edge_weight = value,
                "height_weight" => w.height_weight = value,
                "aspect_penalty_weight" => w.aspect_penalty_weight = value,
                "low_height_penalty_weight" => w.low_height_penalty_weight = value,
                _ => panic!("unknown weight {key}"),
            }
        }
        strategy = Strategy::SolidBaseHighTop(w);
    }
    strategy
}

#[test]
fn strategy_decisions_match_python() {
    let mut strategy = Strategy::PlaceRandomly;
    let mut board = Board::new();
    let mut games = 0;
    for (line, kind, words) in records(STRATEGIES) {
        match kind {
            "strategy" => strategy = parse_strategy(&words),
            "game" => {
                board = Board::new();
                games += 1;
            }
            "decide" => {
                let value: u8 = words[0].parse().unwrap();
                let decision = strategy.decide(&board, value);
                let tied: Vec<Move> = decision.candidate_moves().collect();
                assert_eq!(
                    digest(&decision.moves),
                    words[1],
                    "line {line}: legal moves"
                );
                assert_eq!(digest(&tied), words[2], "line {line}: tie set");
                let chosen = as_move(&words[3..7]);
                assert!(tied.contains(&chosen), "line {line}: chosen move not tied");
                board.place(value, chosen).unwrap();
            }
            "end" => {
                assert_eq!(
                    board.score(),
                    words[0].parse::<i64>().unwrap(),
                    "line {line}"
                );
                assert_eq!(board_hash(&board), words[1], "line {line}: board");
            }
            other => panic!("line {line}: unknown record {other:?}"),
        }
    }
    assert!(games >= 50, "fixtures look truncated");
}
