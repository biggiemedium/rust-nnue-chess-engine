
//! Material evaluation.
//!
//! Evaluates the relative value of pieces on the board.
//!
//! Positive score = advantage for White
//! Negative score = advantage for Black


use crate::board::{Color, PieceType};
use crate::board::shakmaty_adapter::ShakmatyAdapter;
use crate::evaluation::evaluator::EvaluationTerm;

pub struct MaterialEvaluator;

/// Centipawn value of each piece type
/// Weights: https://www.chessprogramming.org/Material

/// King not included because each side has exactly one,
/// If the king is gone it implies that the game has been lost or won
const PIECE_VALUES: [(PieceType, i32); 5] = [
    (PieceType::Pawn, 100),
    (PieceType::Knight, 320),
    (PieceType::Bishop, 330), // biships are *SLIGHTLY* Better than nights, therefore value +10
    (PieceType::Rook, 500),
    (PieceType::Queen, 900),
];

impl MaterialEvaluator {
    pub fn new() -> Self {
        Self
    }
}

// Default state / restarted state
impl Default for MaterialEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationTerm for MaterialEvaluator {
    fn evaluate(&self, board: &ShakmatyAdapter) -> i32 {
        // For each piece type -> take White's count minus Black's count
        // and weight it by that piece's value -> then sum the results.
        PIECE_VALUES
            .iter()
            .map(|&(piece, value)| {
                let white = board.count(Color::White, piece) as i32;
                let black = board.count(Color::Black, piece) as i32;
                (white - black) * value
            })
            .sum()
    }
}