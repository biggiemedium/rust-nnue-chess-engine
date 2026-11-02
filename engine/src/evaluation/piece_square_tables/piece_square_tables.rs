//! Source
//! https://www.chessprogramming.org/Piece-Square_Tables
//! https://adamberent.com/piece-square-table/
//!

use crate::board::{Color, PieceType, Square};
use crate::board::Position;
use crate::evaluation::evaluator::EvaluationTerm;
use crate::evaluation::piece_square_tables::tables;

pub struct PieceSquareTables;

impl PieceSquareTables {

    fn new() -> PieceSquareTables {
        Self
    }

    /// Score, in centipawns, always from the owning side's point of view
    /// (i.e. always non-negated — the caller applies the sign).
    pub fn value(piece: PieceType, color: Color, square: Square) -> i16 {
        let table = match piece {
            PieceType::Pawn => &tables::PAWN_TABLE,
            PieceType::Knight => &tables::KNIGHT_TABLE,
            PieceType::Bishop => &tables::BISHOP_TABLE,
            PieceType::Rook => &tables::ROOK_TABLE,
            PieceType::Queen => &tables::QUEEN_TABLE,
            PieceType::King => &tables::KING_MIDGAME_TABLE,
        };

        table[table_index(color, square)]
    }
}

// Default state / restarted state
impl Default for PieceSquareTables {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationTerm for PieceSquareTables {
    fn evaluate(&self, board: &Position) -> i32 {
        // Walk every piece and add its placement bonus,
        // White positive -> Black negative.
        let mut score = 0i32;

        board.for_each_piece(|square, piece| {
            let value = Self::value(piece.piece_type, piece.color, square) as i32;

            match piece.color {
                Color::White => score += value,
                Color::Black => score -= value,
            }
        });

        score
    }
}

/// Maps a board square to a table slot.
/// `^ 56` flips the rank and leaves the file alone (index = rank * 8 + file)
#[inline]
fn table_index(color: Color, square: Square) -> usize {
    match color {
        Color::White => square.index() ^ 56,
        Color::Black => square.index(),
    }
}
