//! Source
//! https://www.chessprogramming.org/Piece-Square_Tables
//! https://adamberent.com/piece-square-table/
//!

use crate::board::{Color, PieceType, Square};
use crate::board::shakmaty_adapter::ShakmatyAdapter;
use crate::evaluation::evaluator::EvaluationTerm;
use crate::evaluation::piece_square_tables;
use crate::evaluation::piece_square_tables::tables;

pub struct PieceSquareTables;

impl PieceSquareTables {
    /// Score, in centipawns, always from the owning side's point of view
    /// (i.e. always non-negated — the caller applies the sign).
    pub fn value(piece: PieceType, color: Color, square: Square) -> i16 {
        let table = match piece {
            PieceType::Pawn => &tables::PAWN_TABLE,
            PieceType::Knight => &tables::KNIGHT_TABLE,
            PieceType::Bishop => &tables::BISHOP_TABLE,
            PieceType::Rook => &tables::ROOK_TABLE,
            PieceType::Queen => &tables::QUEEN_TABLE,
            PieceType::King => &tables::KING_TABLE,
        };

        table[table_index(color, square)]
    }
}

impl EvaluationTerm for PieceSquareTables {
    /// White-relative: positive favours White.
    fn evaluate(&self, board: &ShakmatyAdapter) -> i32 {
        let mut score = 0i32;

        board.for_each_piece(|square, piece| {
            let value = PieceSquareTables::value(piece.piece_type, piece.color, square) as i32;

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
