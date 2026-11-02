//! NNUE input features.
//!
//! This is the contract between the engine and the python trainer,
//! training/nnue/features.py MUST compute the exact same index or the net reads garbage.
//!
//! 768 inputs = 2 sides * 6 piece types * 64 squares
//! Each side gets its own view of the board ("perspective"):
//!   own pieces   -> 0..384
//!   enemy pieces -> 384..768
//! Black flips the board vertically (sq ^ 56) so both sides "see" themselves from the bottom,
//! that way one set of weights works for both colours.
//!
//! https://www.chessprogramming.org/NNUE

use crate::board::{Color, Piece, PieceType, Square};

pub const INPUTS: usize = 768;

/// index = side * 384 + piece_type * 64 + square
#[inline]
pub fn feature_index(perspective: Color, piece: Piece, square: Square) -> usize {
    let side = if piece.color == perspective { 0 } else { 1 };

    // ^ 56 flips the rank and leaves the file alone
    let sq = match perspective {
        Color::White => square.index(),
        Color::Black => square.index() ^ 56,
    };

    side * 384 + piece_type_index(piece.piece_type) * 64 + sq
}

#[inline]
pub fn piece_type_index(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // same numbers are checked in training/tests/test_features.py
    #[test]
    fn white_pawn_e2() {
        let pawn = Piece { color: Color::White, piece_type: PieceType::Pawn };
        let e2 = Square(12);

        assert_eq!(feature_index(Color::White, pawn, e2), 12); // own pawn, e2
        assert_eq!(feature_index(Color::Black, pawn, e2), 384 + 52); // enemy pawn, flipped to e7
    }

    #[test]
    fn black_king_e8() {
        let king = Piece { color: Color::Black, piece_type: PieceType::King };
        let e8 = Square(60);

        assert_eq!(feature_index(Color::Black, king, e8), 5 * 64 + 4); // own king, flipped to e1
        assert_eq!(feature_index(Color::White, king, e8), 384 + 5 * 64 + 60);
    }
}
