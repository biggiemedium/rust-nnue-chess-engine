//! Source
//! https://www.chessprogramming.org/Piece-Square_Tables
//! https://adamberent.com/piece-square-table/
//!

use crate::board::{Color, PieceType, Square};

pub struct PieceSquareTables {

}

impl PieceSquareTables {

}


// TODO: Move to Utility class
/// Flip board vertically for black pieces.
///
/// a8 -> a1
/// h8 -> h1
fn mirror_square(square: Square) -> usize {
    let file = square.file();
    let rank = square.rank();

    ((7 - rank) * 8 + file) as usize
}