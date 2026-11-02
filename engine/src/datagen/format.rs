//! Training data record format, 32 bytes per position.
//!
//! Fixed size so python can np.memmap the whole file and never parse a FEN.
//! training/nnue/data.py reads this, if you change it change both.
//!
//! offset  size  field
//!   0      8    occupancy  u64 LE, bit i set = piece on square i (a1 = 0, h8 = 63)
//!   8     16    pieces     one 4 bit nibble per piece, in occupancy order (lowest square first),
//!                          low nibble first. nibble = colour << 3 | piece type (pawn 0 .. king 5)
//!  24      2    score      i16 LE, centipawns, from side to move's POV
//!  26      1    result     u8, 0 = loss, 1 = draw, 2 = win, side to move's POV
//!  27      1    stm        u8, 0 = white, 1 = black
//!  28      4    reserved   zero for now

use crate::board::{Color, Position, Square};
use crate::nnue::features::piece_type_index;

pub const RECORD_SIZE: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub fn pack(pos: &Position, score: i16, result: GameResult) -> [u8; RECORD_SIZE] {
    let mut out = [0u8; RECORD_SIZE];
    let occupancy = pos.occupied_bitboard();
    out[0..8].copy_from_slice(&occupancy.to_le_bytes());

    // walk the set bits lowest -> highest, that's the order python unpacks them in
    let mut bits = occupancy;
    let mut i = 0;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1; // clear lowest bit

        let piece = pos.piece_at(Square(sq)).expect("occupied square has no piece");
        let colour = match piece.color {
            Color::White => 0u8,
            Color::Black => 1u8,
        };
        let nibble = colour << 3 | piece_type_index(piece.piece_type) as u8;

        // two pieces per byte, even one in the low half
        out[8 + i / 2] |= if i % 2 == 0 { nibble } else { nibble << 4 };
        i += 1;
    }

    out[24..26].copy_from_slice(&score.to_le_bytes());
    out[26] = result as u8;
    out[27] = if pos.white_to_move() { 0 } else { 1 };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // golden bytes, training/tests/test_data.py checks python produces the exact same thing
    pub const STARTPOS_DRAW: [u8; RECORD_SIZE] = [
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, // occupancy
        0x13, 0x42, 0x25, 0x31, // R N B Q K B N R
        0x00, 0x00, 0x00, 0x00, // white pawns
        0x88, 0x88, 0x88, 0x88, // black pawns
        0x9B, 0xCA, 0xAD, 0xB9, // r n b q k b n r
        0x00, 0x00, // score 0
        0x01, // draw
        0x00, // white to move
        0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn startpos_matches_golden() {
        let record = pack(&Position::startpos(), 0, GameResult::Draw);
        assert_eq!(record, STARTPOS_DRAW);
    }
}
