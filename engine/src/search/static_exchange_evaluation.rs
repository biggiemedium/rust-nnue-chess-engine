//! Static exchange evaluation (SEE)
//! https://www.chessprogramming.org/Static_Exchange_Evaluation
//! https://www.chessprogramming.org/SEE_-_The_Swap_Algorithm
//! https://mediocrechess.blogspot.com/2007/03/guide-static-exchange-evaluation-see.html
//!
//! Yes we may capture a piece but will they counter and take ours?
//! SEE plays out every capture on that ONE square, cheapest attacker first, without
//! actually making moves -> material we win or lose.
//! Example: QxP defended by pawn -> we win 100 lose 900 -> -800, don't bother
//!
//! Done with bitboards this time instead of looping every piece + checking isValidMove.
//! Taking a piece off the occupancy uncovers whatever slider was behind it (x-rays)
//! e.g. two rooks stacked on a file, the back one joins in after the front one captures
//!
//! Ignores pins + checks, it's a fast estimate not a real search.

use crate::board::{Color, Move, PieceType, Position};
use crate::evaluation::material::material::piece_value;

const ATTACKER_ORDER: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

/// Material won (+) or lost (-) by `mv` once the exchange on its square plays out, side to move's POV.
pub fn see(pos: &Position, mv: Move) -> i32 {
    let Some(captured) = mv.captured_piece_type() else {
        return 0; // not a capture
    };
    let Some(from) = mv.from_square() else {
        return 0;
    };
    let to = mv.to_square();

    // gain[i] = what the side making capture i has won so far, assuming the exchange stops there
    let mut gain = [0i32; 32];
    let mut i = 0; // this will double as our depth

    gain[0] = piece_value(captured);
    let mut attacker_value = piece_value(mv.moving_piece_type()); // what's sitting on `to` now, next to get taken

    // our first attacker leaves its square
    // TODO: en passant, the captured pawn isn't on `to` so it stays in occupied (can block a slider, rare)
    let mut occupied = pos.occupied_bitboard() & !(1u64 << from.index());
    let mut side = pos.side_to_move().flip(); // defenders turn -> us = !us

    while i < 31 {
        let attackers = pos.attackers_to(to, occupied) & occupied;
        let ours = attackers & pos.color_bitboard(side);
        if ours == 0 {
            break; // No more attackers
        }

        let (piece_type, piece_bb) = cheapest_attacker(pos, side, ours);

        // the king can only take if nobody can take it back
        if piece_type == PieceType::King && attackers & pos.color_bitboard(side.flip()) != 0 {
            break;
        }

        i += 1;
        gain[i] = attacker_value - gain[i - 1];

        attacker_value = piece_value(piece_type);
        occupied &= !(piece_bb & piece_bb.wrapping_neg()); // take the lowest bit = that attacker off the board
        side = side.flip();
    }

    // walk back up -> at every step the side to move can choose to stop capturing
    // if carrying on is worse than stopping they just won't do it
    while i > 0 {
        gain[i - 1] = -(-gain[i - 1]).max(gain[i]);
        i -= 1;
    }

    gain[0]
}

// least valuable attacker first, a pawn recapturing beats a queen recapturing
fn cheapest_attacker(pos: &Position, side: Color, attackers: u64) -> (PieceType, u64) {
    for piece_type in ATTACKER_ORDER {
        let bb = attackers & pos.pieces_bitboard(side, piece_type);
        if bb != 0 {
            return (piece_type, bb);
        }
    }
    unreachable!("attackers wasn't empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn see_uci(fen: &str, uci: &str) -> i32 {
        let pos = Position::from_fen(fen).unwrap();
        let mv = pos.legal_moves().into_iter().find(|m| m.to_uci_string() == uci).unwrap();
        see(&pos, mv)
    }

    #[test]
    fn free_pawn() {
        // Rxe5, nothing defends it
        assert_eq!(see_uci("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5"), 100);
    }

    #[test]
    fn queen_takes_defended_pawn() {
        // QxP, pxQ -> 100 - 900
        assert_eq!(see_uci("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5"), -800);
    }

    #[test]
    fn even_knight_trade() {
        // NxN, pxN -> 320 - 320
        assert_eq!(see_uci("4k3/8/3p4/4n3/8/5N2/8/4K3 w - - 0 1", "f3e5"), 0);
    }

    #[test]
    fn xray_rook_behind_rook() {
        // RxP, rxR, RxR -> the back rook on e1 only counts if x-rays work
        // without x-ray this comes out as -400
        assert_eq!(see_uci("4k3/4r3/8/4p3/8/8/4R3/4R2K w - - 0 1", "e2e5"), 100);
    }

    #[test]
    fn quiet_move_is_zero() {
        assert_eq!(see_uci("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2"), 0);
    }
}
