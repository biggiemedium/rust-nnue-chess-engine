// most_valuable_victim_least_valuable_attacker
// https://www.chessprogramming.org/MVV-LVA
//
// PxQ first, QxP last -> taking a queen with a pawn is almost always good,
// taking a pawn with a queen might just hang the queen

use crate::board::Move;
use crate::evaluation::material::material::piece_value;

pub fn mvv_lva(mv: Move) -> i32 {
    let victim = mv.captured_piece_type().map(piece_value).unwrap_or(0);
    let attacker = piece_value(mv.moving_piece_type());

    // victim * 10 so the victim always matters more than the attacker
    victim * 10 - attacker
}
