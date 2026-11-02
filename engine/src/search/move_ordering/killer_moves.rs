// Killer moves
// https://www.chessprogramming.org/Killer_Heuristic
//
// Cache moves that might cause beta cutoffs
// A quiet move that refuted one line at this ply will probably refute its siblings too
// e.g. a fork that works no matter which pawn move the opponent made first
//
// Indexed by ply (distance from root), not depth -> two nodes at the same ply are
// "siblings" in the tree, depth changes with extensions so it's not as reliable

use crate::board::Move;
use crate::search::searcher::MAX_PLY;

pub struct KillerMoves {
    table: [[Option<Move>; 2]; MAX_PLY],
}

impl KillerMoves {

    pub fn new() -> Self {
        Self { table: [[None; 2]; MAX_PLY] }
    }

    pub fn clear(&mut self) {
        self.table = [[None; 2]; MAX_PLY];
    }

    #[inline]
    pub fn get(&self, ply: usize) -> [Option<Move>; 2] {
        self.table[ply]
    }

    #[inline]
    pub fn store(&mut self, ply: usize, mv: Move) {
        let slot = &mut self.table[ply];

        // already the main killer -> don't push the backup out with a copy of itself
        if slot[0] != Some(mv) {
            slot[1] = slot[0];
            slot[0] = Some(mv);
        }
    }
}

// Default state / restarted state
impl Default for KillerMoves {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;

    #[test]
    fn newest_killer_goes_first() {
        let moves = Position::startpos().legal_moves();
        let mut killers = KillerMoves::new();

        killers.store(3, moves[0]);
        killers.store(3, moves[1]);
        assert_eq!(killers.get(3), [Some(moves[1]), Some(moves[0])]);

        killers.store(3, moves[1]); // same again, nothing moves
        assert_eq!(killers.get(3), [Some(moves[1]), Some(moves[0])]);
        assert_eq!(killers.get(4), [None, None]);
    }
}
