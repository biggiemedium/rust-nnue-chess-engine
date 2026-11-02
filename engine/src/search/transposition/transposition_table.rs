//! Transposition table
//! https://www.chessprogramming.org/Transposition_Table
//!
//! The same position shows up over and over through different move orders
//! (1. e4 e5 2. Nf3 == 1. Nf3 e5 2. e4) so we cache what we found the first time.
//! It's a big hash map without the map part -> index = zobrist key & (size - 1)
//!
//! Every entry gives us 2 things:
//!   1. a score we can maybe return straight away (if it was searched deep enough)
//!   2. the best move from last time -> try it first, even if the score is useless

use crate::board::Move;
use crate::search::searcher::MATE_BOUND;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Exact, // score is the real score (alpha < score < beta)
    Lower, // failed high (beta cutoff) -> real score is AT LEAST this
    Upper, // failed low (nothing beat alpha) -> real score is AT MOST this
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64, // full key so 2 positions landing on the same slot don't get mixed up
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub bound: Bound,
}

// key 0 = empty slot (a real position hashing to exactly 0 is basically impossible)
const EMPTY: Entry = Entry { key: 0, best_move: None, score: 0, depth: -1, bound: Bound::Upper };

pub struct TranspositionTable {
    entries: Vec<Entry>,
    mask: usize,
}

impl TranspositionTable {

    pub fn new(megabytes: usize) -> Self {
        // round down to a power of 2 so the index is a cheap & instead of a %
        let wanted = (megabytes * 1024 * 1024 / size_of::<Entry>()).max(1);
        let size = if wanted.is_power_of_two() { wanted } else { wanted.next_power_of_two() / 2 };

        Self {
            entries: vec![EMPTY; size],
            mask: size - 1,
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(EMPTY);
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let entry = self.entries[key as usize & self.mask];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    #[inline]
    pub fn store(&mut self, key: u64, best_move: Option<Move>, score: i32, depth: i32, bound: Bound, ply: usize) {
        let slot = &mut self.entries[key as usize & self.mask];

        // same position searched shallower now -> keep the deeper result, it's worth more
        // TODO: aging so old searches don't hog slots forever
        if slot.key == key && depth < slot.depth && bound != Bound::Exact {
            return;
        }

        // don't throw away a known best move just because this search didn't find one
        let best_move = best_move.or(if slot.key == key { slot.best_move } else { None });

        *slot = Entry { key, best_move, score: score_to_tt(score, ply), depth, bound };
    }

    // how full the table is in permille, uci "hashfull"
    pub fn hashfull(&self) -> usize {
        let sample = self.entries.len().min(1000);
        self.entries[..sample].iter().filter(|e| e.key != 0).count() * 1000 / sample
    }
}

// Mate scores are "mate in N plies from the ROOT", but the same position can show up at a
// different ply next time. So store them as "mate in N from HERE" and convert back on the way out.
// Without this the engine thinks a mate is closer / further than it really is
#[inline]
pub fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score + ply as i32
    } else if score < -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
pub fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score - ply as i32
    } else if score < -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::search::searcher::MATE;

    #[test]
    fn store_then_probe() {
        let mut tt = TranspositionTable::new(1);
        let pos = Position::startpos();
        let mv = pos.legal_moves()[0];

        tt.store(pos.key(), Some(mv), 42, 5, Bound::Exact, 0);
        let entry = tt.probe(pos.key()).unwrap();

        assert_eq!((entry.best_move, entry.score, entry.depth, entry.bound), (Some(mv), 42, 5, Bound::Exact));
        assert!(tt.probe(pos.key() ^ 1).is_none());
    }

    #[test]
    fn mate_score_survives_different_ply() {
        // mate found 3 plies from here, stored at ply 4 -> read back at ply 10
        let at_ply_4 = MATE - (4 + 3);
        let stored = score_to_tt(at_ply_4, 4);
        assert_eq!(score_from_tt(stored, 10), MATE - (10 + 3));
    }

    #[test]
    fn size_is_power_of_two() {
        let tt = TranspositionTable::new(3);
        assert!(tt.entries.len().is_power_of_two());
    }
}
