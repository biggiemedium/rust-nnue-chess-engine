// History heuristic
// https://www.chessprogramming.org/History_Heuristic
//
// Killers only remember 2 moves per ply, history remembers every quiet move that
// ever caused a cutoff, anywhere in the tree -> table[side][from][to] += depth * depth
//
// depth * depth because a cutoff near the root saves a huge subtree,
// a cutoff 1 ply from the leaves saves almost nothing

use crate::board::{Color, Move};

// keep scores under the killer move score in move_ordering.rs so killers still go first
const MAX_HISTORY: i32 = 50_000;

pub struct HistoryHeuristic {
    table: Box<[[[i32; 64]; 64]; 2]>, // 32KB, boxed so it lives on the heap not the stack
}

impl HistoryHeuristic {

    pub fn new() -> Self {
        Self { table: Box::new([[[0; 64]; 64]; 2]) }
    }

    pub fn clear(&mut self) {
        *self.table = [[[0; 64]; 64]; 2];
    }

    #[inline]
    pub fn get(&self, side: Color, mv: Move) -> i32 {
        match indices(side, mv) {
            Some((s, from, to)) => self.table[s][from][to],
            None => 0,
        }
    }

    pub fn update(&mut self, side: Color, mv: Move, depth: i32) {
        let Some((s, from, to)) = indices(side, mv) else {
            return;
        };

        self.table[s][from][to] += depth * depth;

        // one move hit the cap -> halve everything, keeps the order between moves the same
        if self.table[s][from][to] > MAX_HISTORY {
            self.age();
        }
    }

    // called between searches, old info is still useful but shouldn't dominate
    pub fn age(&mut self) {
        for side in self.table.iter_mut() {
            for from in side.iter_mut() {
                for score in from.iter_mut() {
                    *score /= 2;
                }
            }
        }
    }
}

// Default state / restarted state
impl Default for HistoryHeuristic {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn indices(side: Color, mv: Move) -> Option<(usize, usize, usize)> {
    let from = mv.from_square()?.index(); // None only for crazyhouse drops, never here
    let s = match side {
        Color::White => 0,
        Color::Black => 1,
    };
    Some((s, from, mv.to_square().index()))
}
