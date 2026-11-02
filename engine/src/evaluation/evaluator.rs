//! Based off https://www.chessprogramming.org/Evaluation


use crate::board::Position;
use crate::evaluation::material::material::MaterialEvaluator;
use crate::evaluation::piece_square_tables::piece_square_tables::PieceSquareTables;

// Send -> the searcher (and its evaluator) gets moved onto the search thread
pub trait EvaluationTerm: Send {
    fn evaluate(&self, board: &Position) -> i32;
}


pub struct Evaluator {
    terms: Vec<Box<dyn EvaluationTerm>>,
}

impl Evaluator {

    pub fn new() -> Self {
        Self {
            terms: vec![
                Box::new(MaterialEvaluator::new()),
                Box::new(PieceSquareTables::default()),
                // NNUE replaces these once net-001 exists, they stay for gen 0 + baseline
            ],
        }
    }

    pub fn evaluate(&self, board: &Position) -> i32 {

        let score: i32 = self.terms
            .iter()
            .map(|term| term.evaluate(board))
            .sum();


        match board.side_to_move() {
            crate::board::Color::White => score,
            crate::board::Color::Black => -score,
        }
    }

}

// Default state / restarted state
impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}
