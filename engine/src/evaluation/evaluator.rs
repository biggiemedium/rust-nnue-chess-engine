//! Based off https://www.chessprogramming.org/Evaluation


use crate::board::shakmaty_adapter::ShakmatyAdapter;

pub trait EvaluationTerm {
    fn evaluate(&self, board: &ShakmatyAdapter) -> i32;
}


pub struct Evaluator {
    terms: Vec<Box<dyn EvaluationTerm>>,
}

impl Evaluator {

    pub fn new() -> Self {
        Self {
            terms: vec![
               // TODO: Add our evaluators
            ],
        }
    }

    pub fn evaluate(&self, board: &ShakmatyAdapter) -> i32 {

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