//! Evaluation plan (NNUE).
//!
//! The real evaluation is the neural network in src/nnue, trained in training/.
//! The hand written eval only sticks around for two jobs:
//!
//!   1. Gen 0 - the net needs data before it exists, so the first round of
//!      self play (src/datagen) runs on material + piece-square tables.
//!   2. Baseline - every net gets tested against it so we can show the Elo gain.
//!
//! Things like king safety, pawn structure, mobility etc. are NOT written by hand,
//! that's exactly what the net learns from the data.
//!
//! Still needed but they live in search, not eval:
//!   - Static exchange evaluation (search/static_exchange_evaluation.rs) - capture ordering + qsearch pruning
//!   - Piece values (material.rs) - MVV-LVA and delta pruning
//!
//! TODO:
//!   - Add MaterialEvaluator + PieceSquareTables to Evaluator::new()
//!   - Taper the king table between KING_MIDGAME_TABLE and KING_ENDGAME_TABLE
