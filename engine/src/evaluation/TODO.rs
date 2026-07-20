//! Evaluation priorities:
//!
//! The evaluation function estimates how favorable a chess position is.
//! Evaluators are ranked by importance and expected impact on playing strength.
//!
//! Priority 1 - Material Evaluation
//!     Measures the value of pieces remaining on the board.
//!     This is the foundation of evaluation because winning material is the
//!     most direct and reliable advantage.
//!
//! Priority 2 - Piece-Square Tables
//!     Evaluates where pieces are located on the board.
//!     A piece's value changes depending on its position (for example,
//!     centralized knights are generally stronger than corner knights).
//!
//! Priority 3 - King Safety
//!     Evaluates threats around the king, including pawn shields and enemy
//!     attack zones. Preventing checkmate threats is more important than
//!     gaining small positional advantages.
//!
//! Priority 4 - Mobility / Activity
//!     Measures how many useful moves pieces have available.
//!     Active pieces generally provide more control and future opportunities.
//!
//! Priority 5 - Pawn Structure
//!     Evaluates long-term pawn weaknesses and strengths:
//!     passed pawns, isolated pawns, doubled pawns, and backward pawns.
//!
//! Priority 6 - Center Control
//!     Evaluates control of important central squares.
//!     Central control improves piece activity and mobility.
//!
//! Priority 7 - Special Positional Features
//!     Smaller bonuses that refine evaluation:
//!    - Bishop pair advantage
//!     - Rook placement (open files, 7th rank)
//!     - Trapped pieces
//!     - Endgame-specific adjustments
//!
//! Evaluation terms should be implemented independently and combined into a
//! weighted final score. Material provides the baseline, while positional
//! evaluators refine the estimate.