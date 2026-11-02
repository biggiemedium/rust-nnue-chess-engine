//! Self play data generation.
//!
//! The engine plays itself and writes every useful position to disk
//! for the python trainer. Gen 0 uses the material + PST eval, after that
//! each new net generates the data for the next one.
//!
//! Plan:
//!   - start from a few random moves so games aren't all the same opening
//!   - search each move to a fixed node count (~5k nodes)
//!   - record (position, search score) and fill in the result once the game ends
//!   - skip noisy positions: in check, best move is a capture, first few plies
//!   - one thread per game, each writing its own file

// TODO: needs search + uci first
