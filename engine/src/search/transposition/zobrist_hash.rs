//! Zobrist hashing
//! https://www.chessprogramming.org/Zobrist_Hashing
//!
//! A random u64 for every (piece, square), castling right, ep file and side to move.
//! XOR together the ones that are true for a position -> its key.
//! XOR is its own undo so a move only needs a couple of XORs to update the key.
//!
//! shakmaty already does this (Zobrist64) and Position caches the key, so nothing lives here yet.
//! TODO: incremental update in Position::play instead of recomputing (see position.rs)
