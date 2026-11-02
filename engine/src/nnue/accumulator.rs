//! The accumulator, the "efficiently updatable" part of NNUE.
//!
//! The first layer is just a sum of weight rows for every active feature.
//! A move only changes 2-4 features so instead of re-summing all ~32 pieces
//! we add the new rows and subtract the old ones.
//!   quiet move: -from +to
//!   capture:    -from +to -captured
//!   castling / promotion / en passant: a couple more
//!
//! One accumulator per perspective (white view + black view).

// TODO: Accumulator { white: [i16; HIDDEN], black: [i16; HIDDEN] }
// TODO: refresh(pos) - full recompute, used at root and in tests
// TODO: update(mv) - incremental add/sub
// TODO: test that update() after a random move sequence == refresh()
