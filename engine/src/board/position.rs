//! `Position` — the engine's board, and the type search/eval actually use.
//!
//! It composes the shakmaty wrapper (`ShakmatyAdapter`) and adds the engine's
//! own concerns: a cached zobrist key so the transposition table doesn't
//! recompute it every probe. Repetition history is deliberately NOT stored here
//! — under copy-make, cloning a history Vec at every node would wreck NPS, so
//! the search layer tracks visited keys in its own stack instead.
//!
//! Everything downstream imports the chess vocabulary (Color, Move, Square, …)
//! from here, re-exported from the adapter, so no other module names shakmaty.

pub use super::shakmaty_adapter::{Color, Move, MoveList, Piece, PieceType, Square};
use super::shakmaty_adapter::ShakmatyAdapter;

/// A chess position plus its cached hash. Cheap to clone (copy-make).
#[derive(Clone)]
pub struct Position {
    board: ShakmatyAdapter,
    key: u64,
}

impl Position {

    #[inline]
    pub fn startpos() -> Self {
        let board = ShakmatyAdapter::startpos();
        let key = board.zobrist_key();
        Self { board, key }
    }

    #[inline]
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let board = ShakmatyAdapter::from_fen(fen)?;
        let key = board.zobrist_key();
        Ok(Self { board, key })
    }

    #[inline]
    pub fn to_fen(&self) -> String {
        self.board.to_fen()
    }

    // =========== hashing ===========================

    /// Cached zobrist key for TT probing.
    #[inline]
    pub fn key(&self) -> u64 {
        self.key
    }

    // ================= move generation ==========================

    #[inline]
    pub fn legal_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    #[inline]
    pub fn capture_moves(&self) -> MoveList {
        self.board.capture_moves()
    }

    // === make move =======================
    //
    // FUTURE: replace the full `zobrist_key()` recompute with an incremental
    // update (shakmaty's `update_zobrist_hash` returns Option and may fall back,
    // so handle the None case) once profiling says it matters.

    #[inline]
    pub fn play(&self, mv: Move) -> Self {
        let board = self.board.play(mv);
        let key = board.zobrist_key();
        Self { board, key }
    }

    #[inline]
    pub fn play_null(&self) -> Option<Self> {
        self.board.play_null().map(|board| {
            let key = board.zobrist_key();
            Self { board, key }
        })
    }

    #[inline]
    pub fn play_uci(&self, uci: &str) -> Option<Self> {
        self.board.play_uci(uci).map(|board| {
            let key = board.zobrist_key();
            Self { board, key }
        })
    }

    // ====== state queries ===================

    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    #[inline]
    pub fn white_to_move(&self) -> bool {
        self.board.side_to_move() == Color::White
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.board.in_check()
    }

    #[inline]
    pub fn is_checkmate(&self) -> bool {
        self.board.is_checkmate()
    }

    #[inline]
    pub fn is_stalemate(&self) -> bool {
        self.board.is_stalemate()
    }

    #[inline]
    pub fn is_insufficient_material(&self) -> bool {
        self.board.is_insufficient_material()
    }

    #[inline]
    pub fn is_game_over(&self) -> bool {
        self.board.is_game_over()
    }

    #[inline]
    pub fn halfmove_clock(&self) -> u32 {
        self.board.halfmove_clock()
    }

    // ========== evaluation access ============

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.board.piece_at(sq)
    }

    #[inline]
    pub fn for_each_piece(&self, f: impl FnMut(Square, Piece)) {
        self.board.for_each_piece(f)
    }

    #[inline]
    pub fn pieces_bitboard(&self, color: Color, piece_type: PieceType) -> u64 {
        self.board.pieces_bitboard(color, piece_type)
    }

    #[inline]
    pub fn color_bitboard(&self, color: Color) -> u64 {
        self.board.color_bitboard(color)
    }

    #[inline]
    pub fn occupied_bitboard(&self) -> u64 {
        self.board.occupied_bitboard()
    }

    #[inline]
    pub fn count(&self, color: Color, piece_type: PieceType) -> u32 {
        self.board.count(color, piece_type)
    }

    #[inline]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        self.board.king_square(color)
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}