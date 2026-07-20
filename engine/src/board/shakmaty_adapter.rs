//! The wrapper class around the shakmaty API.
//!
//! The ONLY file allowed to `use shakmaty::`. Exposes engine-neutral types
//! (our own Move/Color/PieceType/Square, raw u64 bitboards). Nothing shakmaty
//! leaks past this module.
//!
//! Verified against the shakmaty 0.30 `position` source:
//!   * `zobrist_hash::<Zobrist64>(mode)` is a Position-trait method; returns a
//!     ZobristValue newtype, `.0` is the u64. No `ZobristHash` trait to import.
//!   * `to_setup(&self, EnPassantMode) -> Setup` (there is NO `into_setup`).
//!   * `play_unchecked(&mut self, m: Move)` and `is_legal(self, m: Move)` take
//!     Move BY VALUE (Move is Copy).
//!   * `legal_moves()` / `capture_moves()` return shakmaty's `MoveList`.

use shakmaty::{fen::Fen, uci::UciMove, zobrist::Zobrist64, CastlingMode, Chess, Color as SmColor, EnPassantMode, FromSetup, Move as SmMove, Position as SmPosition, Role as SmRole, Setup, Square as SmSquare};

// ---------------------------------------------------------------------------
// Engine-neutral types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Square(pub u8);

impl Square {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
    #[inline]
    pub fn file(self) -> u8 {
        self.0 & 7
    }
    #[inline]
    pub fn rank(self) -> u8 {
        self.0 >> 3
    }
}

/// Opaque move handle wrapping shakmaty's `Move` (which is `Copy`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move(SmMove);

impl Move {
    /// The wrapped shakmaty move, by value (it's Copy). For the adapter only.
    #[inline]
    pub(super) fn inner(self) -> SmMove {
        self.0
    }

    #[inline]
    pub fn is_capture(self) -> bool {
        self.0.is_capture()
    }

    #[inline]
    pub fn is_promotion(self) -> bool {
        self.0.promotion().is_some()
    }

    #[inline]
    pub fn moving_piece_type(self) -> PieceType {
        from_sm_role(self.0.role())
    }

    #[inline]
    pub fn captured_piece_type(self) -> Option<PieceType> {
        self.0.capture().map(from_sm_role)
    }

    #[inline]
    pub fn promotion_piece_type(self) -> Option<PieceType> {
        self.0.promotion().map(from_sm_role)
    }

    #[inline]
    pub fn from_square(self) -> Option<Square> {
        self.0.from().map(from_sm_square)
    }

    #[inline]
    pub fn to_square(self) -> Square {
        from_sm_square(self.0.to())
    }

    #[inline]
    pub fn to_uci_string(self) -> String {
        self.0.to_uci(CastlingMode::Standard).to_string()
    }
}

/// Engine-neutral move list. `Vec` avoids a direct `arrayvec` dependency; the
/// per-node allocation is negligible next to a copy-make clone, and can be
/// swapped for a stack list later if profiling ever flags it.
pub type MoveList = Vec<Move>;

// ---------------------------------------------------------------------------
// The wrapper class
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ShakmatyAdapter {
    board: Chess,
}

impl ShakmatyAdapter {
    // --- construction ------------------------------------------------------

    #[inline]
    pub fn startpos() -> Self {
        Self {
            board: Chess::default(),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parsed: Fen = fen.parse().map_err(|e| format!("bad FEN: {e}"))?;
        let board = parsed
            .into_position::<Chess>(CastlingMode::Standard)
            .map_err(|e| format!("illegal position: {e}"))?;
        Ok(Self { board })
    }

    pub fn to_fen(&self) -> String {
        Fen::try_from_setup(self.board.to_setup(EnPassantMode::Legal))
            .expect("position cannot be represented as FEN")
            .to_string()
    }


    // generate all legal moves
    // O(N) Complexity?
    #[inline]
    pub fn legal_moves(&self) -> MoveList {
        self.board.legal_moves().into_iter().map(Move).collect()
    }

    #[inline]
    pub fn capture_moves(&self) -> MoveList {
        self.board.capture_moves().into_iter().map(Move).collect()
    }


    // Make move
    #[inline]
    pub fn play(&self, mv: Move) -> Self {
        let mut board = self.board.clone();
        board.play_unchecked(mv.inner()); // Move is Copy → pass by value
        Self { board }
    }

    /// Null move. `None` if in check (illegal) or the swap is invalid.
    pub fn play_null(&self) -> Option<Self> {
        if self.in_check() {
            return None;
        }
        let mut setup = self.board.to_setup(EnPassantMode::Always);
        setup.swap_turn(); // Setup has swap_turn(); clears ep, flips side
        Chess::from_setup(setup, CastlingMode::Standard)
            .ok()
            .map(|board| Self { board })
    }

    pub fn play_uci(&self, uci: &str) -> Option<Self> {
        let uci_move: UciMove = uci.parse().ok()?;
        let mv = uci_move.to_move(&self.board).ok()?;
        let mut board = self.board.clone();
        board.play_unchecked(mv);
        Some(Self { board })
    }

    // --- state queries -----------------------------------------------------

    #[inline]
    pub fn side_to_move(&self) -> Color {
        from_sm_color(self.board.turn())
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.board.is_check()
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
        self.board.halfmoves()
    }

    #[inline]
    pub fn zobrist_key(&self) -> u64 {
        self.board.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
    }

    // --- board access for evaluation --------------------------------------

    #[inline]
    pub fn pieces_bitboard(&self, color: Color, piece_type: PieceType) -> u64 {
        let b = self.board.board();
        (b.by_color(to_sm_color(color)) & b.by_role(to_sm_role(piece_type))).0
    }

    #[inline]
    pub fn color_bitboard(&self, color: Color) -> u64 {
        self.board.board().by_color(to_sm_color(color)).0
    }

    #[inline]
    pub fn occupied_bitboard(&self) -> u64 {
        self.board.board().occupied().0
    }

    #[inline]
    pub fn count(&self, color: Color, piece_type: PieceType) -> u32 {
        self.pieces_bitboard(color, piece_type).count_ones()
    }

    #[inline]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        self.board
            .board()
            .king_of(to_sm_color(color))
            .map(from_sm_square)
    }

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.board.board().piece_at(to_sm_square(sq)).map(|p| Piece {
            color: from_sm_color(p.color),
            piece_type: from_sm_role(p.role),
        })
    }

    pub fn for_each_piece(&self, mut f: impl FnMut(Square, Piece)) {
        let b = self.board.board();
        for sq in b.occupied() {
            if let Some(p) = b.piece_at(sq) {
                f(
                    from_sm_square(sq),
                    Piece {
                        color: from_sm_color(p.color),
                        piece_type: from_sm_role(p.role),
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions (private)
// ---------------------------------------------------------------------------

#[inline]
fn to_sm_color(c: Color) -> SmColor {
    match c {
        Color::White => SmColor::White,
        Color::Black => SmColor::Black,
    }
}

#[inline]
fn from_sm_color(c: SmColor) -> Color {
    match c {
        SmColor::White => Color::White,
        SmColor::Black => Color::Black,
    }
}

#[inline]
fn to_sm_role(pt: PieceType) -> SmRole {
    match pt {
        PieceType::Pawn => SmRole::Pawn,
        PieceType::Knight => SmRole::Knight,
        PieceType::Bishop => SmRole::Bishop,
        PieceType::Rook => SmRole::Rook,
        PieceType::Queen => SmRole::Queen,
        PieceType::King => SmRole::King,
    }
}

#[inline]
fn from_sm_role(r: SmRole) -> PieceType {
    match r {
        SmRole::Pawn => PieceType::Pawn,
        SmRole::Knight => PieceType::Knight,
        SmRole::Bishop => PieceType::Bishop,
        SmRole::Rook => PieceType::Rook,
        SmRole::Queen => PieceType::Queen,
        SmRole::King => PieceType::King,
    }
}

#[inline]
fn from_sm_square(s: SmSquare) -> Square {
    Square(u8::from(s))
}

#[inline]
fn to_sm_square(s: Square) -> SmSquare {
    SmSquare::new(s.0 as u32)
}