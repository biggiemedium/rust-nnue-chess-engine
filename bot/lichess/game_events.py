"""Event decoding and dispatch for the Lichess streams exposed by ``api.py``.

Responsibilities:

* Turn raw stream dicts into typed, validated event objects.
* Normalise berserk's datetime/timedelta conversion back into plain
  milliseconds, so downstream clock arithmetic is unsurprising.
* Own the *lifetime* of a stream -- reconnect on drop, back off on rate
  limits, stop cleanly on shutdown.

It deliberately holds no board state and knows no chess rules. Handlers
implement :class:`EventHandler` (usually ``bot.py``) and the actual position
lives in ``game.py``.

Usage::

    from lichess.api import LichessAPI
    from lichess.game_events import AccountEventStream, EventHandler

    class MyBot(EventHandler):
        def on_challenge(self, event):
            if event.variant == "standard":
                self.api.accept_challenge(event.id)

    api = LichessAPI()
    AccountEventStream(api, MyBot()).run()
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

from .api import (
    LichessAPI,
    LichessAuthError,
    LichessError,
    LichessRateLimitError,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Event type vocabulary
# --------------------------------------------------------------------------- #
class AccountEvent(str, Enum):
    """Types emitted by the account-wide event stream."""

    CHALLENGE = "challenge"
    CHALLENGE_CANCELED = "challengeCanceled"
    CHALLENGE_DECLINED = "challengeDeclined"
    GAME_START = "gameStart"
    GAME_FINISH = "gameFinish"


class GameEvent(str, Enum):
    """Types emitted by a single game's state stream."""

    GAME_FULL = "gameFull"
    GAME_STATE = "gameState"
    CHAT_LINE = "chatLine"
    OPPONENT_GONE = "opponentGone"


#: Game statuses that mean the game is over and the stream will close.
TERMINAL_STATUSES = frozenset(
    {
        "aborted",
        "mate",
        "resign",
        "stalemate",
        "timeout",
        "draw",
        "outoftime",
        "cheat",
        "noStart",
        "unknownFinish",
        "variantEnd",
    }
)


# --------------------------------------------------------------------------- #
# Value normalisation
# --------------------------------------------------------------------------- #
def to_millis(value: Any) -> Optional[int]:
    """Coerce a berserk clock value back into integer milliseconds.

    berserk converts time values to ``timedelta`` automatically, but the
    conversion varies across releases and endpoints -- some fields arrive as
    raw ints. This accepts either and always returns milliseconds.
    """
    if value is None:
        return None
    if isinstance(value, _dt.timedelta):
        return int(value.total_seconds() * 1000)
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        return int(value)
    logger.debug("Unrecognised clock value %r (%s)", value, type(value).__name__)
    return None


def _split_moves(moves: Any) -> list[str]:
    """Lichess sends the move list as one space-delimited UCI string."""
    if not moves:
        return []
    if isinstance(moves, list):
        return [str(m) for m in moves]
    return str(moves).split()


# --------------------------------------------------------------------------- #
# Account events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChallengeEvent:
    """An incoming or outgoing challenge."""

    id: str
    challenger: str
    challenger_rating: Optional[int]
    destination: Optional[str]
    rated: bool
    variant: str
    speed: str
    time_control: dict[str, Any]
    color: str
    status: Optional[str]
    decline_reason: Optional[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "ChallengeEvent":
        challenge = raw.get("challenge", raw)
        challenger = challenge.get("challenger") or {}
        dest = challenge.get("destUser") or {}
        return cls(
            id=challenge.get("id", ""),
            challenger=challenger.get("id") or challenger.get("name") or "",
            challenger_rating=challenger.get("rating"),
            destination=dest.get("id") or dest.get("name"),
            rated=bool(challenge.get("rated", False)),
            variant=(challenge.get("variant") or {}).get("key", "standard"),
            speed=challenge.get("speed", "unknown"),
            time_control=challenge.get("timeControl") or {},
            color=challenge.get("color", "random"),
            status=challenge.get("status"),
            decline_reason=challenge.get("declineReason"),
            raw=raw,
        )

    @property
    def is_bot_challenger(self) -> bool:
        return bool((self.raw.get("challenge", {}).get("challenger") or {}).get("title") == "BOT")

    @property
    def initial_seconds(self) -> Optional[int]:
        return self.time_control.get("limit")

    @property
    def increment_seconds(self) -> Optional[int]:
        return self.time_control.get("increment")


@dataclass(frozen=True)
class GameStartEvent:
    """A game we're now playing has begun (or resumed after a restart)."""

    id: str
    color: str
    fen: Optional[str]
    opponent: str
    rated: bool
    variant: str
    speed: str
    is_my_turn: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "GameStartEvent":
        game = raw.get("game", raw)
        opponent = game.get("opponent") or {}
        return cls(
            id=game.get("gameId") or game.get("id", ""),
            color=game.get("color", ""),
            fen=game.get("fen"),
            opponent=opponent.get("id") or opponent.get("username") or "",
            rated=bool(game.get("rated", False)),
            variant=(game.get("variant") or {}).get("key", "standard"),
            speed=game.get("speed", "unknown"),
            is_my_turn=bool(game.get("isMyTurn", False)),
            raw=raw,
        )


@dataclass(frozen=True)
class GameFinishEvent:
    """A game we were playing has ended."""

    id: str
    status: Optional[str]
    winner: Optional[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "GameFinishEvent":
        game = raw.get("game", raw)
        status = game.get("status")
        if isinstance(status, dict):
            status = status.get("name")
        return cls(
            id=game.get("gameId") or game.get("id", ""),
            status=status,
            winner=game.get("winner"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Game events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GameStateEvent:
    """A position update: the move list plus both clocks."""

    moves: list[str]
    wtime_ms: Optional[int]
    btime_ms: Optional[int]
    winc_ms: Optional[int]
    binc_ms: Optional[int]
    status: str
    winner: Optional[str]
    white_draw_offer: bool
    black_draw_offer: bool
    white_takeback_offer: bool
    black_takeback_offer: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "GameStateEvent":
        return cls(
            moves=_split_moves(raw.get("moves")),
            wtime_ms=to_millis(raw.get("wtime")),
            btime_ms=to_millis(raw.get("btime")),
            winc_ms=to_millis(raw.get("winc")),
            binc_ms=to_millis(raw.get("binc")),
            status=raw.get("status", "unknown"),
            winner=raw.get("winner"),
            white_draw_offer=bool(raw.get("wdraw", False)),
            black_draw_offer=bool(raw.get("bdraw", False)),
            white_takeback_offer=bool(raw.get("wtakeback", False)),
            black_takeback_offer=bool(raw.get("btakeback", False)),
            raw=raw,
        )

    @property
    def ply(self) -> int:
        return len(self.moves)

    @property
    def turn(self) -> str:
        """Which colour is to move, derived from ply parity."""
        return "white" if self.ply % 2 == 0 else "black"

    def is_my_turn(self, my_color: str) -> bool:
        return self.turn == my_color

    @property
    def is_finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def time_left_ms(self, color: str) -> Optional[int]:
        return self.wtime_ms if color == "white" else self.btime_ms

    def increment_ms(self, color: str) -> Optional[int]:
        return self.winc_ms if color == "white" else self.binc_ms

    def draw_offered_by_opponent(self, my_color: str) -> bool:
        return self.black_draw_offer if my_color == "white" else self.white_draw_offer

    def takeback_offered_by_opponent(self, my_color: str) -> bool:
        return (
            self.black_takeback_offer
            if my_color == "white"
            else self.white_takeback_offer
        )


@dataclass(frozen=True)
class GameFullEvent:
    """The opening frame of a game stream: metadata plus the initial state."""

    id: str
    white: str
    black: str
    rated: bool
    variant: str
    speed: str
    initial_fen: str
    clock_initial_ms: Optional[int]
    clock_increment_ms: Optional[int]
    state: GameStateEvent
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "GameFullEvent":
        white = raw.get("white") or {}
        black = raw.get("black") or {}
        clock = raw.get("clock") or {}
        return cls(
            id=raw.get("id", ""),
            white=white.get("id") or white.get("name") or "",
            black=black.get("id") or black.get("name") or "",
            rated=bool(raw.get("rated", False)),
            variant=(raw.get("variant") or {}).get("key", "standard"),
            speed=raw.get("speed", "unknown"),
            initial_fen=raw.get("initialFen") or "startpos",
            clock_initial_ms=to_millis(clock.get("initial")),
            clock_increment_ms=to_millis(clock.get("increment")),
            state=GameStateEvent.parse(raw.get("state") or {}),
            raw=raw,
        )

    def my_color(self, my_username: str) -> Optional[str]:
        """Which side we are playing, or None if we're only spectating."""
        target = (my_username or "").lower()
        if self.white.lower() == target:
            return "white"
        if self.black.lower() == target:
            return "black"
        return None

    def opponent(self, my_username: str) -> str:
        return self.black if self.my_color(my_username) == "white" else self.white


@dataclass(frozen=True)
class ChatLineEvent:
    username: str
    text: str
    room: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "ChatLineEvent":
        return cls(
            username=raw.get("username", ""),
            text=raw.get("text", ""),
            room=raw.get("room", "player"),
            raw=raw,
        )

    def is_from(self, username: str) -> bool:
        return self.username.lower() == (username or "").lower()


@dataclass(frozen=True)
class OpponentGoneEvent:
    gone: bool
    claim_win_in_seconds: Optional[int]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "OpponentGoneEvent":
        return cls(
            gone=bool(raw.get("gone", False)),
            claim_win_in_seconds=raw.get("claimWinInSeconds"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Handler interface
# --------------------------------------------------------------------------- #
class EventHandler:
    """Subclass and override whichever callbacks matter.

    Every method defaults to a no-op, so a handler that only cares about
    ``on_game_state`` can ignore the rest. Exceptions raised inside a callback
    are logged and swallowed by the dispatcher -- one bad move calculation
    should not kill the stream.
    """

    # -- account stream -- #
    def on_challenge(self, event: ChallengeEvent) -> None: ...

    def on_challenge_canceled(self, event: ChallengeEvent) -> None: ...

    def on_challenge_declined(self, event: ChallengeEvent) -> None: ...

    def on_game_start(self, event: GameStartEvent) -> None: ...

    def on_game_finish(self, event: GameFinishEvent) -> None: ...

    # -- game stream -- #
    def on_game_full(self, game_id: str, event: GameFullEvent) -> None: ...

    def on_game_state(self, game_id: str, event: GameStateEvent) -> None: ...

    def on_chat_line(self, game_id: str, event: ChatLineEvent) -> None: ...

    def on_opponent_gone(self, game_id: str, event: OpponentGoneEvent) -> None: ...

    # -- lifecycle -- #
    def on_stream_error(self, error: Exception) -> None:
        logger.error("Stream error: %s", error)

    def on_unknown_event(self, raw: dict[str, Any]) -> None:
        logger.debug("Unhandled event type %r", raw.get("type"))


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def dispatch_account_event(raw: dict[str, Any], handler: EventHandler) -> None:
    """Route one raw account-stream object to the matching callback."""
    event_type = raw.get("type")
    if event_type == AccountEvent.CHALLENGE:
        handler.on_challenge(ChallengeEvent.parse(raw))
    elif event_type == AccountEvent.CHALLENGE_CANCELED:
        handler.on_challenge_canceled(ChallengeEvent.parse(raw))
    elif event_type == AccountEvent.CHALLENGE_DECLINED:
        handler.on_challenge_declined(ChallengeEvent.parse(raw))
    elif event_type == AccountEvent.GAME_START:
        handler.on_game_start(GameStartEvent.parse(raw))
    elif event_type == AccountEvent.GAME_FINISH:
        handler.on_game_finish(GameFinishEvent.parse(raw))
    else:
        handler.on_unknown_event(raw)


def dispatch_game_event(
    game_id: str, raw: dict[str, Any], handler: EventHandler
) -> bool:
    """Route one raw game-stream object. Returns False once the game is over."""
    event_type = raw.get("type")
    if event_type == GameEvent.GAME_FULL:
        event = GameFullEvent.parse(raw)
        handler.on_game_full(game_id, event)
        return not event.state.is_finished
    if event_type == GameEvent.GAME_STATE:
        state = GameStateEvent.parse(raw)
        handler.on_game_state(game_id, state)
        return not state.is_finished
    if event_type == GameEvent.CHAT_LINE:
        handler.on_chat_line(game_id, ChatLineEvent.parse(raw))
        return True
    if event_type == GameEvent.OPPONENT_GONE:
        handler.on_opponent_gone(game_id, OpponentGoneEvent.parse(raw))
        return True
    handler.on_unknown_event(raw)
    return True


# --------------------------------------------------------------------------- #
# Stream runners
# --------------------------------------------------------------------------- #
class _BaseStream:
    """Shared reconnect/backoff machinery.

    Lichess closes idle streams routinely, so exhaustion is normal and means
    "reconnect", not "fail". Auth errors are fatal and stop the loop; rate
    limits pause for the full minute Lichess asks for.
    """

    #: Seconds to wait after a 429 before reconnecting.
    rate_limit_pause = 60.0
    #: Backoff bounds for ordinary reconnects.
    min_backoff = 1.0
    max_backoff = 60.0

    def __init__(self, api: LichessAPI, handler: EventHandler) -> None:
        self.api = api
        self.handler = handler
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask the loop to exit after the current iteration."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _sleep(self, seconds: float) -> bool:
        """Interruptible sleep. Returns True if we should keep going."""
        return not self._stop.wait(seconds)

    def _source(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def _handle(self, raw: dict[str, Any]) -> bool:
        """Process one event. Return False to end the loop for good."""
        raise NotImplementedError

    def run(self) -> None:
        backoff = self.min_backoff
        while not self._stop.is_set():
            try:
                for raw in self._source():
                    if self._stop.is_set():
                        return
                    backoff = self.min_backoff  # a live stream resets backoff
                    try:
                        if not self._handle(raw):
                            return
                    except Exception as exc:  # a bad callback must not kill us
                        logger.exception("Handler raised on %r", raw.get("type"))
                        self.handler.on_stream_error(exc)
            except LichessAuthError as exc:
                logger.error("Authentication failed, not retrying: %s", exc)
                self.handler.on_stream_error(exc)
                return
            except LichessRateLimitError as exc:
                logger.warning("Rate limited; pausing %.0fs", self.rate_limit_pause)
                self.handler.on_stream_error(exc)
                if not self._sleep(self.rate_limit_pause):
                    return
                continue
            except LichessError as exc:
                logger.warning("Stream dropped (%s); reconnecting in %.0fs", exc, backoff)
                self.handler.on_stream_error(exc)
                if not self._sleep(backoff):
                    return
                backoff = min(backoff * 2, self.max_backoff)
                continue

            if not self._reconnect_on_exhaustion():
                return
            logger.debug("Stream closed by server; reconnecting in %.0fs", backoff)
            if not self._sleep(backoff):
                return
            backoff = min(backoff * 2, self.max_backoff)

    def _reconnect_on_exhaustion(self) -> bool:
        return True


class AccountEventStream(_BaseStream):
    """Long-lived stream of challenges and game start/finish events.

    Run this on the main thread; spawn a :class:`GameEventStream` per game.
    """

    def _source(self) -> Iterator[dict[str, Any]]:
        return self.api.stream_events()

    def _handle(self, raw: dict[str, Any]) -> bool:
        dispatch_account_event(raw, self.handler)
        return True


class GameEventStream(_BaseStream):
    """State stream for one game. Ends by itself when the game finishes."""

    def __init__(
        self, api: LichessAPI, game_id: str, handler: EventHandler
    ) -> None:
        super().__init__(api, handler)
        self.game_id = game_id
        self._finished = False

    def _source(self) -> Iterator[dict[str, Any]]:
        return self.api.stream_game(self.game_id)

    def _handle(self, raw: dict[str, Any]) -> bool:
        alive = dispatch_game_event(self.game_id, raw, self.handler)
        if not alive:
            self._finished = True
            logger.info("Game %s finished; closing stream", self.game_id)
        return alive

    def _reconnect_on_exhaustion(self) -> bool:
        # Don't reconnect to a game that's already over.
        return not self._finished

    def run_in_thread(self, daemon: bool = True) -> threading.Thread:
        thread = threading.Thread(
            target=self.run, name=f"game-{self.game_id}", daemon=daemon
        )
        thread.start()
        return thread


__all__ = [
    "AccountEvent",
    "GameEvent",
    "TERMINAL_STATUSES",
    "to_millis",
    "ChallengeEvent",
    "GameStartEvent",
    "GameFinishEvent",
    "GameFullEvent",
    "GameStateEvent",
    "ChatLineEvent",
    "OpponentGoneEvent",
    "EventHandler",
    "dispatch_account_event",
    "dispatch_game_event",
    "AccountEventStream",
    "GameEventStream",
]