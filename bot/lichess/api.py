"""Lichess connection wrapper, built on the official ``berserk`` SDK.

This module owns the connection: credentials, session construction, error
translation and stream lifetime. It deliberately keeps no chess state -- that
belongs in ``game.py``, and dispatch belongs in ``game_events.py``.

The public surface is stable even though berserk's kwargs shift between
releases, so ``bot.py`` should call these methods rather than reaching for
``api.client`` directly.

Usage::

    from lichess.api import LichessAPI

    api = LichessAPI()                     # reads LICHESS_API_KEY from secrets.env
    print(api.get_account()["username"])

    for event in api.stream_events():
        if event["type"] == "challenge":
            api.accept_challenge(event["challenge"]["id"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import berserk
from berserk.exceptions import ApiError, ResponseError

logger = logging.getLogger(__name__)

SECRETS_FILENAME = "secrets.env"
API_KEY_NAME = "LICHESS_API_KEY"


# --------------------------------------------------------------------------- #
# Exceptions -- berserk's errors are re-raised as these so that bot.py never
# has to import berserk itself.
# --------------------------------------------------------------------------- #
class LichessError(Exception):
    """Base class for every error raised by this module."""


class LichessConfigError(LichessError):
    """Missing or malformed credentials."""


class LichessAuthError(LichessError):
    """401 / 403 -- token is missing, expired, or lacks the bot:play scope."""


class LichessNotFoundError(LichessError):
    """404 -- game, challenge or user does not exist."""


class LichessRateLimitError(LichessError):
    """429 -- Lichess wants us to back off for a full minute."""


class LichessResponseError(LichessError):
    """Any other non-2xx response."""

    def __init__(self, status_code: Optional[int], message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


# --------------------------------------------------------------------------- #
# Credential loading
# --------------------------------------------------------------------------- #
def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser -- avoids a python-dotenv dependency."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip("'\"")
    return values


def find_secrets_file(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from ``start`` looking for ``secrets.env``.

    Defaults to this file's directory, so ``bot/lichess/api.py`` finds
    the repo root ``secrets.env`` regardless of the working directory at launch.
    """
    current = (start or Path(__file__).resolve().parent).resolve()
    for directory in (current, *current.parents):
        candidate = directory / SECRETS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_api_key(secrets_path: Optional[Path] = None) -> str:
    """Resolve the token from the environment first, then from secrets.env."""
    token = os.environ.get(API_KEY_NAME)
    if token:
        return token.strip()

    path = secrets_path or find_secrets_file()
    if path is None:
        raise LichessConfigError(
            f"{API_KEY_NAME} is not set and no {SECRETS_FILENAME} was found in "
            f"any parent of {Path(__file__).resolve().parent}."
        )

    token = _parse_env_file(path).get(API_KEY_NAME, "").strip()
    if not token:
        raise LichessConfigError(f"{API_KEY_NAME} is missing or empty in {path}.")
    logger.debug("Loaded %s from %s", API_KEY_NAME, path)
    return token


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class LichessAPI:
    """Authenticated Lichess connection backed by ``berserk.Client``.

    Args:
        token: Explicit API token. Falls back to the ``LICHESS_API_KEY``
            environment variable, then to ``secrets.env``.
        secrets_path: Explicit path to the secrets file.
        base_url: Override for testing against a local lila instance.
        session: Inject a pre-built session (e.g. a ``requests_oauth`` one).
            berserk accepts any ``requests.Session``-like object.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        secrets_path: Optional[os.PathLike[str] | str] = None,
        base_url: Optional[str] = None,
        session: Any = None,
    ) -> None:
        self._token = token or load_api_key(
            Path(secrets_path) if secrets_path else None
        )
        self.session = session or berserk.TokenSession(self._token)

        kwargs: dict[str, Any] = {"session": self.session}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = berserk.Client(**kwargs)

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "LichessAPI":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        closer = getattr(self.session, "close", None)
        if callable(closer):
            closer()

    def __repr__(self) -> str:  # never leak the token
        return "<LichessAPI berserk client, token=***>"

    # -- error translation -------------------------------------------------- #
    @staticmethod
    def _translate(exc: Exception) -> LichessError:
        """Map a berserk exception onto this module's exception hierarchy."""
        status = getattr(exc, "status_code", None)
        detail = str(getattr(exc, "reason", None) or exc)[:500]

        if status in (401, 403):
            return LichessAuthError(
                f"HTTP {status}: {detail or 'token rejected'}. Check the token "
                "is valid and carries the bot:play scope."
            )
        if status == 404:
            return LichessNotFoundError(f"HTTP 404: {detail}")
        if status == 429:
            return LichessRateLimitError(
                f"Rate limited: {detail}. Lichess expects a 60s pause."
            )
        return LichessResponseError(status, detail)

    def _call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke a berserk method, re-raising failures as LichessError."""
        try:
            return func(*args, **kwargs)
        except ResponseError as exc:
            raise self._translate(exc) from exc
        except ApiError as exc:
            raise LichessError(str(exc)) from exc

    def _stream(self, func: Any, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Wrap a berserk generator so mid-stream errors are translated too.

        berserk yields decoded NDJSON objects and swallows the keep-alive blank
        lines itself. The generator ends when Lichess closes the connection --
        ``bot.py`` should treat exhaustion as "reconnect", not as fatal.
        """
        try:
            for item in self._call(func, *args, **kwargs):
                yield item
        except ResponseError as exc:
            raise self._translate(exc) from exc
        except ApiError as exc:
            raise LichessError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    def get_account(self) -> dict[str, Any]:
        """Profile of the authenticated user."""
        return self._call(self.client.account.get)

    def get_email(self) -> str:
        result = self._call(self.client.account.get_email)
        return result if isinstance(result, str) else result.get("email", "")

    def get_preferences(self) -> dict[str, Any]:
        return self._call(self.client.account.get_preferences)

    def upgrade_to_bot(self) -> Any:
        """Irreversibly convert this account to a BOT account.

        Only works on an account that has never played a game.
        """
        return self._call(self.client.account.upgrade_to_bot)

    def is_bot_account(self) -> bool:
        return self.get_account().get("title") == "BOT"

    def get_ongoing_games(self, count: int = 10) -> list[dict[str, Any]]:
        return self._call(self.client.games.get_ongoing, count=count)

    # ------------------------------------------------------------------ #
    # Event stream
    # ------------------------------------------------------------------ #
    def stream_events(self) -> Iterator[dict[str, Any]]:
        """Incoming challenges and game start/finish events.

        Event ``type`` is one of ``challenge``, ``challengeCanceled``,
        ``challengeDeclined``, ``gameStart``, ``gameFinish``.
        """
        yield from self._stream(self.client.bots.stream_incoming_events)

    # ------------------------------------------------------------------ #
    # Challenges
    # ------------------------------------------------------------------ #
    def create_challenge(
        self,
        username: str,
        *,
        rated: bool = False,
        clock_limit: Optional[int] = None,
        clock_increment: Optional[int] = None,
        days: Optional[int] = None,
        color: str = "random",
        variant: str = "standard",
        position: Optional[str] = None,
    ) -> dict[str, Any]:
        """Challenge a user. ``clock_limit`` in seconds; ``days`` for correspondence."""
        return self._call(
            self.client.challenges.create,
            username,
            rated=rated,
            clock_limit=clock_limit,
            clock_increment=clock_increment,
            days=days,
            color=color,
            variant=variant,
            position=position,
        )

    def create_ai_challenge(
        self,
        *,
        level: int = 1,
        clock_limit: Optional[int] = None,
        clock_increment: Optional[int] = None,
        color: str = "random",
        variant: str = "standard",
    ) -> dict[str, Any]:
        """Play the Stockfish sparring partner -- handy for smoke tests."""
        return self._call(
            self.client.challenges.create_ai,
            level=level,
            clock_limit=clock_limit,
            clock_increment=clock_increment,
            color=color,
            variant=variant,
        )

    def accept_challenge(self, challenge_id: str) -> Any:
        return self._call(self.client.bots.accept_challenge, challenge_id)

    def decline_challenge(self, challenge_id: str, reason: str = "generic") -> Any:
        """Reasons: generic, later, tooFast, tooSlow, timeControl, rated,
        casual, standard, variant, noBot, onlyBot."""
        return self._call(
            self.client.bots.decline_challenge, challenge_id, reason=reason
        )

    def cancel_challenge(self, challenge_id: str) -> Any:
        return self._call(self.client.challenges.cancel, challenge_id)

    def list_challenges(self) -> dict[str, Any]:
        return self._call(self.client.challenges.get_mine)

    # ------------------------------------------------------------------ #
    # Game play
    # ------------------------------------------------------------------ #
    def stream_game(self, game_id: str) -> Iterator[dict[str, Any]]:
        """Full state of one game as it happens.

        First object is ``gameFull``; subsequent ones are ``gameState`` or
        ``chatLine``. Note berserk converts timestamps to datetimes, so clock
        fields arrive as ``timedelta``/``datetime``, not raw milliseconds.
        """
        yield from self._stream(self.client.bots.stream_game_state, game_id)

    def make_move(
        self, game_id: str, move: str, offering_draw: bool = False
    ) -> Any:
        """Play a move in UCI format, e.g. ``e2e4`` or ``e7e8q``."""
        return self._call(
            self.client.bots.make_move, game_id, move, offering_draw=offering_draw
        )

    def resign_game(self, game_id: str) -> Any:
        return self._call(self.client.bots.resign_game, game_id)

    def abort_game(self, game_id: str) -> Any:
        """Only valid before either side has played two moves."""
        return self._call(self.client.bots.abort_game, game_id)

    def handle_draw(self, game_id: str, accept: bool = True) -> Any:
        """Offer/accept a draw when ``accept`` is True, otherwise decline."""
        return self._call(self.client.bots.handle_draw_offer, game_id, accept)

    def handle_takeback(self, game_id: str, accept: bool = True) -> Any:
        return self._call(self.client.bots.handle_takeback_offer, game_id, accept)

    def send_chat(self, game_id: str, text: str, spectator: bool = False) -> Any:
        return self._call(
            self.client.bots.post_message, game_id, text, spectator=spectator
        )

    def get_chat(self, game_id: str) -> list[dict[str, Any]]:
        return self._call(self.client.board.get_game_chat, game_id)

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def get_user(self, username: str) -> dict[str, Any]:
        return self._call(self.client.users.get_public_data, username)

    def get_online_bots(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._call(self.client.bots.get_online_bots, limit))

    def export_game(self, game_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._call(self.client.games.export, game_id, **kwargs)


__all__ = [
    "LichessAPI",
    "LichessError",
    "LichessConfigError",
    "LichessAuthError",
    "LichessNotFoundError",
    "LichessRateLimitError",
    "LichessResponseError",
    "load_api_key",
    "find_secrets_file",
]