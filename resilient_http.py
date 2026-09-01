"""An async HTTP client for services that rate limit you and fail on you.

The interesting failure is not "the request failed". It is "the request may have
succeeded and we did not hear about it" — so retries here are method-aware, and
the classifier reports whether a failure could have left work applied on the
server (`Applied`). A POST that may already have run is not replayed unless the
caller supplied an idempotency key.

Scope: JSON over aiohttp, one host per client, retries and rate limiting.
Deliberately not here: circuit breaking, per-host budgets, connection-pool
tuning, streaming bodies, response caching. Python 3.12.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from enum import Enum
from math import isfinite
from typing import Any, Self

import aiohttp
from pydantic import BaseModel
from structlog import get_logger

type JSON = Mapping[str, JSON] | list[JSON] | str | int | float | bool | None

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[Any]]

logger = get_logger(__name__)

#: Methods the HTTP spec defines as idempotent: replaying one is, by contract,
#: indistinguishable from sending it once. POST and PATCH are not on this list.
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"})


class TransportError(Exception):
    """A request that will not be attempted again.

    Carries the context as attributes rather than only in the message, so a
    caller can branch on ``status`` or log ``attempts`` without parsing text.
    The originating exception is always chained.
    """

    def __init__(
        self,
        method: str,
        url: str,
        *,
        attempts: int,
        reason: str,
        status: int | None = None,
    ) -> None:
        super().__init__(f"{method} {url} gave up after {attempts} attempt(s): {reason}")
        self.method = method
        self.url = url
        self.attempts = attempts
        self.reason = reason
        self.status = status


class ProtocolError(TransportError):
    """A response arrived, but could not be read as the caller expects."""


# --------------------------------------------------------------------------- #
# Classifying a failure.
# --------------------------------------------------------------------------- #


class Applied(Enum):
    """Whether a failed request could have left work applied on the server."""

    NOT_SENT = "not_sent"  #: provably never processed — safe to replay anything
    UNKNOWN = "unknown"  #: may have run; replaying it may duplicate the effect


@dataclass(frozen=True, slots=True)
class Retryable:
    reason: str
    applied: Applied
    retry_after: float | None = None


def classify(exc: BaseException) -> Retryable | None:
    """Describe a failure, or return ``None`` if retrying cannot help."""
    match exc:
        case aiohttp.ClientResponseError(status=429, headers=headers):
            # An explicit rejection: the server declined to do the work.
            return Retryable("rate_limited", Applied.NOT_SENT, parse_retry_after(headers))
        case aiohttp.ClientResponseError(status=int(status), headers=headers) if status >= 500:
            # A 500 can be raised after the handler has already committed.
            return Retryable(f"server_error:{status}", Applied.UNKNOWN, parse_retry_after(headers))
        case aiohttp.ClientResponseError():
            return None  # 4xx is our bug; replaying it only burns quota.
        case aiohttp.ClientConnectorError():
            # No connection was ever established, so nothing was transmitted.
            return Retryable("connect_failed", Applied.NOT_SENT)
        case aiohttp.ServerDisconnectedError() | asyncio.TimeoutError():
            # The bytes went out. Whether they were acted on is unknowable.
            return Retryable("connection_lost", Applied.UNKNOWN)
        case aiohttp.ClientConnectionError():
            return Retryable("connection", Applied.UNKNOWN)
        case _:
            return None


def parse_retry_after(
    headers: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Both forms RFC 9110 allows: delta-seconds and HTTP-date.

    Returns ``None`` for anything unusable — absent, malformed, negative, NaN,
    infinite. ``None`` means "we have no instruction", never "retry now".
    """
    raw = (headers or {}).get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    if not isfinite(seconds) or seconds < 0:
        return None
    return seconds


# --------------------------------------------------------------------------- #
# Retry policy.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Immutable, and responsible for its own invariants.

    ``jitter`` returns a fraction in ``[0, 1)`` and is injectable so that tests
    can pin the delay instead of asserting on a distribution.
    """

    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {self.attempts}")
        for name in ("base_delay", "max_delay", "multiplier"):
            value: float = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if self.multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {self.multiplier}")
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay ({self.base_delay})",
            )

    def backoff(self, attempt: int) -> float:
        """Full jitter: sample ``[0, window)`` so a recovering fleet spreads out."""
        window = min(self.max_delay, self.base_delay * self.multiplier**attempt)
        return window * self.jitter()


# --------------------------------------------------------------------------- #
# Rate limiting.
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Admits at most ``rate`` calls per second, FIFO, across concurrent tasks.

    Callers reserve the next free slot on a virtual clock while holding the
    lock, then sleep *outside* it. The naive version — read the cursor, sleep,
    write it back — lets every waiter observe the same free slot and fire at
    once, which is exactly the burst the limiter exists to prevent.
    """

    __slots__ = ("_clock", "_interval", "_lock", "_next_slot", "_sleep")

    def __init__(
        self,
        rate: float,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        if not isfinite(rate) or rate <= 0:
            raise ValueError(f"rate must be finite and positive, got {rate}")
        self._interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._next_slot = 0.0
        self._clock = clock  # monotonic by default: immune to NTP steps and DST
        self._sleep = sleep

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        if (delay := slot - now) > 0:
            await self._sleep(delay)


# --------------------------------------------------------------------------- #
# Serialization.
# --------------------------------------------------------------------------- #


def _encode(value: Any) -> JSON:
    match value:
        case Decimal():
            return str(value)  # str, not float: a float round-trip loses cents
        case datetime() | date():
            return value.isoformat()
        case BaseModel():
            return value.model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_encode, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# The client.
# --------------------------------------------------------------------------- #


class HttpClient:
    """Async JSON client for one host. Use as an async context manager."""

    def __init__(
        self,
        base_url: str,
        *,
        rate: float = 10.0,
        policy: RetryPolicy | None = None,
        timeout: float = 30.0,
        auth: aiohttp.BasicAuth | None = None,
        limiter: RateLimiter | None = None,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._policy = policy or RetryPolicy()
        # Accepting a limiter lets several clients share one budget for a host.
        self._limiter = limiter or RateLimiter(rate, sleep=sleep)
        self._timeout = timeout
        self._auth = auth
        self._sleep = sleep
        self._session: aiohttp.ClientSession | None = None
        self._log = logger.bind(host=self._base_url)

    def _ensure_session(self) -> aiohttp.ClientSession:
        # Built on first use: a session must be created inside a running loop.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                raise_for_status=True,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                json_serialize=dumps,
                auth=self._auth,
            )
        return self._session

    async def request(
        self,
        method: str,
        path: str = "",
        *,
        idempotency_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> JSON:
        """Send a request, retrying only where a retry is safe.

        ``idempotency_key`` makes a non-idempotent method replayable: it is sent
        as ``Idempotency-Key`` so the server can collapse duplicates, and it is
        what permits this client to retry a POST whose outcome is unknown.
        """
        method = method.upper()
        url = f"{self._base_url}/{path.lstrip('/')}"
        if idempotency_key is not None:
            headers = {**(headers or {}), "Idempotency-Key": idempotency_key}

        session = self._ensure_session()
        last_exc: BaseException | None = None
        last_status: int | None = None

        for attempt in range(self._policy.attempts):
            await self._limiter.acquire()
            try:
                async with session.request(method, url, headers=headers, **kwargs) as response:
                    return await self._read(response, method, url, attempt + 1)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc, last_status = exc, getattr(exc, "status", None)
                delay, reason = self._plan(exc, attempt, method, idempotency_key)
                if delay is None:
                    raise TransportError(
                        method, url, attempts=attempt + 1, reason=reason, status=last_status,
                    ) from exc
                self._log.warning(
                    "request.retry",
                    method=method,
                    url=url,
                    reason=reason,
                    attempt=attempt + 1,
                    delay=round(delay, 3),
                )
                await self._sleep(delay)

        raise TransportError(
            method, url, attempts=self._policy.attempts,
            reason="retries exhausted", status=last_status,
        ) from last_exc

    def _plan(
        self,
        exc: BaseException,
        attempt: int,
        method: str,
        idempotency_key: str | None,
    ) -> tuple[float | None, str]:
        """How long to wait before the next attempt, or ``None`` to stop, and why.

        Every reason to give up is spelled out here rather than spread across
        the request loop, so the retry rules can be read in one place.
        """
        decision = classify(exc)
        if decision is None:
            return None, "not retryable"

        if attempt >= self._policy.attempts - 1:
            return None, f"{decision.reason}: no attempts left"

        if decision.applied is Applied.UNKNOWN and not _replayable(method, idempotency_key):
            return None, (
                f"{decision.reason}: {method} may already have been applied and "
                f"carries no idempotency key"
            )

        if decision.retry_after is not None:
            if decision.retry_after > self._policy.max_delay:
                # Coming back before the server permitted is worse than failing:
                # it is what turns a soft rate limit into a hard ban.
                return None, (
                    f"{decision.reason}: server asked for {decision.retry_after:.0f}s, "
                    f"beyond max_delay of {self._policy.max_delay:.0f}s"
                )
            return decision.retry_after, decision.reason

        return self._policy.backoff(attempt), decision.reason

    @staticmethod
    async def _read(
        response: aiohttp.ClientResponse,
        method: str,
        url: str,
        attempts: int,
    ) -> JSON:
        try:
            return await response.json()
        except aiohttp.ContentTypeError:
            return await response.text()  # some gateways lie about Content-Type
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Otherwise this escapes as a bare ValueError, bypassing the
            # error type every caller of this module is told to expect.
            raise ProtocolError(
                method, url, attempts=attempts,
                reason="malformed response body", status=response.status,
            ) from exc

    async def get(self, path: str = "", **kwargs: Any) -> JSON:
        return await self.request("GET", path, **kwargs)

    async def put(self, path: str = "", **kwargs: Any) -> JSON:
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str = "", **kwargs: Any) -> JSON:
        return await self.request("DELETE", path, **kwargs)

    async def post(
        self,
        path: str = "",
        *,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> JSON:
        """Not retried after an ambiguous failure unless a key is supplied."""
        return await self.request("POST", path, idempotency_key=idempotency_key, **kwargs)

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _replayable(method: str, idempotency_key: str | None) -> bool:
    return method in IDEMPOTENT_METHODS or idempotency_key is not None


# --------------------------------------------------------------------------- #
# Structured concurrency.
# --------------------------------------------------------------------------- #


async def map_concurrent[T, R](
    fn: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int = 10,
) -> list[R]:
    """Apply ``fn`` to every item, at most ``limit`` in flight, results in order.

    Exactly ``limit`` tasks are created and they pull from the source lazily, so
    memory tracks the concurrency limit rather than the length of the input.
    (An infinite iterable still never returns — it is bounded in space, not in
    time.) ``next()`` on the shared iterator never awaits, so no lock is needed
    to hand out work on a single event loop.

    ``TaskGroup`` over ``gather``: the first failure cancels its siblings rather
    than leaving them to finish into a void.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    source = enumerate(items)
    results: dict[int, R] = {}

    async def worker() -> None:
        for index, item in source:
            results[index] = await fn(item)

    async with asyncio.TaskGroup() as group:
        for _ in range(limit):
            group.create_task(worker())

    return [results[index] for index in range(len(results))]
