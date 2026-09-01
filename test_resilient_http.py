"""Tests for resilient_http.

The retry loop is exercised end to end against a real aiohttp server on
localhost rather than a mocked session: real status codes, real headers, real
dropped connections. What is faked is only what makes tests slow or
non-deterministic — the clock, the sleeper, and the jitter source.

Run: pytest -q
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import BaseModel

from resilient_http import (
    Applied,
    HttpClient,
    ProtocolError,
    RateLimiter,
    RetryPolicy,
    TransportError,
    classify,
    dumps,
    map_concurrent,
    parse_retry_after,
)

NO_JITTER = RetryPolicy(attempts=3, base_delay=1.0, jitter=lambda: 1.0)


class Recorder:
    """Stands in for asyncio.sleep: records the delay, returns immediately."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture
async def serve():
    """Start a one-route server and hand back a client pointed at it."""
    servers: list[TestServer] = []
    clients: list[HttpClient] = []

    async def start(handler: Callable[..., Any], **client_kwargs: Any) -> HttpClient:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        client = HttpClient(
            str(server.make_url("")),
            limiter=RateLimiter(1e6, sleep=_instant),
            **client_kwargs,
        )
        clients.append(client)
        return client

    yield start

    for client in clients:
        await client.aclose()
    for server in servers:
        await server.close()


async def _instant(_: float) -> None:
    return None


def counting(*responses: Callable[[], web.StreamResponse]):
    """A handler that plays the given responses in order, then repeats the last."""
    seen: list[web.Request] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        seen.append(request)
        factory = responses[min(len(seen) - 1, len(responses) - 1)]
        return factory()

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def ok(payload: Any = None) -> Callable[[], web.StreamResponse]:
    return lambda: web.json_response(payload if payload is not None else {"ok": True})


def status(code: int, **headers: str) -> Callable[[], web.StreamResponse]:
    return lambda: web.Response(status=code, headers=headers)


# --------------------------------------------------------------------------- #
# The retry loop, end to end.
# --------------------------------------------------------------------------- #


async def test_server_error_is_retried_until_it_succeeds(serve):
    handler = counting(status(503), status(503), ok({"value": 42}))
    sleeper = Recorder()
    client = await serve(handler, policy=NO_JITTER, sleep=sleeper)

    assert await client.get("/thing") == {"value": 42}
    assert len(handler.seen) == 3
    assert sleeper.delays == [1.0, 2.0]  # base_delay, then multiplied


async def test_client_error_is_not_retried(serve):
    handler = counting(status(400))
    sleeper = Recorder()
    client = await serve(handler, policy=NO_JITTER, sleep=sleeper)

    with pytest.raises(TransportError) as excinfo:
        await client.get("/thing")

    assert len(handler.seen) == 1, "a 400 replayed is a 400 again"
    assert excinfo.value.status == 400
    assert excinfo.value.attempts == 1
    assert excinfo.value.reason == "not retryable"
    assert isinstance(excinfo.value.__cause__, aiohttp.ClientResponseError)


async def test_exhausting_attempts_reports_how_many_were_made(serve):
    handler = counting(status(500))
    client = await serve(handler, policy=NO_JITTER, sleep=Recorder())

    with pytest.raises(TransportError) as excinfo:
        await client.get("/thing")

    assert len(handler.seen) == 3
    assert excinfo.value.attempts == 3
    assert "no attempts left" in excinfo.value.reason


async def test_retry_after_is_waited_exactly(serve):
    handler = counting(status(429, **{"Retry-After": "7"}), ok())
    sleeper = Recorder()
    client = await serve(handler, policy=NO_JITTER, sleep=sleeper)

    await client.get("/thing")
    assert sleeper.delays == [7.0], "the server's instruction beats our backoff"


async def test_retry_after_beyond_max_delay_gives_up_instead_of_returning_early(serve):
    handler = counting(status(429, **{"Retry-After": "3600"}))
    sleeper = Recorder()
    policy = RetryPolicy(attempts=3, base_delay=1.0, max_delay=30.0, jitter=lambda: 1.0)
    client = await serve(handler, policy=policy, sleep=sleeper)

    with pytest.raises(TransportError) as excinfo:
        await client.get("/thing")

    assert sleeper.delays == [], "returning before the server permitted is worse than failing"
    assert len(handler.seen) == 1
    assert "beyond max_delay" in excinfo.value.reason


async def test_timeout_is_retried_for_a_safe_method(serve):
    async def slow(request: web.Request) -> web.StreamResponse:
        slow.seen.append(request)
        if len(slow.seen) == 1:
            await asyncio.sleep(5)
        return web.json_response({"ok": True})

    slow.seen = []
    client = await serve(slow, policy=NO_JITTER, sleep=Recorder(), timeout=0.05)

    assert await client.get("/thing") == {"ok": True}
    assert len(slow.seen) == 2


async def test_malformed_json_surfaces_as_protocol_error(serve):
    handler = counting(lambda: web.Response(text="{not json", content_type="application/json"))
    client = await serve(handler, policy=NO_JITTER, sleep=Recorder())

    with pytest.raises(ProtocolError) as excinfo:
        await client.get("/thing")

    assert isinstance(excinfo.value, TransportError), "one error type escapes this module"
    assert excinfo.value.status == 200
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
    assert len(handler.seen) == 1, "a malformed body is not a transport failure"


async def test_wrong_content_type_still_yields_the_body(serve):
    handler = counting(lambda: web.Response(text="plain", content_type="text/plain"))
    client = await serve(handler, policy=NO_JITTER, sleep=Recorder())
    assert await client.get("/thing") == "plain"


# --------------------------------------------------------------------------- #
# Duplicate side effects: the reason retries are method-aware.
# --------------------------------------------------------------------------- #


async def test_ambiguous_post_is_not_replayed(serve):
    """A 500 may be raised after the handler committed. Replaying it may pay twice."""
    handler = counting(status(500), ok())
    sleeper = Recorder()
    client = await serve(handler, policy=NO_JITTER, sleep=sleeper)

    with pytest.raises(TransportError) as excinfo:
        await client.post("/transfer", json={"amount": 100})

    assert len(handler.seen) == 1, "the transfer must not be sent twice"
    assert sleeper.delays == []
    assert "may already have been applied" in excinfo.value.reason


async def test_an_idempotency_key_makes_the_post_replayable(serve):
    handler = counting(status(500), ok({"transfer": "done"}))
    client = await serve(handler, policy=NO_JITTER, sleep=Recorder())

    result = await client.post("/transfer", json={"amount": 100}, idempotency_key="abc-123")

    assert result == {"transfer": "done"}
    assert len(handler.seen) == 2
    assert {r.headers["Idempotency-Key"] for r in handler.seen} == {"abc-123"}


async def test_a_rejected_post_is_replayed_without_a_key(serve):
    """429 is an explicit refusal: the server did not do the work."""
    handler = counting(status(429, **{"Retry-After": "1"}), ok())
    client = await serve(handler, policy=NO_JITTER, sleep=Recorder())

    await client.post("/transfer", json={"amount": 100})
    assert len(handler.seen) == 2


async def test_a_post_that_never_left_is_replayed_without_a_key():
    """Connection refused: nothing was transmitted, so nothing was applied."""
    sleeper = Recorder()
    client = HttpClient(
        "http://127.0.0.1:1",  # reserved, nothing listens
        policy=NO_JITTER,
        sleep=sleeper,
        limiter=RateLimiter(1e6, sleep=_instant),
    )
    with pytest.raises(TransportError) as excinfo:
        await client.post("/transfer", json={"amount": 100})

    assert len(sleeper.delays) == 2, "attempted three times"
    assert "connect_failed" in excinfo.value.reason
    await client.aclose()


async def test_idempotent_methods_are_replayed_without_a_key(serve):
    for method in ("get", "put", "delete"):
        handler = counting(status(500), ok())
        client = await serve(handler, policy=NO_JITTER, sleep=Recorder())
        await getattr(client, method)("/thing")
        assert len(handler.seen) == 2, method


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


def response_error(status_code: int, **headers: str) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(None, (), status=status_code, headers=headers)


def test_rate_limiting_means_nothing_was_applied():
    decision = classify(response_error(429, **{"Retry-After": "7"}))
    assert decision is not None
    assert (decision.reason, decision.applied, decision.retry_after) == (
        "rate_limited", Applied.NOT_SENT, 7.0,
    )


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_server_errors_are_retryable_but_ambiguous(code: int):
    decision = classify(response_error(code))
    assert decision is not None and decision.applied is Applied.UNKNOWN


def test_server_errors_also_honour_retry_after():
    decision = classify(response_error(503, **{"Retry-After": "12"}))
    assert decision is not None and decision.retry_after == 12.0


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422])
def test_client_errors_are_not_retryable(code: int):
    assert classify(response_error(code)) is None


def test_a_timeout_is_ambiguous():
    decision = classify(asyncio.TimeoutError())
    assert decision is not None and decision.applied is Applied.UNKNOWN


def test_unrelated_exceptions_are_not_retryable():
    assert classify(ValueError("bad input")) is None


# --------------------------------------------------------------------------- #
# Retry-After parsing
# --------------------------------------------------------------------------- #


def test_delta_seconds_form():
    assert parse_retry_after({"Retry-After": "120"}) == 120.0


def test_http_date_form():
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    headers = {"Retry-After": "Mon, 01 Jan 2024 12:02:00 GMT"}
    assert parse_retry_after(headers, now=now) == 120.0


def test_an_http_date_in_the_past_is_ignored():
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    headers = {"Retry-After": "Mon, 01 Jan 2024 11:00:00 GMT"}
    assert parse_retry_after(headers, now=now) is None


@pytest.mark.parametrize("raw", ["-1", "nan", "inf", "-inf", "soon", "", "1e400"])
def test_unusable_values_yield_no_instruction(raw: str):
    assert parse_retry_after({"Retry-After": raw}) is None


def test_absent_header():
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None


# --------------------------------------------------------------------------- #
# RetryPolicy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempts": 0},
        {"attempts": -1},
        {"base_delay": 0},
        {"base_delay": -1.0},
        {"base_delay": float("nan")},
        {"max_delay": float("inf")},
        {"multiplier": 0.5},
        {"base_delay": 60.0, "max_delay": 30.0},
    ],
)
def test_a_policy_that_cannot_work_cannot_be_built(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_backoff_grows_then_flattens_at_max_delay():
    policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=8.0, jitter=lambda: 1.0)
    assert [policy.backoff(i) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_scales_the_whole_window():
    policy = RetryPolicy(base_delay=4.0, jitter=lambda: 0.25)
    assert policy.backoff(0) == 1.0


def test_the_default_jitter_actually_spreads_the_retries():
    policy = RetryPolicy(base_delay=4.0)
    assert len({policy.backoff(2) for _ in range(50)}) > 45


def test_a_policy_is_immutable():
    with pytest.raises(Exception):
        RetryPolicy().attempts = 99


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


async def test_concurrent_callers_are_spaced_not_bursted():
    """The regression this class exists for: ten tasks must not fire as one."""
    slept: list[float] = []
    limiter = RateLimiter(
        2.0,
        clock=lambda: 0.0,  # time never advances, so every caller must wait
        sleep=lambda d: slept.append(d) or asyncio.sleep(0),
    )

    await asyncio.gather(*(limiter.acquire() for _ in range(5)))

    assert slept == [0.5, 1.0, 1.5, 2.0], "each caller waits one more interval"


async def test_the_first_caller_is_not_delayed():
    slept: list[float] = []
    limiter = RateLimiter(1.0, sleep=lambda d: slept.append(d) or asyncio.sleep(0))
    await limiter.acquire()
    assert slept == []


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_rate_must_be_finite_and_positive(rate: float):
    with pytest.raises(ValueError):
        RateLimiter(rate)


def test_limiter_has_no_instance_dict():
    assert not hasattr(RateLimiter(1.0), "__dict__")


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


class Order(BaseModel):
    amount: Decimal


def test_decimals_do_not_go_through_float():
    assert dumps({"fee": Decimal("0.10")}) == '{"fee":"0.10"}'


def test_dates_are_isoformatted():
    assert dumps({"at": datetime(2024, 1, 2, 3, 4, 5)}) == '{"at":"2024-01-02T03:04:05"}'
    assert dumps({"on": date(2024, 1, 2)}) == '{"on":"2024-01-02"}'


def test_models_are_encoded_in_json_mode():
    assert dumps({"order": Order(amount=Decimal("1.5"))}) == '{"order":{"amount":"1.5"}}'


def test_unknown_types_are_reported_by_name():
    with pytest.raises(TypeError, match="object is not JSON serializable"):
        dumps({"x": object()})


# --------------------------------------------------------------------------- #
# map_concurrent
# --------------------------------------------------------------------------- #


async def test_results_keep_input_order_regardless_of_completion_order():
    async def slow_for_small_n(n: int) -> int:
        await asyncio.sleep(0.01 * (5 - n))
        return n * n

    assert await map_concurrent(slow_for_small_n, range(5), limit=3) == [0, 1, 4, 9, 16]


async def test_limit_is_respected():
    in_flight = peak = 0

    async def tracked(n: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return n

    await map_concurrent(tracked, range(20), limit=4)
    assert peak == 4


async def test_the_source_is_consumed_lazily():
    """Memory tracks the limit, not the length of the input."""
    outstanding = 0

    def counted():
        nonlocal outstanding
        for n in range(50):
            outstanding += 1
            yield n

    async def slow(n: int) -> int:
        await asyncio.sleep(0.005)
        return n

    task = asyncio.create_task(map_concurrent(slow, counted(), limit=3))
    await asyncio.sleep(0.001)
    assert outstanding <= 3, "all 50 items were pulled up front"
    assert await task == list(range(50))


@pytest.mark.parametrize("limit", [0, -1])
async def test_a_limit_that_would_deadlock_is_rejected(limit: int):
    async def noop(n: int) -> int:
        return n

    with pytest.raises(ValueError, match="limit must be >= 1"):
        await map_concurrent(noop, range(3), limit=limit)


async def test_empty_input():
    async def noop(n: int) -> int:
        return n

    assert await map_concurrent(noop, [], limit=4) == []


async def test_first_failure_cancels_its_siblings():
    cancelled = 0

    async def fail_on_two(n: int) -> int:
        nonlocal cancelled
        if n == 2:
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return n

    with pytest.raises(BaseExceptionGroup) as excinfo:
        await map_concurrent(fail_on_two, range(6), limit=6)

    assert isinstance(excinfo.value.exceptions[0], RuntimeError)
    assert cancelled >= 1


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


async def test_closing_an_unused_client_does_not_open_a_session():
    client = HttpClient("https://api.example.com")
    await client.aclose()
    assert client._session is None


async def test_close_is_idempotent(serve):
    client = await serve(counting(ok()))
    await client.get("/thing")
    await client.aclose()
    await client.aclose()


async def test_a_client_can_be_reused_after_closing(serve):
    client = await serve(counting(ok({"n": 1})))
    assert await client.get("/thing") == {"n": 1}
    await client.aclose()
    assert await client.get("/thing") == {"n": 1}
