# Resilient HTTP Client — Python work sample

A retry layer for a service that moves money. Extracted from a production RabbitMQ
worker that executes Solana swaps, then rewritten for Python 3.12.

The hard problem in a retry layer is not that a request failed. It is that **it may
have succeeded and you did not hear about it**. A connection dropped after the bytes
went out looks identical, from the client, to one that was refused — but replaying
the first can pay a transfer twice. So every failure here is classified by whether
work could already have been applied on the server, and a POST that might have run is
not replayed unless the caller supplied an idempotency key.

    resilient_http.py         the module — 461 lines, fully annotated
    test_resilient_http.py    71 tests against a real server, 0.6s
    requirements.txt          aiohttp, pydantic, structlog, pytest
    pytest.ini                asyncio auto mode

## Running it

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest

Requires Python 3.12 (PEP 695 type-parameter syntax).

## The design decision

`classify()` reports two things about a failure: why it happened, and whether the
server could already have acted on it.

- `ClientConnectorError` — no connection was established, so nothing was transmitted.
  `Applied.NOT_SENT`. Safe to replay anything, including a POST.
- `429` — an explicit refusal. The server declined to do the work. `NOT_SENT`.
- A timeout, or a disconnect mid-flight — the bytes went out and the outcome is
  unknowable. `Applied.UNKNOWN`.
- `5xx` — can be raised after the handler has already committed. `UNKNOWN`.
- `4xx` — our bug. Not retryable at all; replaying it only burns quota.

`HttpClient._plan()` then holds every reason to stop, in one place, and returns the
reason it stopped so it can become the `TransportError` message: not retryable, no
attempts left, the method may already have been applied and carries no idempotency
key, or the server asked for longer than this policy is willing to wait.

## Trade-offs

**A 5xx is treated as ambiguous, which costs availability.** A 502 from a gateway
almost certainly means the request never reached the application, so a POST fails here
that could safely have been retried. I would rather explain a failed transfer than a
doubled one. Per-status tuning belongs in the policy, not hard-coded in the client.

**`Retry-After` beyond `max_delay` aborts rather than waits.** Capping the server's
instruction at your own ceiling looks like a safety valve and is the opposite: told to
wait an hour, you return in thirty seconds — earlier than permitted, by a client that
believes it is being polite. The cap decides *whether* to retry, never *when*.

**`map_concurrent` is bounded in space, not in time.** It creates exactly `limit`
tasks that pull from the source lazily, so a million-item input costs a million items
of iteration rather than a million `Task` objects. An infinite iterable still never
returns; the docstring says so.

**Clock, sleeper and jitter are constructor parameters.** Nothing in the test suite
patches a module global or asserts on a distribution, which is why 71 tests finish in
under a second and do not go flaky on a loaded CI box.

## What this deliberately does not do

- Circuit breaking, per-host request budgets, connection-pool tuning. This layer
  retries; it does not shed load.
- Streaming bodies. Responses are buffered — right for JSON APIs, wrong for large
  downloads.
- Deduplicate anything. The idempotency key is passed through as a header; collapsing
  duplicates is the server's job. The client only decides whether replaying is allowed.
- Serve more than one host per instance, since the rate limiter's budget is
  per-instance. Pass a shared `RateLimiter` to give several clients one budget.

## On the tests

The retry loop runs against a real `aiohttp` server on localhost — real status codes,
real headers, real refused connections, real timeouts, real malformed bodies. Nothing
about the transport is mocked. What is substituted is only what makes tests slow or
non-deterministic: the clock, the sleeper, and the jitter source, all injected through
constructors.

That makes the assertions exact rather than statistical. `sleeper.delays == [1.0, 2.0]`
after two 503s. `== [7.0]` after a `Retry-After: 7`. `== []` when the server asks for
an hour and the policy caps at thirty seconds.

The test that matters most is `test_ambiguous_post_is_not_replayed`: it fails if anyone
ever makes the retry loop method-blind again.

## Defects found in the version that shipped

Revisiting the original turned up four, all of which only appear under load or during
an outage — the conditions a retry layer exists for.

1. **The backoff was computed and then dropped.** The error handler parsed
   `Retry-After`, chose a delay, logged it, and returned to a loop that reissued the
   request immediately. Under a 429 that is three requests in a few milliseconds,
   which is how a soft rate limit becomes a hard ban.
2. **The throttler could be beaten by its own callers.** It read a shared cursor,
   compared it to the clock, then wrote it back. Any number of coroutines could pass
   the comparison before the first one wrote, and all fire together.
3. **A fixed delay meant a synchronised stampede.** Every worker retried after exactly
   five seconds, so one outage brought the whole fleet back in lockstep.
4. **Responses were released by hand in a `finally`.** `async with session.request(…)`
   covers every path, including cancellation.
