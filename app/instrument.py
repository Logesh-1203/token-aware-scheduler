"""instrument_stream: wrap a backend's NDJSON stream so every request is measured and recorded.

Wrap the iterator you already hand to StreamingResponse in the gateway:

    StreamingResponse(instrument_stream(upstream, store=store, hub=hub, request_id=rid, ...),
                      media_type="application/x-ndjson")

What it does, per request:
  * records IN_PROGRESS at the start and publishes a live event
  * measures TTFT = time to the first NON-EMPTY `token` event (never header arrival)
  * takes token counts / durations from the backend's final `done` event (Ollama metadata)
  * writes exactly one final row (SUCCESS or FAILED) - also when the backend crashes
    mid-stream or the client disconnects
  * refuses to call a REAL request successful unless the backend says mode=REAL and sent
    authoritative metadata (so mock `token1token2...` output can never pass as Qwen3)
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Optional

log = logging.getLogger("horizon.metrics")

_META_KEYS = ("prompt_tokens", "output_tokens", "tokens_per_second",
              "prompt_eval_s", "load_s", "ollama_total_s")


def _line(ev: dict) -> str:
    return json.dumps(ev, ensure_ascii=False) + "\n"


def _text(raw) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "replace")
    return raw if isinstance(raw, str) else json.dumps(raw)


def _parse(raw) -> Optional[dict]:
    if isinstance(raw, dict):  # backend iterator may yield event dicts instead of NDJSON text
        return raw
    try:
        ev = json.loads(_text(raw))
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


def _outcome(final: dict, mode: str):
    if final.get("type") == "error":
        return "FAILED", {}, f"{final.get('kind', 'error')}: {final.get('error', 'unknown error')}"
    if mode == "REAL" and (final.get("mode") != "REAL" or final.get("output_tokens") is None):
        return ("FAILED", {},
                "invalid_real_response: real mode was requested but the backend did not return "
                "authoritative Ollama metadata (mock/simulated output?)")
    return "SUCCESS", {k: final.get(k) for k in _META_KEYS}, None


async def instrument_stream(inner: AsyncIterator, *, store, hub, request_id: str, prompt: str,
                            model: Optional[str], mode: str, backend: Optional[str] = None,
                            policy: Optional[str] = None, profile: Optional[str] = None,
                            retries: int = 0, t0: Optional[float] = None) -> AsyncIterator[str]:
    """`mode` is "REAL" or "SIMULATED". Pass `t0=time.perf_counter()` captured at the top of the
    route handler if you also want scheduler queue-wait included in server-side latency."""
    t0 = time.perf_counter() if t0 is None else t0
    rid = request_id
    try:
        rid = store.start(request_id, prompt=prompt, model=model, mode=mode, backend=backend,
                          policy=policy, profile=profile)["request_id"]
    except Exception:  # metrics must never break inference
        log.exception("metrics: could not record request start")
    _publish(hub, {"event": "start", "request_id": rid, "mode": mode})

    ttft: Optional[float] = None
    final: Optional[dict] = None
    finalized = False

    def finalize(status: str, fields: dict) -> None:
        nonlocal finalized
        finalized = True
        try:
            store.finish(rid, status=status, retries=retries, **fields)
        except Exception:
            log.exception("metrics: could not record request result")
        _publish(hub, {"event": "finish", "request_id": rid, "status": status})

    def envelope(ev: dict, latency: float) -> dict:
        return {**ev, "request_id": rid, "model": model, "mode": mode,
                "server_latency_s": latency, "server_ttft_s": ttft}

    try:
        async for item in inner:
            ev = _parse(item)
            if ev is None:  # not JSON: pass through untouched
                raw = _text(item)
                if raw.strip():
                    yield raw.rstrip("\n") + "\n"
                continue
            kind = ev.get("type")
            if kind == "token" and ttft is None and ev.get("text"):
                ttft = time.perf_counter() - t0
            if kind in ("done", "error"):
                final = ev
                break
            yield _line(ev)

        latency = time.perf_counter() - t0
        if final is None:
            final = {"type": "error", "kind": "truncated",
                     "error": "backend stream ended without a final 'done' event"}
        status, fields, err = _outcome(final, mode)
        finalize(status, {"latency_s": latency, "ttft_s": ttft, "error": err, **fields})
        out = envelope(final, latency)
        out["status"] = status
        if status == "FAILED":
            out["type"] = "error"
            out["error"] = err
            out.setdefault("kind", "invalid_real_response")
        yield _line(out)
    except Exception as exc:  # backend crashed mid-stream: record it AND tell the client
        if finalized:
            raise
        latency = time.perf_counter() - t0
        err = f"stream_error: {type(exc).__name__}: {exc}"
        finalize("FAILED", {"latency_s": latency, "ttft_s": ttft, "error": err})
        yield _line(envelope({"type": "error", "kind": "stream_error", "error": err, "status": "FAILED"}, latency))
    finally:
        if not finalized:  # client disconnected / task cancelled before completion
            finalize("FAILED", {"latency_s": time.perf_counter() - t0, "ttft_s": ttft,
                                "error": "interrupted: client disconnected or request cancelled before completion"})
        aclose = getattr(inner, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


def _publish(hub, event: dict) -> None:
    try:
        hub.publish(event)
    except Exception:
        log.exception("metrics: could not publish live event")
