"""Real Ollama streaming + authoritative metric extraction.

`parse_ollama_lines` is a pure transform (no network) so it is unit-testable.
`stream_ollama` is the thin httpx transport around it.

Events yielded (one dict each; the backend writes them as NDJSON):
  {"type": "token",    "text": "..."}                     visible answer text
  {"type": "thinking", "text": "..."}                     Qwen3 reasoning text (only if think=True)
  {"type": "done",  "mode": "REAL", prompt_tokens, output_tokens, tokens_per_second, ...}
  {"type": "error", "kind": "...", "error": "..."}
Nothing here ever fabricates text or numbers: if Ollama fails, you get an error event.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator, Optional

NS = 1_000_000_000


def tokens_per_second(eval_count, eval_duration_ns) -> Optional[float]:
    """eval_count / (eval_duration / 1e9). Ollama durations are NANOSECONDS. None if unusable."""
    try:
        count, dur = int(eval_count), int(eval_duration_ns)
    except (TypeError, ValueError):
        return None
    if count <= 0 or dur <= 0:
        return None
    return count / (dur / NS)


def _seconds(ns) -> Optional[float]:
    try:
        v = int(ns)
    except (TypeError, ValueError):
        return None
    return v / NS if v >= 0 else None


def final_metadata(chunk: dict) -> dict:
    """Authoritative numbers from Ollama's final (done=true) chunk. /api/chat and
    /api/generate use the same metadata keys."""
    return {
        "prompt_tokens": chunk.get("prompt_eval_count"),   # may be absent when the prompt was cached
        "output_tokens": chunk.get("eval_count"),          # includes thinking tokens, if any
        "tokens_per_second": tokens_per_second(chunk.get("eval_count"), chunk.get("eval_duration")),
        "prompt_eval_s": _seconds(chunk.get("prompt_eval_duration")),
        "load_s": _seconds(chunk.get("load_duration")),
        "ollama_total_s": _seconds(chunk.get("total_duration")),
        "eval_s": _seconds(chunk.get("eval_duration")),
        "done_reason": chunk.get("done_reason"),
    }


def resolve_endpoint(url: str) -> str:
    """Accept either a full endpoint (.../api/chat, .../api/generate) or a bare base URL."""
    u = (url or "http://127.0.0.1:11434").rstrip("/")
    return u if "/api/" in u else u + "/api/chat"


def is_chat(url: str) -> bool:
    return not resolve_endpoint(url).endswith("/api/generate")


def build_payload(prompt: str, *, model: str, url: str, max_output_tokens: Optional[int] = None,
                  think: Optional[bool] = None) -> dict:
    if think is None:  # Qwen3 "thinks" by default; leave it off unless explicitly enabled
        think = os.environ.get("HORIZON_OLLAMA_THINK", "0") == "1"
    payload: dict = {"model": model, "stream": True, "think": think, "options": {}}
    if is_chat(url):
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if max_output_tokens:
        payload["options"]["num_predict"] = int(max_output_tokens)   # max_output_tokens -> num_predict
    return payload


def _err(kind: str, message: str) -> dict:
    return {"type": "error", "kind": kind, "error": message, "mode": "REAL"}


async def parse_ollama_lines(lines: AsyncIterator[str], *, chat: bool = True) -> AsyncIterator[dict]:
    """Ollama NDJSON lines -> token / thinking / done / error events."""
    async for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except ValueError:
            chunk = None
        if not isinstance(chunk, dict):
            yield _err("malformed_response", f"Ollama sent a line that is not a JSON object: {raw[:120]!r}")
            return
        if chunk.get("error"):
            yield _err("ollama_error", str(chunk["error"]))
            return
        msg = chunk.get("message") if chat else None
        msg = msg if isinstance(msg, dict) else {}
        thinking = msg.get("thinking") or chunk.get("thinking")
        text = msg.get("content") if chat else chunk.get("response")
        if thinking:
            yield {"type": "thinking", "text": thinking}
        if text:
            yield {"type": "token", "text": text}
        if chunk.get("done"):
            yield {"type": "done", "mode": "REAL", **final_metadata(chunk)}
            return
    yield _err("truncated", "Ollama stream ended before its final 'done' event")


async def stream_ollama(prompt: str, *, model: str, url: str, max_output_tokens: Optional[int] = None,
                        think: Optional[bool] = None, timeout_s: float = 300.0,
                        connect_timeout_s: float = 5.0) -> AsyncIterator[dict]:
    """Stream a real generation from Ollama. Connection, timeout, HTTP-status and
    missing-model problems come back as `error` events (never as fake output)."""
    import httpx  # imported here so the pure parts above work without httpx installed

    endpoint = resolve_endpoint(url)
    payload = build_payload(prompt, model=model, url=endpoint, max_output_tokens=max_output_tokens, think=think)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s)) as client:
            async with client.stream("POST", endpoint, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    if resp.status_code == 404:
                        yield _err("model_not_found",
                                   f"Ollama has no model '{model}' ({body}). Check `ollama list`, then `ollama pull {model}`.")
                    else:
                        yield _err("ollama_http_error", f"Ollama returned HTTP {resp.status_code}: {body}")
                    return
                async for ev in parse_ollama_lines(resp.aiter_lines(), chat=is_chat(endpoint)):
                    yield ev
    except httpx.ConnectError:
        yield _err("ollama_unreachable", f"Cannot connect to Ollama at {endpoint}. Is it running? (try: ollama list)")
    except httpx.TimeoutException:
        yield _err("timeout", f"Ollama did not respond within {timeout_s:.0f}s (model still loading? try again)")
    except httpx.HTTPError as exc:
        yield _err("ollama_http_error", f"{type(exc).__name__}: {exc}")
