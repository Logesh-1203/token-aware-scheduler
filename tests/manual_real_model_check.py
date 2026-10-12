"""MANUAL real-model check - needs Ollama running with your model installed.

    .\\.venv\\Scripts\\python.exe tests\\manual_real_model_check.py

Talks to Ollama directly through app/ollama_stream.py (no gateway) and verifies that a REAL answer
and authoritative metadata come back. Exit code 0 = pass, 1 = fail. Offline unit tests can't prove this.
Override with: $env:HORIZON_MODEL="..."  $env:OLLAMA_URL="http://127.0.0.1:11434/api/chat"
"""
import asyncio
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.ollama_stream import stream_ollama  # noqa: E402

MODEL = os.environ.get("HORIZON_MODEL", "qwen3:4b-q4_K_M")
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")


async def main() -> int:
    print(f"Model: {MODEL}\nEndpoint: {URL}\nAsking: 'Name three planets, one per line.'\n")
    t0, ttft, text, done = time.perf_counter(), None, [], None
    async for ev in stream_ollama("Name three planets, one per line.", model=MODEL, url=URL, max_output_tokens=128):
        if ev["type"] == "token":
            if ttft is None:
                ttft = time.perf_counter() - t0
            text.append(ev["text"])
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "done":
            done = ev
        elif ev["type"] == "error":
            print(f"\nFAIL: {ev['kind']}: {ev['error']}")
            return 1
    latency = time.perf_counter() - t0
    answer = "".join(text).strip()
    print(f"\n\nlatency={latency:.3f}s  ttft={ttft if ttft is None else round(ttft, 3)}s  metadata={done}")
    problems = []
    if not answer:
        problems.append("empty answer")
    if re.fullmatch(r"(token\d+)+", answer or "x"):
        problems.append("answer looks like simulated token1token2... output")
    if not done:
        problems.append("no final 'done' event")
    elif not done.get("output_tokens") or not done.get("tokens_per_second"):
        problems.append("missing eval_count / eval_duration metadata")
    print("PASS: real model answered with authoritative metadata" if not problems else "FAIL: " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
