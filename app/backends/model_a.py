import asyncio
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

from app.config import (
    TIME_SCALE,
    BACKEND_MODEL,
    BATCHED_ALPHA,
    BATCHED_BETA,
    OLLAMA_URL,
    OLLAMA_TAGS_URL,
    resolve_ollama_model,
)

# Configuration
TOKEN_DELAY_MS = 50  # milliseconds per token for Model A
_active_workers = 0

@app.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "ok",
        "model": "A",
        "real_model_backend": "ollama",
        "ollama_available": ollama_ok,
    }

@app.post("/generate")
async def generate(prompt: dict):
    global _active_workers
    prompt_text = prompt.get("prompt")
    max_tokens = prompt.get("max_output_tokens", 128)
    failure = prompt.get("simulate_failure", "none")
    sim_true_tokens = prompt.get("sim_true_output_tokens")
    use_real = prompt.get("use_real_model", False)

    if not isinstance(prompt_text, str) or not prompt_text:
        raise HTTPException(status_code=400, detail="Invalid prompt")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise HTTPException(status_code=400, detail="Invalid max_output_tokens")

    if failure == "before":
        raise HTTPException(status_code=500, detail="Simulated failure before first token")

    # 1. Real Ollama Qwen3 Generation Path
    if use_real:
        resolved_model = resolve_ollama_model()

        # Probe Ollama connection before starting stream with fast connect timeout
        probe_timeout = httpx.Timeout(connect=2.0, read=45.0, write=5.0, pool=5.0)
        test_client = httpx.AsyncClient(timeout=probe_timeout)
        try:
            probe_req = test_client.build_request(
                "POST",
                OLLAMA_URL,
                json={"model": resolved_model, "messages": [{"role": "user", "content": prompt_text}], "stream": True, "options": {"num_predict": 1}}
            )
            probe_resp = await test_client.send(probe_req, stream=True)
            if probe_resp.status_code >= 400:
                err_text = await probe_resp.aread()
                await test_client.aclose()
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama returned HTTP {probe_resp.status_code}: {err_text.decode('utf-8', errors='ignore')}"
                )
            await probe_resp.aclose()
            await test_client.aclose()
        except (httpx.ConnectError, httpx.NetworkError, httpx.ConnectTimeout, httpx.HTTPError) as ce:
            await test_client.aclose()
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Ensure Ollama is running (`ollama serve`). Error: {ce}"
            )
        except Exception as exc:
            await test_client.aclose()
            if isinstance(exc, HTTPException):
                raise
            err_msg = str(exc) or type(exc).__name__
            raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama: {err_msg}")

        _active_workers += 1

        async def real_token_stream():
            global _active_workers
            try:
                from app.ollama_stream import stream_ollama
                async for ev in stream_ollama(
                    prompt_text,
                    model=resolved_model,
                    url=OLLAMA_URL,
                    max_output_tokens=max_tokens,
                ):
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
            finally:
                _active_workers = max(0, _active_workers - 1)

        return StreamingResponse(real_token_stream(), media_type="application/x-ndjson")

    # 2. Simulated Model Generation Path (for deterministic benchmarks & tests)
    if sim_true_tokens is not None and isinstance(sim_true_tokens, int) and sim_true_tokens > 0:
        tokens_to_gen = min(max_tokens, sim_true_tokens)
    else:
        tokens_to_gen = max_tokens

    base_delay_sec = (TOKEN_DELAY_MS / 1000.0) * TIME_SCALE
    is_batched = (BACKEND_MODEL == "batched")
    prefill_delay = (BATCHED_BETA * len(prompt_text) * TIME_SCALE) if is_batched else 0.0

    _active_workers += 1

    async def token_stream():
        global _active_workers
        try:
            if prefill_delay > 0:
                await asyncio.sleep(prefill_delay)

            for i in range(tokens_to_gen):
                # Concurrency degradation under batched model
                if is_batched and _active_workers > 1:
                    step_delay = base_delay_sec * (1.0 + BATCHED_ALPHA * (_active_workers - 1))
                else:
                    step_delay = base_delay_sec

                if step_delay > 0:
                    await asyncio.sleep(step_delay)
                if failure == "partial" and i == tokens_to_gen // 2:
                    raise RuntimeError("Simulated failure after partial output")
                chunk = {"type": "token", "text": f"token{i+1}"}
                yield json.dumps(chunk) + "\n"
            done = {
                "type": "done",
                "mode": "SIMULATED",
                "request_id": prompt.get("request_id", "placeholder"),
                "tokens_generated": tokens_to_gen,
                "output_tokens": tokens_to_gen,
            }
            yield json.dumps(done) + "\n"
        finally:
            _active_workers = max(0, _active_workers - 1)

    return StreamingResponse(token_stream(), media_type="application/x-ndjson")

