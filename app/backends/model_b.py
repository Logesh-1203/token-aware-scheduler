import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

from app.config import TIME_SCALE, BACKEND_MODEL, BATCHED_ALPHA, BATCHED_BETA

TOKEN_DELAY_MS = 100  # milliseconds per token for Model B
_active_workers = 0

@app.get("/health")
async def health():
    return {"status": "ok", "model": "B"}

@app.post("/generate")
async def generate(request: dict):
    global _active_workers
    prompt = request.get("prompt")
    max_tokens = request.get("max_output_tokens")
    failure = request.get("simulate_failure", "none")
    sim_true_tokens = request.get("sim_true_output_tokens")

    if not isinstance(prompt, str) or not prompt:
        raise HTTPException(status_code=400, detail="Invalid prompt")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise HTTPException(status_code=400, detail="Invalid max_output_tokens")

    if failure == "before":
        raise HTTPException(status_code=500, detail="Simulated failure before first token")

    if sim_true_tokens is not None and isinstance(sim_true_tokens, int) and sim_true_tokens > 0:
        tokens_to_gen = min(max_tokens, sim_true_tokens)
    else:
        tokens_to_gen = max_tokens

    base_delay_sec = (TOKEN_DELAY_MS / 1000.0) * TIME_SCALE
    is_batched = (BACKEND_MODEL == "batched")
    prefill_delay = (BATCHED_BETA * len(prompt) * TIME_SCALE) if is_batched else 0.0

    _active_workers += 1

    async def token_stream():
        global _active_workers
        try:
            if prefill_delay > 0:
                await asyncio.sleep(prefill_delay)

            for i in range(tokens_to_gen):
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
                "request_id": request.get("request_id", "placeholder"),
                "tokens_generated": tokens_to_gen,
                "output_tokens": tokens_to_gen,
            }
            yield json.dumps(done) + "\n"
        finally:
            _active_workers = max(0, _active_workers - 1)

    return StreamingResponse(token_stream(), media_type="application/x-ndjson")
