import json
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Ollama Real Model Backend")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b-q4_K_M"

BACKEND_URLS = {
    "model_a": "http://127.0.0.1:8001",
    "model_b": "http://127.0.0.1:8002",
    "model_c": "http://127.0.0.1:8003",
    "ollama_qwen3": "http://127.0.0.1:8004",
}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                "http://127.0.0.1:11434/api/tags"
            )
            response.raise_for_status()

        return {"status": "ok", "model": MODEL_NAME}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {exc}"
        )


@app.post("/generate")
async def generate(request: dict):
    prompt = request.get("prompt")
    max_tokens = request.get("max_output_tokens", 128)

    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="Invalid prompt")

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid max_output_tokens"
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "options": {"num_predict": max_tokens}
    }

    async def token_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", OLLAMA_URL, json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        item = json.loads(line)
                        text = item.get("message", {}).get("content", "")

                        if text:
                            yield json.dumps({
                                "type": "token",
                                "text": text
                            }) + "\n"

                        if item.get("done"):
                            yield json.dumps({
                                "type": "done",
                                "request_id": request.get("request_id"),
                                "tokens_generated": item.get(
                                    "eval_count", 0
                                )
                            }) + "\n"

        except Exception as exc:
            yield json.dumps({
                "type": "error",
                "error": str(exc)
            }) + "\n"

    return StreamingResponse(
        token_stream(),
        media_type="application/x-ndjson"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backends.model_ollama:app", host="0.0.0.0", port=8004, reload=True)
