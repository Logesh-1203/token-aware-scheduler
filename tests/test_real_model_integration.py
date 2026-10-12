import pytest
import httpx
import json
from app.gateway.main import app as gateway_app
from app.backends.model_a import app as model_a_app
from app.config import resolve_ollama_model


@pytest.mark.asyncio
async def test_model_a_health_endpoint():
    """Verify Model A reports health status and Ollama availability."""
    transport = httpx.ASGITransport(app=model_a_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "A"
        assert "ollama_available" in data


@pytest.mark.asyncio
async def test_resolve_ollama_model():
    """Verify Ollama model resolution dynamically finds installed tags."""
    resolved = resolve_ollama_model()
    assert isinstance(resolved, str)
    assert len(resolved) > 0


@pytest.mark.asyncio
async def test_chat_endpoint_simulated(monkeypatch):
    """Verify POST /chat returns a complete JSON response."""
    class FakeResponse:
        async def aiter_text(self):
            yield '{"type":"token","text":"Hello "}\n'
            yield '{"type":"token","text":"World"}\n'
            yield '{"type":"done","request_id":"test-1","tokens_generated":2}\n'

    class FakeClient:
        async def aclose(self):
            pass

    async def fake_stream_from_backend(url, payload):
        return FakeClient(), FakeResponse()

    monkeypatch.setattr("app.gateway.router._stream_from_backend", fake_stream_from_backend)
    monkeypatch.setattr("app.gateway.router.BACKEND_URLS", {"model_a": "http://dummy"})
    monkeypatch.setattr("app.gateway.router.get_backends_status", lambda: {"model_a": {"active_requests": 0}})

    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "Test scheduling chat endpoint",
            "profile": "interactive",
            "max_output_tokens": 5,
            "policy": "fifo",
            "use_real_model": False,
        }
        resp = await client.post("/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "Hello World"
        assert "prompt" in data
        assert "tokens_generated" in data
        assert data["tokens_generated"] == 2
        assert data["use_real_model"] is False


@pytest.mark.asyncio
async def test_infer_streaming_with_real_model_if_available():
    """Verify live Ollama inference if Ollama is running."""
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as check_client:
            r = await check_client.get("http://127.0.0.1:11434/api/tags")
            if r.status_code == 200:
                ollama_online = True
    except Exception:
        ollama_online = False

    if not ollama_online:
        pytest.skip("Ollama is not running locally; skipping live real-model test.")

    # Call Model A directly with use_real_model=True
    transport = httpx.ASGITransport(app=model_a_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "Say test in one word",
            "max_output_tokens": 10,
            "use_real_model": True,
        }
        async with client.stream("POST", "/generate", json=payload) as resp:
            assert resp.status_code == 200
            tokens = []
            done_received = False
            async for line in resp.aiter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item.get("type") == "token":
                    tokens.append(item.get("text", ""))
                elif item.get("type") == "done":
                    done_received = True

            assert done_received is True
            assert len(tokens) > 0
            full_text = "".join(tokens)
            assert len(full_text.strip()) > 0


@pytest.mark.asyncio
async def test_gateway_chat_with_real_model_if_available(monkeypatch):
    """Verify POST /chat routes to Model A with live Ollama when available."""
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as check_client:
            r = await check_client.get("http://127.0.0.1:11434/api/tags")
            if r.status_code == 200:
                ollama_online = True
    except Exception:
        ollama_online = False

    if not ollama_online:
        pytest.skip("Ollama is not running locally; skipping live real-model test.")

    # Route backend calls to model_a ASGI app
    async def fake_stream_from_backend(url, payload):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=model_a_app), base_url="http://model-a")
        req = client.build_request("POST", "/generate", json=payload)
        resp = await client.send(req, stream=True)
        return client, resp

    monkeypatch.setattr("app.gateway.router._stream_from_backend", fake_stream_from_backend)

    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "Say test in one word",
            "profile": "interactive",
            "max_output_tokens": 10,
            "policy": "adaptive",
            "use_real_model": True,
        }
        resp = await client.post("/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"].strip()) > 0
        assert "token1" not in data["response"]
        assert data["use_real_model"] is True


@pytest.mark.asyncio
async def test_error_when_ollama_unavailable(monkeypatch):
    """Verify that when Ollama is unavailable, real-model mode errors with 503 instead of falling back to fake tokens."""
    monkeypatch.setattr("app.backends.model_a.OLLAMA_URL", "http://127.0.0.1:59999/api/chat")
    transport = httpx.ASGITransport(app=model_a_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prompt": "Should fail",
            "max_output_tokens": 5,
            "use_real_model": True,
        }
        resp = await client.post("/generate", json=payload)
        assert resp.status_code in (502, 503)
        assert "Cannot connect to Ollama" in resp.json().get("detail", "")
