import pytest
from httpx import AsyncClient, ASGITransport
from app.gateway.main import app as gateway_app
from app.gateway.scheduler import reset_scheduler_state, get_metrics
from app.gateway.health import health_manager, STATE_HEALTHY, STATE_OPEN

@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    yield
    reset_scheduler_state()

@pytest.mark.asyncio
async def test_failure_recovery_pre_token_retry(monkeypatch):
    """Verify that when a backend fails BEFORE the first token,
    the gateway retries once on an alternate backend and completes the stream.
    """
    transport = ASGITransport(app=gateway_app)

    # Mock backend stream: first call fails before first token, second call succeeds
    calls = []
    class FakeStream:
        async def aiter_text(self):
            yield '{"type":"token","text":"Recovered"}\n'
            yield '{"type":"done","request_id":"retry_test"}\n'

    class FakeClient:
        async def aclose(self):
            pass

    async def mock_stream(url, payload):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("Backend connection refused before token 1")
        return FakeClient(), FakeStream()

    monkeypatch.setattr("app.gateway.router._stream_from_backend", mock_stream)
    monkeypatch.setattr("app.gateway.router.BACKEND_URLS", {
        "model_a": "http://127.0.0.1:8001",
        "model_b": "http://127.0.0.1:8002",
    })

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/infer", json={
            "prompt": "Test recovery",
            "profile": "interactive",
            "max_output_tokens": 1,
            "policy": "fifo",
            "simulate_failure": "before"
        })
        assert resp.status_code == 200
        lines = [line async for line in resp.aiter_lines()]
        assert len(lines) == 2
        assert "Recovered" in lines[0]

    # Verification:
    # 1. Two backend calls were made (primary + retry)
    assert len(calls) == 2
    # 2. Metrics recorded 1 submitted, 1 completed, 1 retry, 0 failed
    metrics = get_metrics()
    assert metrics["total_submitted"] == 1
    assert metrics["total_completed"] == 1
    assert metrics["total_retried"] == 1
    assert metrics["total_failed"] == 0

@pytest.mark.asyncio
async def test_no_retry_after_partial_output(monkeypatch):
    """Verify that if a failure occurs AFTER tokens have already been delivered,
    the gateway MUST NOT retry on another backend (strict idempotency guard).
    """
    transport = ASGITransport(app=gateway_app)

    calls = []
    class FailingMidwayStream:
        async def aiter_text(self):
            yield '{"type":"token","text":"Part1"}\n'
            # Abort midway through generation
            raise RuntimeError("Backend process died after partial output")

    class FakeClient:
        async def aclose(self):
            pass

    async def mock_stream(url, payload):
        calls.append(url)
        return FakeClient(), FailingMidwayStream()

    monkeypatch.setattr("app.gateway.router._stream_from_backend", mock_stream)
    monkeypatch.setattr("app.gateway.router.BACKEND_URLS", {
        "model_a": "http://127.0.0.1:8001",
        "model_b": "http://127.0.0.1:8002",
    })

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Since client.post reads until stream completion, the mid-stream failure
        # propagates to client.post directly
        with pytest.raises(Exception) as exc_info:
            await client.post("/infer", json={
                "prompt": "Test partial",
                "profile": "normal",
                "max_output_tokens": 5,
                "policy": "fifo",
                "simulate_failure": "partial"
            })
        assert "Backend process died after partial output" in str(exc_info.value)

    # Strictly only 1 backend call made; no retry attempted after partial emission
    assert len(calls) == 1
    metrics = get_metrics()
    assert metrics["total_retried"] == 0
    assert metrics["total_failed"] == 1
