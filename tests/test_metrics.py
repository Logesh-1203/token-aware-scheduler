import pytest
from httpx import AsyncClient, ASGITransport
from app.gateway.main import app as gateway_app
from app.gateway.scheduler import _metrics_increment, get_metrics

@pytest.mark.asyncio
async def test_metrics_completed(monkeypatch):
    # Reset metrics to zero
    # Subtract current counts to bring them to zero
    current = get_metrics()
    _metrics_increment('total_submitted', -current['total_submitted'])
    _metrics_increment('total_completed', -current['total_completed'])
    _metrics_increment('total_failed', -current['total_failed'])
    _metrics_increment('total_retried', -current['total_retried'])
    # Fake backend streaming response
    class FakeResponse:
        def __init__(self, chunks):
            self._chunks = chunks
        async def aiter_text(self):
            for c in self._chunks:
                yield c
    class FakeClient:
        async def aclose(self):
            pass
    chunks = ["{\"type\":\"token\",\"text\":\"Hi\"}\n", "{\"type\":\"done\",\"request_id\":\"1\"}\n"]
    fake_resp = FakeResponse(chunks)
    fake_client = FakeClient()
    async def fake_stream_from_backend(url, payload):
        return fake_client, fake_resp
    monkeypatch.setattr('app.gateway.router._stream_from_backend', fake_stream_from_backend)
    monkeypatch.setattr('app.gateway.router.BACKEND_URLS', {'model_a': 'http://a'})
    monkeypatch.setattr('app.gateway.router.get_backends_status', lambda: {'model_a': {'active_requests': 0}})
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        resp = await client.post('/infer', json={"prompt":"test","profile":"interactive","max_output_tokens":1,"policy":"fifo"})
        assert resp.status_code == 200
        async for _ in resp.aiter_text():
            pass
    metrics = get_metrics()
    assert metrics['total_submitted'] == 1
    assert metrics['total_completed'] == 1
    assert metrics['total_failed'] == 0
