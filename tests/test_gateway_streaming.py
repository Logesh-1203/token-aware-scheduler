import pytest
from httpx import AsyncClient, HTTPError, ASGITransport
from app.gateway.main import app as gateway_app

# Helper fake response
class FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks
    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk

class FakeClient:
    async def aclose(self):
        pass

@pytest.mark.asyncio
async def test_infer_stream_success(monkeypatch):
    # Prepare fake streaming response with 3 token lines and done line
    chunks = [
        '{"type":"token","text":"Hello"}\n',
        '{"type":"token","text":"World"}\n',
        '{"type":"done","request_id":"123"}\n',
    ]
    fake_resp = FakeResponse(chunks)
    fake_client = FakeClient()
    async def fake_stream_from_backend(url, payload):
        return fake_client, fake_resp
    monkeypatch.setattr('app.gateway.router._stream_from_backend', fake_stream_from_backend)
    # Mock healthy backends and status
    monkeypatch.setattr('app.gateway.router.BACKEND_URLS', {'model_a': 'http://dummy'})
    monkeypatch.setattr('app.gateway.router.get_backends_status', lambda: {'model_a': {'active_requests': 0}})
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post('/infer', json={"prompt":"Hi","profile":"interactive","max_output_tokens":2,"policy":"fifo"})
        assert response.status_code == 200
        streamed = [line + "\n" async for line in response.aiter_lines()]
        assert streamed == chunks

@pytest.mark.asyncio
async def test_infer_retry_before_first_token(monkeypatch):
    # First call raises error, second call returns response
    fake_resp = FakeResponse(['{"type":"token","text":"Ok"}\n'])
    fake_client = FakeClient()
    calls = []
    async def fake_stream_from_backend(url, payload):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError("backend down")
        return fake_client, fake_resp
    monkeypatch.setattr('app.gateway.router._stream_from_backend', fake_stream_from_backend)
    # Two backends in config
    monkeypatch.setattr('app.gateway.router.BACKEND_URLS', {'model_a': 'http://a', 'model_b': 'http://b'})
    monkeypatch.setattr('app.gateway.router.get_backends_status', lambda: {'model_a': {'active_requests': 0}, 'model_b': {'active_requests': 0}})
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post('/infer', json={"prompt":"Hi","profile":"interactive","max_output_tokens":1,"policy":"fifo"})
        assert response.status_code == 200
        streamed = [line async for line in response.aiter_text()]
        assert streamed == ['{"type":"token","text":"Ok"}\n']
        assert calls == ['http://a', 'http://b']
