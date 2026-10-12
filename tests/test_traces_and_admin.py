import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from app.gateway.main import app as gateway_app
from app.gateway.scheduler import reset_scheduler_state, get_metrics, _queue, _enqueue
from app.gateway.traces import trace_manager

@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    yield
    reset_scheduler_state()

@pytest.mark.asyncio
async def test_admin_reset_endpoint():
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Enqueue something and verify metrics
        _enqueue({"prompt": "t", "profile": "interactive", "max_output_tokens": 5, "policy": "fifo"})
        assert get_metrics()["total_submitted"] == 1

        # Call reset endpoint
        resp = await client.post("/admin/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify metrics reset
        metrics = get_metrics()
        assert metrics["total_submitted"] == 0
        assert metrics["queue_length"] == 0

@pytest.mark.asyncio
async def test_decision_trace_generation():
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Record a test trace
        trace_manager.record_trace({
            "request_id": "test_req_42",
            "policy": "adaptive",
            "profile": "interactive",
            "max_output_tokens": 10,
            "selected_backend": "model_a",
            "queue_wait": 0.05,
            "ttft": 0.08,
            "completion_latency": 0.15,
            "explanation": "Adaptive policy scheduled interactive request to model_a.",
            "final_status": "completed",
        })

        # Fetch traces list
        resp = await client.get("/traces")
        assert resp.status_code == 200
        traces = resp.json()
        assert len(traces) >= 1
        assert traces[0]["request_id"] == "test_req_42"

        # Fetch single trace
        resp_single = await client.get("/traces/test_req_42")
        assert resp_single.status_code == 200
        assert resp_single.json()["selected_backend"] == "model_a"

def test_explanation_generator():
    explanation = trace_manager.generate_explanation(
        policy="adaptive",
        profile="interactive",
        selected_backend="model_a",
        queue_wait=0.12,
        score_breakdown={"total_score": 102.5},
        candidate_estimates={"model_a": {"estimated_completion_sec": 0.45}},
        rejected_reasons={"model_c": "circuit_open"},
        success=True,
    )
    assert "Adaptive policy scheduled interactive request" in explanation
    assert "model_a" in explanation
    assert "model_c (circuit_open)" in explanation
