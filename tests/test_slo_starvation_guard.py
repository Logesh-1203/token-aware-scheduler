"""Comprehensive test suite for SLO-Aware Starvation Prevention & Guard."""
import pytest
import asyncio
from httpx import AsyncClient, ASGITransport

from app.gateway.main import app as gateway_app
from app.gateway.scheduler import (
    reset_scheduler_state,
    schedule_and_acquire,
    release_slot,
    get_metrics,
    get_starvation_guard_status,
    _queue,
    _active_requests,
    BACKEND_CONCURRENCY,
    BACKEND_URLS,
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_ADAPTIVE_PRED,
)
from app.gateway.starvation_guard import starvation_guard
from app.gateway.benchmark import run_starvation_guard_comparison


@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    starvation_guard.reset()
    yield
    reset_scheduler_state()
    starvation_guard.reset()


# ---------------------------------------------------------------------------
# 1. Settings Validation and Reset
# ---------------------------------------------------------------------------
def test_dashboard_settings_validation_and_reset():
    # Sensible defaults exist
    cfg = starvation_guard.get_config()
    assert cfg["enabled"] is True
    assert cfg["max_queue_wait_threshold_sec"] > 0
    assert cfg["slo_latency_target_sec"] > 0

    # Valid update
    ok, msg = starvation_guard.update_config({
        "enabled": False,
        "max_queue_wait_threshold_sec": 15.0,
        "aging_boost_rate": 8.0,
        "aging_interval_sec": 0.5,
        "slo_latency_target_sec": 12.0,
    })
    assert ok is True
    updated = starvation_guard.get_config()
    assert updated["enabled"] is False
    assert updated["max_queue_wait_threshold_sec"] == 15.0
    assert updated["aging_boost_rate"] == 8.0
    assert updated["slo_latency_target_sec"] == 12.0

    # Invalid updates rejected safely
    bad_ok, _ = starvation_guard.update_config({"max_queue_wait_threshold_sec": -5.0})
    assert bad_ok is False

    bad_ok2, _ = starvation_guard.update_config({"aging_boost_rate": -1.0})
    assert bad_ok2 is False

    bad_ok3, _ = starvation_guard.update_config({"slo_latency_target_sec": 0.0})
    assert bad_ok3 is False

    bad_ok4, _ = starvation_guard.update_config({"aging_interval_sec": 100.0})
    assert bad_ok4 is False

    # Reset restores initial configuration
    starvation_guard.reset()
    reset_cfg = starvation_guard.get_config()
    assert reset_cfg["enabled"] is True
    assert reset_cfg["max_queue_wait_threshold_sec"] == reset_cfg["defaults"]["max_queue_wait_threshold_sec"]


# ---------------------------------------------------------------------------
# 2. Queue Aging Over Time & Priority Changes
# ---------------------------------------------------------------------------
def test_queue_aging_and_priority_escalation():
    starvation_guard.reset()
    starvation_guard.update_config({"enabled": True, "aging_boost_rate": 10.0, "max_queue_wait_threshold_sec": 20.0})

    now = 1000.0
    # Enqueued 5 seconds ago
    req = {
        "request_id": "req-batch",
        "profile": "batch",
        "policy": POLICY_TOKEN_AWARE,
        "max_output_tokens": 100,
        "enqueued_at": now - 5.0,
        "arrival_seq": 1,
    }

    # With protection enabled: aging boost is 5s * 10 pts/s = 50 pts
    inspected = starvation_guard.inspect_queue_items([req], now=now)
    assert len(inspected) == 1
    item = inspected[0]
    assert item["queue_wait_sec"] == 5.0
    assert item["aging_adjustment"] == 50.0
    assert item["effective_priority"] == item["original_priority"] + 50.0
    assert item["is_protected_by_aging"] is True
    assert item["starvation_risk"] == "Healthy"

    # When protection is disabled: aging boost is 0
    starvation_guard.enabled = False
    inspected_disabled = starvation_guard.inspect_queue_items([req], now=now)
    item_dis = inspected_disabled[0]
    assert item_dis["aging_adjustment"] == 0.0
    assert item_dis["effective_priority"] == item_dis["original_priority"]
    assert item_dis["is_protected_by_aging"] is False


# ---------------------------------------------------------------------------
# 3. Batch Starvation Prevention Across Policies
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_starvation_prevention_token_aware():
    starvation_guard.reset()
    starvation_guard.update_config({
        "enabled": True,
        "max_queue_wait_threshold_sec": 5.0,
        "starvation_bonus": 1000.0,
    })

    backend_urls = {"model_a": "http://a"}

    # Saturate backend slot
    r0 = {"prompt": "init", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_TOKEN_AWARE}
    b0, _, _, _ = await schedule_and_acquire(r0, backend_urls=backend_urls)
    assert _active_requests["model_a"] == 1

    # 1. Enqueue batch request that has waited long enough to breach starvation threshold
    batch_req = {
        "request_id": "starved-batch",
        "prompt": "heavy batch",
        "profile": "batch",
        "max_output_tokens": 150,
        "policy": POLICY_TOKEN_AWARE,
    }
    t_batch = asyncio.create_task(schedule_and_acquire(batch_req, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    # Simulate that batch request has been waiting in queue for 6.0 seconds (> 5.0s threshold)
    _queue[0]["enqueued_at"] -= 6.0

    # 2. Enqueue fresh interactive request (normally higher priority: 100 vs 10)
    inter_req = {
        "request_id": "new-interactive",
        "prompt": "short query",
        "profile": "interactive",
        "max_output_tokens": 10,
        "policy": POLICY_TOKEN_AWARE,
    }
    t_inter = asyncio.create_task(schedule_and_acquire(inter_req, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    assert len(_queue) == 2

    # Release slot: Starvation Guard MUST prioritize starved-batch over new-interactive!
    await release_slot(b0)
    b_dispatched, _, wait_sec, meta = await t_batch

    assert b_dispatched == "model_a"
    assert meta["score_breakdown"]["starvation_bonus"] == 1000.0
    assert get_metrics()["requests_protected_by_aging"] >= 1

    # Clean up second request
    await release_slot(b_dispatched)
    await t_inter


@pytest.mark.asyncio
async def test_batch_starvation_prevention_disabled():
    starvation_guard.reset()
    # DISABLE starvation protection
    starvation_guard.enabled = False

    backend_urls = {"model_a": "http://a"}

    # Saturate backend slot
    r0 = {"prompt": "init", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_TOKEN_AWARE}
    b0, _, _, _ = await schedule_and_acquire(r0, backend_urls=backend_urls)

    # 1. Enqueue batch request
    batch_req = {
        "request_id": "old-batch",
        "prompt": "heavy batch",
        "profile": "batch",
        "max_output_tokens": 150,
        "policy": POLICY_TOKEN_AWARE,
    }
    t_batch = asyncio.create_task(schedule_and_acquire(batch_req, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    # Simulate wait time of 10s
    _queue[0]["enqueued_at"] -= 10.0

    # 2. Enqueue fresh interactive request
    inter_req = {
        "request_id": "new-interactive",
        "prompt": "short query",
        "profile": "interactive",
        "max_output_tokens": 10,
        "policy": POLICY_TOKEN_AWARE,
    }
    t_inter = asyncio.create_task(schedule_and_acquire(inter_req, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    # Release slot: Without protection, interactive request wins because batch has token penalty and 0 aging!
    await release_slot(b0)
    b_dispatched, _, _, _ = await t_inter

    assert b_dispatched == "model_a"
    # Batch is still in queue waiting (starving!)
    assert len(_queue) == 1
    assert _queue[0]["request_id"] == "old-batch"

    # Finish batch
    await release_slot(b_dispatched)
    await t_batch


# ---------------------------------------------------------------------------
# 4. Maximum Queue Wait & Risk Classification
# ---------------------------------------------------------------------------
def test_max_queue_wait_and_risk_kpis():
    starvation_guard.reset()
    starvation_guard.update_config({"max_queue_wait_threshold_sec": 10.0})

    now = 500.0
    items = [
        {"request_id": "q1", "profile": "interactive", "policy": POLICY_FIFO, "max_output_tokens": 10, "enqueued_at": now - 1.0},
        {"request_id": "q2", "profile": "normal", "policy": POLICY_TOKEN_AWARE, "max_output_tokens": 20, "enqueued_at": now - 7.0},
        {"request_id": "q3", "profile": "batch", "policy": POLICY_ADAPTIVE, "max_output_tokens": 100, "enqueued_at": now - 12.0},
    ]

    inspected = starvation_guard.inspect_queue_items(items, now)
    assert inspected[0]["starvation_risk"] == "Healthy"
    assert inspected[1]["starvation_risk"] == "Approaching Threshold (Warning)"
    assert inspected[2]["starvation_risk"] == "Threshold Exceeded (Starved)"

    kpis = starvation_guard.compute_kpis(items, now, {}, recent_latencies=[])
    assert kpis["queue_depth"] == 3
    assert kpis["longest_wait_sec"] == 12.0
    assert kpis["starvation_risk_count"] == 2  # q2 (7s >= 6s warning) and q3 (12s >= 10s)


# ---------------------------------------------------------------------------
# 5. Concurrency Limits & Duplicate Dispatch Prevention
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_backend_concurrency_and_single_dispatch():
    backend_urls = {"model_a": "http://a", "model_b": "http://b", "model_c": "http://c"}

    # Acquire all 3 slots (concurrency limit = 1 per backend)
    r1 = {"prompt": "1", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_ADAPTIVE}
    r2 = {"prompt": "2", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_ADAPTIVE}
    r3 = {"prompt": "3", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_ADAPTIVE}

    b1, _, _, _ = await schedule_and_acquire(r1, backend_urls=backend_urls)
    b2, _, _, _ = await schedule_and_acquire(r2, backend_urls=backend_urls)
    b3, _, _, _ = await schedule_and_acquire(r3, backend_urls=backend_urls)

    assert sum(_active_requests.values()) == 3

    # Launch 4th request (must queue)
    r4 = {"request_id": "r4", "prompt": "4", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_ADAPTIVE}
    t4 = asyncio.create_task(schedule_and_acquire(r4, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    assert len(_queue) == 1

    # Release slot b1 once: exactly 1 request dispatched, queue becomes empty
    await release_slot(b1)
    b4, _, _, _ = await t4

    assert len(_queue) == 0
    assert b4 == b1
    assert _active_requests[b4] == 1

    # Clean up
    await release_slot(b2)
    await release_slot(b3)
    await release_slot(b4)


# ---------------------------------------------------------------------------
# 6. SLO Monitor: Sufficient vs Missing Metrics
# ---------------------------------------------------------------------------
def test_slo_monitor_missing_and_sufficient_data():
    starvation_guard.reset()
    starvation_guard.update_config({"slo_latency_target_sec": 5.0})

    # Fewer than 3 completed requests -> Insufficient data
    empty_slo = starvation_guard.compute_slo_monitor([])
    assert empty_slo["has_sufficient_data"] is False
    assert empty_slo["rolling_slo_status"] == "Insufficient data"

    two_samples = starvation_guard.compute_slo_monitor([1.2, 2.3])
    assert two_samples["has_sufficient_data"] is False

    # 4 completed requests under target -> Healthy
    healthy_slo = starvation_guard.compute_slo_monitor([1.0, 1.5, 2.0, 2.5])
    assert healthy_slo["has_sufficient_data"] is True
    assert healthy_slo["rolling_slo_status"] == "Healthy"
    assert healthy_slo["requests_exceeding_target"] == 0
    assert healthy_slo["slo_compliance_pct"] == 100.0

    # Samples exceeding target -> Breached
    breached_slo = starvation_guard.compute_slo_monitor([1.0, 2.0, 6.0, 7.5])
    assert breached_slo["has_sufficient_data"] is True
    assert breached_slo["is_rolling_p95_breach"] is True
    assert breached_slo["rolling_slo_status"] == "Breached"
    assert breached_slo["requests_exceeding_target"] == 2
    assert breached_slo["slo_compliance_pct"] == 50.0


# ---------------------------------------------------------------------------
# 7. Benchmark Comparison: Starvation Protection OFF vs ON
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_starvation_guard_benchmark_comparison():
    class MockStreamClient:
        async def aclose(self):
            pass

    class MockStreamResponse:
        async def aiter_text(self):
            yield '{"type":"token","text":"hello "}\n'
            yield '{"type":"done","tokens_generated":1}\n'

    async def mock_stream(url, payload):
        return MockStreamClient(), MockStreamResponse()

    res = await run_starvation_guard_comparison(
        policy="adaptive",
        workload_size=4,
        seed=42,
        stream_from_fn=mock_stream,
    )

    assert "protection_off" in res
    assert "protection_on" in res
    assert "comparison" in res
    assert "tradeoff_analysis" in res

    comp = res["comparison"]
    assert "max_queue_wait_sec" in comp
    assert "starvation_risk_requests" in comp
    assert comp["max_queue_wait_sec"]["favorable_direction"] == "lower"


# ---------------------------------------------------------------------------
# 8. REST API Endpoints Integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_starvation_guard_api_endpoints():
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /scheduler/starvation-guard
        r_get = await client.get("/scheduler/starvation-guard")
        assert r_get.status_code == 200
        data = r_get.json()
        assert "config" in data
        assert "kpis" in data
        assert "slo_monitor" in data
        assert "queued_requests" in data

        # POST /scheduler/starvation-guard (valid)
        r_update = await client.post("/scheduler/starvation-guard", json={
            "enabled": False,
            "max_queue_wait_threshold_sec": 25.0,
            "aging_boost_rate": 6.5,
        })
        assert r_update.status_code == 200
        assert r_update.json()["config"]["enabled"] is False
        assert r_update.json()["config"]["max_queue_wait_threshold_sec"] == 25.0

        # POST /scheduler/starvation-guard (invalid)
        r_invalid = await client.post("/scheduler/starvation-guard", json={
            "max_queue_wait_threshold_sec": -10.0,
        })
        assert r_invalid.status_code == 400

        # POST /scheduler/starvation-guard/reset
        r_reset = await client.post("/scheduler/starvation-guard/reset")
        assert r_reset.status_code == 200
        assert r_reset.json()["config"]["enabled"] is True
