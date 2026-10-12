import pytest
import asyncio
from app.gateway.scheduler import (
    reset_scheduler_state,
    schedule_and_acquire,
    release_slot,
    get_metrics,
    _queue,
)
from app.gateway.benchmark import generate_deterministic_workload

@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    yield
    reset_scheduler_state()

def test_workload_determinism():
    # Calling workload generator with same seed must return identical requests
    w1 = generate_deterministic_workload(size=20, seed=42)
    w2 = generate_deterministic_workload(size=20, seed=42)
    assert len(w1) == len(w2) == 20
    for r1, r2 in zip(w1, w2):
        assert r1["request_id"] == r2["request_id"]
        assert r1["profile"] == r2["profile"]
        assert r1["max_output_tokens"] == r2["max_output_tokens"]

@pytest.mark.asyncio
async def test_queue_forms_under_concurrency():
    # Backend concurrency limit is 1 per backend.
    # 3 backends in BACKEND_URLS (model_a, model_b, model_c) -> 3 active slots total.
    # If 4 requests arrive, the 4th MUST be queued in _queue!
    backend_urls = {"model_a": "http://a", "model_b": "http://b", "model_c": "http://c"}

    r1 = {"prompt": "1", "profile": "interactive", "max_output_tokens": 10, "policy": "fifo"}
    r2 = {"prompt": "2", "profile": "interactive", "max_output_tokens": 10, "policy": "fifo"}
    r3 = {"prompt": "3", "profile": "interactive", "max_output_tokens": 10, "policy": "fifo"}
    r4 = {"prompt": "4", "profile": "interactive", "max_output_tokens": 10, "policy": "fifo"}

    # Acquire slots for 1, 2, 3
    b1, _, _, _ = await schedule_and_acquire(r1, backend_urls=backend_urls)
    b2, _, _, _ = await schedule_and_acquire(r2, backend_urls=backend_urls)
    b3, _, _, _ = await schedule_and_acquire(r3, backend_urls=backend_urls)

    assert get_metrics()["active_requests"] == 3

    # Now launch 4th request in background task (it will queue up because all slots are taken)
    task_4 = asyncio.create_task(schedule_and_acquire(r4, backend_urls=backend_urls))
    await asyncio.sleep(0.01)

    # Verify a real queue actually formed!
    assert get_metrics()["queue_length"] == 1
    assert len(_queue) == 1
    assert _queue[0]["prompt"] == "4"

    # Releasing one backend slot should immediately dispatch request 4
    await release_slot(b1)
    b4, _, wait_sec, _ = await task_4

    assert get_metrics()["queue_length"] == 0
    assert wait_sec >= 0.0

    # Cleanup remaining slots
    await release_slot(b2)
    await release_slot(b3)
    await release_slot(b4)
    assert get_metrics()["active_requests"] == 0
