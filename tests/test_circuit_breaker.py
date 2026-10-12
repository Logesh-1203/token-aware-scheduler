import pytest
import time
from app.gateway.health import (
    BackendHealthManager,
    STATE_HEALTHY,
    STATE_DEGRADED,
    STATE_OPEN,
    STATE_HALF_OPEN,
)
from app.config import CB_FAILURE_THRESHOLD, CB_OPEN_DURATION

def test_circuit_breaker_lifecycle():
    hm = BackendHealthManager()
    backend = "model_c"

    # 1. Starts healthy
    assert hm.get_state(backend) == STATE_HEALTHY
    assert hm.is_available(backend) is True

    # 2. First failure -> degraded
    hm.record_failure(backend, "connection timeout")
    assert hm.get_state(backend) == STATE_DEGRADED
    assert hm.is_available(backend) is True

    # 3. Consecutive failures up to threshold -> trips to open
    for _ in range(CB_FAILURE_THRESHOLD - 1):
        hm.record_failure(backend, "connection refused")
    assert hm.get_state(backend) == STATE_OPEN
    assert hm.is_available(backend) is False

    # 4. Open duration elapses -> transitions to half-open
    # Simulate time elapsed
    hm._last_state_change[backend] = time.monotonic() - (CB_OPEN_DURATION + 0.1)
    assert hm.get_state(backend) == STATE_HALF_OPEN
    assert hm.is_available(backend) is True

    # 5. Successful probe restores healthy state
    hm.record_success(backend, latency_sec=0.1, tokens=5)
    assert hm.get_state(backend) == STATE_HEALTHY
    assert hm.is_available(backend) is True

def test_circuit_breaker_probe_failure_reopens():
    hm = BackendHealthManager()
    backend = "model_b"

    # Trip to open
    for _ in range(CB_FAILURE_THRESHOLD):
        hm.record_failure(backend, "failed")
    assert hm.get_state(backend) == STATE_OPEN

    # Transition to half-open
    hm._last_state_change[backend] = time.monotonic() - (CB_OPEN_DURATION + 0.1)
    assert hm.get_state(backend) == STATE_HALF_OPEN

    # Probe fails -> immediately re-trips to OPEN
    hm.record_failure(backend, "probe failed")
    assert hm.get_state(backend) == STATE_OPEN
    assert hm.is_available(backend) is False
