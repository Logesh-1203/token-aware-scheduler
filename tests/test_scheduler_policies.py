import pytest
import time
from app.gateway.scheduler import (
    _queue,
    _enqueue,
    _dequeue,
    reset_scheduler_state,
    compute_token_aware_score,
    compute_adaptive_score,
    choose_backend_for_request,
)
from app.config import (
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    STARVATION_WAIT_THRESHOLD,
    STARVATION_BONUS,
)

@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    yield
    reset_scheduler_state()

def test_fifo_ordering():
    # Enqueue 3 requests with different profiles and token counts
    r1 = {"prompt": "1", "profile": "batch", "max_output_tokens": 500, "policy": POLICY_FIFO}
    r2 = {"prompt": "2", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_FIFO}
    r3 = {"prompt": "3", "profile": "normal", "max_output_tokens": 50, "policy": POLICY_FIFO}

    _enqueue(r1)
    _enqueue(r2)
    _enqueue(r3)

    # In FIFO, r1 arrived first and MUST be dequeued first regardless of profile/tokens
    selected_1 = _dequeue(POLICY_FIFO)
    assert selected_1["prompt"] == "1"

    selected_2 = _dequeue(POLICY_FIFO)
    assert selected_2["prompt"] == "2"

    selected_3 = _dequeue(POLICY_FIFO)
    assert selected_3["prompt"] == "3"

def test_token_aware_priority_over_fifo():
    # Enqueue a large batch request first, then an interactive request
    r_batch = {"prompt": "batch", "profile": "batch", "max_output_tokens": 500, "policy": POLICY_TOKEN_AWARE}
    r_interactive = {"prompt": "chat", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_TOKEN_AWARE}

    _enqueue(r_batch)
    _enqueue(r_interactive)

    # In token_aware, interactive with 10 tokens has much higher score than batch with 500 tokens
    selected = _dequeue(POLICY_TOKEN_AWARE)
    assert selected["prompt"] == "chat"
    assert selected["profile"] == "interactive"

def test_stable_tie_breaking():
    # Two identical requests arrive in sequence
    r1 = {"prompt": "first", "profile": "interactive", "max_output_tokens": 20, "policy": POLICY_ADAPTIVE}
    r2 = {"prompt": "second", "profile": "interactive", "max_output_tokens": 20, "policy": POLICY_ADAPTIVE}

    _enqueue(r1)
    _enqueue(r2)

    # Equal priority scores must deterministically tie-break by arrival sequence
    selected = _dequeue(POLICY_ADAPTIVE)
    assert selected["prompt"] == "first"

def test_aging_and_starvation_prevention():
    # A batch request enqueued 40s ago (exceeding starvation threshold)
    # vs a freshly arrived interactive request
    old_batch = {
        "prompt": "old_batch",
        "profile": "batch",
        "max_output_tokens": 200,
        "policy": POLICY_ADAPTIVE,
    }
    new_chat = {
        "prompt": "new_chat",
        "profile": "interactive",
        "max_output_tokens": 10,
        "policy": POLICY_ADAPTIVE,
    }

    _enqueue(old_batch)
    # Manually backdate the enqueued_at time
    old_batch["enqueued_at"] = time.monotonic() - (STARVATION_WAIT_THRESHOLD + 5.0)

    _enqueue(new_chat)

    # The starved batch request should receive +1000 starvation bonus and win!
    selected = _dequeue(POLICY_ADAPTIVE)
    assert selected["prompt"] == "old_batch"
    assert selected["score_breakdown"]["starvation_bonus"] == STARVATION_BONUS

def test_cost_aware_routing():
    # Adaptive cost-aware routing selects fastest estimated completion
    best_b, estimates, rejected = choose_backend_for_request(
        policy=POLICY_ADAPTIVE,
        max_output_tokens=100,
        candidate_backends=["model_a", "model_b", "model_c"]
    )
    # Model A is 50ms/token, Model B is 100ms, Model C is 200ms
    # When all are idle, Model A should have the lowest estimated completion time
    assert best_b == "model_a"
    assert estimates["model_a"]["estimated_completion_sec"] < estimates["model_c"]["estimated_completion_sec"]
