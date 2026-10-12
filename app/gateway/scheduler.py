import asyncio
import math
import time
from collections import deque
from typing import Deque, Dict, Any, List, Optional, Tuple

from ..config import (
    QUEUE_MAX_SIZE,
    BACKEND_CONCURRENCY,
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_ADAPTIVE_LEAST_CONN,
    POLICY_PRIORITY_ONLY,
    POLICY_SJF,
    POLICY_ADAPTIVE_ORACLE,
    POLICY_ADAPTIVE_PRED,
    POLICY_ADAPTIVE_NO_AGING,
    POLICY_ADAPTIVE_NO_COST,
    PROFILE_SCORES,
    BACKEND_URLS,
    ADAPTIVE_PROFILE_WEIGHT,
    ADAPTIVE_WAIT_WEIGHT,
    ADAPTIVE_COST_WEIGHT,
    STARVATION_WAIT_THRESHOLD,
    STARVATION_BONUS,
    TOKEN_DELAY_MS,
    TIME_SCALE,
    SLO_TARGETS,
    ENABLE_ADMISSION_CONTROL,
)
from .health import health_manager
from .traces import trace_manager
from .starvation_guard import starvation_guard

# In-memory queue and lock
_queue: Deque[Dict[str, Any]] = deque()
_queue_lock = asyncio.Lock()
# Monotonic sequence counter for stable tie-breaking
_arrival_seq_counter = 0

# Active request counters per backend
_active_requests: Dict[str, int] = {name: 0 for name in BACKEND_URLS.keys()}

# Detailed Metrics Storage
_metrics: Dict[str, Any] = {
    "total_submitted": 0,
    "total_completed": 0,
    "total_failed": 0,
    "total_retried": 0,
    "max_queue_wait": 0.0,
    "max_queue_length": 0,
    "starvation_incidents": 0,
    "requests_protected_by_aging": 0,
}

# Per-profile statistics
_profile_stats: Dict[str, Dict[str, Any]] = {
    p: {"submitted": 0, "completed": 0, "failed": 0, "wait_times": [], "latencies": [], "ttfts": []}
    for p in ("interactive", "normal", "batch")
}

# Per-policy statistics
_policy_stats: Dict[str, Dict[str, Any]] = {
    p: {"submitted": 0, "completed": 0, "failed": 0, "wait_times": [], "latencies": [], "ttfts": []}
    for p in (POLICY_FIFO, POLICY_TOKEN_AWARE, POLICY_ADAPTIVE, POLICY_ADAPTIVE_LEAST_CONN)
}

_benchmark_start_time: Optional[float] = None
_benchmark_end_time: Optional[float] = None

def _now() -> float:
    return time.monotonic()


def estimate_prompt_tokens(prompt: str) -> Dict[str, Any]:
    """Estimate prompt token count with an explicit methodology label.

    Uses standard English NLP word-to-token heuristic (1.33 tokens per word)
    and character heuristic (~4 characters per token), clearly documented and
    labelled as an approximation rather than exact model tokenizer output.
    """
    if not prompt or not str(prompt).strip():
        return {
            "estimated_tokens": 1,
            "method": "fallback_empty",
            "is_exact": False,
            "char_count": 0,
            "word_count": 0,
        }
    clean = str(prompt).strip()
    words = len(clean.split())
    chars = len(clean)
    approx_from_words = int(math.ceil(words * 1.33))
    approx_from_chars = max(1, chars // 4)
    est = max(approx_from_words, approx_from_chars, 1)
    return {
        "estimated_tokens": est,
        "method": "word_char_heuristic_1.33w_4c",
        "is_exact": False,
        "char_count": chars,
        "word_count": words,
    }

def reset_scheduler_state() -> None:
    """Reset all queues, active counts, metrics, health, and traces to initial state."""
    global _arrival_seq_counter, _benchmark_start_time, _benchmark_end_time
    _queue.clear()
    _arrival_seq_counter = 0
    for k in _active_requests:
        _active_requests[k] = 0
    for k in _metrics:
        _metrics[k] = 0 if isinstance(_metrics[k], int) else 0.0
    for p in _profile_stats:
        _profile_stats[p] = {"submitted": 0, "completed": 0, "failed": 0, "wait_times": [], "latencies": [], "ttfts": []}
    for p in _policy_stats:
        _policy_stats[p] = {"submitted": 0, "completed": 0, "failed": 0, "wait_times": [], "latencies": [], "ttfts": []}
    _benchmark_start_time = None
    _benchmark_end_time = None
    health_manager.reset()
    trace_manager.reset()
    starvation_guard.reset()

def _increment_metric(key: str, amount: int = 1) -> None:
    _metrics[key] = _metrics.get(key, 0) + amount

_metrics_increment = _increment_metric

def record_queue_wait(wait_seconds: float) -> None:
    if wait_seconds > _metrics.get("max_queue_wait", 0.0):
        _metrics["max_queue_wait"] = wait_seconds
    thresh = starvation_guard.max_queue_wait_threshold_sec
    if wait_seconds > thresh:
        _metrics["starvation_incidents"] += 1

def record_request_submitted(profile: str, policy: str) -> None:
    _metrics["total_submitted"] += 1
    if profile in _profile_stats:
        _profile_stats[profile]["submitted"] += 1
    if policy in _policy_stats:
        _policy_stats[policy]["submitted"] += 1

def record_request_completed(profile: str, policy: str, wait_sec: float, latency_sec: float, ttft_sec: float) -> None:
    _metrics["total_completed"] += 1
    if profile in _profile_stats:
        s = _profile_stats[profile]
        s["completed"] += 1
        s["wait_times"].append(wait_sec)
        s["latencies"].append(latency_sec)
        if ttft_sec > 0:
            s["ttfts"].append(ttft_sec)
    if policy in _policy_stats:
        s = _policy_stats[policy]
        s["completed"] += 1
        s["wait_times"].append(wait_sec)
        s["latencies"].append(latency_sec)
        if ttft_sec > 0:
            s["ttfts"].append(ttft_sec)

def record_request_failed(profile: str, policy: str) -> None:
    _metrics["total_failed"] += 1
    if profile in _profile_stats:
        _profile_stats[profile]["failed"] += 1
    if policy in _policy_stats:
        _policy_stats[policy]["failed"] += 1

def get_metrics() -> Dict[str, Any]:
    active_total = sum(_active_requests.values())
    return {
        **_metrics,
        "queue_length": len(_queue),
        "active_requests": active_total,
        "backend_active": dict(_active_requests),
        "starvation_threshold_sec": starvation_guard.max_queue_wait_threshold_sec,
        "starvation_protection_enabled": starvation_guard.enabled,
        "requests_protected_by_aging": _metrics.get("requests_protected_by_aging", 0),
        "time_scale": TIME_SCALE,
        "backend_concurrency_limit": BACKEND_CONCURRENCY,
    }

def get_detailed_stats() -> Dict[str, Any]:
    return {
        "metrics": get_metrics(),
        "profiles": _profile_stats,
        "policies": _policy_stats,
        "backends": health_manager.get_status_summary(),
    }

def get_backends_status() -> Dict[str, Any]:
    return {
        name: {
            "active_requests": _active_requests.get(name, 0),
            "concurrency_limit": BACKEND_CONCURRENCY,
            "health": health_manager.get_state(name),
            "is_available": health_manager.is_available(name),
            "observed_ms_per_token": round(health_manager.get_observed_ms_per_token(name), 2),
        }
        for name in BACKEND_URLS
    }

def increment_active(backend_name: str) -> None:
    _active_requests[backend_name] = _active_requests.get(backend_name, 0) + 1

def decrement_active(backend_name: str) -> None:
    if _active_requests.get(backend_name, 0) > 0:
        _active_requests[backend_name] -= 1

def get_available_backends(backend_list: Optional[List[str]] = None) -> List[str]:
    """Return healthy backends that currently have an open concurrency slot."""
    targets = backend_list if backend_list is not None else list(BACKEND_URLS.keys())
    available = []
    for name in targets:
        if health_manager.is_available(name):
            if _active_requests.get(name, 0) < BACKEND_CONCURRENCY:
                available.append(name)
    return available

def choose_backend_for_request(
    policy: str,
    max_output_tokens: int,
    candidate_backends: Optional[List[str]] = None,
    prompt: str = "",
    profile: str = "normal",
    request_id: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """Select the optimal backend according to policy.
    Returns: (selected_backend_name, candidate_estimates_dict, rejected_reasons_dict)
    """
    candidates = candidate_backends if candidate_backends is not None else list(BACKEND_URLS.keys())
    estimates = {}
    rejected = {}

    eligible = []
    for b in candidates:
        if not health_manager.is_available(b):
            state = health_manager.get_state(b)
            rejected[b] = f"circuit_{state}"
            continue
        eligible.append(b)

    if not eligible:
        # Fall back to any backend if all are degraded/circuit-open
        eligible = candidates

    # 1. Least-Connection Routing (for FIFO, Token-Aware, Adaptive-Least-Conn, and Priority-Only)
    if policy in (POLICY_FIFO, POLICY_TOKEN_AWARE, POLICY_ADAPTIVE_LEAST_CONN, POLICY_PRIORITY_ONLY):
        for b in eligible:
            act = _active_requests.get(b, 0)
            ms_tok = health_manager.get_observed_ms_per_token(b)
            tps = round(1000.0 / max(ms_tok, 1.0), 2)
            estimates[b] = {
                "active": act,
                "ms_per_token": round(ms_tok, 2),
                "observed_tps": tps,
                "routing_score": -float(act),
            }
        best_b = min(eligible, key=lambda b: (_active_requests.get(b, 0), TOKEN_DELAY_MS.get(b, 100)))
        return best_b, estimates, rejected

    # 2. Cost-Aware Routing (for Adaptive, Adaptive-Oracle, Adaptive-Pred, SJF, and Ablations)
    best_b = eligible[0]
    best_est_completion = float("inf")

    for b in eligible:
        active = _active_requests.get(b, 0)
        ms_per_tok = health_manager.get_observed_ms_per_token(b)
        # Expected slot wait: active requests ahead in slot * estimated duration
        token_work_sec = (max_output_tokens * ms_per_tok) / 1000.0
        slot_wait_sec = active * token_work_sec
        # Startup connection overhead (simulated ~0.02s scaled)
        startup_overhead_sec = 0.02 * TIME_SCALE
        est_completion_sec = slot_wait_sec + startup_overhead_sec + token_work_sec
        observed_tps = round(1000.0 / max(ms_per_tok, 1.0), 2)
        routing_score = round(100.0 / max(est_completion_sec, 0.001) - (active * 10.0), 2)

        estimates[b] = {
            "active": active,
            "ms_per_token": round(ms_per_tok, 2),
            "observed_tps": observed_tps,
            "slot_wait_sec": round(slot_wait_sec, 3),
            "token_work_sec": round(token_work_sec, 3),
            "estimated_completion_sec": round(est_completion_sec, 3),
            "routing_score": routing_score,
        }

        if est_completion_sec < best_est_completion:
            best_est_completion = est_completion_sec
            best_b = b

    return best_b, estimates, rejected

# Priority Calculation functions
def compute_priority_only_score(req: Dict[str, Any], now: float) -> Tuple[float, Dict[str, Any]]:
    profile_score = PROFILE_SCORES.get(req.get("profile", "normal"), 50)
    return float(profile_score), {
        "profile_score": profile_score,
        "total_score": float(profile_score),
    }

def compute_sjf_score(req: Dict[str, Any], now: float) -> Tuple[float, Dict[str, Any]]:
    tokens = req.get("max_output_tokens", 5)
    # Negate tokens so smaller token request gets higher score
    score = -float(tokens)
    return score, {
        "tokens": tokens,
        "total_score": score,
    }

def compute_token_aware_score(req: Dict[str, Any], now: float) -> Tuple[float, Dict[str, Any]]:
    waiting = now - req["enqueued_at"]
    profile_score = PROFILE_SCORES.get(req["profile"], 50)
    tokens = req["max_output_tokens"]

    if starvation_guard.enabled:
        aging = waiting * starvation_guard.aging_boost_rate
        starvation = starvation_guard.starvation_bonus if waiting >= starvation_guard.max_queue_wait_threshold_sec else 0.0
    else:
        aging = 0.0
        starvation = 0.0

    score = profile_score + aging - (tokens * 0.2) + starvation
    breakdown = {
        "profile_score": profile_score,
        "aging_score": round(aging, 2),
        "token_penalty": round(tokens * 0.2, 2),
        "starvation_bonus": starvation,
        "total_score": round(score, 2),
    }
    return score, breakdown

def compute_adaptive_score(
    req: Dict[str, Any],
    now: float,
    policy_variant: str = POLICY_ADAPTIVE,
) -> Tuple[float, Dict[str, Any]]:
    waiting = now - req["enqueued_at"]
    profile_score = PROFILE_SCORES.get(req.get("profile", "normal"), 50)

    # Determine effective tokens for estimation
    if policy_variant == POLICY_ADAPTIVE_ORACLE and "sim_true_output_tokens" in req:
        tokens = req["sim_true_output_tokens"]
    elif policy_variant == POLICY_ADAPTIVE_PRED and "predicted_tokens" in req:
        tokens = req["predicted_tokens"]
    else:
        tokens = req.get("max_output_tokens", 5)

    avg_ms = 100.0 * TIME_SCALE
    est_cost_sec = (tokens * avg_ms) / 1000.0

    p_part = profile_score * ADAPTIVE_PROFILE_WEIGHT
    c_part = (est_cost_sec * ADAPTIVE_COST_WEIGHT) if policy_variant != POLICY_ADAPTIVE_NO_COST else 0.0

    if starvation_guard.enabled:
        w_part = (waiting * starvation_guard.aging_boost_rate) if policy_variant != POLICY_ADAPTIVE_NO_AGING else 0.0
        starvation = starvation_guard.starvation_bonus if (policy_variant != POLICY_ADAPTIVE_NO_AGING and waiting >= starvation_guard.max_queue_wait_threshold_sec) else 0.0

        # Deadline / Slack awareness (from starvation_guard & SLO_TARGETS)
        profile = req.get("profile", "normal")
        slo_completion = starvation_guard.slo_latency_target_sec if profile == "normal" else SLO_TARGETS.get(profile, {}).get("completion", starvation_guard.slo_latency_target_sec)
        slack = slo_completion - (waiting + est_cost_sec)
        slack_boost = 25.0 if slack < 0 else 0.0
    else:
        w_part = 0.0
        starvation = 0.0
        slack = 0.0
        slack_boost = 0.0

    score = p_part + w_part - c_part + starvation + slack_boost

    breakdown = {
        "profile_score": round(p_part, 2),
        "aging_score": round(w_part, 2),
        "cost_penalty": round(c_part, 2),
        "starvation_bonus": starvation,
        "slack_sec": round(slack, 3),
        "slack_boost": slack_boost,
        "effective_tokens": tokens,
        "total_score": round(score, 2),
    }
    return score, breakdown

def check_admission_control(profile: str) -> bool:
    """Returns True if request is admitted, False if shed (HTTP 429)."""
    from .. import config
    if not config.ENABLE_ADMISSION_CONTROL:
        return True
    q_len = len(_queue)
    if q_len >= 15 and profile == "batch":
        return False
    if q_len >= 25 and profile != "interactive":
        return False
    return True

def _is_request_eligible(req: Dict[str, Any], open_slots: Optional[List[str]] = None) -> bool:
    if open_slots is None:
        return True
    allowed = req.get("allowed_backends")
    if allowed is None and req.get("use_real_model"):
        allowed = ["model_a"]
    if allowed is not None:
        return any(b in allowed for b in open_slots)
    return True

def _get_eligible_candidates(req: Dict[str, Any], open_slots: List[str]) -> List[str]:
    allowed = req.get("allowed_backends")
    if allowed is None and req.get("use_real_model"):
        allowed = ["model_a"]
    if allowed is not None:
        return [b for b in open_slots if b in allowed]
    return open_slots

def _dequeue(policy: str, open_slots: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Select the best waiting request from _queue according to policy that is eligible for open_slots.
    Uses arrival_seq for deterministic tie-breaking.
    """
    if not _queue:
        return None

    # Filter by eligibility for open slots if provided
    eligible_indices = [
        idx for idx, req in enumerate(_queue)
        if _is_request_eligible(req, open_slots)
    ]
    if not eligible_indices:
        return None

    now = _now()

    if policy == POLICY_FIFO:
        # If starvation protection is enabled, check if any request has breached starvation threshold
        starved_candidates = []
        if starvation_guard.enabled:
            for idx in eligible_indices:
                r = _queue[idx]
                if (now - r["enqueued_at"]) >= starvation_guard.max_queue_wait_threshold_sec:
                    starved_candidates.append(idx)

        if starved_candidates:
            # Pick earliest arrived among starved requests
            best_idx = min(starved_candidates, key=lambda i: _queue[i]["arrival_seq"])
            selected = _queue[best_idx]
            del _queue[best_idx]
            wait_sec = now - selected["enqueued_at"]
            starvation_guard.record_protection_event(
                selected.get("request_id", ""), selected.get("profile", "normal"),
                wait_sec, starvation_guard.starvation_bonus
            )
            _metrics["requests_protected_by_aging"] = _metrics.get("requests_protected_by_aging", 0) + 1
            selected["score_breakdown"] = {"policy": "fifo", "arrival_seq": selected["arrival_seq"], "starvation_override": True}
            return selected

        # Pure arrival order among eligible requests
        best_idx = eligible_indices[0]
        best_seq = _queue[best_idx]["arrival_seq"]
        for idx in eligible_indices:
            req = _queue[idx]
            if req["arrival_seq"] < best_seq:
                best_seq = req["arrival_seq"]
                best_idx = idx
        selected = _queue[best_idx]
        del _queue[best_idx]
        selected["score_breakdown"] = {"policy": "fifo", "arrival_seq": selected["arrival_seq"]}
        return selected

    best_idx = eligible_indices[0]
    best_score = -float("inf")
    best_seq = float("inf")
    best_breakdown = {}

    for idx in eligible_indices:
        req = _queue[idx]
        req_policy = req.get("policy", policy)
        if req_policy == POLICY_PRIORITY_ONLY:
            score, breakdown = compute_priority_only_score(req, now)
        elif req_policy == POLICY_SJF:
            score, breakdown = compute_sjf_score(req, now)
        elif req_policy in (
            POLICY_ADAPTIVE,
            POLICY_ADAPTIVE_LEAST_CONN,
            POLICY_ADAPTIVE_ORACLE,
            POLICY_ADAPTIVE_PRED,
            POLICY_ADAPTIVE_NO_AGING,
            POLICY_ADAPTIVE_NO_COST,
        ):
            score, breakdown = compute_adaptive_score(req, now, policy_variant=req_policy)
        else:
            score, breakdown = compute_token_aware_score(req, now)

        seq = req["arrival_seq"]
        # Higher score wins; on tie, earlier arrival sequence wins
        if score > best_score or (score == best_score and seq < best_seq):
            best_score = score
            best_seq = seq
            best_idx = idx
            best_breakdown = breakdown

    selected = _queue[best_idx]
    del _queue[best_idx]
    selected["score_breakdown"] = best_breakdown

    if starvation_guard.enabled:
        aging = best_breakdown.get("aging_score", 0.0)
        starv = best_breakdown.get("starvation_bonus", 0.0)
        if aging > 0.0 or starv > 0.0:
            starvation_guard.record_protection_event(
                selected.get("request_id", ""), selected.get("profile", "normal"),
                now - selected["enqueued_at"], aging + starv
            )
            _metrics["requests_protected_by_aging"] = _metrics.get("requests_protected_by_aging", 0) + 1

    return selected

def _enqueue(request: Dict[str, Any]) -> None:
    global _arrival_seq_counter
    if len(_queue) >= QUEUE_MAX_SIZE:
        raise asyncio.QueueFull()
    _arrival_seq_counter += 1
    request["arrival_seq"] = _arrival_seq_counter
    request["enqueued_at"] = _now()
    _queue.append(request)
    if len(_queue) > _metrics.get("max_queue_length", 0):
        _metrics["max_queue_length"] = len(_queue)
    record_request_submitted(request.get("profile", "normal"), request.get("policy", POLICY_FIFO))

async def schedule_and_acquire(
    request_dict: Dict[str, Any],
    backend_urls: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, float, Dict[str, Any]]:
    """Core scheduling entry point.
    If concurrency slots are open and queue is empty, dispatches immediately.
    Otherwise, queues the request and suspends until a slot is freed.
    Returns: (backend_name, backend_url, queue_wait_seconds, scheduling_metadata)
    """
    urls = backend_urls if backend_urls is not None else BACKEND_URLS
    policy = request_dict.get("policy", POLICY_FIFO)
    tokens = request_dict.get("max_output_tokens", 5)
    prompt_text = request_dict.get("prompt", "")
    est_input = estimate_prompt_tokens(prompt_text)
    request_dict["estimated_input_tokens"] = est_input["estimated_tokens"]
    request_dict["token_estimation_method"] = est_input["method"]

    if "allowed_backends" not in request_dict:
        if request_dict.get("use_real_model"):
            request_dict["allowed_backends"] = ["model_a"]
        elif backend_urls is not None:
            request_dict["allowed_backends"] = list(backend_urls.keys())

    async with _queue_lock:
        open_slots = get_available_backends(list(urls.keys()))
        # Fast-path: if slot is open and no requests are waiting ahead, dispatch immediately
        if open_slots and len(_queue) == 0:
            backend_name, estimates, rejected = choose_backend_for_request(
                policy, tokens, open_slots,
                prompt=prompt_text, profile=request_dict.get("profile", "normal"),
                request_id=request_dict.get("request_id", "")
            )
            increment_active(backend_name)
            enqueued_at = _now()
            request_dict["enqueued_at"] = enqueued_at
            record_request_submitted(request_dict.get("profile", "normal"), policy)
            wait_sec = 0.0
            record_queue_wait(wait_sec)
            score_b = {"total_score": PROFILE_SCORES.get(request_dict.get("profile"), 50)}
            explanation = trace_manager.generate_explanation(
                policy=policy,
                profile=request_dict.get("profile", "normal"),
                selected_backend=backend_name,
                queue_wait=wait_sec,
                score_breakdown=score_b,
                candidate_estimates=estimates,
                rejected_reasons=rejected,
                success=True,
                prompt_tokens=request_dict["estimated_input_tokens"],
                max_output_tokens=tokens,
                fallback_occurred=bool(request_dict.get("fallback_occurred", False)),
            )
            metadata = {
                "score_breakdown": score_b,
                "candidate_estimates": estimates,
                "rejected_reasons": rejected,
                "dispatched_at": enqueued_at,
                "routing_explanation": explanation,
                "estimated_input_tokens": request_dict["estimated_input_tokens"],
                "routing_score": estimates.get(backend_name, {}).get("routing_score", 0.0),
            }
            return backend_name, urls[backend_name], wait_sec, metadata

        # Slow-path: slot contention exists; must queue and await slot availability
        event = asyncio.Event()
        request_dict["event"] = event
        _enqueue(request_dict)

    # Await dispatch signal
    await event.wait()

    # Request has been dequeued and assigned a backend by release_slot() or dispatcher
    backend_name = request_dict["assigned_backend"]
    wait_sec = request_dict["dispatched_at"] - request_dict["enqueued_at"]
    record_queue_wait(wait_sec)
    explanation = trace_manager.generate_explanation(
        policy=policy,
        profile=request_dict.get("profile", "normal"),
        selected_backend=backend_name,
        queue_wait=wait_sec,
        score_breakdown=request_dict.get("score_breakdown", {}),
        candidate_estimates=request_dict.get("candidate_estimates", {}),
        rejected_reasons=request_dict.get("rejected_reasons", {}),
        success=True,
        prompt_tokens=request_dict.get("estimated_input_tokens", 1),
        max_output_tokens=tokens,
        fallback_occurred=bool(request_dict.get("fallback_occurred", False)),
    )
    metadata = {
        "score_breakdown": request_dict.get("score_breakdown", {}),
        "candidate_estimates": request_dict.get("candidate_estimates", {}),
        "rejected_reasons": request_dict.get("rejected_reasons", {}),
        "dispatched_at": request_dict["dispatched_at"],
        "routing_explanation": explanation,
        "estimated_input_tokens": request_dict.get("estimated_input_tokens", 1),
        "routing_score": request_dict.get("candidate_estimates", {}).get(backend_name, {}).get("routing_score", 0.0),
    }
    url = urls.get(backend_name) or BACKEND_URLS.get(backend_name)
    return backend_name, url, wait_sec, metadata

async def release_slot(backend_name: str) -> None:
    """Invoked when a backend completes a request. Releases slot and dispatches next waiting request."""
    async with _queue_lock:
        decrement_active(backend_name)
        if not _queue:
            return

        # Check all available backends with open slots
        open_slots = get_available_backends()
        if not open_slots:
            return

        # Pop best eligible request from queue according to policy of the queue/request
        policy = _queue[0].get("policy", POLICY_FIFO)
        next_req = _dequeue(policy, open_slots=open_slots)
        if not next_req:
            return

        candidate_slots = _get_eligible_candidates(next_req, open_slots)
        chosen_b, estimates, rejected = choose_backend_for_request(
            next_req.get("policy", policy),
            next_req.get("max_output_tokens", 5),
            candidate_slots,
            prompt=next_req.get("prompt", ""),
            profile=next_req.get("profile", "normal"),
            request_id=next_req.get("request_id", ""),
        )
        increment_active(chosen_b)
        next_req["assigned_backend"] = chosen_b
        next_req["candidate_estimates"] = estimates
        next_req["rejected_reasons"] = rejected
        next_req["dispatched_at"] = _now()
        if "event" in next_req:
            next_req["event"].set()

async def dispatcher_loop() -> None:
    """Background liveness loop ensuring waiting requests are progressed if slots open."""
    while True:
        try:
            await asyncio.sleep(0.05)
            async with _queue_lock:
                if not _queue:
                    continue
                open_slots = get_available_backends()
                if not open_slots:
                    continue
                policy = _queue[0].get("policy", POLICY_FIFO)
                next_req = _dequeue(policy, open_slots=open_slots)
                if not next_req:
                    continue
                candidate_slots = _get_eligible_candidates(next_req, open_slots)
                chosen_b, estimates, rejected = choose_backend_for_request(
                    next_req.get("policy", policy),
                    next_req.get("max_output_tokens", 5),
                    candidate_slots,
                    prompt=next_req.get("prompt", ""),
                    profile=next_req.get("profile", "normal"),
                    request_id=next_req.get("request_id", ""),
                )
                increment_active(chosen_b)
                next_req["assigned_backend"] = chosen_b
                next_req["candidate_estimates"] = estimates
                next_req["rejected_reasons"] = rejected
                next_req["dispatched_at"] = _now()
                if "event" in next_req:
                    next_req["event"].set()
        except asyncio.CancelledError:
            break
        except Exception:
            pass

def get_queue_snapshot() -> List[Dict[str, Any]]:
    return list(_queue)

def get_starvation_guard_status() -> Dict[str, Any]:
    """Retrieve complete status of Starvation Guard, KPIs, queue items, and SLO monitor."""
    now = _now()
    cfg = starvation_guard.get_config()
    queue_items = list(_queue)
    inspected = starvation_guard.inspect_queue_items(queue_items, now)

    # Completed latencies across all profiles
    recent_latencies = []
    for p in _profile_stats.values():
        recent_latencies.extend(p.get("latencies", []))

    kpis = starvation_guard.compute_kpis(queue_items, now, _metrics, recent_latencies)
    slo_mon = starvation_guard.compute_slo_monitor(recent_latencies)

    return {
        "config": cfg,
        "kpis": kpis,
        "slo_monitor": slo_mon,
        "queued_requests": inspected,
        "queue_count": len(queue_items),
        "events_log": list(reversed(starvation_guard.starvation_events_log[-15:])),
    }
